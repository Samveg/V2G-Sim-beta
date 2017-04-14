import driving.basic_powertrain
import charging.uncontrolled
import charging.station
import result
import cPickle
import copy
import datetime


class Project(object):
    """V2G-Sim project holder. It contains all the vehicles, locations,
    car models and charging stations used in the project. It also includes
    statistics on input activities for the vehicles.

    Args:
        vehicles (list): vehicle objects
        car_models (list): car model objects
        locations (list): location objects
        charging_stations (list): charging_station objects
        description (string): description of the project
        vehicle_statistics (pandas.DataFrame): statistics for individual
            vehicles (distance traveled, ...)
        itinerary_statistics (pandas.DataFrame): statistics for specific
            itinerary combinaisons (number of vehicles, ...)
        timestep (int): simulation interval in [seconds]
    """

    def __init__(self, description='no description', timestep=60):
        self.vehicles = []
        self.car_models = []
        self.locations = []
        self.charging_stations = []
        self.description = description
        self.vehicle_statistics = None
        self.itinerary_statistics = None
        self.timestep = timestep
        self.date = datetime.datetime.now().replace(hour=0, minute=0,
                                                    second=0, microsecond=0)

    def check_integrity(self):
        """Launch tests on the project
        """
        pass

    def copy(self):
        """Deep copy the project and return the copy
        """
        return copy.deepcopy(self)

    def save(self, filename):
        """Save the project

        Args:
            filename (string): path to save the project (/../my_project.v2gsim)
        """
        with open(filename, "wb") as output:
            cPickle.dump(self, output, cPickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        """Load a project

        Args:
            filename (string): path to a project (/../my_project.v2gsim)
        """
        with open(filename, "rb") as input:
            project = cPickle.load(input)

        return project


class BasicCarModel(object):
    """ BasicCarModel describes a very basic car model using deterministic
    consumption for typical drivecycles. Equivalent to assigning consumption
    with average speed.

    Args:
        name (string): (required) name of the model
        driving (func): (required) algorithm to compute
            vehicle consumptionwhile driving
        maker (string): name of the maker
        year (string): year of the car model
        UDDS (float):  Consumption on the UDDS drivecycle [wh/km]
        HWFET (float): Consumption on the HWFET drivecycle [wh/km]
        US06 (float): consumption on the US06 drivecycle[wh/km]
        Delhi (float): consumption on the Delhi drivecycle [wh/km]
        battery_capacity (float): battery capacity in [Wh]
        battery_efficiency_charging (float): efficiency while charging
        battery_efficiency_discharging (float): efficiency while discharging
        maximum_SOC (float): maximum state of charge that the battery can reach
        decay (float): state of charge from which the battery charging
               profile is not linear
        minimum_power (float): maximum power rate for the battery [W]
        maximum_power (float): minumin power rate for the battery [W]
    """

    def __init__(self, name,
                 driving=driving.basic_powertrain.consumption,
                 maker=None, year=None, UDDS=145.83, HWFET=163.69, US06=223.62,
                 Delhi=138.3, maximum_SOC=0.95, decay=0.95,
                 battery_capacity=23832, battery_efficiency_charging=1.0,
                 battery_efficiency_discharging=1.0, maximum_power=6600,
                 minimum_power=-6600):
        self.name = name
        self.maker = maker
        self.year = year
        self.UDDS = UDDS  # [wh/km]
        self.HWFET = HWFET  # [wh/km]
        self.US06 = US06  # [wh/km]
        self.Delhi = Delhi  # [wh/km]
        self.battery_capacity = battery_capacity  # [Wh]
        self.battery_efficiency_charging = battery_efficiency_charging
        self.battery_efficiency_discharging = battery_efficiency_discharging
        self.maximum_SOC = maximum_SOC
        self.decay = decay
        self.minimum_power = minimum_power  # [W]
        self.maximum_power = maximum_power  # [W]
        self.driving = driving

    def __repr__(self):
        return "Basic car model: name({}) batteryCap({}Wh)".format(
            self.name, self.battery_capacity)


class Vehicle(object):
    """Vehicle represents a single car with a specific model and set of
    activities throughout the day. Vehicles keep track of their SOC as well
    as their stranding log.

    Args:
        id (int): unique id
        initial_SOC (float): initial state of charge [0, 1]
        SOC (list): state of charge of the battery along the time
        activities (list): activities throughout the day
            (parked, driving, parked, ...)
        weight (float): how representative is a vehicle [0, 1]
        valid_activities (boolean): True if activities don't leave any
            gap of time, the end of an activity must correspond with the start
            of the next one
        stranding_log (list): time at which the vehicle has stranded [hour]
        car_model (BasicCarModel): car model associated to the vehicle
        result_function (func): function describing the way results are saved
        result (any): result structure returned by the result function
        battery_model (BatteryModel): model to represent battery degradation
    """

    def __init__(self, index, car_model, initial_SOC=0.95):
        self.id = index
        self.car_model = car_model
        self.SOC = [initial_SOC]
        self.activities = []
        self.weight = 1
        self.valid_activities = False
        self.stranding_log = []
        self.result_function = result.save_vehicle_state
        self.result = None
        self.battery_model = None

    def check_activities(self, start_date, end_date):
        """Verify if every activity start at the end of the previous activity
        """
        # Check the starting date
        self.valid_activities = True
        if self.activities[0].start != start_date:
            self.valid_activities = False
            return False

        # Check the start - end match of activities
        for i in range(0, len(self.activities) - 1):
            if self.activities[i].end != self.activities[i + 1].start:
                self.valid_activities = False
                return False

        # Check for positive duration
        for i in range(0, len(self.activities)):
            if self.activities[i].end < self.activities[i].start:
                self.valid_activities = False
                return False

        # Check the ending date match
        if self.activities[-1].end != end_date:
            self.valid_activities = False
            return False

        return True

    def __repr__(self):
        string = ("Vehicle: id({}) car_model.name({}) " +
                  "initial_SOC({})").format(self.id,
                                            self.car_model.name,
                                            self.SOC[0])
        for activity in self.activities:
            string += "\n" + activity.__repr__()
        return string + "\n"


class Activity(object):
    """ Activity is an abstract class that is implemented in Driving and Parked.

    Args:
        start (datetime): start time of the activity
        end (datetime): end time of the activity
    """

    def __init__(self, start, end):
        self.start = start
        self.end = end


class Parked(Activity):
    """ Parked activity inherits from Activity. It represents a car parked at
    a location.

    Args:
        location (Location): (required) location object at which the vehicle
            is parked
        charging_station (ChargingStation): charging station at which a
            vehicle is plugged
    """

    def __init__(self, start, end, location,
                 charging_station=None):
        Activity.__init__(self, start, end)
        self.location = location
        self.charging_station = charging_station

    def __repr__(self):
        string = ("Parked Activity: start({}) end({}) " +
                  "location({})").format(
                      self.start, self.end,
                      self.location.category)
        if self.charging_station:
            string += " charging_station({}W)".format(
                self.charging_station.maximum_power)
        return string


class Driving(Activity):
    """ Driving represents a drivecycle, it inherits from Activity.
    Note that units differ from the SI units since distance is in [km]
    and speed in [km/h]. Data is indexed with at project timestep rate.

    Args:
        distance (float): distance traveled in [km]
        speed (list): drive cycle speed in [km/h]
        terrain (list): grade along the drive cycle in [rad]
    """

    def __init__(self, start, end, distance):
        Activity.__init__(self, start, end)
        self.distance = distance
        self.speed = []
        self.terrain = []

    def __repr__(self):
        return "Driving: start({}) end({}) distance({}km)".format(
            self.start, self.end, self.distance)


class Location(object):
    """ Location physical place or a category of place.

    Args:
        name (string): (required) name of the location
        category (string): (required) type of location (Home, Work, ...)
        position (tuple): GPS position
        result (pandas.DataFrame): results
        result_function (func): function to create and update the result
            data frame
        available_charging_station (pandas.DataFrame): describe the
            availability of charging stations (charging_station,
            probability, available, total)
        assign_charging_station_function (func): function to assign a
            charging station to an incoming vehicle
    """

    def __init__(self, category, name, position=(0, 0),
                 assign_charging_station=charging.station.randomly_assign,
                 result_function=result.save_location_state):
        self.category = category
        self.name = name
        self.position = position
        self.result = None
        self.result_function = result_function
        self.available_charging_station = None
        self.assign_charging_station = assign_charging_station

    def __repr__(self):
        return ("Location: type({})" +
                " name({}) GPS({}))\n").format(self.category,
                                               self.name, self.position)


class ChargingStation(object):
    """ Charging station represents a type of infrastructure

    Args:
        name (string): name associated with the infrastructure
        charging (func): function to control
            the charging behavior
        post_simulation (boolean): True station can be subject to post processing
        maximum_power (float): maximum rate at which a vehicle can charge
        minimum_power (float): minimum rate at which a vehicle can charge
    """

    def __init__(self, charging=charging.uncontrolled.consumption,
                 maximum_power=1440, minimum_power=-1440, post_simulation=False,
                 name='charger'):
        self.name = name
        self.post_simulation = post_simulation
        self.maximum_power = maximum_power
        self.minimum_power = minimum_power
        self.charging = charging

    def __repr__(self):
        return ("power rate {}W\n").format(self.maximum_power)


class ItineraryBin(object):
    """Hold statistics for specific itineraries

    Args:
        vehicles (list): vehicles having this itinerary combinaison
        statistics (pandas.DataFrame): statistics for individual activities (
            duration, distance, ...)
        weight (float): weight associated to this itinerary combinaison
    """

    def __init__(self, vehicles):
        self.vehicles = vehicles
        self.statistics = None
        self.weight = None
        # self.parameters = {'startTime': [], 'meanSpeed': [],
        #                    'duration': [], 'durationParked': []}

    def __repr__(self):
        return ("This ItineraryBin is composed of {} vehicles with {} " +
                "locations along the day \n").format(len(self.vehicle),
                                                     len(self.locationName))
