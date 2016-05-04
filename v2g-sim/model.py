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
        self.vehicles = None
        self.car_models = None
        self.locations = None
        self.charging_stations = None
        self.description = description
        self.vehicle_statistics = None
        self.itinerary_statistics = None
        self.timestep = timestep

    def check_integrity(self):
        """Launch tests on the project
        """
        pass

    def save(self, filename):
        """Save the project

        Args:
            filename (string): path to save the project (/../my_project.v2gsim)
        """
        pass

    def load(self, filename):
        """Load a project

        Args:
            filename (string): path to a project (/../my_project.v2gsim)
        """
        pass


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
    """

    def __init__(self, index, car_model, initial_SOC=0.95):
        self.id = index
        self.carModel = car_model
        self.SOC = [initial_SOC]
        self.activities = []
        self.weight = 1
        self.valid_activities = False
        self.strandingLog = []

    def check_activities(self):
        """Verify if every activity start at the end of the previous activity
        """
        self.valid_activities = True
        if self.activities[0].start != 0:
            self.valid_activities = False

        for i in range(0, len(self.activities) - 1):
            if self.activities[i].end != self.activities[i + 1].start:
                self.valid_activities = False

        if self.activities[-1].end != 24:
            self.valid_activities = False

    def set_activities(self, parked, driving):
        """Set activities in order and check if the order has been respected
        """
        self.activities = []
        # Append activities in the right order into self.activities
        for i in range(0, len(driving)):
            self.activities.append(parked[i])
            self.activities.append(driving[i])
        self.activities.append(parked[-1])
        self.check_activities()

    def __repr__(self):
        string = ("Vehicle: id({}) carModel.name({}) initSOC({}) " +
                  "outputInterval({}) \n").format(self.id,
                                                  self.carModel.name,
                                                  self.SOC,
                                                  self.outputInterval)
        for activity in self.activities:
            string += "\n" + activity.__repr__()
        return string


class BasicCarModel(object):
    """ BasicCarModel describes a very basic car model using deterministic
    consumption for typical drivecycles. Equivalent to assigning consumption
    with average speed.

    Args:
        name (string): (required) name of the model
        driving_function (func): (required) algorithm to compute
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

    def __init__(self, name, driving_function, maker=None, year=None,
                 UDDS=145.83, HWFET=163.69, US06=223.62, Delhi=138.3,
                 maximum_SOC=0.95, decay=0.95, battery_capacity=23832,
                 battery_efficiency_charging=1.0,
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
        self.driving_function = driving_function

    def __repr__(self):
        return "Basic car model: name({}) batteryCap({}Wh)".format(
            self.name, self.battery_capacity)


class Activity(object):
    """ Activity is an abstract class that is implemented in Driving and Parked.
    Data is indexed with the project timestep.

    Args:
        start (float): start time of the activity in hours [h]
        end (float): end time of the activity in hours [h]
        power_demand (list): he power consumption during the activity [W]
    """

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.power_demand = []


class Parked(Activity):
    """ Parked activity inherits from Activity. It represents a car parked at
    a location.

    Args:
        location (Location): (required) location object at which the vehicle
            is parked
        plugged_in (boolean): True if the vehicle is plugged to a charging
            infrastructure
        charging_station (ChargingStation): charging station at which a
            vehicle is plugged
    """

    def __init__(self, start, end, location, plugged_in=False,
                 charging_station=None):
        Activity.__init__(self, start, end)
        self.location = location
        self.plugged_in = plugged_in
        self.charging_station = charging_station

    def __repr__(self):
        return ("Parked Activity: start({}) end({}) location({}) " +
                "pluggedIn({}) chargingInfra({})").format(
                    self.start, self.end,
                    self.location.category,
                    self.plugged_in,
                    self.charging_station.maximum_power)


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
        return "Driving: start({}) end({}) distance({})".format(
            self.start, self.end, self.distance)


class Location(object):
    """ Location physical place or a category of place.

    Args:
        name (string): (required) name of the location
        category (string): (required) type of location (Home, Work, ...)
        position (tuple): GPS position
        power_demand (list): power demand [W]
    """

    def __init__(self, category, name, position=(0, 0)):
        self.category = category
        self.name = name
        self.position = position
        self.power_demand = []
        self.available_charging_station = None

    def __repr__(self):
        return ("Location: type({})" +
                " name({}) GPS({}))\n").format(self.category,
                                               self.name, self.position)


class ChargingStation(object):
    """ Charging station represents a type of infrastructure

    Args:
        name (string): name associated with the infrastructure
        charging_function (func): (required) function to control
            the charging behavior
        v2g (boolean): True to allow potential flow from vehicle to the grid
        maximum_power (float): maximum rate at which a vehicle can charge
        minimum_power (float): minimum rate at which a vehicle can charge
    """

    def __init__(self, charging_function, maximum_power=1440,
                 minimum_power=-1440, v2g=False, name='charger'):
        self.name = name
        self.v2g = v2g
        self.maximum_power = maximum_power
        self.minimum_power = minimum_power
        self.charging_function = charging_function

    def __repr__(self):
        return ("Charging infrastructure: v2g({}) powerRateMax({})" +
                "powerRateMin({}) chargingFunc({})\n").format(
                    self.v2g, self.maximum_power,
                    self.minimum_power,
                    str(self.charging_function))


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
