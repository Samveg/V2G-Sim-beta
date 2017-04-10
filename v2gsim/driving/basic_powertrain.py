from __future__ import division
import numpy


def consumption(activity, vehicle, nb_interval, timestep, verbose=False):
    """Calculate the consumption when a vehicle is driving

    Args:
        activity (Driving): a driving activity
        vehicle (Vehicle): a Vehicle object to update with the driving
            activity consumption
        nb_interval (int): number of timestep for the driving activity
        timestep (int): calculation timestep

    Return:
        SOC (list): state of charge/
        power_demand (list): power demand/
        stranded (boolean): True if the vehicle run out of charge during the
        activity
        detail (any type): optional data
    """

    # Calculate the duration
    duration = nb_interval * timestep / 3600

    # Get the mean speed
    if duration > 0:
        mean_speed = activity.distance / duration
    else:
        if verbose:
            print('Activity duration is shorter than timestep')
        return [], [], False, False

    # Get the energy Wh consumption per km
    energyConsumption = _drivecycle_energy_per_distance(vehicle.car_model, mean_speed)

    # Set the total energy needed for the trip (Wh)
    energy = activity.distance * energyConsumption

    # Get last SOC value ((SOCinit*batteryCap)-energyForTheTrip)/batteryCap
    endSOC = (((vehicle.SOC[-1] * vehicle.car_model.battery_capacity) - energy) /
              vehicle.car_model.battery_capacity)

    # Check if not below stranding threshold
    stranded = False
    if endSOC < 0.1:
        stranded = True
        if endSOC < 0:
            endSOC = 0

    # Get SOC list with last SOC-intervalSOC, SOC-2*intervalSOC ...endSOC
    SOC = list(numpy.linspace(vehicle.SOC[-1], endSOC, num=nb_interval, endpoint=True))

    # Set the power demand for this driving activity [J] /
    constant_power_demand = energy * 3600 / (nb_interval * timestep)
    power_demand = [constant_power_demand] * nb_interval

    return SOC, power_demand, stranded, False


def road_consumption_plus_ancillary_load(activity, vehicle, nb_interval, timestep, verbose=False):
    """Calculate the consumption when a vehicle is driving

    Args:
        activity (Driving): a driving activity
        vehicle (Vehicle): a Vehicle object to update with the driving
            activity consumption
        nb_interval (int): number of timestep for the driving activity
        timestep (int): calculation timestep

    Return:
        SOC (list): state of charge/
        power_demand (list): power demand/
        stranded (boolean): True if the vehicle run out of charge during the
        activity
        detail (any type): optional data
    """

    # Calculate the duration
    duration = nb_interval * timestep / 3600

    # Get the mean speed
    if duration > 0:
        mean_speed = activity.distance / duration
    else:
        if verbose:
            print('Activity duration is shorter than timestep')
        return [], [], False, False

    # Get the energy Wh consumption per km
    energyConsumption = _drivecycle_energy_per_distance(vehicle.car_model, mean_speed)

    # Set the total energy needed for the trip (Wh)
    energy = (activity.distance * energyConsumption +
              duration * vehicle.car_model.ancillary_load)

    # Get last SOC value ((SOCinit*batteryCap)-energyForTheTrip)/batteryCap
    endSOC = (((vehicle.SOC[-1] * vehicle.car_model.battery_capacity) - energy) /
              vehicle.car_model.battery_capacity)

    # Check if not below stranding threshold
    stranded = False
    if endSOC < 0.1:
        stranded = True
        if endSOC < 0:
            endSOC = 0

    # Get SOC list with last SOC-intervalSOC, SOC-2*intervalSOC ...endSOC
    SOC = list(numpy.linspace(vehicle.SOC[-1], endSOC, num=nb_interval, endpoint=True))

    # Set the power demand for this driving activity [J] /
    constant_power_demand = energy * 3600 / (nb_interval * timestep)
    power_demand = [constant_power_demand] * nb_interval

    return SOC, power_demand, stranded, False


def _drivecycle_energy_per_distance(car_model, mean_speed):
    # Get the energy consumption per km

    # UDDS (Urban Dynamometer Driving Schedule) 12.07km
    # with maximum speed 91.25km/h and average speed
    # of 31.5km/h (19.6mph)
    UDDS = 31.5  # (km/h)

    # HWFET (Highway Fuel Economy Test) 16.45km
    # with average speed 77.7km/h (48.3mph)
    HWFET = 77.7  # (km/h)

    # US06 12.8km average speed 77.9km/h (48.4mph)with
    # a maximum speed at 129.2km/h
    # --> above HWFET

    # Delhi (no information) congested city drive cycle
    # --> below UDDS

    # Determine the right consumption (!) Need more linearity
    if mean_speed < UDDS:
        # Consumption for a slow driving cycle
        energy_consumption = car_model.UDDS
    elif mean_speed >= UDDS and mean_speed <= HWFET:
        # Mix between a UDDS and a HWFET drice cycle consumption
        energy_consumption = car_model.HWFET
    elif mean_speed > HWFET:
        # Consumption for a fast driving cycle
        energy_consumption = car_model.US06

    return energy_consumption
