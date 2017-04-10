from __future__ import division
import random

def consumption(activity, vehicle, nb_interval, timestep, charging_option):
    """Calculate the consumption of a plugged in vehicle if no control is
    apply (charge ASAP until full battery).

    Args:
        activity (Parked): Parked activity to get charging station capabilities
        vehicle (Vehicle): Vehicle object to get current SOC and physical
            constraints (maximum SOC, ...)
        nb_interval (int): number of timestep for the parked activity
        timestep (int): calculation timestep
        charging_option (any): not used

    Returns:
        SOC (list): state of charge
        power_demand (list): power demand
    """

    maximum_power = min(activity.charging_station.maximum_power,
                        vehicle.car_model.maximum_power)
    power_at_battery = maximum_power * vehicle.car_model.battery_efficiency_charging
    battery_capacity = vehicle.car_model.battery_capacity * 3600  # from Wh to J

    SOC = [vehicle.SOC[-1]]
    power_demand = []

    # For the duration of the activity
    for i in range(0, nb_interval):
        # If the car still needs to charge
        if SOC[-1] < vehicle.car_model.maximum_SOC:
            # Set the power demand to be the charger station power
            power_demand.append(maximum_power)
            # SOC [0,1] + (power_demand [W] * timestep [s] / totalCap [J])
            SOC.append(SOC[-1] + (power_at_battery * timestep / battery_capacity))
        # Vehicle is not charging
        else:
            power_demand.append(0)
            SOC.append(SOC[-1])

    del SOC[0]  # removed initial value added 17 line above
    return SOC, power_demand


def charge_soc_dependent(activity, vehicle, nb_interval, timestep, charging_option):
    """Calculate the consumption of a plugged in vehicle if no control is
    apply (charge ASAP until full battery).

    Args:
        activity (Parked): Parked activity to get charging station capabilities
        vehicle (Vehicle): Vehicle object to get current SOC and physical
            constraints (maximum SOC, ...)
        nb_interval (int): number of timestep for the parked activity
        timestep (int): calculation timestep
        charging_option (any): not used

    Returns:
        SOC (list): state of charge
        power_demand (list): power demand
    """
    def probability_to_plug(level, activity):
        """Return a probability [0, 1] that someone use the charging infrastructure"""
        soc_charging = activity.location.soc_charging
        soc_no_charging = activity.location.soc_no_charging
        if level < soc_charging:
            return 1.0
        elif level <= soc_no_charging:
            return (1.0 / (soc_charging - soc_no_charging)) * level + soc_no_charging / (soc_no_charging - soc_charging)
        else:
            return 0.0

    # Decide to charge or not
    charge = random.random() <= probability_to_plug(vehicle.SOC[-1], activity)
    if not charge:
        return [vehicle.SOC[-1]] * nb_interval, [0] * nb_interval

    maximum_power = min(activity.charging_station.maximum_power,
                        vehicle.car_model.maximum_power)
    power_at_battery = maximum_power * vehicle.car_model.battery_efficiency_charging
    battery_capacity = vehicle.car_model.battery_capacity * 3600  # from Wh to J

    SOC = [vehicle.SOC[-1]]
    power_demand = []

    # For the duration of the activity
    for i in range(0, nb_interval):
        # If the car still needs to charge
        if SOC[-1] < vehicle.car_model.maximum_SOC:
            # Set the power demand to be the charger station power
            power_demand.append(maximum_power)
            # SOC [0,1] + (power_demand [W] * timestep [s] / totalCap [J])
            SOC.append(SOC[-1] + (power_at_battery * timestep / battery_capacity))
        # Vehicle is not charging
        else:
            power_demand.append(0)
            SOC.append(SOC[-1])

    del SOC[0]  # removed initial value added 17 line above
    return SOC, power_demand
