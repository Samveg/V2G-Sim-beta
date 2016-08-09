from __future__ import division
import random
import datetime
import v2gsim


def demand_response(parked, vehicle, nb_interval, timestep, option):
    """EV Demand response function. Determines how much EV charging load can be shed during a demand response event
    without adversely affecting driver mobility needs.

    Args:
        vehicle (Vehicle): Vehicle object to get current SOC and physical constraints (maximum SOC, ...)
        parked (Parked): Parked activity to get available charging station and charging strategy
        option (dict): DR parameters 'startDR', 'endDR', 'post_DR_window_fraction', 'thresholdSOC' and 'date_limit'

    Return:
        SOC and powerDemand as a list
    """

    # randomly pick the end of the DR event
    endDR = option['endDR']
    endDR += datetime.timedelta(seconds=(random.random() * ((option['endDR'] - option['startDR']).total_seconds() * option['post_DR_window_fraction'])))

    # Decide if the activity is concerned about the DR event
    DR_right_side_activity = option['startDR'] <= parked.start <= option['endDR']
    DR_left_side_activity = option['startDR'] <= parked.end <= option['endDR']
    DR_in_activity = option['startDR'] >= parked.start and option['endDR'] <= parked.end
    # DR_contains_activity is True for both DR_right and DR_left, so the 4 possibilities are covered

    if (DR_right_side_activity or DR_left_side_activity or DR_in_activity):
        # Calculate the duration of the DR event in the parked activity
        if option['startDR'] >= parked.start:
            start = option['startDR']
            insertIndex = int((start - parked.start).total_seconds() / timestep)
        else:
            start = parked.start
            insertIndex = 0
        end = min(endDR, parked.end)

        nbIntervalOutDR = (int((start - parked.start).total_seconds() / timestep) +
                           int((parked.end - end).total_seconds() / timestep))

        nbIntervalDR = int((end - start).total_seconds() / timestep)

        # <-- sometimes int() reduce size by one, investigate further
        if nb_interval != (nbIntervalOutDR + nbIntervalDR):
            nbIntervalDR += 1

        # Get the energy for the non DR event part
        SOCNoDR, powerDemandNoDR = v2gsim.charging.uncontrolled.consumption(parked, vehicle, nbIntervalOutDR, timestep, option)

        # Add the energy obtained with the normal charging
        removeCounter = 0
        if len(SOCNoDR) > 0:
            vehicle.SOC.append(SOCNoDR[-1])
            removeCounter += 1

        # Lookup the next activities and their consumption
        tempSOCList = [vehicle.SOC[-1]]
        for activity in vehicle.activities:
            if activity.start > parked.start and activity.start < option['date_limit']:
                temp_nb_interval = int((activity.end - activity.start).total_seconds() / timestep)
                # Compute consumption for driving and parked activities
                if isinstance(activity, v2gsim.model.Driving):
                    SOC, powerDemandTemp, stranded, detail = vehicle.car_model.driving(activity,
                                                                                       vehicle,
                                                                                       temp_nb_interval,
                                                                                       timestep)
                    if len(SOC) > 0:
                        vehicle.SOC.append(SOC[-1])
                        tempSOCList.append(SOC[-1])
                        removeCounter += 1

                elif isinstance(activity, v2gsim.model.Parked):
                    SOC, powerDemandTemp = v2gsim.charging.uncontrolled.consumption(activity,
                                                                                    vehicle,
                                                                                    temp_nb_interval,
                                                                                    timestep,
                                                                                    option)
                    if len(SOC) > 0:
                        vehicle.SOC.append(SOC[-1])
                        tempSOCList.append(SOC[-1])
                        removeCounter += 1

        # Clean the SOC variables appended previously (to make our life easier)
        for counter in range(0, removeCounter):
            del vehicle.SOC[-1]

        # Get the minimum value of energy
        minSOC = min(tempSOCList)

        # Get the corresponding power consumption
        if minSOC <= option['thresholdSOC']:
            # Calculate the missing energy to reach threshold and get the power rate
            powerRateDR = _constant_power_to_get_min_SOC(option['thresholdSOC'], minSOC, vehicle.car_model.battery_capacity,
                                                         nb_interval * timestep)
            # Get charging rate
            powerRateDR = min(powerRateDR, vehicle.car_model.maximum_power, parked.charging_station.maximum_power)
            # Append power
            powerDemandDR = [powerRateDR for i in range(0, nbIntervalDR)]
        else:
            # Append power
            powerDemandDR = [0 for i in range(0, nbIntervalDR)]

        # Insert DR event charging
        powerDemandNoDR[insertIndex:insertIndex] = powerDemandDR
        powerDemand = powerDemandNoDR

        # Calculate SOC - update powerRate if SOC max is reached
        SOC, powerDemand = _SOC_DR(vehicle, powerDemand, timestep)
    else:
        # Get the energy for the non DR event part
        SOC, powerDemand = v2gsim.charging.uncontrolled.consumption(parked, vehicle, nb_interval, timestep, option)

    # This last piece of code is checking that the lenght of the vectors returned is correct <-- Should not be a problem
    if len(SOC) > nb_interval or len(powerDemand) > nb_interval:
        print("Oops")
        SOC = SOC[0:nb_interval]
        powerDemand = powerDemand[0:nb_interval]
    return SOC, powerDemand


def _constant_power_to_get_min_SOC(threshold, lowestSOC, batteryCap, duration):
    # threshold is always bigger than lowestSOC, battery capacity is in Wh, duration in seconds
    return (threshold - lowestSOC) * batteryCap * 3600 / duration  # conversion from Wh to Joules


def _SOC_DR(vehicle, powerDemand, timestep):
    # Note: there is no decay here
    SOC = [vehicle.SOC[-1]]

    # Go through the newly generated power consumptions
    for index, power in enumerate(powerDemand):
        SOC.append(_SOC_increase(SOC[-1], power, timestep, vehicle.car_model.battery_capacity))
        # If SOC is above maximum SOC
        if SOC[-1] > vehicle.car_model.maximum_SOC:
            SOC[-1] = vehicle.car_model.maximum_SOC
            powerDemand[index] = 0

    del SOC[0]  # Remove initial SOC added line 216
    return SOC, powerDemand


def _SOC_increase(lastSOC, powerRate, duration, batteryCap):
    # lastSOC (1<x<0), powerRate (Watt), duration (seconds), batteryCap (wh)
    return lastSOC + (powerRate * duration / (batteryCap * 3600))  # batteryCap from Wh to Joules


def Q_consumption(activity, vehicle, nb_interval, timestep, charging_option):
    """Calculate the consumption of a plugged in vehicle which charge Q or
    the remaining energy it needs if it's inferior than Q.

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
    # Get duration in [seconds] of activity
    duration = nb_interval * timestep
    if duration <= 0:
        return [], []

    # Get Q the energy in [Wh] to furnish under the L1 charger assumption
    Q = 1440 * duration / 3600

    # Take the minimum of Q or remaining energy to charge [Wh]
    if (Q + vehicle.SOC[-1] * vehicle.car_model.battery_capacity > 
        vehicle.car_model.battery_capacity * vehicle.car_model.maximum_SOC):
        Q = (vehicle.car_model.battery_capacity * vehicle.car_model.maximum_SOC -
            vehicle.SOC[-1] * vehicle.car_model.battery_capacity)

    # Get maximum achievable power
    maximum_power = min(activity.charging_station.maximum_power,
                        vehicle.car_model.maximum_power)

    # Get nominal power
    nominal_power = Q * 3600 / duration  # Q from Wh to Joule
    if nominal_power > maximum_power:
        print('Error nominal power too large ' +
              str(nominal_power) + ' > ' + str(maximum_power))

    power_at_battery = nominal_power * vehicle.car_model.battery_efficiency_charging
    battery_capacity = vehicle.car_model.battery_capacity * 3600  # from Wh to J

    SOC = [vehicle.SOC[-1]]
    power_demand = []

    # For the duration of the activity
    for i in range(0, nb_interval):
        # If the car still needs to charge
        if SOC[-1] < vehicle.car_model.maximum_SOC:
            # Set the power demand to be the charger station power
            power_demand.append(nominal_power)
            # SOC [0,1] + (power_demand [W] * timestep [s] / totalCap [J])
            SOC.append(SOC[-1] + (power_at_battery * timestep / battery_capacity))
        # Vehicle is not charging
        else:
            power_demand.append(0)
            SOC.append(SOC[-1])

    del SOC[0]  # removed initial value added 17 line above
    return SOC, power_demand


def follow_signal(activity, vehicle, nb_interval, timestep, charging_option):
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
    signal = charging_option[activity.start:activity:end].signal.tolist()

    SOC = [vehicle.SOC[-1]]
    power_demand = []

    # For the duration of the activity
    for i in range(0, nb_interval):
        # If the car still needs to charge
        if SOC[-1] < vehicle.car_model.maximum_SOC:
            # Set the power demand to be the charger station power
            power_demand.append(maximum_power * signal[i])
            # SOC [0,1] + (power_demand [W] * timestep [s] / totalCap [J])
            SOC.append(SOC[-1] + (power_at_battery * signal[i] * timestep / battery_capacity))
        # Vehicle is not charging
        else:
            power_demand.append(0)
            SOC.append(SOC[-1])

    del SOC[0]  # removed initial value added 17 line above
    return SOC, power_demand