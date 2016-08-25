from __future__ import division
import numpy
import pandas
import model

def save_location_state(location, timestep, date_from, date_to,
                        vehicle=None, activity=None,
                        power_demand=None, SOC=None, nb_interval=None,
                        init=False, run=False, post=False):
    """Save local results from a parked activity during running
    time. If date_from and date_to, set a fresh pandas DataFrame at locations.

    Args:
        location (Location): location
        timestep (int): calculation timestep
        date_from (datetime.datetime): date to start recording power demand
        date_to (datetime.datetime): date to end recording power demand
        vehicle (Vehicle): vehicle
        activity (Parked): parked activity
        power_demand (list): power demand from parked activity
        SOC (list): state of charge from the parked activity
        nb_interval (int): number of timestep for the parked activity

    Example:
        >>> # Initialize a result DataFrame for each location
        >>> save_power_demand_at_location(location, timestep, date_from=some_date,
                                          date_to=other_date)
        >>> # Save data during run time
        >>> save_power_demand_at_location(location, timestep, vehicle, activity,
                                          power_demand, nb_interval)
    """
    if run:
        activity_index1, activity_index2, location_index1, location_index2, save = _map_index(
            activity.start, activity.end, date_from, date_to, len(power_demand),
            len(location.result['power_demand']), timestep)

        # Save a lot of interesting result
        if save:
            location.result['power_demand'][location_index1:location_index2] += (
                power_demand[activity_index1:activity_index2])

            # Add 'number_of_vehicle_parked' in the initialization section
            location.result['number_of_vehicle_parked'][location_index1:location_index2] += 1

            # Examples:
            # Add 'available_energy' in the initialization section
            # Then location.result['available_energy'][location_index1:location_index2] += (
            #          [soc * vehicle.car_model.battery_capacity
            #           for soc in SOC[activity_index1:activity_index2]])

    elif init:
        # Initiate a dictionary of numpy array to hold result (faster than DataFrame)
        location.result = {'power_demand': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                           'number_of_vehicle_parked': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep))}

    elif post:
        # Convert location result back into pandas DataFrame (faster that way)
        i = pandas.date_range(start=date_from, end=date_to,
                              freq=str(timestep) + 's', closed='left')
        location.result = pandas.DataFrame(index=i, data=location.result)


def location_potential_power_demand(location, timestep, date_from, date_to,
                                    vehicle=None, activity=None,
                                    power_demand=None, SOC=None, nb_interval=None,
                                    init=False, run=False, post=False):
    """Save local power demand for ASAP, 50percent ASAP, nominal,
    50percent ALAP, ALAP.

    Args:
        location (Location): location
        timestep (int): calculation timestep
        date_from (datetime.datetime): date to start recording power demand
        date_to (datetime.datetime): date to end recording power demand
        vehicle (Vehicle): vehicle
        activity (Parked): parked activity
        power_demand (list): power demand from parked activity
        SOC (list): state of charge from the parked activity
        nb_interval (int): number of timestep for the parked activity
    """
    if run:
        activity_index1, activity_index2, location_index1, location_index2, save = _map_index(
            activity.start, activity.end, date_from, date_to, len(power_demand),
            len(location.result['nominal']), timestep)

        # Save a lot of interesting result
        if save:
            # Get duration in [seconds] of activity
            duration = nb_interval * timestep
            if duration <= 0:
                return

            # Get Q the energy in [Wh] to furnish under the L1 charger assumption
            Q = 1440 * duration / 3600

            # Take the minimum of Q or remaining energy to charge [Wh]
            if (Q + SOC[0] * vehicle.car_model.battery_capacity > 
                vehicle.car_model.battery_capacity * vehicle.car_model.maximum_SOC):
                Q = (vehicle.car_model.battery_capacity * vehicle.car_model.maximum_SOC -
                    SOC[0] * vehicle.car_model.battery_capacity)

            # Get maximum achievable power
            maximum_power = min(activity.charging_station.maximum_power,
                                vehicle.car_model.maximum_power)
            # If the vehicle is not plugged then just leave the function
            if maximum_power <= 0:
                return

            # Get nominal power
            nominal_power = Q * 3600 / duration  # Q from Wh to Joule
            if nominal_power > maximum_power:
                print('Error in potential power demand: nominal power too large ' +
                      str(nominal_power) + ' > ' + str(maximum_power))

            # Get the power at 50% of maximum power - nominal
            mid_power = 0.5 * (maximum_power - nominal_power) + nominal_power

            # Get the power vector associated with how long one can charge at a certain power
            time_at_maximum = [maximum_power] * int(Q * 3600 / (maximum_power * timestep))
            time_at_mid = [mid_power] * int(Q * 3600 / (mid_power * timestep))

            # Create the lists of potential power consumption
            ASAP = list(time_at_maximum)
            ASAP.extend([0] * int(nb_interval - len(time_at_maximum)))

            ALAP = [0] * int(nb_interval - len(time_at_maximum))
            ALAP.extend(time_at_maximum)

            nominal = [nominal_power] * int(nb_interval)

            ASAP_nominal = list(time_at_mid)
            ASAP_nominal.extend([0] * int(nb_interval - len(time_at_mid)))

            ALAP_nominal = [0] * int(nb_interval - len(time_at_mid))
            ALAP_nominal.extend(time_at_mid)

            # Double check that the potential consumption have the same lenght as the actual one
            for potential_consumption in [ASAP, ALAP, nominal, ASAP_nominal, ALAP_nominal]:
                if len(potential_consumption) != nb_interval:
                    print('Error in potential power demand: lenght does not match nb_interval=' +
                        str(nb_interval) + ' != ' + str(len(potential_consumption)))

            # Save the final result into the location
            location.result['ASAP'][location_index1:location_index2] += (
                ASAP[activity_index1:activity_index2])

            location.result['ASAP_nominal'][location_index1:location_index2] += (
                ASAP_nominal[activity_index1:activity_index2])

            location.result['nominal'][location_index1:location_index2] += (
                nominal[activity_index1:activity_index2])

            location.result['ALAP_nominal'][location_index1:location_index2] += (
                ALAP_nominal[activity_index1:activity_index2])

            location.result['ALAP'][location_index1:location_index2] += (
                ALAP[activity_index1:activity_index2])

    elif init:
        # Initiate a dictionary of numpy array to hold result (faster than DataFrame)
        location.result = {'ASAP': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                           'ASAP_nominal': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                           'nominal': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                           'ALAP_nominal': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                           'ALAP': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep))}

    elif post:
        # Convert location result back into pandas DataFrame (faster that way)
        i = pandas.date_range(start=date_from, end=date_to,
                              freq=str(timestep) + 's', closed='left')
        location.result = pandas.DataFrame(index=i, data=location.result)


def save_vehicle_state(vehicle, timestep, date_from,
                       date_to, activity=None, power_demand=None, SOC=None,
                       detail=None, nb_interval=None, init=False, run=False, post=False):
    """Placeholder function to save individual vehicle's state. Do nothing.
    """
    if init:
        vehicle.SOC = [vehicle.SOC[0]]
        vehicle.result = None


def save_detailed_vehicle_power_demand(vehicle, timestep, date_from,
                                       date_to, activity=None, power_demand=None, SOC=None,
                                       detail=None, nb_interval=None, init=False, run=False, post=False):
    """Save vehicle detailed powertrain output. Only use with the detailed
    detailed power train model.
    """
    if run:
        activity_index1, activity_index2, location_index1, location_index2, save = _map_index(
            activity.start, activity.end, date_from, date_to, len(power_demand),
            len(vehicle.result['power_demand']), timestep)

        # Save a lot of interesting result
        if save:
            # If parked pmin and pmax are not necessary the same
            if isinstance(activity, model.Parked):
                vehicle.result['power_demand'][location_index1:location_index2] += (
                    power_demand[activity_index1:activity_index2])

    elif init:
        vehicle.SOC = [vehicle.SOC[0]]
        vehicle.result = {'power_demand': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep))}

    elif post:
        # Convert location result back into pandas DataFrame (faster that way)
        i = pandas.date_range(start=date_from, end=date_to,
                              freq=str(timestep) + 's', closed='left')
        vehicle.result = pandas.DataFrame(index=i, data=vehicle.result)


def save_detailed_vehicle_state(vehicle, timestep, date_from,
                                date_to, activity=None, power_demand=None, SOC=None,
                                detail=None, nb_interval=None, init=False, run=False, post=False):
    """Save vehicle detailed powertrain output. Only use with the detailed
    detailed power train model.
    """
    if run:
        activity_index1, activity_index2, location_index1, location_index2, save = _map_index(
            activity.start, activity.end, date_from, date_to, len(power_demand),
            len(vehicle.result['output_current']), timestep)

        # Save a lot of interesting result
        if save:
            # detail means some data is passed from the detailed power-train model
            if detail:
                vehicle.result['output_current'][location_index1:location_index2] += (
                    detail.ess.i_out[activity_index1:activity_index2])

            # if detail is false then it was a parked activity
            else:
                vehicle.result['power_demand'][location_index1:location_index2] += (
                    power_demand[activity_index1:activity_index2])
                
                vehicle.result['parked'][location_index1:location_index2] = (
                    [True] * (activity_index2 - activity_index1))

    elif init:
        vehicle.SOC = [vehicle.SOC[0]]
        vehicle.result = {'output_current': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                          'power_demand': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                          'parked': numpy.array([False] * int((date_to - date_from).total_seconds() / timestep))}

    elif post:
        # Convert location result back into pandas DataFrame (faster that way)
        i = pandas.date_range(start=date_from, end=date_to,
                              freq=str(timestep) + 's', closed='left')
        vehicle.result = pandas.DataFrame(index=i, data=vehicle.result)


def _map_index(activity_start, activity_end, date_from, date_to, vector_size,
               result_size, timestep):
    # Purpose of this function is to avoid slow DataFrame
    # Return activity_index1, activity_index2, location_index1, location_index2, save
    # Start
    start_inside = False
    start_before = False
    if date_from <= activity_start:
        if activity_start < date_to:
            start_inside = True
    else:
        start_before = True

    # End
    end_inside = False
    end_after = False
    if activity_end <= date_to:
        if date_from < activity_end:
            end_inside = True
    else:
        end_after = True

    # Map Index
    if start_inside and end_inside:
        location_index1 = int((activity_start - date_from).total_seconds() / timestep)
        return 0, vector_size, location_index1, location_index1 + vector_size, True

    elif start_before and end_inside:
        activity_index1 = int((date_from - activity_start).total_seconds() / timestep)
        return activity_index1, vector_size, 0, vector_size - activity_index1, True

    elif start_inside and end_after:
        location_index1 = int((activity_start - date_from).total_seconds() / timestep)
        location_index2 = int((activity_end - date_to).total_seconds() / timestep)
        return 0, vector_size - location_index2, location_index1, result_size, True

    elif start_before and end_after:
        activity_index1 = int((date_from - activity_start).total_seconds() / timestep)
        location_index2 = int((activity_end - date_to).total_seconds() / timestep)
        return activity_index1, vector_size - location_index2, 0, result_size, True

    else:
        return 0, 0, 0, 0, False
