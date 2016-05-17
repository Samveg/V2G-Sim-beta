import numpy
import pandas


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

            # Examples:
            # Add 'number_of_vehicle_parked' in the initialization section
            # Then location.result['number_of_vehicle_parked'][location_index1:location_index2] += 1

            # Add 'available_energy' in the initialization section
            # Then location.result['available_energy'][location_index1:location_index2] += (
            #          [soc * vehicle.car_model.battery_capacity
            #           for soc in SOC[activity_index1:activity_index2]])

    elif init:
        # Initiate a dictionary of numpy array to hold result (faster than DataFrame)
        location.result = {'power_demand': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep))}

    elif post:
        # Convert location result back into pandas DataFrame (faster that way)
        i = pandas.date_range(start=date_from, end=date_to,
                              freq=str(timestep) + 's', closed='left')
        location.result = pandas.DataFrame(index=i, data=location.result)


def save_vehicle_state(vehicle, timestep, date_from,
                       date_to, activity=None, power_demand=None, SOC=None,
                       nb_interval=None, init=False, run=False, post=False):
    """Placeholder function to save individual vehicle's state. Do nothing.
    """
    pass


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
