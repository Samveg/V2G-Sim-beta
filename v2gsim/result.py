import pandas


def save_power_demand_at_location(location, timestep, vehicle=None,
                                  activity=None, power_demand=None,
                                  nb_interval=None, date_from=False,
                                  date_to=False):
    """Save local result from a parked activity during running
    time. If date_from and date_to, set a fresh pandas DataFrame at locations.

    Args:
        location (Location): location
        timestep (int): calculation timestep
        vehicle (Vehicle): vehicle
        activity (Parked): parked activity
        power_demand (list): power demand from parked activity
        nb_interval (int): number of timestep for the parked activity
        date_from (datetime.datetime): date to start recording power demand
        date_to (datetime.datetime): date to end recording power demand

    Example:
        >>> # Initialize a result DataFrame for each location
        >>> save_power_demand_at_location(location, timestep, date_from=some_date,
                                          date_to=other_date)
        >>> # Save data during run time
        >>> save_power_demand_at_location(location, timestep, vehicle, activity,
                                          power_demand, nb_interval)
    """
    if date_from and date_to:
        # Initiate data frame for consumption result at location
        i = pandas.date_range(start=date_from, end=date_to, freq=str(timestep) + 's')
        location.result = pandas.DataFrame(index=i, data={'power_demand': [0] * len(i)})

    else:
        # Save power_demand at location
        i = pandas.date_range(start=activity.start,
                              periods=nb_interval,
                              freq=str(timestep) + 's')

        df = pandas.DataFrame(index=i, data={'power_demand': power_demand})
        t = location.result[i[0]:i[-1]]
        if len(t) != 0:
            location.result[t.index[0]:t.index[-1]] = (
                location.result[t.index[0]:t.index[-1]].add(df[t.index[0]:t.index[-1]]))
