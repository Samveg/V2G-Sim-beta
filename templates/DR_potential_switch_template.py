import datetime
import matplotlib.pyplot as plt
import pandas
import os
import sys
import v2gsim


def save_location_state_DR(location, timestep, date_from, date_to,
                        vehicle=None, activity=None,
                        power_demand=None, SOC=None, nb_interval=None,
                        init=False, run=False, post=False):
    """Save local results from a parked activity at run time.
    """
    if run:
        activity_index1, activity_index2, location_index1, location_index2, save = v2gsim.result._map_index(
            activity.start, activity.end, date_from, date_to, len(power_demand),
            len(location.result['power_demand']), timestep)

        # Save a lot of interesting result
        if save:
            if activity.charging_station.post_simulation:
                location.result['controlled_power_demand'][location_index1:location_index2] += (
                    power_demand[activity_index1:activity_index2])

                location.result['pmax'][location_index1:location_index2] += activity.charging_station.maximum_power
            else:
                location.result['uncontrolled_power_demand'][location_index1:location_index2] += (
                    power_demand[activity_index1:activity_index2])

    elif init:
        # Initiate a dictionary of numpy array to hold result (faster than DataFrame)
        location.result = {'uncontrolled_power_demand': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                           'controlled_power_demand': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                           'pmax': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep))}

    elif post:
        # Convert location result back into pandas DataFrame (faster that way)
        i = pandas.date_range(start=date_from, end=date_to,
                              freq=str(timestep) + 's', closed='left')
        location.result = pandas.DataFrame(index=i, data=location.result)


project = v2gsim.model.Project()
project = v2gsim.itinerary.from_excel(project, '../data/NHTS/Tennessee.xlsx')
project = v2gsim.itinerary.copy_append(project, nb_copies=2)

# Assign new result function to all locations
for location in project.locations:
    location.result_function = save_location_state_DR

# Initiate SOC and charging infra
conv = v2gsim.core.initialize_SOC(project, nb_iteration=1)

# Launch the simulation
v2gsim.core.run(project, date_from=project.date + datetime.timedelta(hours=12),
                date_to=project.date + datetime.timedelta(days=2, hours=12))

# Look at the results

