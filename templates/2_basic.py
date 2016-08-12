from __future__ import division
import datetime
import matplotlib.pyplot as plt
import v2gsim

# Create a project and initialize it with someitineraries
project = v2gsim.model.Project()
project = v2gsim.itinerary.from_excel(project, '../data/NHTS/Tennessee.xlsx')

# This function from the itinerary module return all the vehicles that
# start and end their day at the same location (e.g. home)
project.vehicles = v2gsim.itinerary.get_cycling_itineraries(project)

# This function from the itinerary module copy a daily itinerary and
# append it at then end of the existing itinerary. In doing so, it makes
# sure that activities are merged at the junction.
project = v2gsim.itinerary.copy_append(project, nb_of_days_to_add=2)

# Some default infrastructure have been created for you, namely "no_charger",
# "L1" and "L2", you can change the probability of a vehicle to be plugged
# to one of those infrastructures at different locations as follow:
for location in project.locations:
    if location.category == 'Home':
        location.available_charging_station.loc['no_charger', 'probability'] = 0.0
        location.available_charging_station.loc['L1', 'probability'] = 0.3
        location.available_charging_station.loc['L2', 'probability'] = 0.7
    elif location.category == 'Work':
        location.available_charging_station.loc['no_charger', 'probability'] = 0.0
        location.available_charging_station.loc['L1', 'probability'] = 0.0
        location.available_charging_station.loc['L2', 'probability'] = 1.0
    else:
        location.available_charging_station.loc['no_charger', 'probability'] = 1.0
        location.available_charging_station.loc['L1', 'probability'] = 0.0
        location.available_charging_station.loc['L2', 'probability'] = 0.0

# Initiate SOC and charging infrastructures
v2gsim.core.initialize_SOC(project, nb_iteration=2)

# Launch the simulation
# Note that date_from and date_to have been added since 1_basic_template.py
# date_from and date_to allows a user to specify
# the window over which results are saved (reduce memory burden).
v2gsim.core.run(project, date_from=project.date + datetime.timedelta(days=1),
                date_to=project.date + datetime.timedelta(days=2))

# Look at the results
total_power_demand = v2gsim.post_simulation.result.total_power_demand(project)

# Plot the result
plt.figure()
plt.plot(total_power_demand['total'] / (1000 * 1000))
plt.ylabel('Power demand (MW)')
plt.xlabel('Time')
plt.legend()
plt.show()
