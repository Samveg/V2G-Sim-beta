from __future__ import division
import datetime
import matplotlib.pyplot as plt
import pandas
import os
import sys
import v2gsim
import seaborn as sns
sns.set_style("whitegrid")
sns.despine()

project = v2gsim.model.Project()
project = v2gsim.itinerary.from_excel(project, '../data/NHTS/Tennessee.xlsx')

# Putting aside vehicle that does not cycle in one day
project.vehicles = v2gsim.itinerary.get_cycling_itineraries(project)

project = v2gsim.itinerary.copy_append(project, nb_copies=2)

# Set the charging infrastructure at each location
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

# Initiate SOC and charging infra
conv = v2gsim.core.initialize_SOC(project, nb_iteration=1)

# Launch the simulation
v2gsim.core.run(project, date_from=project.date + datetime.timedelta(days=1),
                date_to=project.date + datetime.timedelta(days=2))

# Look at the results
total_power_demand = v2gsim.post_simulation.result.total_power_demand(project)

# Plot the result
fig = plt.figure()
plot = fig.add_subplot(111)
plt.plot(total_power_demand.index.tolist(), total_power_demand['total'] / (1000 * 1000))
plt.ylabel('Power demand (MW)', fontsize=18)
plt.xlabel('Time', fontsize=18)
plot.tick_params(axis='both', which='major', labelsize=16)
plot.tick_params(axis='both', which='minor', labelsize=16)
plt.show()
