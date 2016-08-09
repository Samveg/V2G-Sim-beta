from __future__ import division
import datetime
import matplotlib.pyplot as plt
import pandas
import os
import sys
import pdb
import v2gsim
import pdb, traceback, sys

project = v2gsim.model.Project()
project = v2gsim.itinerary.from_excel(project, '../data/NHTS/Tennessee.xlsx')
project.vehicles = v2gsim.itinerary.get_cycling_itineraries(project)

# Initiate SOC and charging infra
conv = v2gsim.core.initialize_SOC(project, nb_iteration=1)

# Launch the simulation
v2gsim.core.run(project)

# Look at the results
total_power_demand = v2gsim.post_simulation.result.total_power_demand(project)

# Plot the result
plt.plot(total_power_demand['total'], label='original_power_demand')

# Create a new project
project.itinerary_statistics = v2gsim.itinerary.find_all_itinerary_combination(project)
project.itinerary_statistics = v2gsim.itinerary.merge_itinerary_combination(project, ['Home', 'Work'])
project.itinerary_statistics = v2gsim.itinerary.get_itinerary_statistic(project)
new_project = v2gsim.itinerary.new_project_using_stats(project, 2000)


# Initiate SOC and charging infra
conv = v2gsim.core.initialize_SOC(new_project, nb_iteration=1)

# Launch the simulation
import pdb, traceback, sys
try:
    v2gsim.core.run(new_project)
except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)

# Look at the results
total_power_demand2 = v2gsim.post_simulation.result.total_power_demand(new_project)

# Plot the result
plt.plot(total_power_demand2['total'], label='recreated_power_demand')

plt.show()

print('Press c and then enter to quit debugger')
pdb.set_trace()
# project.itinerary_statistics[project.itinerary_statistics.nb_of_vehicles>3]
# index = 4
# v2gsim.itinerary.plot_vehicle_itinerary(project.itinerary_statistics.ix[index].vehicles, title=str(project.itinerary_statistics.ix[index].locations))
# plt.show()
# v2gsim.itinerary.plot_vehicle_itinerary(new_project.vehicles[:20])