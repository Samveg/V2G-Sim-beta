from __future__ import division
import matplotlib.pyplot as plt
import pdb
import v2gsim

# Create a project
project = v2gsim.model.Project()
project = v2gsim.itinerary.from_excel(project, '../data/NHTS/Tennessee.xlsx')
project.vehicles = v2gsim.itinerary.get_cycling_itineraries(project)

# This function from the itinerary module finds all the different combination
# of locations in a project (e.g home-work-restaurant-work-home, ...), and thus different
# itineraries.
# It returns a data frame with a row per combination, including the combination, but also
# the vehicles that have this itinerary and some basics filtering options.
project.itinerary_statistics = v2gsim.itinerary.find_all_itinerary_combination(project)

# This function from the itinerary module reduce the number of itineraries
# by merging the location names that are not provided in the input list into 'other_location'
# It returns a new frame with less row since some of the combination have been merged.
project.itinerary_statistics = v2gsim.itinerary.merge_itinerary_combination(project, ['Home', 'Work'])

# This function from the itinerary module creates statistics about each itinerary.
# For each activity it will create distributions describing duration, etc...
# based on the data from individual vehicle with the same itinerary.
project.itinerary_statistics = v2gsim.itinerary.get_itinerary_statistic(project)

# This function from the itinerary module recreates a new project based on the
# statistics of all the different itineraries. The new project can have
# a chosen number of vehicles.
new_project = v2gsim.itinerary.new_project_using_stats(project, 2000)

# Initiate SOC and charging infrastructures
v2gsim.core.initialize_SOC(new_project, nb_iteration=1)

# Launch the simulation
v2gsim.core.run(new_project)

# Get the results
total_power_demand = v2gsim.post_simulation.result.total_power_demand(new_project)

# Plot the result
plt.plot(total_power_demand['total'], label='recreated_power_demand')
plt.show()

print('Press c and then enter to quit debugger')
pdb.set_trace()
# project.itinerary_statistics[project.itinerary_statistics.nb_of_vehicles>3]
# index = 4
# v2gsim.itinerary.plot_vehicle_itinerary(project.itinerary_statistics.ix[index].vehicles, title=str(project.itinerary_statistics.ix[index].locations))
# plt.show()
# v2gsim.itinerary.plot_vehicle_itinerary(new_project.vehicles[:20])
