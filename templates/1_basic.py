from __future__ import division
import matplotlib.pyplot as plt
import pdb

# Give you access to all the V2G-Sim modules
import v2gsim

# Create a project that will hold other objects such as vehicles, locations
# car models, charging stations and some results. (see model.Project class)
project = v2gsim.model.Project()

# Use the itinerary module to import itineraries from an Excel file.
# Instantiate a project with the necessary information to run a simulation.
# Default values are assumed for the vehicle to model
# and the charging infrastructures to simulate.
project = v2gsim.itinerary.from_excel(project, '../data/NHTS/Tennessee.xlsx')

# At first every vehicle start with a full battery. In order to start from
# a more realistic state of charge (SOC), we run some iterations of a day,
# to find a stable SOC for each vehicle at the end of the day.
# This value is then used as the initial SOC condition to a realistic state.
v2gsim.core.initialize_SOC(project, nb_iteration=3)

# Launch the simulation and save the results
v2gsim.core.run(project)

# Concatenate the power demand for each location into one frame.
# you can access the demand at any location by using "loactionName_demand"
# or access the total demand with "total".
total_power_demand = v2gsim.post_simulation.result.total_power_demand(project)

# Plot the result
plt.figure()
plt.plot(total_power_demand['total'] / (1000 * 1000))
plt.plot(total_power_demand['Home_demand'] / (1000 * 1000))
plt.ylabel('Power demand (MW)')
plt.xlabel('Time')
plt.legend()
plt.show()

# Stop the script at the end, and let you explore the project structure.
# Perhaps you can checkout "project.vehicles[0]"
print('Press c and then enter to quit debugger')
pdb.set_trace()
