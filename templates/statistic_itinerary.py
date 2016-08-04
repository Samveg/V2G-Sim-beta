from __future__ import division
import datetime
import matplotlib.pyplot as plt
import pandas
import os
import sys
import pdb
import v2gsim

project = v2gsim.model.Project()
project = v2gsim.itinerary.from_excel(project, '../data/NHTS/Tennessee_100.xlsx')
project = v2gsim.itinerary.copy_append(project, nb_copies=2)

# Initiate SOC and charging infra
conv = v2gsim.core.initialize_SOC(project, nb_iteration=1)

# Launch the simulation
v2gsim.core.run(project, date_from=project.date + datetime.timedelta(hours=12),
                date_to=project.date + datetime.timedelta(days=2, hours=12))

# Look at the results
total_power_demand = v2gsim.post_simulation.result.total_power_demand(project)

# Plot the result
plt.figure()
plt.plot(total_power_demand['total'])
plt.plot(total_power_demand['Home_demand'])
plt.show()

print('Press c and then enter to quit debugger')
pdb.set_trace()
