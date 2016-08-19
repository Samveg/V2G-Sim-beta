import datetime
import matplotlib.pyplot as plt
import dill as pickle
import pandas
import os
import sys
import traceback
import pdb
import v2gsim


project = v2gsim.model.Project(timestep=60)
project = v2gsim.itinerary.from_excel(project, '../data/NHTS/Tennessee_1.xlsx')
project = v2gsim.itinerary.copy_append(project, nb_copies=2)

# Create a detailed power train model
car_model = v2gsim.driving.detailed.init_model.load_powertrain('/Users/wangdai/V2G-Sim-Beta/v2gsim/driving/detailed/data.xlsx')

# Assign model to all vehicles
for vehicle in project.vehicles:
    vehicle.car_model = car_model
    vehicle.result_function = v2gsim.result.save_detailed_vehicle_state

# Assign drivecycles to all driving activities
v2gsim.driving.drivecycle.generator.assign_EPA_cycle(project)

# # Initiate SOC and charging infra <-- No need since the vehicle recover full charge
# conv = v2gsim.core.initialize_SOC(project, nb_iteration=1)

# Run V2G-Sim
v2gsim.core.run(project, date_from=project.date + datetime.timedelta(hours=12),
                date_to=project.date + datetime.timedelta(days=2, hours=12))

total_power_demand = v2gsim.post_simulation.result.total_power_demand(project)

# Plot the result
plt.figure()
plt.plot(total_power_demand.index.tolist(), total_power_demand['total'].values.tolist())
plt.show()

plt.figure()
project.vehicles[0].result.output_current.plot()
plt.show()

plt.figure()
project.vehicles[0].result.battery_temp.plot()
plt.show()
