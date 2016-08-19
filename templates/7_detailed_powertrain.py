from __future__ import division
import matplotlib.pyplot as plt
import v2gsim

# Create a project
project = v2gsim.model.Project(timestep=60)
project = v2gsim.itinerary.from_excel(project, '../data/NHTS/Tennessee_1.xlsx')

# Create a detailed power train vehicle model from an excel spread sheet
car_model = v2gsim.driving.detailed.init_model.load_powertrain('../v2gsim/driving/detailed/data.xlsx')

# Assign model to all vehicles and also use a new function to reccord detailed
# from the power train model, such as battery temperature.
for vehicle in project.vehicles:
    vehicle.car_model = car_model
    vehicle.result_function = v2gsim.result.save_detailed_vehicle_state

# Assign a drivecycle to all the driving activities since the detail power train
# model uses speed profile to calculate on road consumption
v2gsim.driving.drivecycle.generator.assign_EPA_cycle(project)

# # Initiate SOC and charging infra <-- No need since the vehicle recover full charge
# conv = v2gsim.core.initialize_SOC(project, nb_iteration=1)

# Run V2G-Sim
v2gsim.core.run(project)

total_power_demand = v2gsim.post_simulation.result.total_power_demand(project)

# Plot the result
plt.figure()
plt.plot(total_power_demand['total'])
plt.show()

plt.figure()
project.vehicles[0].result.output_current.plot()
plt.show()

plt.figure()
project.vehicles[0].result.battery_temp.plot()
plt.show()
