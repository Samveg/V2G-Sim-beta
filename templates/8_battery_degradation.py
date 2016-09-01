from __future__ import division
import v2gsim
import v2gsim.battery_degradation.CapacityLoss

# Timestep need to be set to 1 seconds to capture 
# the detailed output of the powertrain model
project = v2gsim.model.Project(timestep=1)
project = v2gsim.itinerary.from_excel(project, '../data/NHTS/Tennessee_1.xlsx')

# Create a detailed power train model
car_model = v2gsim.driving.detailed.init_model.load_powertrain('../v2gsim/driving/detailed/data.xlsx')

# Assign model to all vehicles
for vehicle in project.vehicles:
    vehicle.car_model = car_model
    vehicle.result_function = v2gsim.result.save_detailed_vehicle_state

# Assign drivecycles to all driving activities
v2gsim.driving.drivecycle.generator.assign_EPA_cycle(project)

# Run V2G-Sim
v2gsim.core.run(project)

# input climate data
radiation = open('../data/climate/radm.txt', 'r+')
r = radiation.readlines()
radH = []
for i in range(len(r)):
	for k in range(0,3600):
		radH.append(float(r[i]))

amtem = open('../data/climate/temm.txt', 'r+')
t = amtem.readlines()
ambientT = []
for i in range(len(t)):
	for k in range(0,3600):
		ambientT.append(float(t[i]))

# Call battery degradation calculation function
v2gsim.battery_degradation.CapacityLoss.bd(project.vehicles, radH, ambientT, days=1)

