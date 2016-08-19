from __future__ import division
import datetime
import matplotlib.pyplot as plt
import dill as pickle
import pandas
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../V2G-Sim-Beta"))
import traceback
import pdb
import v2gsim
import v2gsim.battery_degradation.Fixedexample

project = v2gsim.model.Project(timestep=1)
project = v2gsim.itinerary.from_excel(project, '../data/NHTS/Tennessee_1.xlsx')
project = v2gsim.itinerary.copy_append(project, nb_copies=0)

# Create a detailed power train model
car_model = v2gsim.driving.detailed.init_model.load_powertrain('/Users/wangdai/V2G-Sim-Beta/v2gsim/driving/detailed/data.xlsx')

# Assign model to all vehicles
for vehicle in project.vehicles:
    vehicle.car_model = car_model
    vehicle.result_function = v2gsim.result.save_detailed_vehicle_state

# Assign drivecycles to all driving activities
v2gsim.driving.drivecycle.generator.assign_EPA_cycle(project)

# Run V2G-Sim
v2gsim.core.run(project)

total_power_demand = v2gsim.post_simulation.result.total_power_demand(project)



r = open('/Users/wangdai/V2G-Sim-Beta/v2gsim//battery_degradation/radm.txt', 'r+')
dataRead = r.readlines()
for i in range(len(dataRead) - 1):
	dataRead[i] = dataRead[i][:-1]
radH = []

for i in range(len(dataRead)):
	for k in range(3600):
		radH.append(float(dataRead[i]))

t = open('/Users/wangdai/V2G-Sim-Beta/v2gsim//battery_degradation/temm.txt', 'r+')
dataReadt = t.readlines()
for i in range(len(dataReadt) - 1):
	dataReadt[i] = dataReadt[i]
ambientT = []
for i in range(len(dataReadt)):
	for k in range(3600):
		ambientT.append(float(dataReadt[i]))


# Call battery degradation calculation function
v2gsim.battery_degradation.Fixedexample.bd(project.vehicles, radH, ambientT, days=1)
