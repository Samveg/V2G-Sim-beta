from __future__ import division

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../v2g_sim"))
import core
import model
import itinerary.load
import itinerary.itineraryBin
import itinerary.create_vehicles
import result.itinerary
import result.vehicle_consumption
import result.visualization.d3
import result.visualization.matp
import driving.detailed.init_model
import driving.drivecycle.generator
import battery_degradation.Fixedexample
import matplotlib.pyplot as plt
import scipy.io as sio
import pdb, traceback, sys
import v2gsim

project = v2gsim.model.Project(timestep=1)
project = v2gsim.itinerary.from_excel(project, 'Tennessee_1.xlsx')
project = v2gsim.itinerary.copy_append(project, nb_copies=2)

# Create a detailed power train model
car_model = v2gsim.driving.detailed.init_model.load_powertrain('/home/V2G_Sim_beta/v2gsim/driving/detailed/data.xlsx')

# Assign model to all vehicles
for vehicle in project.vehicles:
    vehicle.car_model = car_model
    vehicle.result_function = v2gsim.result.save_detailed_vehicle_state

# Assign drivecycles to all driving activities
v2gsim.driving.drivecycle.generator.assign_EPA_cycle(project)

# Run V2G-Sim
v2gsim.core.run(project)

total_power_demand = v2gsim.post_simulation.result.total_power_demand(project)



r = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 os.path.join("..","..","v2g_sim","battery_degradation","radm.txt",)),
	            'r+')
dataRead = r.readlines()
for i in range(len(dataRead) - 1):
	dataRead[i] = dataRead[i][:-1]
radH = []

for i in range(len(dataRead)):
	for k in range(3600):
		radH.append(float(dataRead[i]))

t = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 os.path.join("..","..","v2g_sim","battery_degradation","temm.txt",)),
	            'r+')
	# t = open('/Users/apple/Documents/V2G-Sim/V2G-Sim/v2g_sim/battery_degradation/temh.txt', 'r+')
dataReadt = t.readlines()
for i in range(len(dataReadt) - 1):
	dataReadt[i] = dataReadt[i]
ambientT = []
for i in range(len(dataReadt)):
	for k in range(3600):
		ambientT.append(float(dataReadt[i]))


# Call battery degradation calculation function
battery_degradation.Fixedexample.bd(project.vehicles, radH, ambientT)
