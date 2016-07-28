from __future__ import division

import os
import sys
import numpy


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
import result.itinerary
import battery_degradation.Fixedexample
import charging.optimized
import grid.apipyiso as grid
from pyiso import client_factory
import scipy.io as sio
import matplotlib.pyplot as plt
import pdb, traceback, sys


os.system('cls' if os.name == 'nt' else 'clear')
print('###############')
print('### V2G-Sim ###')
print('###############')
print('')

# Initialize location and vehicles from excel file
excelFilename = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join("..", "..", "data","NHTS_data", "Tennessee_1.xlsx"))
vehicleList, locationList = itinerary.load.excel(excelFilename)

# Create the vehicle model
carModel = driving.detailed.init_model.load_powertrain(
    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 os.path.join("..", "..", "data", "detailed_power_train", "Leaf.xlsx")), 0.5)

# Set power-train model to every vehicle
driving.detailed.init_model.assign_powertrain_model(vehicleList, carModel)

driving.drivecycle.generator.assign_EPA_cycle(vehicleList, constGrade=0)

# Set the DR charging function at all locations
for location in locationList:
    location.chargingInfra.chargingFunc = ('charging.controlled.demand_response' +
                                           '(vehicle, activity, lastIteration, option)')
optionList = []
#optionList.append({'startDR': -1, 'endDR': -1, 'post_DR_window_fraction': 0.5, 'thresholdSOC': 0.2})
optionList.append({'startDR': 17, 'endDR': 21, 'post_DR_window_fraction': 0.5, 'thresholdSOC': 0.2})
# optionList.append({'startDR': 13, 'endDR': 17, 'post_DR_window_fraction': 0.5, 'thresholdSOC': 0.2})
# optionList.append({'startDR': 14, 'endDR': 18, 'post_DR_window_fraction': 0.5, 'thresholdSOC': 0.2})
# optionList.append({'startDR': 15, 'endDR': 19, 'post_DR_window_fraction': 0.5, 'thresholdSOC': 0.2})
# optionList.append({'startDR': 16, 'endDR': 20, 'post_DR_window_fraction': 0.5, 'thresholdSOC': 0.2})

# powerDemandLoc = []
for option in optionList:
    # Reset vehicles --> initSOC = 0.9
    result.vehicle_consumption.reset_vehicle(vehicleList)
    core.runV2GSim(vehicleList, nbIteration = 2, option = option)

# for vehicle in vehicleList:
#     print(vehicle.SOC[0])
# pdb.set_trace()
vehicleLoad = result.vehicle_consumption.get_total_EV_power_demand(vehicleList, 1)
# for vehicle in vehicleList:
#     print(vehicle.SOC[0])
# for vehicle in vehicleList:
#     powerdemand = []
#     for activity in vehicle.itinerary[0].activity:
#         if isinstance(activity, model.Parked):
#             powerdemand.extend(activity.powerDemand)
#     x = range(len(powerdemand))
#     plt.plot(x,powerdemand,'g')
#     plt.show()
# x = range(len(vehicleLoad))
# plt.plot(x,vehicleLoad,'g')
# plt.show()

# try:
	# r = open('/Users/apple/Documents/V2G-Sim/V2G-Sim/v2g_sim/battery_degradation/radh.txt', 'r+')
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

# t = open('/Users/apple/Documents/V2G-Sim/V2G-Sim/v2g_sim/battery_degradation/temm.txt', 'r+')
dataReadt = t.readlines()
for i in range(len(dataReadt) - 1):
	dataReadt[i] = dataReadt[i][0:2]
ambientT = []
for i in range(len(dataReadt)):
	for k in range(3600):
		ambientT.append(float(dataReadt[i]))

# except:
# 	    type, value, tb = sys.exc_info()
# 	    traceback.print_exc()
# 	    pdb.post_mortem(tb)


# pdb.set_trace()
battery_degradation.Fixedexample.bd(vehicleList, radH, ambientT)

# averageSOC = result.vehicle_consumption.get_average_SOC(vehicleList)
# outputInterval = 60
# x = range(0, len(averageSOC))
# hourlySteps = 3600 / outputInterval
# x = [x[i] / hourlySteps for i in range(0, len(x))]
# plt.plot(x, averageSOC, label='average SOC')
# plt.xlabel('Time of the day (hours)', fontsize=14)
# plt.ylabel('Average SOC', fontsize=14)
# plt.legend(loc=0, frameon=True)

# plt.show()
