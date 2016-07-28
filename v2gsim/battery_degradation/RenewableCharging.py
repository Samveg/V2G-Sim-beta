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
excelFilename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             os.path.join("..", "..", "data","NHTS_data", "California_100.xlsx"))
vehicleList, locationList = itinerary.load.excel(excelFilename)
#Change infrastructure
for location in locationList:
    if location.type == 'Work':
        location.chargingInfra.powerRateMax = 7200
    elif location.type == 'Home':
        location.chargingInfra.powerRateMax = 1440
    else:
        location.chargingInfra.powerRateMax = 0

# Change car model parameters
vehicleList[0].carModel.batteryCap = 23832  # Nissan Leaf --> 23832Wh

# Remove vehicle that could not be initialized correctly
vehicleList = itinerary.load.remove_vehicle_with_empty_itinerary(vehicleList)

# # Run the simplest model with uncontrolled charging
# core.runV2GSim(vehicleList, nbIteration=3)

# Create the vehicle model
carModel = driving.detailed.init_model.load_powertrain(
    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 os.path.join("..", "..", "data", "detailed_power_train", "Leaf.xlsx")), 0.5)

# Set power-train model to every vehicle
driving.detailed.init_model.assign_powertrain_model(vehicleList, carModel)

driving.drivecycle.generator.assign_EPA_cycle(vehicleList, constGrade=0)


try:
    core.runV2GSim(vehicleList, nbIteration=2)
except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)

# for vehicle in vehicleList:
#     powerdemand = []
#     for activity in vehicle.itinerary[0].activity:
#         if isinstance(activity, model.Parked):
#             powerdemand.extend(activity.powerDemand)
#     x = range(len(powerdemand))
#     plt.plot(x,powerdemand,'g')
#     plt.show()

# ------------------ Renewable forecast ----------------------
# Input data
year = 2015
month = 1
day = 17
client = client_factory('CAISO')

# Grid manager
data = grid.get_daily_data(year, month, day, client, categories=['solarpv', 'solarth', 'wind', 'load'])

duckList = []
for index, time in enumerate(data['time']):
    duckList.append(data['load'][index] - data['solar'][index] - data['wind'][index])

# Calculate the scaling factor from the base case - ony once
maxLoad = max(duckList)
maxDesired = 6  # MW
scalingFactor = maxDesired / maxLoad

# Offset the net load and switch to W
duckList = [duckList[j] * scalingFactor * 1000 * 1000 for j in range(0, len(duckList))]  # from MW to W

# Change sampling rate from hours to 60 seconds interval
timeDuck = [index * 3600 for index in range(0, len(data['time']))]
timeOfDay = [60 * index for index in range(0, int(24 * 60))]
duckList = numpy.interp(timeOfDay, timeDuck, duckList)
# -----------------------------------------------------------


# ------------------ Optimization code ----------------------
# Time interval for the optimization in hours
timeInterval = 0.1

# Create an optimization object for each vehicle
vehicleList = charging.optimized.initialize_optimization(vehicleList, timeInterval, minimumSOC=0.1, finalSOC=0.7, v2g = True)

# Get all the vehicle that can participate in the optimization problem
vehicleList_forOptimization, vehicleList_notParticipating = charging.optimized.select_participant(vehicleList,
                                                                                                  timeInterval,
                                                                                                  duckList, 60)
# Force SOC to start at 0.7
for vehicle in vehicleList_forOptimization:
    vehicle.SOC[0] = 0.7

# Update optimization object for vehicle participating in the optimization
vehicleList_forOptimization = charging.optimized.initialize_optimization(vehicleList_forOptimization,
                                                                         timeInterval, minimumSOC=0.1, finalSOC=0.7, v2g = True)

# Launch the optimization code and replace parked activity consumption with optimum
vehicleList_forOptimization, priceSignal, dualGap = charging.optimized.optimization(vehicleList_forOptimization,
                                                                                    timeInterval, duckList,
                                                                                    netLoadSamplingRate=60)
# Re-compute SOC level through out the day
vehicleList_forOptimization = charging.optimized.recompute_SOC_profile(vehicleList_forOptimization)
# -----------------------------------------------------------


vehicleLoad = result.vehicle_consumption.get_total_EV_power_demand(vehicleList, 1)
#print(len(vehicleList_forOptimization))
# x = range(len(vehicleLoad))
# plt.plot(x,vehicleLoad,'g')
# plt.show()

# # Plot net load results
# plt.figure(figsize=(10, 6), dpi=80)
# hourlySteps = 3600 / vehicleList[0].outputInterval
# x = range(0, len(duckList))
# x = [x[i] / hourlySteps for i in range(0, len(x))]
# plt.plot(x, duckList, label='net load')
#
# optimalLoad = [load + vehicle for load, vehicle in zip(duckList, vehicleLoad)]
# label = 'net load + optimal charging(' + str(len(vehicleList_forOptimization) * 100 / len(vehicleList)) + '%)'
# plt.plot(x, optimalLoad, label=label)
#
# # temp = [load + vehicle for load, vehicle in zip(duckList, vehicleLoadBefore)]
# # plt.plot(x, temp, label='net load + uncontrolled charging(100%)')
#
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Power demand (W)', fontsize=12)
# plt.title('Net load')
# plt.legend(loc=0, frameon=True)
#
# plt.show()

# try:
	# r = open('/Users/apple/Documents/V2G-Sim/V2G-Sim/v2g_sim/battery_degradation/radh.txt', 'r+')
r = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
             os.path.join("..","..","v2g_sim","battery_degradation","radm.txt",)),
            'r+')
dataRead = r.readlines()
for i in range(len(dataRead) - 1):
	dataRead[i] = dataRead[i]
radH = []

for i in range(len(dataRead)):
	for k in range(3600):
		radH.append(float(dataRead[i]))

t = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
             os.path.join("..","..","v2g_sim","battery_degradation","temm.txt",)),
            'r+')
dataReadt = t.readlines()
for i in range(len(dataReadt) - 1):
	dataReadt[i] = dataReadt[i]
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
