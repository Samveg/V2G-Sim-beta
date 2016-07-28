from __future__ import division

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../v2g_sim")) 
import core
import itinerary.load
import itinerary.itineraryBin
import itinerary.create_vehicles
import result.itinerary
import result.vehicle_consumption
import result.visualization.d3
import result.visualization.matp
import charging.controlled
import driving.detailed.init_model
import driving.drivecycle.generator
import pdb
import matplotlib.pyplot as plt
import battery_degradation.Fixedexample

os.system('cls' if os.name == 'nt' else 'clear')
print('###############')
print('### V2G-Sim ###')
print('###############')
print('')

# Initialize location and vehicles from excel file
# excelFilename = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join("..", "NHTS_data",
#                                                                                        "Tennessee_1.xlsx"))
excelFilename = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join("..", "..", "data","NHTS_data", "California_100.xlsx"))
vehicleList, locationList = itinerary.load.excel(excelFilename)
# Create the vehicle model
carModel = driving.detailed.init_model.load_powertrain(
    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 os.path.join("..", "..", "data", "detailed_power_train", "Leaf.xlsx")), 0.5)
# Set power-train model to every vehicle
driving.detailed.init_model.assign_powertrain_model(vehicleList, carModel)

driving.drivecycle.generator.assign_EPA_cycle(vehicleList, constGrade=0)

chargingSignal = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join("..", "..", "v2g_sim",
                                                                                        "charging", "basicSignal2.txt"))
signal = charging.controlled.load_basic_signal(chargingSignal)

# pdb.set_trace()
# Set the DR charging function at all locations
for location in locationList:
    location.chargingInfra.chargingFunc = ('charging.controlled.consumption_modified_signal' + 
                                                    '(vehicle, activity, lastIteration, option)')
    if location.type in 'Home':
        location.chargingInfra.powerRateMax = 7200
    elif location.type in 'Work':
        location.chargingInfra.powerRateMax = 7200
    # else:
    #     location.chargingInfra.powerRateMax = 0
    # location.chargingInfra.chargingFunc = 'Hello Teng'
# for vehicle in vehicleList:
#     vehicle.SOC[0]=0.7

optionList = []
# optionList.append({'startDR': -1, 'endDR': -1, 'post_DR_window_fraction': 0.5, 'thresholdSOC': 0.2})
optionList.append({'startDR': 19, 'endDR': 21, 'post_DR_window_fraction': 0, 'thresholdSOC': 0.2, 'signal':signal})
# optionList.append({'startDR': 13, 'endDR': 24, 'post_DR_window_fraction': 0.5, 'thresholdSOC': 0.2})
# optionList.append({'startDR': 14, 'endDR': 18, 'post_DR_window_fraction': 0.5, 'thresholdSOC': 0.2})
# optionList.append({'startDR': 15, 'endDR': 19, 'post_DR_window_fraction': 0.5, 'thresholdSOC': 0.2})
# optionList.append({'startDR': 16, 'endDR': 20, 'post_DR_window_fraction': 0.5, 'thresholdSOC': 0.2})

powerDemandLoc = []
for option in optionList:
    # Reset vehicles --> initSOC = 0.9
    result.vehicle_consumption.reset_vehicle(vehicleList)
    # Launch the simulation with the DR event function
    core.runV2GSim(vehicleList, nbIteration=2, option=option)


# pdb.set_trace()
vehicleLoad = result.vehicle_consumption.get_total_EV_power_demand(vehicleList, 1)
# with open('savetotalload.txt','a+') as totalload_file:
# 	totalload_file.write(' '.join(str(e) for e in vehicleLoad))
# x = range(len(vehicleLoad))
# plt.plot(x,vehicleLoad,'r')
# plt.show()
#
r = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
             os.path.join("..","..","v2g_sim","battery_degradation","radh.txt",)),
            'r+')
dataRead = r.readlines()
for i in range(len(dataRead) - 1):
	dataRead[i] = dataRead[i]
radH = []

for i in range(len(dataRead)):
	for k in range(3600):
		radH.append(float(dataRead[i]))

t = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                      os.path.join("..","..","v2g_sim","battery_degradation","temh.txt",)),
         'r+')
#t = open('/Users/apple/Documents/V2G-Sim/V2G-Sim/v2g_sim/battery_degradation/temm.txt', 'r+')
dataReadt = t.readlines()
for i in range(len(dataReadt) - 1):
	dataReadt[i] = dataReadt[i]
ambientT = []
for i in range(len(dataReadt)):
	for k in range(3600):
		ambientT.append(float(dataReadt[i]))
battery_degradation.Fixedexample.bd(vehicleList, radH, ambientT)
# Plot Results --------------------------------------------------
# import matplotlib.pyplot as plt
# import seaborn
# # Avoid frame around plot
# seaborn.set_style("whitegrid")
# seaborn.despine()
#
# outputInterval = 60
# for simulation, option in zip(powerDemandLoc, optionList):
#     # Create the x axis data
#     x = range(0, len(simulation['Total']))
#     hourlySteps = 3600 / outputInterval
#     x = [x[i] / hourlySteps for i in range(0, len(x))]
#
#     # Plot the data
#     label = str(option['startDR']) + ' - ' + str(option['endDR'])
#     plt.plot(x, simulation['Total'], label=label)
#
# plt.xlabel('Time of the day (hours)', fontsize=14)
# plt.ylabel('Power demand (W)', fontsize=14)
# plt.legend(loc=0, frameon=True)
#
# plt.show()
