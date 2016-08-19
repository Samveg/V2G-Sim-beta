from __future__ import division
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../battery_degradation"))  # YOUR PATH
import v2gsim.core as core
import v2gsim.model as model
import v2gsim.battery_degradation.Fixeddegradation as Fixeddegradation
import v2gsim
import matplotlib.pyplot as plt

import numpy as np



class BatteryModel(object):
	"""create a class to store all the
	   degradation information
	"""

	def __init__(self):
		self.cabinT = [20]
		self.batteryT = [18]
		self.batteryLoss = {'cycleLoss': [0], 'calendarLoss': [0]}
		# input parameters of EV thermal network and degradation model
		self.coefTemp = {'q_havc': 4500, 'M_c': 101771, 'M_b': 182000, 'K_ab': 4.343, 'K_ac': 22.6, 'K_bc': 3.468}
		self.coefLoss = {'a': 8.888888888889532 * 10 ** (-6), 'b': -0.005288888888889, 'c': 0.787113333333394,
		                 'd': -0.0067, 'e': 2.35, 'f': 8720, 'E': 24500, 'R': 8.314}


def bd(vehicleList, radH, ambientT, days):
	""" battery degradation function

	Args:
	    vehicleList: vehicleList (list of vehicles): vehicles to simulate
	    radH: solar radiation
	    ambientT: ambient temperature
	Returns:

	"""
	for vehicle in vehicleList:
		vehicle.battery_model = BatteryModel()
		DrivingCurrent = vehicle.result.output_current.tolist()   #create this variable to store all the current vector,
		ChargingDemand = vehicle.result.power_demand.tolist()   #create this variable to store all the current vector
		ChargingCurrent = [-i / 380 for i in ChargingDemand]
		AllDayCurrent = [x + y for x,y in zip(DrivingCurrent, ChargingCurrent)]
		
		#######
		soc = vehicle.SOC
		deltasoc = [y - x for x,y in zip(([soc[0]]+soc[:]),soc[:]+[soc[-1]])] # calculate delta soc
		deltasoc = tempdsoc[:-1]  # delta soc which will be used to calculate cycle life loss
		######

		# deltasoc = numpy.diff(vehicle.SOC).tolist()
		DrivingCharge = []
		flag = vehicle.result.parked.tolist() # mark if the car is parked or driving
		R = 0.15 # internal resistance


		for i in range(0, len(AllDayCurrent)):
			DrivingCharge.append(R * AllDayCurrent[i] ** 2) # heat generate from battery
			# Calculate whole day temperature.
			if not flag[i]: # vehicle is driving
				# Calculate battery temeprature when the vehicle is driving
				Fixeddegradation.driving_temperature(vehicle, ambientT[i], radH[i], DrivingCharge[-1],
				                                     vehicle.battery_model.coefTemp)

				# Calculate cycle life loss caused by driving
				Fixeddegradation.cycle_loss_drive(vehicle, vehicle.battery_model.batteryT[-1], AllDayCurrent[i], deltasoc[i], vehicle.battery_model.coefLoss)

			elif flag[i]: # vehicle is parked
				if AllDayCurrent[i] == 0:  # The car is parked but not charging/discharging

					# Calculate battery temperature when the vehicle is parked but not charging/discharging, battery thermal mananage system (BTMS) works but AC does not work
					Fixeddegradation.idle_temperature(vehicle, ambientT[i], radH[i],
				                                     vehicle.battery_model.coefTemp)
					# no cycle life loss, since current=0
					Fixeddegradation.cycle_loss_drive(vehicle, vehicle.battery_model.batteryT[-1], AllDayCurrent[i], deltasoc[i], vehicle.battery_model.coefLoss) # cycleloss=0
				else: # The car is parked and charge/discharge

					# Calculate battery temperature when the vehicle is parked but not charging/discharging, battery thermal mananage system (BTMS) works but AC does not work
					Fixeddegradation.charging_temperature(vehicle, ambientT[i], radH[i], DrivingCharge[-1],
				                                     vehicle.battery_model.coefTemp)

					# Calculate cycle life loss caused by charging/discharging
					Fixeddegradation.cycle_loss_drive(vehicle, vehicle.battery_model.batteryT[-1], AllDayCurrent[i], deltasoc[i], vehicle.battery_model.coefLoss)

		# calculate calendar_loss
		Fixeddegradation.calendar_loss(vehicle, vehicle.battery_model.coefLoss, days)

		print(vehicle.battery_model.batteryLoss['calendarLoss'][-1])
		print(sum(vehicle.battery_model.batteryLoss['cycleLoss']))
