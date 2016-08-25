from __future__ import division
import numpy as np
from math import exp
import sys, os


class BatteryModel(object):
	"""Create a class to store all the
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


def driving_temperature(vehicle, ambientT, rad, charge, coefTemp):
	"""Description of the function

	Args:
		arg1 (arg1_type): arg1 description
		arg2 ...
	"""
	# when the EV is driving, calculate the battery temperature second by second.
	# AC starts to work if cambin temperature is higher than 25, battery cooling system starts to work if battery temperature is higher than 20.
	if vehicle.battery_model.cabinT[-1] >= 25 and vehicle.battery_model.batteryT[-1] >= 20:
		cabinTemp = vehicle.battery_model.cabinT[-1]+(coefTemp['K_ac']*(ambientT-vehicle.battery_model.cabinT[-1])+coefTemp['K_bc']*(vehicle.battery_model.batteryT[-1]-vehicle.battery_model.cabinT[-1])+rad-4500)/coefTemp['M_c']
		batteryTemp = vehicle.battery_model.batteryT[-1]+(coefTemp['K_ab']*(ambientT-vehicle.battery_model.batteryT[-1])+coefTemp['K_bc']*(vehicle.battery_model.cabinT[-1]-vehicle.battery_model.batteryT[-1])+charge-354*(vehicle.battery_model.batteryT[-1]-20))/coefTemp['M_b']
	elif vehicle.battery_model.cabinT[-1] < 25 and vehicle.battery_model.batteryT[-1] < 20:
		cabinTemp = vehicle.battery_model.cabinT[-1]+(coefTemp['K_ac']*(ambientT-vehicle.battery_model.cabinT[-1])+coefTemp['K_bc']*(vehicle.battery_model.batteryT[-1]-vehicle.battery_model.cabinT[-1])+rad)/coefTemp['M_c']
		batteryTemp = vehicle.battery_model.batteryT[-1]+(coefTemp['K_ab']*(ambientT-vehicle.battery_model.batteryT[-1])+coefTemp['K_bc']*(vehicle.battery_model.cabinT[-1]-vehicle.battery_model.batteryT[-1])+charge)/coefTemp['M_b']
	elif vehicle.battery_model.cabinT[-1] >= 25 and vehicle.battery_model.batteryT[-1] < 20:
		cabinTemp = vehicle.battery_model.cabinT[-1]+(coefTemp['K_ac']*(ambientT-vehicle.battery_model.cabinT[-1])+coefTemp['K_bc']*(vehicle.battery_model.batteryT[-1]-vehicle.battery_model.cabinT[-1])+rad-4500)/coefTemp['M_c']
		batteryTemp = vehicle.battery_model.batteryT[-1]+(coefTemp['K_ab']*(ambientT-vehicle.battery_model.batteryT[-1])+coefTemp['K_bc']*(vehicle.battery_model.cabinT[-1]-vehicle.battery_model.batteryT[-1])+charge)/coefTemp['M_b']
	else:
		cabinTemp = vehicle.battery_model.cabinT[-1]+(coefTemp['K_ac']*(ambientT-vehicle.battery_model.cabinT[-1])+coefTemp['K_bc']*(vehicle.battery_model.batteryT[-1]-vehicle.battery_model.cabinT[-1])+rad)/coefTemp['M_c']
		batteryTemp = vehicle.battery_model.batteryT[-1]+(coefTemp['K_ab']*(ambientT-vehicle.battery_model.batteryT[-1])+coefTemp['K_bc']*(vehicle.battery_model.cabinT[-1]-vehicle.battery_model.batteryT[-1])+charge-354*(vehicle.battery_model.batteryT[-1]-20))/coefTemp['M_b']

        # Save cabin and battery temperature
	vehicle.battery_model.cabinT.append(cabinTemp)
	vehicle.battery_model.batteryT.append(batteryTemp)


def charging_temperature(vehicle, ambientT, rad, charge, coefTemp):
	"""Description of the function

	Args:
		arg1 (arg1_type): arg1 description
		arg2 ...
	"""
	# seperately calculate the battery temperature when it is charging/discharging, because there is heat generated from the battery
	if vehicle.battery_model.batteryT[-1] >= 20:
		cabinTemp = vehicle.battery_model.cabinT[-1]+(coefTemp['K_ac']*(ambientT-vehicle.battery_model.cabinT[-1])+coefTemp['K_bc']*(vehicle.battery_model.batteryT[-1]-vehicle.battery_model.cabinT[-1])+rad)/coefTemp['M_c']
		batteryTemp = vehicle.battery_model.batteryT[-1]+(coefTemp['K_ab']*(ambientT-vehicle.battery_model.batteryT[-1])+coefTemp['K_bc']*(vehicle.battery_model.cabinT[-1]-vehicle.battery_model.batteryT[-1])+charge-354*(vehicle.battery_model.batteryT[-1]-20))/coefTemp['M_b']
	else:
		cabinTemp = vehicle.battery_model.cabinT[-1]+(coefTemp['K_ac']*(ambientT-vehicle.battery_model.cabinT[-1])+coefTemp['K_bc']*(vehicle.battery_model.batteryT[-1]-vehicle.battery_model.cabinT[-1])+rad)/coefTemp['M_c']
		batteryTemp = vehicle.battery_model.batteryT[-1]+(coefTemp['K_ab']*(ambientT-vehicle.battery_model.batteryT[-1])+coefTemp['K_bc']*(vehicle.battery_model.cabinT[-1]-vehicle.battery_model.batteryT[-1])+charge)/coefTemp['M_b']


	vehicle.battery_model.cabinT.append(cabinTemp)
	vehicle.battery_model.batteryT.append(batteryTemp)


def idle_temperature(vehicle, ambientT, rad, coefTemp):
	"""Description of the function

	Args:
		arg1 (arg1_type): arg1 description
		arg2 ...
	"""
	# seperately calculate the battery temperature when it is idle, because there is no heat generated from the battery.
	# Assume key-off, so AC and thermal management system do not work
	cabinTemp = vehicle.battery_model.cabinT[-1]+(coefTemp['K_ac']*(ambientT-vehicle.battery_model.cabinT[-1])+coefTemp['K_bc']*(vehicle.battery_model.batteryT[-1]-vehicle.battery_model.cabinT[-1])+rad)/coefTemp['M_c']
	batteryTemp = vehicle.battery_model.batteryT[-1]+(coefTemp['K_ab']*(ambientT-vehicle.battery_model.batteryT[-1])+coefTemp['K_bc']*(vehicle.battery_model.cabinT[-1]-vehicle.battery_model.batteryT[-1]))/coefTemp['M_b']
	vehicle.battery_model.cabinT.append(cabinTemp)
	vehicle.battery_model.batteryT.append(batteryTemp)


def calendar_loss(vehicle, coefLoss, days):
	"""Description of the function

	Args:
		arg1 (arg1_type): arg1 description
		arg2 ...
	"""
    # copy temperature
	timeSpand = 86400*days
	temperature = vehicle.battery_model.batteryT*days

	# cumulative calendar life loss
	for i in range(1, timeSpand):
		calendarLoss = vehicle.battery_model.batteryLoss['calendarLoss'][-1] + 1/3600/24*0.5*coefLoss['f']*exp(-coefLoss['E']/coefLoss['R']/(temperature[i]+273.15))*((i/3600/24)**(-0.5))
		vehicle.battery_model.batteryLoss['calendarLoss'].append(calendarLoss)


def cycle_loss_drive(vehicle, bt,current, deltsoc, coefLoss):
	"""Description of the function

	Args:
		arg1 (arg1_type): arg1 description
		arg2 ...
	"""
	# cycle life loss at every time step
	loss = (coefLoss['a']*(bt+273.15)**2+coefLoss['b']*(bt+273.15)+coefLoss['c'])*exp((coefLoss['d']*(bt+273.15)+coefLoss['e'])*abs(deltsoc)*3600)*(abs(current))/2/2/3600
	vehicle.battery_model.batteryLoss['cycleLoss'].append(loss)


def bd(vehicleList, radH, ambientT, days):
	""" battery degradation function

	Args:
	    vehicleList (list of vehicles): vehicles to simulate
	    radH: solar radiation
	    ambientT: ambient temperature
	"""
	for vehicle in vehicleList:
		vehicle.battery_model = BatteryModel()
		DrivingCurrent = vehicle.result.output_current.tolist()   #create this variable to store all the current vector,
		ChargingDemand = vehicle.result.power_demand.tolist()   #create this variable to store all the current vector
		ChargingCurrent = [-i / 380 for i in ChargingDemand]
		AllDayCurrent = [x + y for x,y in zip(DrivingCurrent, ChargingCurrent)]

		deltasoc = [0] + np.diff(vehicle.SOC).tolist()
		DrivingCharge = []
		flag = vehicle.result.parked.tolist() # mark if the car is parked or driving
		R = 0.15 # internal resistance


		for i in range(0, len(AllDayCurrent)):
			DrivingCharge.append(R * AllDayCurrent[i] ** 2) # heat generate from battery
			# Calculate whole day temperature.
			if not flag[i]: # vehicle is driving
				# Calculate battery temeprature when the vehicle is driving
				driving_temperature(vehicle, ambientT[i], radH[i], DrivingCharge[-1],
				                                     vehicle.battery_model.coefTemp)

				# Calculate cycle life loss caused by driving
				cycle_loss_drive(vehicle, vehicle.battery_model.batteryT[-1], AllDayCurrent[i], deltasoc[i], vehicle.battery_model.coefLoss)

			elif flag[i]: # vehicle is parked
				if AllDayCurrent[i] == 0:  # The car is parked but not charging/discharging

					# Calculate battery temperature when the vehicle is parked but not charging/discharging, battery thermal mananage system (BTMS) works but AC does not work
					idle_temperature(vehicle, ambientT[i], radH[i],
				                                     vehicle.battery_model.coefTemp)
					# no cycle life loss, since current=0
					cycle_loss_drive(vehicle, vehicle.battery_model.batteryT[-1], AllDayCurrent[i], deltasoc[i], vehicle.battery_model.coefLoss) # cycleloss=0
				else: # The car is parked and charge/discharge

					# Calculate battery temperature when the vehicle is parked but not charging/discharging, battery thermal mananage system (BTMS) works but AC does not work
					charging_temperature(vehicle, ambientT[i], radH[i], DrivingCharge[-1],
				                                     vehicle.battery_model.coefTemp)

					# Calculate cycle life loss caused by charging/discharging
					cycle_loss_drive(vehicle, vehicle.battery_model.batteryT[-1], AllDayCurrent[i], deltasoc[i], vehicle.battery_model.coefLoss)

		# calculate calendar_loss
		calendar_loss(vehicle, vehicle.battery_model.coefLoss, days)

		print(vehicle.battery_model.batteryLoss['calendarLoss'][-1])
		print(sum(vehicle.battery_model.batteryLoss['cycleLoss'])*days)

