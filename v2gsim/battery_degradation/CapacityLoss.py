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
	"""Calculate the temperature when the EV is driving

	Args:
		vehicle (Vehicle): vehicle object to get current SOC and physical constraints (maximum SOC, ...)
		ambientT (float): ambient temperature at this time step
		rad (float) : solar radiation at this time step
		charge (float): heat generate at this time step from the battery when the EV is driving
		coefTemp (dict) : Coefficients of EV thermal model
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
	"""Calculate the temperature when the EV is charging

	Args:
		vehicle (Vehicle): vehicle object to get current SOC and physical constraints (maximum SOC, ...)
		ambientT (float): ambient temperature at this time step
		rad (float) : solar radiation at this time step
		charge (float): heat generate from the battery when the EV is charging/discharging at this time step
		coefTemp (dict) : Coefficients of EV thermal model
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
	"""Calculate the temperature when the EV is idle

	Args:
		vehicle (Vehicle): vehicle object to get current SOC and physical constraints (maximum SOC, ...)
		ambientT (float): ambient temperature at this time step
		rad (float) : solar radiation at this time step
		coefTemp (dict) : Coefficients of EV thermal model
	"""
	# seperately calculate the battery temperature when it is idle, because there is no heat generated from the battery.
	# Assume key-off, so AC and thermal management system do not work
	cabinTemp = vehicle.battery_model.cabinT[-1]+(coefTemp['K_ac']*(ambientT-vehicle.battery_model.cabinT[-1])+coefTemp['K_bc']*(vehicle.battery_model.batteryT[-1]-vehicle.battery_model.cabinT[-1])+rad)/coefTemp['M_c']
	batteryTemp = vehicle.battery_model.batteryT[-1]+(coefTemp['K_ab']*(ambientT-vehicle.battery_model.batteryT[-1])+coefTemp['K_bc']*(vehicle.battery_model.cabinT[-1]-vehicle.battery_model.batteryT[-1]))/coefTemp['M_b']
	vehicle.battery_model.cabinT.append(cabinTemp)
	vehicle.battery_model.batteryT.append(batteryTemp)


def calendar_loss(vehicle, coefLoss, days):
	"""Calculate battery capacity loss caused by calendar aging

	Args:
		vehicle (Vehicle): vehicle object to get current SOC and physical constraints (maximum SOC, ...)
		coefLoss (dict) : Coefficients of EV capacity loss model
		days (int): the number of days for EV battery calendar aging
	"""
    # copy temperature
	timeSpand = 86400*days
	temperature = vehicle.battery_model.batteryT*days

	# cumulative calendar life loss
	for i in range(1, timeSpand):
		calendarLoss = vehicle.battery_model.batteryLoss['calendarLoss'][-1] + 1/3600/24*0.5*coefLoss['f']*exp(-coefLoss['E']/coefLoss['R']/(temperature[i]+273.15))*((i/3600/24)**(-0.5))
		vehicle.battery_model.batteryLoss['calendarLoss'].append(calendarLoss)


def cycle_loss_drive(vehicle, bt,current, deltsoc, coefLoss):
	"""Calculate battery capacity loss caused by cycling at this time step

	Args:
		vehicle (Vehicle): vehicle object to get current SOC and physical constraints (maximum SOC, ...)
		bt (float) : battery temperature at this time step
		current (float): battery current input/output at this time step
		deltsoc (float): incremental soc at this time step
		coefLoss (dict): Coefficients of EV capacity loss model
	"""
	# cycle life loss at current time step
	loss = (coefLoss['a']*(bt+273.15)**2+coefLoss['b']*(bt+273.15)+coefLoss['c'])*exp((coefLoss['d']*(bt+273.15)+coefLoss['e'])*abs(deltsoc)*3600)*(abs(current))/2/2/3600
	vehicle.battery_model.batteryLoss['cycleLoss'].append(loss)


def bd(vehicleList, radH, ambientT, days):
	""" battery degradation function

	Args:
	    vehicleList (list of vehicles): vehicles to simulate
	    radH (list): solar radiation
	    ambientT (list): ambient temperature
	    days (int): the number of days for EV battery calendar aging
	"""
	for indexV, vehicle in enumerate(vehicleList):
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


		#print the capacity loss caused by calendar aging and cycling
		print "The calendar life loss of Vehicle %s is %s"  % (indexV+1, vehicle.battery_model.batteryLoss['calendarLoss'][-1])
		print "The cycle life loss of Vehicle %s is %s" %(indexV+1, sum(vehicle.battery_model.batteryLoss['cycleLoss'])*days)

