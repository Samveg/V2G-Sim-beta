from __future__ import division
from math import exp
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  


def driving_temperature(vehicle, ambientT, rad, charge, coefTemp):
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
	# seperately calculate the battery temperature when it is idle, because there is no heat generated from the battery.
	# Assume key-off, so AC and thermal management system do not work

	cabinTemp = vehicle.battery_model.cabinT[-1]+(coefTemp['K_ac']*(ambientT-vehicle.battery_model.cabinT[-1])+coefTemp['K_bc']*(vehicle.battery_model.batteryT[-1]-vehicle.battery_model.cabinT[-1])+rad)/coefTemp['M_c']
	batteryTemp = vehicle.battery_model.batteryT[-1]+(coefTemp['K_ab']*(ambientT-vehicle.battery_model.batteryT[-1])+coefTemp['K_bc']*(vehicle.battery_model.cabinT[-1]-vehicle.battery_model.batteryT[-1]))/coefTemp['M_b']
	vehicle.battery_model.cabinT.append(cabinTemp)
	vehicle.battery_model.batteryT.append(batteryTemp)


def calendar_loss(vehicle, coefLoss,days):
	"""
	Important!!! Set the number of days for the calendar life loss calculation
	"""

    # copy temperature
	timeSpand = 86400*days
	temperature = vehicle.battery_model.batteryT*days

	# cumulative calendar life loss
	for i in range(1,timeSpand):
		calendarLoss = vehicle.battery_model.batteryLoss['calendarLoss'][-1] + 1/3600/24*0.5*coefLoss['f']*exp(-coefLoss['E']/coefLoss['R']/(temperature[i]+273.15))*((i/3600/24)**(-0.5))
		vehicle.battery_model.batteryLoss['calendarLoss'].append(calendarLoss)



def cycle_loss_drive(vehicle, bt,current, deltsoc, coefLoss):
	# cycle life loss at every time step
	loss = (coefLoss['a']*(bt+273.15)**2+coefLoss['b']*(bt+273.15)+coefLoss['c'])*exp((coefLoss['d']*(bt+273.15)+coefLoss['e'])*abs(deltsoc)*3600)*(abs(current))/2/2/3600
	vehicle.battery_model.batteryLoss['cycleLoss'].append(loss)

