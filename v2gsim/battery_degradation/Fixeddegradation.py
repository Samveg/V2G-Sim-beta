from __future__ import division
from math import exp
import pdb
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  
import pdb, traceback, sys
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def driving_temperature(vehicle, driving, ambientT, rad, charge, coefTemp):
	# seperately calculate the battery temperature when it is driving, because there is heat generated from the battery
	ambientT = ambientT[int(driving.start*3600):int(driving.end*3600)+1]
	rad = rad[int(driving.start*3600):int(driving.end*3600)+1]
	duration = int(driving.end*3600)-int(driving.start*3600)

	# when the EV is driving, calculate the battery temperature second by second.
	# AC starts to work if cambin temperature is higher than 25, battery cooling system starts to work if battery temperature is higher than 20.
	for i in range(duration):
		if vehicle.extra.cabinT[-1] >= 25 and vehicle.extra.batteryT[-1] >= 20:
			cabinTemp = vehicle.extra.cabinT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.extra.cabinT[-1])+coefTemp['K_bc']*(vehicle.extra.batteryT[-1]-vehicle.extra.cabinT[-1])+rad[i]-4500)/coefTemp['M_c']
			batteryTemp = vehicle.extra.batteryT[-1]+(coefTemp['K_ab']*(ambientT[i]-vehicle.extra.batteryT[-1])+coefTemp['K_bc']*(vehicle.extra.cabinT[-1]-vehicle.extra.batteryT[-1])+charge[i]-354*(vehicle.extra.batteryT[-1]-20))/coefTemp['M_b']
		elif vehicle.extra.cabinT[-1] < 25 and vehicle.extra.batteryT[-1] < 20:
			cabinTemp = vehicle.extra.cabinT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.extra.cabinT[-1])+coefTemp['K_bc']*(vehicle.extra.batteryT[-1]-vehicle.extra.cabinT[-1])+rad[i])/coefTemp['M_c']
			batteryTemp = vehicle.extra.batteryT[-1]+(coefTemp['K_ab']*(ambientT[i]-vehicle.extra.batteryT[-1])+coefTemp['K_bc']*(vehicle.extra.cabinT[-1]-vehicle.extra.batteryT[-1])+charge[i])/coefTemp['M_b']
		elif vehicle.extra.cabinT[-1] >= 25 and vehicle.extra.batteryT[-1] < 20:
			cabinTemp = vehicle.extra.cabinT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.extra.cabinT[-1])+coefTemp['K_bc']*(vehicle.extra.batteryT[-1]-vehicle.extra.cabinT[-1])+rad[i]-4500)/coefTemp['M_c']
			batteryTemp = vehicle.extra.batteryT[-1]+(coefTemp['K_ab']*(ambientT[i]-vehicle.extra.batteryT[-1])+coefTemp['K_bc']*(vehicle.extra.cabinT[-1]-vehicle.extra.batteryT[-1])+charge[i])/coefTemp['M_b']
		else:
			cabinTemp = vehicle.extra.cabinT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.extra.cabinT[-1])+coefTemp['K_bc']*(vehicle.extra.batteryT[-1]-vehicle.extra.cabinT[-1])+rad[i])/coefTemp['M_c']
			batteryTemp = vehicle.extra.batteryT[-1]+(coefTemp['K_ab']*(ambientT[i]-vehicle.extra.batteryT[-1])+coefTemp['K_bc']*(vehicle.extra.cabinT[-1]-vehicle.extra.batteryT[-1])+charge[i]-354*(vehicle.extra.batteryT[-1]-20))/coefTemp['M_b']

        # Save cabin and battery temperature
		vehicle.extra.cabinT.append(cabinTemp)
		vehicle.extra.batteryT.append(batteryTemp)



def charging_temperature(vehicle, charging, ambientT, rad, charge, coefTemp):
	# seperately calculate the battery temperature when it is charging/discharging, because there is heat generated from the battery
	if int(charging.end) == 24:
		ambientT = ambientT[int(charging.start*3600):int(charging.end*3600)]
		rad = rad[int(charging.start*3600):int(charging.end*3600)]
		duration = int(charging.end * 3600) - int(charging.start * 3600)-1
	else:
		ambientT = ambientT[int(charging.start*3600):int(charging.end*3600)+1]
		rad = rad[int(charging.start*3600):int(charging.end*3600)+1]
		duration = int(charging.end * 3600) - int(charging.start * 3600)

	for i in range(duration):
		if vehicle.extra.batteryT[-1] >= 20:
			cabinTemp = vehicle.extra.cabinT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.extra.cabinT[-1])+coefTemp['K_bc']*(vehicle.extra.batteryT[-1]-vehicle.extra.cabinT[-1])+rad[i])/coefTemp['M_c']
			batteryTemp = vehicle.extra.batteryT[-1]+(coefTemp['K_ab']*(ambientT[i]-vehicle.extra.batteryT[-1])+coefTemp['K_bc']*(vehicle.extra.cabinT[-1]-vehicle.extra.batteryT[-1])+charge[i]-354*(vehicle.extra.batteryT[-1]-20))/coefTemp['M_b']
		else:
			cabinTemp = vehicle.extra.cabinT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.extra.cabinT[-1])+coefTemp['K_bc']*(vehicle.extra.batteryT[-1]-vehicle.extra.cabinT[-1])+rad[i])/coefTemp['M_c']
			batteryTemp = vehicle.extra.batteryT[-1]+(coefTemp['K_ab']*(ambientT[i]-vehicle.extra.batteryT[-1])+coefTemp['K_bc']*(vehicle.extra.cabinT[-1]-vehicle.extra.batteryT[-1])+charge[i])/coefTemp['M_b']


		vehicle.extra.cabinT.append(cabinTemp)
		vehicle.extra.batteryT.append(batteryTemp)




def idle_temperature(vehicle, parked, ambientT, rad, coefTemp):
	# seperately calculate the battery temperature when it is idle, because there is no heat generated from the battery.
	# Assume key-off, so AC and thermal management system do not work
	if int(parked.end) == 24:
		ambientT = ambientT[int(parked.start*3600):int(parked.end*3600)]
		rad = rad[int(parked.start*3600):int(parked.end*3600)]
		duration = int(parked.end * 3600) - int(parked.start * 3600)-1
	else:
		ambientT = ambientT[int(parked.start*3600):int(parked.end*3600)+1]
		rad = rad[int(parked.start*3600):int(parked.end*3600)+1]
		duration = int(parked.end * 3600) - int(parked.start * 3600)

	for i in range(duration):
		cabinTemp = vehicle.extra.cabinT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.extra.cabinT[-1])+coefTemp['K_bc']*(vehicle.extra.batteryT[-1]-vehicle.extra.cabinT[-1])+rad[i])/coefTemp['M_c']
		batteryTemp = vehicle.extra.batteryT[-1]+(coefTemp['K_ab']*(ambientT[i]-vehicle.extra.batteryT[-1])+coefTemp['K_bc']*(vehicle.extra.cabinT[-1]-vehicle.extra.batteryT[-1]))/coefTemp['M_b']
		vehicle.extra.cabinT.append(cabinTemp)
		vehicle.extra.batteryT.append(batteryTemp)


def calendar_loss(vehicle, coefLoss,days=365*5):
	"""
	Important!!! Set the number of days for the calendar life loss calculation
	"""

    # copy temperature
	timeSpand = 86400*days
	temperature = vehicle.extra.batteryT*days

	for i in range(1,timeSpand):
		calendarLoss = vehicle.extra.batteryLoss['calendarLoss'][-1] + 1/3600/24*0.5*coefLoss['f']*exp(-coefLoss['E']/coefLoss['R']/(temperature[i]+273.15))*((i/3600/24)**(-0.5))
		vehicle.extra.batteryLoss['calendarLoss'].append(calendarLoss)



def cycle_loss_drive(vehicle, activity, current, soc, coefLoss):
	"""

     Calculate one day cycle life loss.

	"""
	setup1 = [soc[0]] + soc[:]
	setup2 = soc[:] + [soc[-1]]
	deltsoctemp = []
	for i in range(len(setup1)):
		deltsoctemp.append((setup1[i]-setup2[i]))
	deltsoc = deltsoctemp[:-2]


	for i in range(len(deltsoc)):
		loss = (coefLoss['a']*(vehicle.extra.batteryT[int(activity.start*3600)+i]+273.15)**2+coefLoss['b']*(vehicle.extra.batteryT[int(activity.start*3600)+i]+273.15)+coefLoss['c'])*exp((coefLoss['d']*(vehicle.extra.batteryT[int(activity.start*3600)+i]+273.15)+coefLoss['e'])*abs(deltsoc[i])*3600)*(abs(current[i]))/2/2/3600
		vehicle.extra.batteryLoss['cycleLoss'].append(loss)

