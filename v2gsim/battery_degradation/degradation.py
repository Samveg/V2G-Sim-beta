from __future__ import division
from math import exp
import pdb
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  
import pdb, traceback, sys

# ambientT[activity.start:activity.end]
# every vehicle.outputInterval
# def driving_temperature(vehicle, driving, ambientT, rad, resistance, current, coefTemp):
def driving_temperature(vehicle, driving, ambientT, rad, charge, coefTemp):
	""" coefTemp is a dictionnary of known keys
    *******VERY IMPORTANT	NEED TWO TEMPERATURE*****
    *******ONE IS THE CABIN TEMP AND ONE IS BATTERY TEMP*****

    ****Assume that ambientT and rad already knows the duration of the activity****
    ***ambientT and rad are the length of the day but the resistance  and the current are the length of the duration
	"""
	# duration in hours = driving.end - driving.start
	# hourlySteps=3600/vehicle.outputInterval
	# duration in interval which could be second or minutes = int(parked.end*hourlySteps)-int(parked.start*hourlySteps)
# ****NOTE: make sure () is [] and make sure that the slicing on the ambienT and rad is correct sicne they have the lenght of a day.
# 10.22: is the end time included in the activity??
	try:
	
		ambientT = ambientT[int(driving.start*3600)-1:int(driving.end*3600)-1]
		rad = rad[int(driving.start*3600)-1:int(driving.end*3600)-1]
		# print(rad)
		
		duration = int(driving.end*3600) - int(driving.start*3600)-1
		for i in range(duration):
		# calculate temperature at each time due to differnt time interval
			if vehicle.cabinT[-1] >= 25 and vehicle.batteryT[-1] >= 20:
				cabinTemp = vehicle.cabinT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.cabinT[-1])+coefTemp['K_bc']*(vehicle.batteryT[-1]-vehicle.cabinT[-1])+rad[i]-4500)/coefTemp['M_c']
				# batteryTemp = vehicle.batteryT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.batteryT[-1])+coefTemp['K_bc']*(vehicle.cabinT[i]-vehicle.batteryT[i])+resistance*current^2-354*(vehicle.batteryT[-1]-20))/coefTemp['M_b']
				batteryTemp = vehicle.batteryT[-1]+(coefTemp['K_ab']*(ambientT[i]-vehicle.batteryT[-1])+coefTemp['K_bc']*(vehicle.cabinT[-1]-vehicle.batteryT[-1])+charge[i]-354*(vehicle.batteryT[-1]-20))/coefTemp['M_b']
			elif vehicle.cabinT[-1] < 25 and vehicle.batteryT[-1] < 20:
				cabinTemp = vehicle.cabinT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.cabinT[-1])+coefTemp['K_bc']*(vehicle.batteryT[-1]-vehicle.cabinT[-1])+rad[i])/coefTemp['M_c']
				# batteryTemp = vehicle.batteryT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.batteryT[-1])+coefTemp['K_bc']*(vehicle.cabinT[i]-vehicle.batteryT[i])+resistance*current^2)/coefTemp['M_b']
				batteryTemp = vehicle.batteryT[-1]+(coefTemp['K_ab']*(ambientT[i]-vehicle.batteryT[-1])+coefTemp['K_bc']*(vehicle.cabinT[-1]-vehicle.batteryT[-1])+charge[i])/coefTemp['M_b']
			elif vehicle.cabinT[-1] >= 25 and vehicle.batteryT[-1] < 20:
				cabinTemp = vehicle.cabinT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.cabinT[-1])+coefTemp['K_bc']*(vehicle.batteryT[-1]-vehicle.cabinT[-1])+rad[i]-4500)/coefTemp['M_c']
				# batteryTemp = vehicle.batteryT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.batteryT[-1])+coefTemp['K_bc']*(vehicle.cabinT[i]-vehicle.batteryT[i])+resistance*current^2)/coefTemp['M_b']
				batteryTemp = vehicle.batteryT[-1]+(coefTemp['K_ab']*(ambientT[i]-vehicle.batteryT[-1])+coefTemp['K_bc']*(vehicle.cabinT[-1]-vehicle.batteryT[-1])+charge[i])/coefTemp['M_b']
			else:
				cabinTemp = vehicle.cabinT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.cabinT[-1])+coefTemp['K_bc']*(vehicle.batteryT[-1]-vehicle.cabinT[-1])+rad[i])/coefTemp['M_c']
				# batteryTemp = vehicle.batteryT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.batteryT[-1])+coefTemp['K_bc']*(vehicle.cabinT[i]-vehicle.batteryT[i])+resistance*current^2-354*(vehicle.batteryT[-1]-20))/coefTemp['M_b']
				batteryTemp = vehicle.batteryT[-1]+(coefTemp['K_ab']*(ambientT[i]-vehicle.batteryT[-1])+coefTemp['K_bc']*(vehicle.cabinT[-1]-vehicle.batteryT[-1])+charge[i]-354*(vehicle.batteryT[-1]-20))/coefTemp['M_b']
			# print(batteryTemp)
			vehicle.cabinT.append(cabinTemp)
			vehicle.batteryT.append(batteryTemp)
			
	except:
		    type, value, tb = sys.exc_info()
		    traceback.print_exc()
		    pdb.post_mortem(tb)
	

# def charging_temperature(vehicle, charging, ambientT, rad, resistance, current, coefTemp):
def charging_temperature(vehicle, charging, ambientT, rad, charge, coefTemp):
	try:
		ambientT = ambientT[int(charging.start*3600)-1:int(charging.end*3600)-1]
		rad = rad[int(charging.start*3600)-1:int(charging.end*3600)-1]
		duration = int(charging.end*3600) - int(charging.start*3600)-1
		for i in range(duration):
			cabinTemp = vehicle.cabinT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.cabinT[-1])+coefTemp['K_bc']*(vehicle.batteryT[-1]-vehicle.cabinT[-1])+rad[i])/coefTemp['M_c']
			# batteryTemp = vehicle.batteryT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.batteryT[-1])+coefTemp['K_bc']*(vehicle.cabinT[i]-vehicle.batteryT[i])+resistance*current^2)/coefTemp['M_b']
			batteryTemp = vehicle.batteryT[-1]+(coefTemp['K_ab']*(ambientT[i]-vehicle.batteryT[-1])+coefTemp['K_bc']*(vehicle.cabinT[-1]-vehicle.batteryT[-1])+charge[i])/coefTemp['M_b']
			vehicle.cabinT.append(cabinTemp)
			vehicle.batteryT.append(batteryTemp)

	except:
		    type, value, tb = sys.exc_info()
		    traceback.print_exc()
		    pdb.post_mortem(tb)
def idle_temperature(vehicle, parked, ambientT, rad, coefTemp):
	if int(parked.end) == 24:
		ambientT = ambientT[int(parked.start*3600)-1:int(parked.end*3600)]
		rad = rad[int(parked.start*3600)-1:int(parked.end*3600)]
		duration = parked.end*3600 - parked.start*3600
	else:
		ambientT = ambientT[int(parked.start*3600):int(parked.end*3600)-1]
		rad = rad[int(parked.start*3600):int(parked.end*3600)-1]
		duration = parked.end*3600 - parked.start*3600-1
	try:

		for i in range(int(duration)):
			cabinTemp = vehicle.cabinT[-1]+(coefTemp['K_ac']*(ambientT[i]-vehicle.cabinT[-1])+coefTemp['K_bc']*(vehicle.batteryT[-1]-vehicle.cabinT[-1])+rad[i])/coefTemp['M_c']
			batteryTemp = vehicle.batteryT[-1]+(coefTemp['K_ab']*(ambientT[i]-vehicle.batteryT[-1])+coefTemp['K_bc']*(vehicle.cabinT[-1]-vehicle.batteryT[-1]))/coefTemp['M_b']
			vehicle.cabinT.append(cabinTemp)
			vehicle.batteryT.append(batteryTemp)
	except:
		    type, value, tb = sys.exc_info()
		    traceback.print_exc()
		    pdb.post_mortem(tb)
	

# you would call this functin like this calendar_loss(vehicle.batteryT)
# translation: from the matlab codes, calender loss assumes that the length of the calender is already known
#             therefore, the codes iterate over each interval to calculate the degradation.
def calendar_loss(vehicle, coefLoss,days):
	"""
	****Assume temperature has been converted to celsius
	"""
	dayLength = len(vehicle.batteryT)
	temperature = vehicle.batteryT*days
	timeSpand = dayLength*days
	for i in range(1,timeSpand):
		calendarLoss = vehicle.batteryLoss['calendarLoss'][-1] +1/3600/24*0.5*coefLoss['f']*exp(-coefLoss['E']/coefLoss['R']/(temperature[i]+273.15))*((i/3600/24)**(-0.5))
		vehicle.batteryLoss['calendarLoss'].append(calendarLoss)

	for k in range(1,days+1):
		vehicle.batteryLoss['cumlossDaily'].append(vehicle.batteryLoss['calendarLoss'][k*dayLength-1])


	#10.27 T here is actually the battery temperature
# cycle loss assumes the idle and the charge do not degrade the battery
# translation: cyle_loss assumes that the activity idel and chargin won't have effects on the battery degradaiton.
# 			so I could incoporate the cycle_loss codes into the driving activity.
# for cycle_loss assume given soc with known and right range
# maybe for the cycle_loss we don't need to measure it second-wise? Since by hour to hour, the degradation won't change significantly.

# ****NOTE: vehicle.SOC 
def cycle_loss_drive(vehicle, activity, current, soc, coefLoss):
	# try:
	setup1 = [soc[0]] + soc[:]
	setup2 = soc[:] + [soc[-1]]
	deltsoctemp = []
	cycleLoss =  []
	for i in range(len(setup1)):
		deltsoctemp.append((setup1[i]-setup2[i]))
	deltsoc = deltsoctemp[:-1]
	duration = activity.end*3600 - activity.start*3600-1
	
	for i in range(int(duration)):
		# print(vehicle.batteryT[int(activity.start*3600)+i]+273.15)
		loss = (coefLoss['a']*(vehicle.batteryT[int(activity.start*3600)+i]+273.15)**2+coefLoss['b']*(vehicle.batteryT[int(activity.start*3600)+i]+273.15)+coefLoss['c'])*exp((coefLoss['d']*(vehicle.batteryT[int(activity.start*3600)+i]+273.15)+coefLoss['e'])*deltsoc[i]*3600)*current[i]/3600/2
		cycleLoss.append(loss)
	vehicle.batteryLoss['cycleLoss'].append(sum(cycleLoss))
	
	# for i in range(int(duration)):
	# # for i in range(1):
	# 	try:	
	# 		loss = (coefLoss['a']*((T[int(activity.start*3600)+i])**2)+coefLoss['b']*(T[int(activity.start*3600)+i])+coefLoss['c'])*exp((coefLoss['d']*(T[int(activity.start*3600)+i])+coefLoss['e'])*deltsoc[i]*3600)*(current[i])/3600/2
	# 		first =(coefLoss['a']*(T[int(activity.start*3600)+i])**2+coefLoss['b']*(T[int(activity.start*3600)+i])+coefLoss['c'])
	# 		second = (exp((coefLoss['d']*(T[int(activity.start*3600)+i])+coefLoss['e'])*deltsoc[i]*3600))
	# 		# print(loss)
	# 		cycleLoss.append(second)
	# 	except:
	# 	    type, value, tb = sys.exc_info()
	# 	    traceback.print_exc()
	# 	    pdb.post_mortem(tb)
	# vehicle.batteryLoss['cycleLoss'].append(sum(cycleLoss))
	# print(sum(cycleLoss))
# 11.19 comments: every number is the same but results are different.
	# print(vehicle.batteryLoss['cycleLoss'])

	# except:
	#     type, value, tb = sys.exc_info()
	#     traceback.print_exc()
	#     pdb.post_mortem(tb)




# Class name : VehicleClass
# variable name : vehicleClass
# function : vehicle_class




