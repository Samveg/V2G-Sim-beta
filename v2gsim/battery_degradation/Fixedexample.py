import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../battery_degradation"))  # YOUR PATH
import core
import model
import itinerary.load
import Fixeddegradation
import driving.detailed.init_model
import driving.drivecycle.generator
import pdb, traceback, sys
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn

import scipy.interpolate as sp
import numpy as np
import pylab

os.system('cls' if os.name == 'nt' else 'clear')
print('###############')
print('### V2G-Sim ###')
print('###############')
print('')


class Degradation(object):
	"""create a class to store all the
	   degradation information
	"""

	def __init__(self):
		self.cabinT = [20]
		self.batteryT = [18]
		self.batteryT_forCalander = []
		self.batteryCurrent_forCalender = []
		self.coefCalender = []
		self.batteryLoss = {'cycleLoss': [0], 'calendarLoss': [0], 'cumlossDaily': [],'calendarLifeLoss':[]}
		# input parameters of EV thermal network and degradation model
		self.coefTemp = {'q_havc': 4500, 'M_c': 101771, 'M_b': 182000, 'K_ab': 4.343, 'K_ac': 22.6, 'K_bc': 3.468}
		self.coefLoss = {'a': 8.888888888889532 * 10 ** (-6), 'b': -0.005288888888889, 'c': 0.787113333333394,
		                 'd': -0.0067, 'e': 2.35, 'f': 8720, 'E': 24500, 'R': 8.314}


def bd(vehicleList, radH, ambientT, vehicleLoad=None):
	""" battery degradation function

	Args:
	    vehicleList: vehicleList (list of vehicles): vehicles to simulate
	    radH: solar radiation
	    ambientT: ambient temperature
	    vehicleLoad:

	Returns:

	"""
	for vehicle in vehicleList:
		vehicle.extra = Degradation()
		totalDrivingCurrent = []   #create this variable to store all the current from driving activities. this will help calculate the driving cycle loss
		totalChargingCurrent = []  #create this variable to store all the current from charging activities. this will help calculate the charging cycle loss
		R =0.15
		soc = vehicle.SOC
		allDayCurrent = [] #create this variable to store all the current. Then to calculate calender loss, determine the time point at which the current is zero.

		#Calculate Temperature at different status: driving, parked, idle
		for activity in vehicle.itinerary[0].activity:

			# Driving
			if isinstance(activity, model.Driving):
				drivingCurrent = activity.extra.ess.i_out
				allDayCurrent.extend(drivingCurrent)
				drivingCharge = []

				if len(drivingCurrent)==1&vehicle.outputInterval != 1:
					# if outputinterval is not 1 and driving activity has just one value
					duration = int(activity.end * 3600) - int(activity.start * 3600)
					interpcurrent=drivingCurrent*duration
					totalDrivingCurrent.extend(interpcurrent)
					drivingCharge=[R*totalDrivingCurrent[0]**2]*duration

				elif len(drivingCurrent) > 1 & vehicle.outputInterval != 1:
					# if outputinterval is not 1 and driving activity has more than one value
					duration = int(activity.end * 3600) - int(activity.start * 3600)
					x = np.linspace(0,len(drivingCurrent),len(drivingCurrent))
					xval = np.linspace(0,duration,duration)
					interpcurrent = np.interp(xval, x, drivingCurrent)
					totalDrivingCurrent.extend(interpcurrent)
					for i in range(len(totalDrivingCurrent)):
						drivingCharge.append(R*totalDrivingCurrent[i]**2)

				else:
					# if outputinterval is 1 
					for i in range(len(drivingCurrent)):
						drivingCharge.append(R*drivingCurrent[i]**2)
					drivingCharge.append(R*drivingCharge[-1]**2)
					totalDrivingCurrent.extend(drivingCurrent)
				Fixeddegradation.driving_temperature(vehicle, activity, ambientT, radH, drivingCharge,
				                                     vehicle.extra.coefTemp)

			# Parked
			elif isinstance(activity, model.Parked):
				# plugged in charging
				if activity.pluggedIn:
					if vehicleLoad == None:
						power = activity.powerDemand
					chargingcurrent =[]
					Qcharge=[]
					for i in range(0, len(power)):
						chargingcurrent.append(-power[i]/380)

					if vehicle.outputInterval == 1:
						for i in range(len(chargingcurrent)):
							Qcharge.append(R*chargingcurrent[i]**2)
					else:
						duration = int(activity.end * 3600) - int(activity.start * 3600)
						x = np.linspace(0,len(power),len(power))
						xval = np.linspace(0,duration,duration)
						chargingcurrent = np.interp(xval, x, chargingcurrent)
						for i in range(len(chargingcurrent)):
							Qcharge.append(R*chargingcurrent[i]**2)
					totalChargingCurrent.extend(chargingcurrent)
					allDayCurrent.extend(chargingcurrent)
					Fixeddegradation.charging_temperature(vehicle, activity, ambientT, radH, Qcharge,
				                                     vehicle.extra.coefTemp)
				# plugged in idle
				else:
					duration = int(activity.end * 3600) - int(activity.start * 3600)
					current=[0]*duration
					allDayCurrent.extend(current)
					Fixeddegradation.idle_temperature(vehicle, activity, ambientT, radH, vehicle.extra.coefTemp)
				# pdb.set_trace()

		# Interpolate the SOC if output interval != 1
		drivingstart = 0
		chargingstart = 0
		if vehicle.outputInterval != 1:
			x = np.linspace(0,len(soc),len(soc))
			xval = np.linspace(0,86400,86401)
			soc = np.interp(xval, x, soc).tolist()

		# Calculate the capacity loss caused by cycling.
		for activity in vehicle.itinerary[0].activity:
			if isinstance(activity, model.Driving):
				duration = int(activity.end * 3600) - int(activity.start * 3600)
				Fixeddegradation.cycle_loss_drive(vehicle, activity, totalDrivingCurrent[drivingstart: drivingstart + duration], soc[int(activity.start*3600): int(activity.start*3600) + duration], vehicle.extra.coefLoss)
				drivingstart = drivingstart + duration
			if isinstance(activity, model.Parked):
				if activity.pluggedIn:
					duration = int(activity.end * 3600) - int(activity.start * 3600)
					Fixeddegradation.cycle_loss_drive(vehicle, activity, totalChargingCurrent[chargingstart: chargingstart + duration], soc[int(activity.start*3600): int(activity.start*3600) + duration], vehicle.extra.coefLoss)
					chargingstart = chargingstart + duration

		# After getting temperature, call calendar loss function
		Fixeddegradation.calendar_loss(vehicle, vehicle.extra.coefLoss)

		# Print the calendar life loss and cycle life loss by the end of 10th yeaer
		print(vehicle.extra.batteryLoss['calendarLoss'][-1])
		print(sum(vehicle.extra.batteryLoss['cycleLoss'])*3650)


        # # Save single day cycle life loss
		# y=[]
		# y=sum(vehicle.extra.batteryLoss['cycleLoss'])
        # # write cycle life loss data for one day
		# with open('savecycleloss.txt','a+') as cycleloss_file:
		# 	cycleloss_file.write(str(y) + '\n')


#		# Save ten year cycle life loss
# 		y=[]
# 		x=np.arange(0,3650,10)
# 		for i in x:
# 			y.append(sum(vehicle.extra.batteryLoss['cycleLoss'])* beta * i)
# 		with open('savecycleloss.txt','a+') as cycleloss_file:
# 			cycleloss_file.write(' '.join(str(e) for e in y) + '\n')

#       #save 10 year calendar life loss
# 		x=np.arange(0,3649,365)
# 		y=[]
# 		for i in x:
# 			y.append(vehicle.extra.batteryLoss['calendarLoss'][i * 86400])
#
# 		with open('savecalendarloss.txt','a+') as calendarloss_file:
# 			calendarloss_file.write(' '.join(str(e) for e in y))

#       #save total 10 year capacity loss
		# y=[]
		# x=np.arange(0,3650,10)
		# for i in x:
		# 	y.append(sum(vehicle.extra.batteryLoss['cycleLoss'])* beta * i + vehicle.extra.batteryLoss['calendarLoss'][i * 86400])
		# with open('savetotolloss.txt','a+') as totalloss_file:
		# 	totalloss_file.write(' '.join(str(e) for e in y) + '\n')



		#plot total
		# x = range(0,3650)
		# y=[]
		# # # pdb.set_trace()
		# for i in x:
		# 	y.append(sum(vehicle.extra.batteryLoss['cycleLoss']) * beta * x[i] + vehicle.extra.batteryLoss['calendarLoss'][i * len(vehicle.extra.batteryT_forCalander) - 1])
		# plt.plot(x,y, 'k')
		# #labels=range(0,11)
		# #ax.set_xticklabels(labels)
        #
		# plt.ylabel('Total Capacity Loss (%)',fontsize =14)
		# plt.xlabel('Number of years',fontsize =14)
		# plt.title('Total Loss vs. Time(s)',fontsize =18)
		# plt.show()

		# # x = range(len(vehicle.extra.batteryLoss['calendarLoss']))
		# x  = range(1,11)
		# y=[]
		# for i in x: 
		# 	y.append(vehicle.extra.batteryLoss['calendarLoss'][i * 365 * len(vehicle.extra.batteryT_forCalander) - 1])
		# # plt.plot(x,vehicle.extra.batteryLoss['calendarLoss'],'g')
		# plt.plot(x,y,'g')
		# plt.ylabel('Calendarloss',fontsize =14)
		# plt.xlabel('number of years',fontsize =14)
		# plt.title('Calendarloss vs. Time(s)',fontsize =18)
		# plt.xticks([1,2,3,4,5,6,7,8,9,10])
		# plt.show()

		# x = range(1,11)
		# y = [sum(vehicle.extra.batteryLoss['cycleLoss']) * 365 * beta * x[i] for i in range(len(x))]
		# plt.plot(x,y,'r')
		# plt.xticks([1,2,3,4,5,6,7,8,9,10])
		# plt.ylabel('Cycleloss',fontsize =14)
		# plt.xlabel('number of years',fontsize =14)
		# plt.title('Cycleloss vs. Time(s)',fontsize =18)
		# plt.show()


		# x = range(len(vehicle.extra.batteryLoss['cycleLoss']))
		# plt.plot(x,vehicle.extra.batteryLoss['cycleLoss'],'r')
		# plt.show()
		
		# print(sum(vehicle.extra.batteryLoss['cycleLoss']))
		# print(vehicle.extra.batteryLoss['calendarLoss'][-1])
		# print(vehicle.extra.batteryLoss['calendarLoss'][-1]+sum(vehicle.extra.batteryLoss['cycleLoss'])*beta)
		# print(vehicle.extra.batteryLoss['calendarLoss'][-1]+sum(vehicle.extra.batteryLoss['cycleLoss'])*300*beta)
		# print(vehicle.extra.batteryLoss['cycleLoss'])
		# x = range(len(vehicle.extra.batteryT))
		# x = [x[i]/60 for i in range(len(x))]
		# plt.figure(1)
		# plt.title('Battery Temperature(C) vs. Time(s)',fontsize =18)
		# plt.ylabel('Battery Temperature(C)',fontsize =14)
		# plt.xlabel('time',fontsize =14)
		# plt.plot(x,vehicle.extra.batteryT,'g')
		# plt.show()
		# print(vehicle.extra.batteryLoss['cumlossDaily'])

#
