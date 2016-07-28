import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # YOUR PATH
import core
import model
import itinerary.load
import degradation
import driving.detailed.init_model
import driving.drivecycle.generator
import pdb, traceback, sys
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn
import pdb

os.system('cls' if os.name == 'nt' else 'clear')
print('###############')
print('### V2G-Sim ###')
print('###############')
print('')


# Initialize location and vehicles from excel file
excelFilename = '../../data/NHTS_data/test_simple.xlsx'  # YOUR PATH
vehicleList, locationList = itinerary.load.excel(excelFilename)


coefTemp = {'q_havc':4500, 'M_c':101771,'M_b':182000,'K_ab':4.343,'K_ac':22.6,'K_bc':3.468}
coefLoss = {'a':8.888888888889532*10**(-6), 'b' : -0.005288888888889,'c': 0.787113333333394, 'd': -0.0067, 'e':2.35,'f':10720,'E':24500,'R':8.314}
# Load the weather data
# use a function to load .mat file from Dai

r = open('radh.txt','r')
dataRead = r.readlines()
for i in range(len(dataRead)-1):
	dataRead[i] = dataRead[i][:-1]
radH = []
for i in range(len(dataRead)):
	for k in range(3600):
		radH.append(float(dataRead[i]))
Q_drive1 = sio.loadmat('Q_drive.mat')['Q'][0]
Q_drive = []
for i in range(len(Q_drive1)):
	Q_drive.append(Q_drive1[i])
soc1 = sio.loadmat('correct soc.mat')['soc'][0][0][0]
soc = []
for i in range(len(soc1)):
	soc.append(soc1[i])
Qcharge = sio.loadmat('Qcharge.mat')['Qcharge'][0]
current = sio.loadmat('I_current.mat')['I'][0]
T = sio.loadmat('T.mat')['T'][0]

days = 10

# l = len(resistance)/10

# 10.21: one flaw is that th output interval is hard to control here as the file default interval is one hour.

# use a function to interpolate wther data at the right time sample (vehicle.outputInterval)

# 10.22: Now from above I have the right size of the radiation.
# 10.22: Below is to get right size of data from ambientT
t = open('temh.txt','r')
dataReadt = t.readlines()
for i in range(len(dataReadt)-1):
	dataReadt[i] = dataReadt[i][0:2]
ambientT = []
for i in range(len(dataReadt)):
	for k in range(3600):
		ambientT.append(float(dataReadt[i]))


DaiCabinT1 = sio.loadmat('DaiCabinT.mat')['x'][0]
DaiCabinT = []
for i in range(len(DaiCabinT1)):
	DaiCabinT.append(DaiCabinT1[i])


DaiBT1 = sio.loadmat('DaiBatteryT.mat')['y'][0]
DaiBT = []
for i in range(len(DaiCabinT1)):
	DaiBT.append(DaiBT1[i])

# length: radH: 86400, Q_drive:1369, soc:1369, Qcharge: 1369, 30
#    		current:1369, ambientT: 86400)

# Load Dai's output from the power-train model
# In generator file there is an example showing how to load mat file.
# transform the txt file into mat as well.
# use a function to interpolate data at the right time sample (vehicle.outputInterval)
# # Init the power-train model
# carModel = driving.detailed.init_model.load_powertrain('../driving/detailed/data.xlsx') #***driving.detailed.data
# driving.detailed.init_model.assign_powertrain_model(vehicleList, carModel)

# driving.drivecycle.generator.assign_EPA_cycle(vehicleList, constGrade=0)

# # Run the simulation for a full day
# core.runV2GSim(vehicleList, nbIteration=3)

# For every activity inside the vehicle itinerary calculate battery loss

for vehicle in vehicleList:
	for activity in vehicle.itinerary[0].activity:
		if isinstance(activity, model.Driving):
# 			# degradation.driving_temperature(vehicleList[0], activity, ambientT, radH, resistance, current, coefTemp)
			degradation.driving_temperature(vehicle, activity, ambientT, radH, Q_drive, coefTemp)
			# degradation.cycle_loss_drive(vehicle, activity, current, soc, coefLoss,T)
		elif isinstance(activity,model.Parked):
			if activity.pluggedIn:
				degradation.charging_temperature(vehicle, activity, ambientT, radH, Qcharge, coefTemp)
			else:
				degradation.idle_temperature(vehicle, activity, ambientT, radH, coefTemp)
	for activity in vehicle.itinerary[0].activity:
		if isinstance(activity, model.Driving):
			degradation.cycle_loss_drive(vehicle, activity, current, soc, coefLoss)
	# 		current: cyclelifeloss ---> Itemp ---> why average? Is current ==
	degradation.calendar_loss(vehicle,coefLoss,10)
	
	           
	print("The cumulative daily cycle life loss is " + str(sum(vehicle.batteryLoss['cycleLoss'])) + ", which is "+str(sum(vehicle.batteryLoss['cycleLoss'])*100)+ "% \n");
	print("The loss of each day " + str(vehicle.batteryLoss['cumlossDaily']))
	# print(sum(vehicle.batteryLoss['calendarLoss']))

	# x = range(len(vehicle.batteryT))
	# x = [x[i]/60 for i in range(len(x))]

	# plt.figure(1)
	# plt.title('Battery Temperature(C) vs. Time(s)',fontsize =18)
	# plt.ylabel('Battery Temperature(C)',fontsize =14)
	# plt.xlabel('time',fontsize =14)
	# plt.plot(x,vehicle.batteryT,'g')
	# plt.plot(x,DaiBT[:-6],'b')
	# plt.show()



	# plt.figure(2)
	# plt.title('Cabin Temperature(C) vs. Time(s)',fontsize =18)
	# plt.ylabel('Cabin Temperature(C)',fontsize=14)
	# plt.xlabel('time(T)',fontsize=14)
	# plt.plot(x, vehicle.cabinT,'b-')
	# plt.plot(x,DaiCabinT[:-6],'r-')
	# plt.show()


	# x = [i for i in range(1,days+1)]
	# cyclossDaily = sum(vehicle.batteryLoss['cycleLoss'])
	# cyclossTot = cyclossDaily*days
	# cyclossCum = []
	# for i in range(1,days+1):
	# 	cyclossCum.append(cyclossDaily*i)
	# cyclossCum.append(cyclossTot)
	# # fix calendar life coefficient
	# beta1 = 0.9343
	# totalLoss=[]
	# for i in range(days):
	# 	total = cyclossCum[i]+vehicle.batteryLoss['cumlossDaily'][i]*0.72*beta1
	# 	totalLoss.append(total)
	# plt.figure(3)
	# plt.title('Total Capacity Loss',fontsize=18)
	# plt.ylabel('Total capacity loss(%)',fontsize=20)
	# plt.xlabel('Time(days)',fontsize=20)
	# plt.plot(x,totalLoss,'b-')
	# plt.show()


	# except:
	#     type, value, tb = sys.exc_info()
	#     traceback.print_exc()
	#     pdb.post_mortem(tb)
# try:
# 	plt.figure(3)
# 	plt.title('Battery Temperature(C) vs. Time(s)',fontsize =18)
# 	plt.ylabel('Battery Temperature(C)',fontsize =14)
# 	plt.xlabel('time',fontsize =14)
# 	plt.plot(x, vehicle.)
# 	plt.show()

# except:
# 	    type, value, tb = sys.exc_info()
# 	    traceback.print_exc()
# 	    pdb.post_mortem(tb)
# try to plot stuff
#     pdb.post_mortem(tb) end before the function, stop it from jumping out but check if anything in batteryLoss


# Debug usefull tools
# try:

# except:
#     type, value, tb = sys.exc_info()
#     traceback.print_exc()
#     pdb.post_mortem(tb)

# import pdb; pdb.set_trace()