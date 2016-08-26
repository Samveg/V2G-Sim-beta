from __future__ import division
import datetime
import matplotlib.pyplot as plt
import cPickle as pickle
import v2gsim
import pandas

# ### Require gurobi or CPLEX #####
# Create a project and initialize it with someitineraries
project = v2gsim.model.Project()
project = v2gsim.itinerary.from_excel(project, '../data/NHTS/Tennessee.xlsx')
project = v2gsim.itinerary.copy_append(project, nb_of_days_to_add=2)

# This function from the itinerary module return all the vehicles that
# start and end their day at the same location (e.g. home)
project.vehicles = v2gsim.itinerary.get_cycling_itineraries(project)

# Reduce the number of vehicles
project.vehicles = project.vehicles[0:100]

# Create some new charging infrastructures, append those new
# infrastructures to the project list of infrastructures
charging_stations = []
charging_stations.append(
    v2gsim.model.ChargingStation(name='L2', maximum_power=7200, minimum_power=0))
charging_stations.append(
    v2gsim.model.ChargingStation(name='L1_V1G', maximum_power=1400, minimum_power=0, post_simulation=True))
charging_stations.append(
    v2gsim.model.ChargingStation(name='L2_V2G', maximum_power=7200, minimum_power=-7200, post_simulation=True))
project.charging_stations.extend(charging_stations)

# Create a data frame with the new infrastructures mix and
# apply this mix at all the locations
df = pandas.DataFrame(index=['L2', 'L1_V1G', 'L2_V2G'],
                      data={'charging_station': charging_stations,
                            'probability': [0.0, 0.4, 0.6]})
for location in project.locations:
    if location.category in ['Work', 'Home']:
        location.available_charging_station = df.copy()

# Initiate SOC and charging infrastructures
v2gsim.core.initialize_SOC(project, nb_iteration=2)

# Assign a basic result function to save power demand
for vehicle in project.vehicles:
    vehicle.result_function = v2gsim.post_simulation.netload_optimization.save_vehicle_state_for_optimization

# Launch the simulation
v2gsim.core.run(project, date_from=project.date + datetime.timedelta(days=1),
                date_to=project.date + datetime.timedelta(days=2),
                reset_charging_station=False)

# Look at the results
total_power_demand = v2gsim.post_simulation.result.total_power_demand(project)

# Optimization
myopti = v2gsim.post_simulation.netload_optimization.CentralOptimization(project, 10,
                                                                         project.date + datetime.timedelta(days=1),
                                                                         project.date + datetime.timedelta(days=2),
                                                                         minimum_SOC=0.1, maximum_SOC=0.95)
# Load the net load data
finalResult = pandas.DataFrame()
filename = '../data/netload/2025.pickle'
with open(filename, 'rb') as fp:
    net_load = pickle.load(fp)
day = datetime.datetime(2025, 6, 17)
net_load = pandas.DataFrame(net_load[day: day + datetime.timedelta(days=1)]['netload'])

myresult = myopti.solve(project, net_load * 1000000,
                        1500000, peak_shaving='peak_shaving', SOC_margin=0.05)

# Get the result in the right format
temp_vehicle = pandas.DataFrame(
    (total_power_demand['total'] - myresult['vehicle_before'] + myresult['vehicle_after']) *
    (1500000 / len(project.vehicles)) / (1000 * 1000))  # Scale up and W to MW
temp_vehicle = temp_vehicle.rename(columns={0: 'vehicle'})
temp_vehicle['index'] = range(0, len(temp_vehicle))
temp_vehicle = temp_vehicle.set_index(['index'], drop=True)

temp_netload = net_load.copy()
temp_netload = temp_netload.resample('60S')
temp_netload = temp_netload.fillna(method='ffill').fillna(method='bfill')
temp_netload = temp_netload.head(len(temp_vehicle))
tempIndex = temp_netload.index
temp_netload['index'] = range(0, len(temp_vehicle))
temp_netload = temp_netload.set_index(['index'], drop=True)

temp_result = pandas.DataFrame(temp_netload['netload'] + temp_vehicle['vehicle'])
temp_result = temp_result.rename(columns={0: 'netload'})
temp_result = temp_result.set_index(tempIndex)
temp_netload = temp_netload.set_index(tempIndex)
temp_vehicle = temp_vehicle.set_index(tempIndex)

plt.plot(temp_netload['netload'], label='netload')
plt.plot(temp_result['netload'], label='netload + vehicles')
plt.ylabel('Power (MW)')
plt.legend()
plt.show()

import pdb
pdb.set_trace()
