from __future__ import division
import datetime
import matplotlib.pyplot as plt
import cPickle as pickle
import v2gsim
import pandas

# Create a project and initialize it with someitineraries
project = v2gsim.model.Project()
project = v2gsim.itinerary.from_excel(project, '../data/NHTS/Tennessee.xlsx')

# This function from the itinerary module return all the vehicles that
# start and end their day at the same location (e.g. home)
project.vehicles = v2gsim.itinerary.get_cycling_itineraries(project)

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
    vehicle.result_function = v2gsim.result.save_detailed_vehicle_power_demand

# Assign different result functions to the vehicle that are plugged
# at least once to a controllable infrastructure
for vehicle in project.vehicles:
    for activity in vehicle.activities:
        if isinstance(activity, v2gsim.model.Parked):
            if activity.charging_station.post_simulation:
                vehicle.result_function = v2gsim.post_simulation.netload_optimization.save_vehicle_state_for_decentralized_optimization
                # pass to the next vehicle
                break

# Launch the simulation
v2gsim.core.run(project)

# Look at the results
total_power_demand = v2gsim.post_simulation.result.total_power_demand(project)

# Create the optimization object
opti = v2gsim.post_simulation.netload_optimization.DecentralizedOptimization(
    project, 10, max_iteration=15, sigma=50)

# Load the net load data
finalResult = pandas.DataFrame()
filename = '../data/netload/2025.pickle'
with open(filename,'rb') as fp:
    net_load = pickle.load(fp)
day = datetime.datetime(2025, 6, 17)
net_load = pandas.DataFrame(net_load[day: day + datetime.timedelta(days=1)]['netload'])

# Initialize the optimization
new_net_load, opti_vehicle_load, total_vehicle_load, net_load_with_vehicles, dual_gap = opti.solve(
    project, net_load * (1000 * 1000), real_number_of_vehicle=1500000)

try:
    plt.plot(dual_gap)
    plt.show()

    plt.plot(opti_vehicle_load)
    plt.show()

    plt.plot(new_net_load, label='netload')
    plt.plot(net_load_with_vehicles, label='vehicles')
    plt.show()
except:
    import pdb
    pdb.set_trace()  
import pdb
pdb.set_trace()