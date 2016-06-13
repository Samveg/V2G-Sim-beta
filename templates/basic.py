import datetime
import matplotlib.pyplot as plt
import dill as pickle
import pandas
import os
import sys
import traceback
import pdb
sys.path.append(os.path.join(os.path.dirname(__file__), "../../V2G_Sim_beta/"))
import v2gsim


project = v2gsim.model.Project()
project = v2gsim.itinerary.from_excel(project, '../data/NHTS/Tennessee.xlsx')
project = v2gsim.itinerary.copy_append(project, nb_copies=2)

# Add charging infra
charging_stations = []
charging_stations.append(
    v2gsim.model.ChargingStation(name='L1_V1G', maximum_power=1400, minimum_power=0, post_simulation=True))
charging_stations.append(
    v2gsim.model.ChargingStation(name='L2_V2G', maximum_power=7200, minimum_power=-7200, post_simulation=True))
project.charging_stations.extend(charging_stations)

df = pandas.DataFrame(index=['L1_V1G', 'L2_V2G'],
                      data={'charging_station': charging_stations,
                            'probability': [0.5, 0.5]})
for location in project.locations:
    if location.category in ['Work', 'Home']:
        location.available_charging_station = df

# Initiate SOC and charging infra
conv = v2gsim.core.initialize_SOC(project, nb_iteration=1)

for station in project.charging_stations:
    station.charging = v2gsim.charging.controlled.demand_response

# Run V2G-Sim
v2gsim.core.run(project, date_from=project.date + datetime.timedelta(hours=12),
                date_to=project.date + datetime.timedelta(days=2, hours=12),
                charging_option={'startDR': project.date + datetime.timedelta(days=1, hours=17),
                                 'endDR': project.date + datetime.timedelta(days=1, hours=19),
                                 'date_limit': project.date + datetime.timedelta(days=2, hours=12),
                                 'post_DR_window_fraction': 1.5,
                                 'thresholdSOC': 0.2})

total_power_demand = v2gsim.result.total_power_demand(project)

# Plot the result
plt.figure()
plt.plot(total_power_demand.index.tolist(), total_power_demand['total'].values.tolist())
plt.show()
