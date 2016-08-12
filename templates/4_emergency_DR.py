import datetime
import matplotlib.pyplot as plt
import pandas
import v2gsim

project = v2gsim.model.Project()
project = v2gsim.itinerary.from_excel(project, '../data/NHTS/Tennessee.xlsx')
project = v2gsim.itinerary.copy_append(project, nb_of_days_to_add=2)

# Create some new charging infrastructures, append those new
# infrastructures to the project list of infrastructures
charging_stations = []
charging_stations.append(
    v2gsim.model.ChargingStation(name='L1_V1G', maximum_power=1400, minimum_power=0, post_simulation=True))
charging_stations.append(
    v2gsim.model.ChargingStation(name='L2_V2G', maximum_power=7200, minimum_power=-7200, post_simulation=True))
project.charging_stations.extend(charging_stations)

# Create a data frame with the new infrastructures mix and
# apply this mix at all the locations
df = pandas.DataFrame(index=['L1_V1G', 'L2_V2G'],
                      data={'charging_station': charging_stations,
                            'probability': [0.5, 0.5]})
for location in project.locations:
    if location.category in ['Work', 'Home']:
        location.available_charging_station = df.copy()

# Initiate SOC and charging infrastructures
conv = v2gsim.core.initialize_SOC(project, nb_iteration=1)

# Change the charging function to use the demand response controller
for station in project.charging_stations:
    station.charging = v2gsim.charging.controlled.demand_response

# Run V2G-Sim with a charging_option parameter.
# this parameter will be passed to the charging function at every charging
# events.
v2gsim.core.run(project, date_from=project.date + datetime.timedelta(hours=12),
                date_to=project.date + datetime.timedelta(days=2, hours=12),
                charging_option={'startDR': project.date + datetime.timedelta(days=1, hours=17),
                                 'endDR': project.date + datetime.timedelta(days=1, hours=19),
                                 'date_limit': project.date + datetime.timedelta(days=2, hours=12),
                                 'post_DR_window_fraction': 1.5,
                                 'thresholdSOC': 0.2})

total_power_demand = v2gsim.post_simulation.result.total_power_demand(project)

# Plot the result
plt.figure()
plt.plot(total_power_demand.index.tolist(), total_power_demand['total'].values.tolist())
plt.show()
