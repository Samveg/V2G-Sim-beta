from __future__ import division
import datetime
import matplotlib.pyplot as plt
import v2gsim

project = v2gsim.model.Project()
project = v2gsim.itinerary.from_excel(project, '../data/NHTS/Tennessee.xlsx')
project.vehicles = v2gsim.itinerary.get_cycling_itineraries(project)
project = v2gsim.itinerary.copy_append(project, nb_of_days_to_add=2)

# Assign a new charging function to all the charging stations
# contained in the project. The function assigned is a variation
# of the default uncontrolled charging.
for station in project.charging_stations:
    station.charging = v2gsim.charging.controlled.Q_consumption

# Assign new result function to all locations in the project
# so DR potential can be reccorded in a specific manner. This function
# leads to a different result structure for each location.
for location in project.locations:
    location.result_function = v2gsim.result.location_potential_power_demand

# Initiate SOC and charging infra
v2gsim.core.initialize_SOC(project, nb_iteration=2)

# Launch the simulation
v2gsim.core.run(project, date_from=project.date + datetime.timedelta(hours=12),
                date_to=project.date + datetime.timedelta(days=2, hours=12))

# Sum up the results
result = project.locations[0].result.copy()
for location in project.locations[1:]:
    result += location.result
result = result / (1000 * 1000)  # to MW

# Plot the result
plt.figure()
plt.plot(result.ASAP)
plt.plot(result.nominal)
plt.plot(result.ALAP)
plt.legend(loc=0)
plt.ylabel('Power demand (MW)')
plt.xlabel('Time')
plt.show()
