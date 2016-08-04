from __future__ import division
import datetime
import matplotlib.pyplot as plt
import pandas
import numpy
import os
import sys
import v2gsim
import seaborn as sns
sns.set_style("whitegrid")
sns.despine()


def save_location_state_DR(location, timestep, date_from, date_to,
                        vehicle=None, activity=None,
                        power_demand=None, SOC=None, nb_interval=None,
                        init=False, run=False, post=False):
    """Save local results from a parked activity at run time.
    """
    if run:
        activity_index1, activity_index2, location_index1, location_index2, save = v2gsim.result._map_index(
            activity.start, activity.end, date_from, date_to, len(power_demand),
            len(location.result['pmax']), timestep)

        # Save a lot of interesting result
        if save:
            if activity.charging_station.post_simulation:
                location.result['controlled_power_demand'][location_index1:location_index2] += (
                    power_demand[activity_index1:activity_index2])

                location.result['pmax'][location_index1:location_index2] += activity.charging_station.maximum_power
            else:
                location.result['uncontrolled_power_demand'][location_index1:location_index2] += (
                    power_demand[activity_index1:activity_index2])

    elif init:
        # Initiate a dictionary of numpy array to hold result (faster than DataFrame)
        location.result = {'uncontrolled_power_demand': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                           'controlled_power_demand': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                           'pmax': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep))}

    elif post:
        # Convert location result back into pandas DataFrame (faster that way)
        i = pandas.date_range(start=date_from, end=date_to,
                              freq=str(timestep) + 's', closed='left')
        location.result = pandas.DataFrame(index=i, data=location.result)


def initialize_project():
    """Initialize a project with all infrastructures and vehicles loaded

    Return:
        project object
    """
    # Create the project structure
    project = v2gsim.model.Project()
    project = v2gsim.itinerary.from_excel(project, '../data/NHTS/California.xlsx')
    
    # Purge the vehicle with non-cycling itineraries
    project.vehicles = v2gsim.itinerary.get_cycling_itineraries(project)
    # # Extend the timeframe
    # project = v2gsim.itinerary.copy_append(project, nb_copies=2)

    # Create two new charging infrastructure and append to the project list of infrastructures
    charging_stations = []
    charging_stations.append(
        v2gsim.model.ChargingStation(name='L1_controllable', maximum_power=1400, minimum_power=0, post_simulation=True))
    charging_stations.append(
        v2gsim.model.ChargingStation(name='L2_controllable', maximum_power=7200, minimum_power=0, post_simulation=True))
    charging_stations.append(
        v2gsim.model.ChargingStation(name='L3_controllable', maximum_power=120000, minimum_power=0, post_simulation=True))
    project.charging_stations.extend(charging_stations)


    # Add the new charging infrastructure at each location
    temp1 = pandas.DataFrame(index=['L1_controllable'],
                    data={'charging_station': charging_stations[0],
                    'probability': 0.0, 'available': float('inf'), 'total': float('inf')})
    temp2 = pandas.DataFrame(index=['L2_controllable'],
                    data={'charging_station': charging_stations[1],
                    'probability': 0.0, 'available': float('inf'), 'total': float('inf')})
    temp3 = pandas.DataFrame(index=['L3_controllable'],
                    data={'charging_station': charging_stations[2],
                    'probability': 0.0, 'available': float('inf'), 'total': float('inf')})
    for location in project.locations:
        location.available_charging_station = pandas.concat([location.available_charging_station, temp1], axis=0)
        location.available_charging_station = pandas.concat([location.available_charging_station, temp2], axis=0)
        location.available_charging_station = pandas.concat([location.available_charging_station, temp3], axis=0)

    # Assign new result function to all locations so DR potential can be reccorded
    for location in project.locations:
        location.result_function = save_location_state_DR

    return project


def set_infrastructure_probabilities(project, home, work, other_location):
    """Set new probability for charging infrastructure
    """
    # Set the charging infrastructure at each location
    for location in project.locations:
        if location.category == 'Home':
            for key, value in home.iteritems():
                location.available_charging_station.loc[key, 'probability'] = value

        elif location.category == 'Work':
            for key, value in work.iteritems():
                location.available_charging_station.loc[key, 'probability'] = value
        else:
            for key, value in other_location.iteritems():
                location.available_charging_station.loc[key, 'probability'] = value


def run_model(project):
    """Run initialization function and run final simulation
    """
    # Initiate SOC and charging infra
    conv = v2gsim.core.initialize_SOC(project, nb_iteration=1)

    # Launch the simulation
    v2gsim.core.run(project)


def plot_result(results, yearList):
    """Plot the result we are interested in
    """
    # Look at the results
    for result, year in zip(results, yearList):
        plt.plot(result.uncontrolled_power_demand + result.controlled_power_demand, label='power_demand_' + str(year))
    plt.legend(loc=0)
    plt.title('Total power demand')
    plt.ylabel('Power (GW)')
    plt.xlabel('Time')
    plt.show()

    for result, year in zip(results, yearList):
        plt.plot(result.uncontrolled_power_demand, label='pmin_' + str(year))
    plt.legend(loc=0)
    plt.title('Pmin')
    plt.ylabel('Power (GW)')
    plt.xlabel('Time')
    plt.show()

    for result, year in zip(results, yearList):
        plt.plot(result.pmax, label='pmax_' + str(year))
    plt.legend(loc=0)
    plt.title('Pmax')
    plt.ylabel('Power (GW)')
    plt.xlabel('Time')
    plt.show()

# ############################################################################################
# Create the project with the right number of vehicles and the available chargers
project = initialize_project()
yearList = [2016, 2020, 2030, 2040, 2050]

# Fleet mix
fleet_mix = pandas.DataFrame(index=yearList, data={
    'PHEV': [182223, 652662, 3172894, 9027923, 12000000],
    'BEV': [133550, 315235, 1023833, 2753486, 5416570],
    'battery_capcity_increase': [1.0, 2.0, 3.0, 3.5, 4.0],
    'BEV_performance_increase' : [1.0, 1.26, 1.4375, 1.6125, 1.7875],
    'PHEV_performance_increase' : [1.0, 1.19125, 1.30625, 1.4475, 1.575]})

# Charging infrastructure assumptions
home = pandas.DataFrame(index=yearList, data={
    'no_charger':[0.0, 0.0, 0.0, 0.0, 0.0],
    'L1': [0.7, 0.0, 0.0, 0.0, 0.0],
    'L2': [0.3, 0.0, 0.0, 0.0, 0.0],
    'L1_controllable': [0.0, 0.3, 0.2, 0.1, 0.0],
    'L2_controllable': [0.0, 0.7, 0.8, 0.9, 1.0],
    'L3_controllable': [0.0, 0.0, 0.0, 0.0, 0.0]})
work = pandas.DataFrame(index=yearList, data={
    'no_charger':[0.5, 0.2, 0.0, 0.0, 0.0],
    'L1': [0.0, 0.0, 0.0, 0.0, 0.0],
    'L2': [0.5, 0.0, 0.0, 0.0, 0.0],
    'L1_controllable': [0.0, 0.0, 0.0, 0.0, 0.0],
    'L2_controllable': [0.0, 0.7, 0.8, 0.6, 0.4],
    'L3_controllable': [0.0, 0.1, 0.2, 0.4, 0.6]})
other_location = pandas.DataFrame(index=yearList, data={
    'no_charger':[1.0, 0.8, 0.6, 0.4, 0.2],
    'L1': [0.0, 0.0, 0.0, 0.0, 0.0],
    'L2': [0.0, 0.0, 0.0, 0.0, 0.0],
    'L1_controllable': [0.0, 0.0, 0.0, 0.0, 0.0],
    'L2_controllable': [0.0, 0.2, 0.3, 0.4, 0.5],
    'L3_controllable': [0.0, 0.0, 0.1, 0.2, 0.3]})

# Place holder for the results
results = []

for year in yearList:
    # Set the fleet mix
    total_number_of_vehicles = fleet_mix.ix[year]['BEV'] + fleet_mix.ix[year]['PHEV']
    battery_size = fleet_mix.ix[year]['battery_capcity_increase']
    BEV_perf = fleet_mix.ix[year]['BEV_performance_increase']
    PHEV_perf = fleet_mix.ix[year]['PHEV_performance_increase']
    year_fleet_mix = pandas.DataFrame(data={
        'percentage': [fleet_mix.ix[year]['BEV'] / total_number_of_vehicles,
                       fleet_mix.ix[year]['PHEV'] / total_number_of_vehicles],
        'car_model': [v2gsim.model.BasicCarModel('Leaf',
                                                 battery_capacity=24000 * battery_size,
                                                 maximum_power=120000,
                                                 UDDS=145.83 / BEV_perf,
                                                 HWFET=163.69 / BEV_perf,
                                                 US06=223.62 / BEV_perf,
                                                 Delhi=138.3 / BEV_perf),
                      v2gsim.model.BasicCarModel('Prius',
                                                 battery_capacity=12000 * battery_size,
                                                 maximum_power=120000,
                                                 UDDS=145.83 / PHEV_perf,
                                                 HWFET=163.69 / PHEV_perf,
                                                 US06=223.62 / PHEV_perf,
                                                 Delhi=138.3 / PHEV_perf)
                      ]})
    v2gsim.itinerary.set_fleet_mix(project.vehicles, year_fleet_mix)

    # Set the infrastructure mix
    set_infrastructure_probabilities(project, home.ix[year].to_dict(), 
        work.ix[year].to_dict(), other_location.ix[year].to_dict())

    # Run the model
    run_model(project)

    # Sum up the results
    result = project.locations[0].result.copy()
    for location in project.locations[1:]:
        result += location.result
    result = result * (total_number_of_vehicles / len(project.vehicles)) / (1000 * 1000 * 1000)  # to GW

    # Append the result
    results.append(result)

# Vizualise results
plot_result(results, yearList)

# Save result as a csv file
for result, year in zip(results, yearList):
    result.to_csv('../../SWITCH_EV/' + str(year) + '.csv', ',')

import pdb
pdb.set_trace()
