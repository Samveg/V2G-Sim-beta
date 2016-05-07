from __future__ import division
import pandas
import datetime
from model import (Vehicle, ChargingStation, Location, Parked, Driving,
                   BasicCarModel)


def from_excel(project, filename):
    """Read itineraries from an excel file. Excel header: Vehicle ID,
    Start time (hour), End time (hour), Distance (mi), P_max (W), Location,
    NHTS HH Wt.

    Args:
        project (Project): empty project
        filename (string): relative or absolute path to the excel file

    Return:
        project (Project): project assigned with vehicles
    """
    print('itinerary.from_excel(project, ' + filename + ')')

    # Day of the project
    date = project.date
    tot_sec = (date - datetime.datetime.utcfromtimestamp(0)).total_seconds()

    df = pandas.read_excel(io='Tennessee.xlsx', sheetname='Activity')
    df = df.drop('Nothing', axis=1)
    df = df.rename(columns={'Vehicle ID': 'id', 'State': 'state',
                            'Start time (hour)': 'start',
                            'End time (hour)': 'end',
                            'Distance (mi)': 'distance',
                            'P_max (W)': 'maximum_power',
                            'Location': 'location',
                            'NHTS HH Wt': 'weight'})
    from_mile_to_km = 1.60934

    # Initialize a car model for the project
    project.car_models = [BasicCarModel(name='Leaf')]

    # Initialize itinerary data for each vehicle
    # Group the dataframe by vehicle_id, and for each vehicle_id,
    for vehicle_id, vehicle_data in df.groupby('id'):
        # Create a vehicle instance
        vehicle = Vehicle(vehicle_id, project.car_models[0])
        vehicle.weight = vehicle_data.weight.iloc[0]

        # Create a list of activities parked and driving
        for index, row in vehicle_data.iterrows():
            # Round the start and end time to correspond with project.timestep
            start = datetime.datetime.utcfromtimestamp(tot_sec + int(
                (row['start'] * 3600) / project.timestep) * project.timestep)
            end = datetime.datetime.utcfromtimestamp(tot_sec + int(
                (row['end'] * 3600) / project.timestep) * project.timestep)

            # Accordingly to the state create a driving or parked activity
            if row['state'] in 'Driving':
                vehicle.activities.append(Driving(start, end, row['distance'] * from_mile_to_km))
            elif row['state'] in ['Parked', 'Charging']:
                vehicle.activities.append(
                    Parked(start, end, unique_location_category(row['location'], project)))
            else:
                print('State should either be Driving, Parked or Charging')
                print('State was: ' + str(row['state']))

        # Check time gap before appending vehicle to the project
        if not vehicle.check_activities(start_date=date,
                                        end_date=date + datetime.timedelta(days=1)):
            print('Itinerary does not respect the constraints')
            print(vehicle)
        project.vehicles.append(vehicle)

    # Initialize charging station at each location
    project.charging_stations = [ChargingStation(name='no_charger',
                                                 maximum_power=0,
                                                 minimum_power=0),
                                 ChargingStation(name='L1',
                                                 maximum_power=1440,
                                                 minimum_power=-1440),
                                 ChargingStation(name='L2',
                                                 maximum_power=7200,
                                                 minimum_power=-7200)]
    df2 = pandas.DataFrame(index=['no_charger', 'L1', 'L2'],
                           data={'charging_station': project.charging_stations,
                                 'probability': [0.0, 0.8, 0.2],
                                 'available': [float('inf'), float('inf'),
                                               float('inf')],
                                 'total': [float('inf'), float('inf'),
                                           float('inf')]})
    for location in project.locations:
        location.available_charging_station = df2

    print('')
    return project


def unique_location_category(location_category, project):
    """Compare input location category with locations in the project, return a
    new location and update the project or just return an existing location.

    Args:
        location_category (string): location category (Home, Work, ...)
        project (Project): current project

    Return:
        location (Location): either a new or an existing location
    """
    # Check if this location is matching any existing location in the project
    for existing_location in project.locations:
        if location_category in existing_location.category:
            return existing_location

    # If we reach this part of the code a new location need to be created
    new_location = Location(location_category, name=location_category + '01')
    project.locations.append(new_location)
    return new_location
