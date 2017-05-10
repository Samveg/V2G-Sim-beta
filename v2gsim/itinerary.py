from __future__ import division
import pandas
import scipy.stats
import datetime
import progressbar
import matplotlib.pyplot as plt
import random
from model import (Vehicle, ChargingStation, Location, Parked, Driving,
                   BasicCarModel, Project)


def from_csv(project, filename, number_of_days=1):
    """Read itineraries from an csv file. Excel header: id, start, end,
    distance, location

    Args:
        project (Project): empty project
        filename (string): relative or absolute path to the excel file

    Return:
        project (Project): project assigned with vehicles
    """
    df = pandas.read_csv(filename)
    return _dataframe_to_vehicles(project, df, number_of_days)


def from_excel(project, filename='', number_of_days=1, is_preload=False, df=False):
    """Read itineraries from an excel file. Excel header: Vehicle ID,
    Start time (hour), End time (hour), Distance (mi), P_max (W), Location,
    NHTS HH Wt.

    Args:
        project (Project): empty project
        filename (string): relative or absolute path to the excel file

    Return:
        project (Project): project assigned with vehicles
    """
    if not is_preload:
        df = pandas.read_excel(io=filename, sheetname='Activity')
    df = df.drop('Nothing', axis=1)
    df = df.rename(columns={'Vehicle ID': 'id', 'State': 'state',
                            'Start time (hour)': 'start',
                            'End time (hour)': 'end',
                            'Distance (mi)': 'distance',
                            'P_max (W)': 'maximum_power',
                            'Location': 'location',
                            'NHTS HH Wt': 'weight'})
    return _dataframe_to_vehicles(project, df, number_of_days)


def _dataframe_to_vehicles(project, df, number_of_days):
    # Day of the project
    date = project.date
    tot_sec = (date - datetime.datetime.utcfromtimestamp(0)).total_seconds()

    from_mile_to_km = 1.60934

    # Initialize a car model for the project
    project.car_models = [BasicCarModel(name='Leaf')]

    # Initialize itinerary data for each vehicle
    # Group the dataframe by vehicle_id, and for each vehicle_id,
    for vehicle_id, vehicle_data in df.groupby('id'):
        # Create a vehicle instance
        vehicle = Vehicle(vehicle_id, project.car_models[0])
        try:
            vehicle.weight = vehicle_data.weight.iloc[0]
        except AttributeError:
            vehicle.weight = 0

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
                                        end_date=date + datetime.timedelta(days=number_of_days)):
            print('Itinerary does not respect the constraints')
            print(vehicle)
        project.vehicles.append(vehicle)

    # Initialize charging station at each location
    reset_charging_infrastructures(project)
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


def reset_charging_infrastructures(project):
    """Reset the charging infrastructure at every location to be uncontrolled charging,
    using either L1 or L2 charger.

    Args:
        project (Project): a project
    """
    project.charging_stations = [ChargingStation(name='no_charger',
                                                 maximum_power=0,
                                                 minimum_power=0),
                                 ChargingStation(name='L1',
                                                 maximum_power=1440,
                                                 minimum_power=0),
                                 ChargingStation(name='L2',
                                                 maximum_power=7200,
                                                 minimum_power=0)]
    df2 = pandas.DataFrame(index=['no_charger', 'L1', 'L2'],
                           data={'charging_station': project.charging_stations,
                                 'probability': [0.0, 0.8, 0.2],
                                 'available': [float('inf'), float('inf'),
                                               float('inf')],
                                 'total': [float('inf'), float('inf'),
                                           float('inf')]})
    for location in project.locations:
        location.available_charging_station = df2.copy()


def copy_append(project, nb_of_days_to_add=2):
    """Copy the itinerary of each vehicle in the project.
    The copy is appended and a merge operation is applied to ensure a good blend.

    Args:
        project (Project): project
        nb_of_days_to_add (int): number of copies to append

    Return:
        project (Project): new project with extended itineraries
    """
    number_of_merged = 0
    for vehicle in project.vehicles:
        new_activities = []
        for i_copy in range(0, nb_of_days_to_add):
            # Manually copy all activities
            for activity in vehicle.activities:
                if isinstance(activity, Driving):
                    new_activities.append(Driving(activity.start + datetime.timedelta(days=1 + i_copy),
                                                  activity.end + datetime.timedelta(days=1 + i_copy),
                                                  activity.distance))
                elif isinstance(activity, Parked):
                    new_activities.append(Parked(activity.start + datetime.timedelta(days=1 + i_copy),
                                                 activity.end + datetime.timedelta(days=1 + i_copy),
                                                 activity.location))
        # Add new activities
        vehicle.activities.extend(new_activities)

        # Merge itineraries
        merged_activities = []
        skip = False
        for index, activity in enumerate(vehicle.activities):
            # Check if we have reached the last activity, if so exit loop
            if index == len(vehicle.activities) - 1:
                merged_activities.append(activity)
                break

            # Check if the activity has been merged during the previous iteration
            if skip:
                skip = False
                continue

            # Check for two parked activity in a row
            if (isinstance(activity, Parked) and
                    isinstance(vehicle.activities[index + 1], Parked)):
                # Check for matching names
                if (activity.location.category == vehicle.activities[index + 1].location.category and
                        activity.location.name == vehicle.activities[index + 1].location.name):
                    # Double check separately that times are matching
                    if activity.end == vehicle.activities[index + 1].start:
                        # Merge both activities
                        number_of_merged += 1
                        skip = True
                        activity.end = vehicle.activities[index + 1].end
                        merged_activities.append(activity)
                    else:
                        print activity
                        print vehicle.activities[index + 1]
                        print 'Merging issue'
                        merged_activities.append(activity)
                else:
                    merged_activities.append(activity)
            else:
                merged_activities.append(activity)

        vehicle.activities = merged_activities
        # Check time gap
        if not vehicle.check_activities(
                start_date=project.date, end_date=project.date + datetime.timedelta(days=1 + nb_of_days_to_add)):
            print('Itinerary does not respect the constraints')
            print(vehicle)

    return project


def get_vehicle_statistics(project, verbose=False):
    """This function return 'number_of_trip', 'morning_start', 'total_distance',
    'went_to_work', 'last_trip_time' for each vehicle.
    note: the function do not handle vehicles with only 1 driving activity
    and no parked activities.

    Args:
        project (Project): project

    Return:
        stat (DataFrame): data frame with index vehicle ids
    """

    stat = pandas.DataFrame(index=[vehicle.id for vehicle in project.vehicles],
                            columns=['number_of_trip', 'morning_start', 'total_distance',
                                     'went_to_work', 'last_trip_time'])

    for vehicle in project.vehicles:

        # Count the number of trip in a day
        counter = 0
        for activity in vehicle.activities:
            if isinstance(activity, Driving):
                counter += 1
        stat.loc[vehicle.id, 'number_of_trip'] = counter

        # Find the morning start
        if isinstance(vehicle.activities[0], Parked):
            stat.loc[vehicle.id, 'morning_start'] = vehicle.activities[0].end
        elif isinstance(vehicle.activities[0], Driving):
            if verbose:
                print('Vehicle: ' + str(vehicle.id) + ' started the day driving.')
            stat.loc[vehicle.id, 'morning_start'] = vehicle.activities[1].end

        # Find total distance traveled [km] and average distance
        counter = 0
        for activity in vehicle.activities:
            if isinstance(activity, Driving):
                counter += activity.distance
        stat.loc[vehicle.id, 'total_distance'] = counter

        # Went to work ?
        stat.loc[vehicle.id, 'went_to_work'] = False
        for activity in vehicle.activities:
            if isinstance(activity, Parked):
                if activity.location.category in ['Work']:
                    stat.loc[vehicle.id, 'went_to_work'] = True
                    break

        # When did you go back home ?
        if isinstance(vehicle.activities[-1], Parked):
            stat.loc[vehicle.id, 'last_trip_time'] = vehicle.activities[-1].start
        elif isinstance(vehicle.activities[-1], Driving):
            if verbose:
                print('Vehicle: ' + str(vehicle.id) + ' ended the day driving.')
            stat.loc[vehicle.id, 'last_trip_time'] = vehicle.activities[-2].start

    return stat


def set_fleet_mix(vehicles, mix):
    """Set the fleet mix

    Args:
        vehicles (Vehicle): list of vehicle
        mix (pandas.DataFrame): a row has a car model
            with an associated penetration
    """

    # Calculate the number of vehicles for each model
    fleetMix = mix.copy()
    fleetMix['number_of_vehicle'] = fleetMix['percentage'] * len(vehicles)
    fleetMix['number_of_vehicle'] = fleetMix['number_of_vehicle'].apply(int)
    fleetMix['vehicle_using_it'] = [0] * len(fleetMix)

    # The total number of vehicles is higher than the sum of what was given for each model
    diff = len(vehicles) - fleetMix['number_of_vehicle'].sum()
    if diff != 0:
        # randomly pick a model to increase the number of vehicle associated to it
        for i in range(0, diff):
            fleetMix.loc[random.choice(fleetMix.index.tolist()), 'number_of_vehicle'] += 1

    for vehicle in vehicles:
        # Update the list of available models
        available_model_category = fleetMix[fleetMix['vehicle_using_it'] < fleetMix['number_of_vehicle']].index.tolist()

        # Pick the model index
        model_index = int(random.choice(available_model_category))

        # Set the right model
        vehicle.car_model = fleetMix.ix[model_index]['car_model']
        fleetMix.loc[model_index, 'vehicle_using_it'] += 1


def get_cycling_itineraries(project, verbose=False):
    """Put aside vehicles that does not come back
    at the same location as they started their day from.
    Also putting aside vehicle that ends or starts with a
    driving activity.

    Args:
        project (Project): a project with vehicles

    Return:
        a list of vehicle (list)
    """
    good_vehicles = []
    for vehicle in project.vehicles:
        if (isinstance(vehicle.activities[0], Parked) and
            isinstance(vehicle.activities[-1], Parked)):
            if (vehicle.activities[0].location.category ==
                vehicle.activities[-1].location.category):
                good_vehicles.append(vehicle)

    if verbose:
        previous_count = len(project.vehicles)
        new_count = len(good_vehicles)
        print(str(previous_count - new_count) +
              ' vehicles did not finished at the same location as they have started the day')
        print('')

    return good_vehicles


def find_all_itinerary_combination(project, verbose=True):
    """Find all combination of locations in the vehicle itinerary data.

    Args:
        project (Project): a project

    Returns:
        a pandas DataFrame with combination, vehicles and some high level statistics.
    """
    if verbose:
        progress = progressbar.ProgressBar(widgets=['Parsing: ',
                                                    progressbar.Percentage(), progressbar.Bar()],
                                           maxval=len(project.vehicles)).start()

    # Create the dataframe holding the results
    frame = pandas.DataFrame(columns={'locations', 'vehicles'})

    # Iterate through all the vehicles
    for vehicle_index, vehicle in enumerate(project.vehicles):
        itinerary = []
        existing = False

        # Get the list of locations
        for activity in vehicle.activities:
            if isinstance(activity, Parked):
                itinerary.append(activity.location.category)

        # Is the temporary itinerary already existing?
        for index, row in frame.iterrows():
            if row.locations == itinerary:
                existing = True
                frame.loc[index, 'vehicles'].append(vehicle)
                break

        if not existing:
            # Create a new row in the dataframe
            frame = pandas.concat([
                frame, pandas.DataFrame(
                    data={'locations': [itinerary], 'vehicles': [[vehicle]]})], axis=0,
                ignore_index=True)
        if verbose:
            progress.update(vehicle_index + 1)

    if verbose:
        progress.finish()

    # Get some basic filtering value
    frame['nb_of_parked_activity'] = (
        frame.locations.apply(lambda x: len(x)))

    frame['nb_of_vehicles'] = (
        frame.vehicles.apply(lambda x: len(x)))

    frame['worker'] = (
        frame.locations.apply(lambda x: True if 'Work' in x else False))

    if verbose:
        print('')
    return frame


def merge_itinerary_combination(project, locations_to_save, verbose=True):
    """Merge location combination, do not remove any vehicle.
    itinerary_statistics will only keep the locations in *locations_to_save* or
    *"someWhere"*. Regroup ItineraryBin that are now similar and
    merge their assigned vehicles.

    Args:
        project (Project): a project
        locations_to_save (list): collection of location name to preserve

    Returns:
        a pandas DataFrame with combination, vehicles and some high level statistics.
    """
    def _remove_other_location(locations, locations_to_save):
        new_locations = []
        for location_category in locations:
            if location_category not in locations_to_save:
                new_locations.append('other_location')
            else:
                new_locations.append(location_category)
        return new_locations

    def _remove_vehicle_with_missing_driving(values):
        nb_of_activity = 2 * values['nb_of_parked_activity'] - 1
        new_vehicles = []
        for vehicle in values['vehicles']:
            if len(vehicle.activities) == nb_of_activity:
                new_vehicles.append(vehicle)
        return new_vehicles

    # Copy the frame
    frame = project.itinerary_statistics.copy()

    # Simplify the name of locations
    frame.locations = frame.locations.apply(
        lambda locations: _remove_other_location(locations, locations_to_save))

    # Get all unique combination of locations
    frame.locations = frame.locations.apply(tuple)
    unique_locations = frame.locations.unique().tolist()

    # Create the dataframe with unique locations
    frame2 = pandas.DataFrame(index=range(0, len(unique_locations)),
                              data={'locations': unique_locations,
                                    'vehicles': [[] for i in range(0, len(unique_locations))]})

    # For each of the unique itinerary merge the vehicles from the duplicate itineraries
    for index, row in frame.iterrows():
        # Get the corresponding row in frame 2 and append vehicles
        temp_index = frame2[frame2.locations == row.locations].index
        if len(temp_index) > 1:
            print('Error one the itinerary is not unique')
        frame2.loc[temp_index[0], 'vehicles'].extend(row.vehicles)

    # Get some basic filtering value
    frame2['nb_of_parked_activity'] = (
        frame2.locations.apply(lambda x: len(x)))

    # Check that all vehicles have the same number of activities (perhaps one is missing a driving)
    frame2.vehicles = frame2.apply(_remove_vehicle_with_missing_driving, axis=1)

    frame2['nb_of_vehicles'] = (
        frame2.vehicles.apply(lambda x: len(x)))

    if verbose:
        print(str(len(project.vehicles) - frame2.nb_of_vehicles.sum()) +
              'removed vehicles, because of missing driving cycle')

    frame2['worker'] = (
        frame2.locations.apply(lambda x: True if 'Work' in x else False))

    return frame2


def get_itinerary_statistic(project, verbose=True):
    """For each itinerary creates a specific dataframe containing
    primary statistical metrics on each activity using data from all the
    vehicles in the itinerary's category. Note date are assumed to be UTC.

    Args:
        project (Project): a project

    Returns:
        a pandas DataFrame with locations, vehicles, high level statistics and detailed
        activity statistics.
    """

    # Copy the data
    frame = project.itinerary_statistics.copy()

    # Get the total number of vehicles
    total_vehicle = frame.nb_of_vehicles.sum()
    if total_vehicle != len(project.vehicles):
        print('Warning vehicles are missing: ' +
              str(total_vehicle - len(project.vehicles)) + ' vehicles')

    # For each combination
    epoch = datetime.datetime.utcfromtimestamp(0)  # Needed to change dates into floats (assume dates are UTC)
    temp_frame_holding_stat = pandas.DataFrame()  # Needed because coulc not update dataframe with another frame
    for index, row in frame.iterrows():
        # Create the structure holding the results for the whole day
        total_number_of_activities = (row.nb_of_parked_activity * 2) - 1
        activity_stat = pandas.DataFrame(index=range(0, total_number_of_activities),
                                         columns=['start_loc', 'start_scale',
                                                  'duration_loc', 'duration_scale',
                                                  'end_loc', 'end_scale',
                                                  'distance_loc', 'distance_scale',
                                                  'mean_speed_loc', 'mean_speed_scale'])

        # Iterate through all the activities for each vehicle to gather the data (could skip this)
        vehicle_schedules = []
        for vehicle in row.vehicles:
            vehicle_stat = pandas.DataFrame(index=range(0, len(vehicle.activities)),
                                            columns=['start', 'duration', 'end', 'distance', 'mean_speed'])
            vehicle_stat = vehicle_stat.fillna(0)
            for i, activity in enumerate(vehicle.activities):
                if isinstance(activity, Parked):
                    vehicle_stat.loc[i, 'start'] = (activity.start - epoch).total_seconds()
                    vehicle_stat.loc[i, 'duration'] = (activity.end - activity.start).total_seconds()
                    vehicle_stat.loc[i, 'end'] = (activity.end - epoch).total_seconds()

                elif isinstance(activity, Driving):
                    vehicle_stat.loc[i, 'start'] = (activity.start - epoch).total_seconds()
                    vehicle_stat.loc[i, 'duration'] = (activity.end - activity.start).total_seconds()
                    vehicle_stat.loc[i, 'distance'] = activity.distance
                    if vehicle_stat.loc[i, 'duration'] == 0:
                        vehicle_stat.loc[i, 'mean_speed'] = 0
                    else:
                        vehicle_stat.loc[i, 'mean_speed'] = activity.distance / ((activity.end - activity.start).total_seconds() / 3600)
                    vehicle_stat.loc[i, 'end'] = (activity.end - epoch).total_seconds()
            vehicle_schedules.append(vehicle_stat)

        # Get the best fit for each of the category per activity
        for activity_index in range(0, total_number_of_activities):
            for column in ['start', 'duration', 'end', 'distance', 'mean_speed']:
                temporary_vector = []
                for schedule in vehicle_schedules:
                    temporary_vector.append(schedule.loc[activity_index, column])
                # Find the best fit for the data
                loc, scale = scipy.stats.norm.fit(temporary_vector)
                activity_stat.loc[activity_index, column + str('_loc')] = loc
                activity_stat.loc[activity_index, column + str('_scale')] = scale

        temp_frame_holding_stat = pandas.concat([temp_frame_holding_stat,
                                                 pandas.DataFrame(index=[index],
                                                                  data={'activity_statistics': [activity_stat]})], axis=0)

    frame = pandas.concat([frame, temp_frame_holding_stat], axis=1)

    return frame


def create_vehicles_from_stat(current_id, itinerary, project):
    """Create a new vehicle from a statistical itinerary

    Args:
        current_id (Integer): id assigned to the new vehicle
        itinerary (DataFrame): a row of project.itinerary_statistics
        project (Project): a project to append new locations if necessary

    Return:
        a vehicle object with a full day of activities
    """
    not_completed = True
    max_iteration = 100
    iteration = 0
    # Duration shortening is to avoid itineraries to overpass 24h
    duration_shortening = 0
    iteration_before_reducing_parked_duration = 25

    while not_completed and iteration < max_iteration:
        iteration += 1
        vehicle = Vehicle(current_id, project.car_models[0])

        is_first_activity = lambda x: True if x == 0 else False
        is_last_activity = lambda x, y=len(itinerary.activity_statistics) - 1: True if x == y else False
        is_duration_enough = lambda x: x if x > 60 else 60

        if iteration % iteration_before_reducing_parked_duration == 0 and iteration != 0:
            duration_shortening += 0.1

        # The activities one by one
        location_index = 0
        parked = True
        try_again = False
        for index, activity in itinerary.activity_statistics.iterrows():
            if is_first_activity(index):
                # This activity is assumed to be parked
                start = project.date
                # duration is int() to avoid milliseconds
                duration = int(scipy.stats.norm.rvs(activity.duration_loc, activity.duration_scale, 1)[0])
                # duration is rounded at project.timestep
                duration -= duration % project.timestep
                duration = is_duration_enough(duration)
                duration -= duration * duration_shortening

                end = start + datetime.timedelta(seconds=duration)
                location = itinerary.locations[location_index]
                location_index += 1
                parked = False

                # Append the activity
                vehicle.activities.append(Parked(start, end, unique_location_category(location, project)))

            elif is_last_activity(index):
                # This activity is assumed to be parked
                start = vehicle.activities[-1].end
                end = project.date + datetime.timedelta(days=1)
                location = itinerary.locations[location_index]

                # Append the activity
                vehicle.activities.append(Parked(start, end, unique_location_category(location, project)))

            elif parked:
                start = vehicle.activities[-1].end
                duration = int(scipy.stats.norm.rvs(activity.duration_loc, activity.duration_scale, 1)[0])
                duration -= duration % project.timestep
                duration = is_duration_enough(duration)
                duration -= duration * duration_shortening

                end = start + datetime.timedelta(seconds=duration)
                location = itinerary.locations[location_index]
                location_index += 1
                parked = False

                # Append the activity
                vehicle.activities.append(Parked(start, end, unique_location_category(location, project)))

            elif not parked:
                start = vehicle.activities[-1].end
                duration = int(scipy.stats.norm.rvs(activity.duration_loc, activity.duration_scale, 1)[0])
                duration -= duration % project.timestep
                duration = is_duration_enough(duration)
                mean_speed = scipy.stats.norm.rvs(activity.mean_speed_loc, activity.mean_speed_scale, 1)[0]
                end = start + datetime.timedelta(seconds=duration)
                distance = (duration / 3600) * mean_speed  # duration in seconds and mean_speed in km/h
                vehicle.activities.append(Driving(start, end, distance))
                parked = True

            # Post activity check
            if end > project.date + datetime.timedelta(days=1):
                # Some activity took too much time
                try_again = True
                break

        if not try_again:
            not_completed = False

    return vehicle


def new_project_using_stats(old_project, new_number_of_vehicle, only_worker=False,
                            min_vehicles_per_bin=False, remove_old_project=False, verbose=True,
                            advanced_filtering_func=False, creating_vehicle_func=create_vehicles_from_stat):
    """Creates a new project from statistical itineraries of an existing project.
    Note: In the default version, itinerary should be able to cycle day after day, see
    get_cycling_itineraries.

    Args:
        old_project (Project): project which contains the statistical itineraries
        new_number_of_vehicle (Integer): number of vehicles to create from the statistics
        only_worker (Boolean): remove non worker from the pool
        min_vehicles_per_bin (Integer): remove itineraries with too few vehicle representing it
        remove_old_project (Boolean): remove the old project to free the memory before creating
            the new project.
        advanced_filtering_func (Function): customizable filtering function

    Return:
        a new project with similar itineraries as the old project and chosen number of vehicles
    """
    if verbose:
        progress = progressbar.ProgressBar(widgets=['Parsing: ',
                                                    progressbar.Percentage(), progressbar.Bar()],
                                           maxval=len(old_project.itinerary_statistics)).start()

    # Create a new project
    project = Project()
    project.car_models = [BasicCarModel(name='Leaf')]

    # Copy the data
    frame = old_project.itinerary_statistics.copy()

    # Remove the old project ?
    if remove_old_project:
        del old_project  # Deep del?

    # Filtering the statistical data
    if only_worker:
        frame = frame[frame.worker == True]

    if min_vehicles_per_bin:
        frame = frame[frame.nb_of_vehicles >= min_vehicles_per_bin]

    if advanced_filtering_func:
        frame = advanced_filtering_func(frame)

    # For each itinerary of the frame recreate
    current_id = 0
    total_number_of_vehicle = frame.nb_of_vehicles.sum()
    for index, itinerary in frame.iterrows():
        # How many vehicles should be created
        number_of_vehicle_per_bin = int((itinerary.nb_of_vehicles / total_number_of_vehicle) * new_number_of_vehicle)

        # Recreate the vehicles activities
        for vehicle_index in range(0, number_of_vehicle_per_bin):
            vehicle = creating_vehicle_func(current_id, itinerary, project)

            # Check time gap before appending vehicle to the project
            if not vehicle.check_activities(
                start_date=project.date, end_date=project.date + datetime.timedelta(days=1)):
                print('Itinerary does not respect the constraints')
                print(vehicle)
                import pdb
                pdb.set_trace()
            project.vehicles.append(vehicle)
            current_id += 1

        if verbose:
            progress.update(index + 1)

    if verbose:
        progress.finish()

    # Initialize charging station at each location
    reset_charging_infrastructures(project)

    if verbose:
        print('')
    return project


def plot_vehicle_itinerary(vehicles, title=''):
    """Plot the itinerary of a vehicle through out the day

    Args:
        vehicles (List): list of vehicles
    """
    for index, vehicle in enumerate(vehicles):
        # Create the vector to plot
        x = []
        y = []
        for activity in vehicle.activities:
            x.append(activity.start)
            x.append(activity.end)
            if isinstance(activity, Parked):
                y.append(0 + index)
                y.append(0 + index)
            if isinstance(activity, Driving):
                y.append(1 + index)
                y.append(1 + index)
        plt.plot(x, y, label='vehicle id ' + str(vehicle.id), linewidth=2.0, alpha=0.8)
    plt.legend(loc=0)
    plt.title(title)
