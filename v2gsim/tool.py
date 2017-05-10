from __future__ import division
import random
import numpy
import pandas
import v2gsim


def project_from_excel(df, filename=False, parent_folder_itinerary='', copy_append=True):
    """Create a project including itineraries"""
    # Create a project and initialize it with some itineraries
    project = v2gsim.model.Project()
    project.date = df['project'].start_date[0]
    project.end = df['project'].end_date[0]
    project.nb_days = (project.end - project.date).days
    project.ambient_temperature = df['project'].ambient_temperature[0]

    if filename is False:
        project = v2gsim.itinerary.from_excel(project, parent_folder_itinerary + df['project'].itinerary_filename[0])
    else:
        project = v2gsim.itinerary.from_csv(project, filename, number_of_days=project.nb_days)

    # This function from the itinerary module return all the vehicles that
    # start and end their day at the same location (e.g. home)
    project.vehicles = v2gsim.itinerary.get_cycling_itineraries(project)

    if project.nb_days > 1 and copy_append:
        project = v2gsim.itinerary.copy_append(project, nb_of_days_to_add=project.nb_days - 1)

    # Little check
    print('')
    print('A project has been created with ' + str(len(project.vehicles)) + ' vehicles')
    print('Project start date: ' + str(project.vehicles[0].activities[0].start) + ' - ' +
          'end date: ' + str(project.vehicles[0].activities[-1].end))
    print('')
    return project


def project_from_csv(df, parent_folder_itinerary='', filename=False, verbose=False):
    """Create a project including itineraries"""
    # Create a project and initialize it with some itineraries
    project = v2gsim.model.Project()
    project.date = df['project'].start_date[0]
    project.end = df['project'].end_date[0]
    project.nb_days = (project.end - project.date).days
    project.ambient_temperature = df['project'].ambient_temperature[0]
    if not filename:
        project = v2gsim.itinerary.from_csv(project, parent_folder_itinerary + df['project'].itinerary_filename[0])
    else:
        project = v2gsim.itinerary.from_csv(project, filename)

    # This function from the itinerary module return all the vehicles that
    # start and end their day at the same location (e.g. home)
    project.vehicles = v2gsim.itinerary.get_cycling_itineraries(project)

    if project.nb_days > 1:
        project = v2gsim.itinerary.copy_append(project, nb_of_days_to_add=project.nb_days - 1)

    # Little check
    if verbose:
        print('')
        print('A project has been created with ' + str(len(project.vehicles)) + ' vehicles')
        print('Project start date: ' + str(project.vehicles[0].activities[0].start) + ' - ' +
              'end date: ' + str(project.vehicles[0].activities[-1].end))
        print('')
    return project


def car_model_from_excel(df, ambient_temperature, verbose=False):
    """Create a list of car models"""
    car_models = []
    for row in df['vehicle_characteristic'].itertuples():
        car_model = v2gsim.model.BasicCarModel(row.name)
        car_model.battery_capacity = float(row.battery_wh)
        car_model.battery_efficiency_charging = float(row.battery_eff_charging)
        car_model.maximum_power = float(row.maximum_charging_power_w)
        car_model.maximum_SOC = float(row.maximum_soc)
        car_model.ancillary_load = float(row.ancillary_load_w)
        car_model.driving = v2gsim.driving.basic_powertrain.road_consumption_plus_ancillary_load

        if ambient_temperature in 'cold':
            car_model.UDDS = float(row.UDDS_cold_wh_per_km)
            car_model.HWFET = float(row.HWFET_cold_wh_per_km)
            car_model.US06 = float(row.US06_cold_wh_per_km)

        elif ambient_temperature in 'temperate':
            car_model.UDDS = float(row.UDDS_temperate_wh_per_km)
            car_model.HWFET = float(row.HWFET_temperate_wh_per_km)
            car_model.US06 = float(row.US06_temperate_wh_per_km)

        elif ambient_temperature in 'hot':
            car_model.UDDS = float(row.UDDS_hot_wh_per_km)
            car_model.HWFET = float(row.HWFET_hot_wh_per_km)
            car_model.US06 = float(row.US06_hot_wh_per_km)

        else:
            raise Exception('Ambient temperature setting is wrong')
        car_models.append(car_model)

    # Little check
    if verbose:
        print('')
        for cm in car_models:
            print(cm.name + ' | battery: ' + str(cm.battery_capacity) + ' Wh | ' +
                  'UDDS/HWFET/US06: ' + str(int(cm.UDDS)) + '/' + str(int(cm.HWFET)) + '/' + str(int(cm.US06)) + 'Wh/km | ' +
                  ' added ancillary load: ' + str(cm.ancillary_load) + 'W')
        print('')
    return car_models


def assign_car_model(df, project, verbose=False):
    """Set vehicle.car_model from project.car_models with proportion from vehicle_stock"""
    # Get the total of vehicle
    total_stock = int(df['vehicle_stock'].number_of_vehicles.sum())

    # Get the number of itineraries
    total_itineraries = len(project.vehicles)

    # Scaling factor
    scaling = float(total_itineraries / total_stock)

    # Reset vehicle model
    for vehicle in project.vehicles:
        vehicle.car_model = False

    all_indexes = [ int(i) for i in range(0, len(project.vehicles))]
    for row in df['vehicle_stock'].itertuples():
        # Create a list of random indexes
        nb_indexes = int(row.number_of_vehicles * scaling)
        random_indexes = []
        for i in range(0, nb_indexes):
            random_indexes.append(int(random.choice(all_indexes)))
            all_indexes.remove(random_indexes[-1])

        # Pick corresponding car model
        temp_car_model = False
        for index, car_model in enumerate(project.car_models):
            if car_model.name in row.vehicle_name:
                temp_car_model = project.car_models[index]
        for index in random_indexes:
            if temp_car_model:
                project.vehicles[index].car_model = temp_car_model
            else:
                raise Exception('could not find car model')

    # If a vehicle was forgotten assign a car model
    for vehicle in project.vehicles:
        if not vehicle.car_model:
            vehicle.car_model = project.car_models[0]

    # Double check percentage are right
    if verbose:
        print('')
        car_model_names = [car_model.name for car_model in project.car_models]
        percentages = {key: 0 for key in car_model_names}
        for vehicle in project.vehicles:
            percentages[vehicle.car_model.name] += 1
        for name in percentages:
            print('There is ' + str(percentages[name] * 100 / total_itineraries) + ' % ' +
                  'of ' + str(name))
        print('')
        print('Scaling number: ' + str(scaling))
        print('')
    return scaling


def charging_stations_from_excel(df):
    """Return a list of charging infrastructures"""
    charging_stations = []

    # create the no charger infra
    charging_stations.append(
        v2gsim.model.ChargingStation(name='no_charger', maximum_power=0, minimum_power=0))

    for row in df['charging_station'].itertuples():
        charging_stations.append(
            v2gsim.model.ChargingStation(name=row.name, maximum_power=float(row.maximum_power_w),
                                         minimum_power=0))
    return charging_stations


def set_available_infrastructures_at_locations(df, project, verbose=False):
    """Set Charging infrastructure"""
    # Reset charging infrastructures
    for index, location in enumerate(project.locations):
        location.available_charging_station = pandas.DataFrame(
            columns=['charging_station', 'probability'])

        # Get home and work locations
        if location.category in 'Home':
            home_location = project.locations[index]
        elif location.category in 'Work':
            work_location = project.locations[index]

    # Assign setting
    for row in df['location'].itertuples():
        # Get the charging station
        current_station = False
        for index, st in enumerate(project.charging_stations):
            if st.name == row.charger_name:
                current_station = project.charging_stations[index]
                break
        if not current_station:
            raise Exception('could not find the charging station')

        # Assign home and work locations
        if row.name in 'home':
            size = len(home_location.available_charging_station)
            home_location.available_charging_station.loc[size] = [current_station, float(row.availability)]
            home_location.soc_no_charging = float(row.soc_high_no_charging)
            home_location.soc_charging = float(row.soc_low_need_to_charge)
        elif row.name in 'work':
            size = len(work_location.available_charging_station)
            work_location.available_charging_station.loc[size] = [current_station, float(row.availability)]
            work_location.soc_no_charging = float(row.soc_high_no_charging)
            work_location.soc_charging = float(row.soc_low_need_to_charge)
        elif row.name in 'other':
            # Assign setting for 'other locations'
            for index, location in enumerate(project.locations):
                if location.category not in ['Home', 'Work']:
                    size = len(location.available_charging_station)
                    location.available_charging_station.loc[size] = [current_station, float(row.availability)]
                    location.soc_no_charging = float(row.soc_high_no_charging)
                    location.soc_charging = float(row.soc_low_need_to_charge)
        else:
            raise Exception('Do not support other locations than home/work/other')

    # Add the no charger 'station'
    for index, location in enumerate(project.locations):
        for index, st in enumerate(project.charging_stations):
            if st.name == 'no_charger':
                no_station = project.charging_stations[index]
                break
        size = len(location.available_charging_station)
        probability = 1.0 - float(location.available_charging_station.probability.sum())
        if probability < 0:
            raise('Charger availability cannot be more than 100%')
        location.available_charging_station.loc[size] = [no_station, probability]

    # Double check by printing the first 4 location charging stations
    if verbose:
        print('')
        for i in range(0, 4):
            print(project.locations[i].category)
            print(project.locations[i].available_charging_station)
        print('')


def set_available_infrastructures_at_locations_v2(df, project, verbose=False):
    """Set Charging infrastructure"""
    # Reset charging infrastructures
    for index, location in enumerate(project.locations):
        location.available_charging_station = pandas.DataFrame(
            columns=['charging_station', 'probability'])

        # Get home and work locations
        if location.category in 'Home':
            home_location = project.locations[index]
        elif location.category in 'Work':
            work_location = project.locations[index]

    # Assign setting
    location_with_settings = df['location'].name.unique().tolist()
    for row in df['location'].itertuples():
        # Get the charging station
        current_station = False
        for index, st in enumerate(project.charging_stations):
            if st.name == row.charger_name:
                current_station = project.charging_stations[index]
                break
        if not current_station:
            raise Exception('could not find the charging station')

        # Get the location
        current_location = False
        for index, location in enumerate(project.locations):
            if row.name in 'other':
                current_location = True
            if location.category in row.name:
                current_location = project.locations[index]
                break
        if not current_location:
            print('Looking for ' + str(row.name) + ' in project locations:')
            for index, location in enumerate(project.locations):
                print(location.category)
            raise Exception('could not find the location')

        # Assign home and work locations
        if row.name not in 'other':
            size = len(current_location.available_charging_station)
            current_location.available_charging_station.loc[size] = [current_station, float(row.availability)]
            current_location.soc_no_charging = float(row.soc_high_no_charging)
            current_location.soc_charging = float(row.soc_low_need_to_charge)
        else:
            # Assign setting for 'other locations'
            for index, location in enumerate(project.locations):
                if location.category not in location_with_settings:
                    size = len(location.available_charging_station)
                    location.available_charging_station.loc[size] = [current_station, float(row.availability)]
                    location.soc_no_charging = float(row.soc_high_no_charging)
                    location.soc_charging = float(row.soc_low_need_to_charge)

    # Add the no charger 'station'
    for index, location in enumerate(project.locations):
        for index, st in enumerate(project.charging_stations):
            if st.name == 'no_charger':
                no_station = project.charging_stations[index]
                break
        size = len(location.available_charging_station)
        probability = 1.0 - float(location.available_charging_station.probability.sum())
        if probability < 0:
            raise('Charger availability cannot be more than 100%')
        location.available_charging_station.loc[size] = [no_station, probability]

    # Double check by printing the first 4 location charging stations
    if verbose:
        print('')
        for i in range(0, 4):
            print(project.locations[i].category)
            print(project.locations[i].available_charging_station)
        print('')



def custom_save_location_state(location, timestep, date_from, date_to,
                               vehicle=None, activity=None,
                               power_demand=None, SOC=None, nb_interval=None,
                               init=False, run=False, post=False):
    """Save local results from a parked activity during running
    time. If date_from and date_to, set a fresh pandas DataFrame at locations.

    Args:
        location (Location): location
        timestep (int): calculation timestep
        date_from (datetime.datetime): date to start recording power demand
        date_to (datetime.datetime): date to end recording power demand
        vehicle (Vehicle): vehicle
        activity (Parked): parked activity
        power_demand (list): power demand from parked activity
        SOC (list): state of charge from the parked activity
        nb_interval (int): number of timestep for the parked activity
    """
    if run:
        activity_index1, activity_index2, location_index1, location_index2, save = v2gsim.result._map_index(
            activity.start, activity.end, date_from, date_to, len(power_demand),
            len(location.result['power_demand']), timestep)

        # Save a lot of interesting result
        if save:
            location.result['power_demand'][location_index1:location_index2] += (
                power_demand[activity_index1:activity_index2])

            # Add 'number_of_vehicle_parked' in the initialization section
            location.result['number_of_vehicle_parked'][location_index1:location_index2] += 1

            # Number of vehicle currently charging
            location.result['number_of_vehicle_charging'][location_index1:location_index2] += (
                [1 if power != 0.0 else 0 for power in power_demand])

    elif init:
        # Initiate a dictionary of numpy array to hold result (faster than DataFrame)
        location.result = {'power_demand': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                           'number_of_vehicle_parked': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                           'number_of_vehicle_charging': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep))}

    elif post:
        # Convert location result back into pandas DataFrame (faster that way)
        i = pandas.date_range(start=date_from, end=date_to,
                              freq=str(timestep) + 's', closed='left')
        location.result = pandas.DataFrame(index=i, data=location.result)
