from __future__ import division
import model
import pandas
import numpy
import datetime
import progressbar


def run(project, charging_option=None, date_from=None, date_to=None,
        reset_charging_station=False):
    """Launch a simulation in a decoupled manner, vehicle don't take other
    vehicle's actions into consideration. The simulation goes throught each
    activity one by one for each vehicle and number of iteration desired.
    When the simulation enters a driving activity, it computes the power demand
    and save time a which vehicles have stranded. When the simulation enters a
    parked activity, it first determines what is the charging station assigned
    and then compute the corresponding power demand from the grid.

    Args:
        project (Project): project to simulate
        date_from (datetime.datetime): date to start recording power demand
        date_to (datetime.datetime): date to end recording power demand
        charging_option (any): pass some object to the charging function
        reset_charging_station (boolean): reset charging stations to None
    """
    if date_from is None:
        date_from = project.date
    if date_to is None:
        date_to = project.date + datetime.timedelta(days=1)

    # Itinitialize result placeholders and reset charging stations to None
    project = _pre_run(project, date_from, date_to, reset=reset_charging_station)

    # Create the progress bar
    progress = progressbar.ProgressBar(widgets=['core.run: ',
                                                progressbar.Percentage(),
                                                progressbar.Bar()],
                                       maxval=len(project.vehicles)).start()

    # ####################### Simulation #####################################
    # For each vehicle
    for indexV, vehicle in enumerate(project.vehicles):
        # For each activity
        for indexA, activity in enumerate(vehicle.activities):
            # Calculate the duration of the activity
            nb_interval = int((activity.end - activity.start).total_seconds() / project.timestep)
            if isinstance(activity, model.Driving):
                SOC, power_demand, stranded, detail = vehicle.car_model.driving(activity,
                                                                                vehicle,
                                                                                nb_interval,
                                                                                project.timestep)
                vehicle.SOC.extend(SOC)
                # Log stranded vehicles
                if stranded:
                    vehicle.stranding_log.append(activity.end)

            elif isinstance(activity, model.Parked):
                # Get the charging station if not already assigned
                if activity.charging_station is None:
                    activity.charging_station = activity.location.assign_charging_station(indexA,
                                                                                          activity,
                                                                                          vehicle)

                # Compute the consumption at the charging station
                detail = False
                SOC, power_demand = activity.charging_station.charging(activity,
                                                                       vehicle,
                                                                       nb_interval,
                                                                       project.timestep,
                                                                       charging_option)
                vehicle.SOC.extend(SOC)
                # Save power demand at location
                if len(power_demand) != 0:
                    activity.location.result_function(activity.location,
                                                      project.timestep,
                                                      date_from, date_to,
                                                      vehicle, activity,
                                                      power_demand, SOC,
                                                      nb_interval, run=True)
            vehicle.result_function(vehicle, project.timestep, date_from,
                                    date_to, activity, power_demand, SOC,
                                    detail, nb_interval, run=True)
        # Remove initial SOC
        del vehicle.SOC[0]
        progress.update(indexV + 1)
    # ########################################################################

    # Post process result (change format, ...)
    project = _post_run(project, date_from, date_to)
    progress.finish()


def _pre_run(project, date_from, date_to, reset):
    # Reset location result before starting computation
    for location in project.locations:
        location.result_function(location, project.timestep,
                                 date_from, date_to, init=True)

    # Reset activity charging station and vehicle results
    for vehicle in project.vehicles:
        vehicle.result_function(vehicle, project.timestep, date_from,
                                date_to, init=True)
        if reset:
            for activity in vehicle.activities:
                if isinstance(activity, model.Parked):
                    activity.charging_station = None

    return project


def _post_run(project, date_from, date_to):
    # Result post processing
    for location in project.locations:
        location.result_function(location, project.timestep, date_from, date_to,
                                 post=True)

    for vehicle in project.vehicles:
        vehicle.result_function(vehicle, project.timestep, date_from,
                                date_to, post=True)

    return project


def initialize_SOC(project, nb_iteration=1, charging_option=None, verbose=True):
    """Initialize the state of charge of each vehicle by running a simulation
    on previous days. Reset any charging infrastructure assigned to None.

    Args:
        project (Project): project to simulate
        nb_iteration (int): number of iteration to converge on vehicle's
            initial state of charge
        charging_option (any): pass some object to the charging function
    Returns:
        convergence (pandas.DataFrame): convergence rate per iteration
    """
    # Create the progress bar
    progress = progressbar.ProgressBar(widgets=['core.initialize_SOC: ',
                                                progressbar.Percentage(),
                                                progressbar.Bar()],
                                       maxval=nb_iteration * len(project.vehicles)).start()

    # Reset assigned charging station and SOC to maximum value
    for vehicle in project.vehicles:
        vehicle.SOC = [vehicle.car_model.maximum_SOC]
        for activity in vehicle.activities:
            if isinstance(activity, model.Parked):
                activity.charging_station = None

    convergence = pandas.DataFrame(
        index=[0],
        data={'mean': numpy.mean([v.SOC[0] for v in project.vehicles]),
              'std': numpy.std([v.SOC[0] for v in project.vehicles]),
              'mean_rate': [0], 'std_rate': [0]})

    # For each iteration
    count = 0
    for indexI in range(0, nb_iteration):
        # For each vehicle
        for vehicle in project.vehicles:
            # For each activity
            for indexA, activity in enumerate(vehicle.activities):
                # Calculate the duration of the activity
                nb_interval = int((activity.end - activity.start).total_seconds() / project.timestep)
                if isinstance(activity, model.Driving):
                    SOC, _1, _2, _3 = vehicle.car_model.driving(activity, vehicle,
                                                                nb_interval,
                                                                project.timestep)
                    if len(SOC) != 0:
                        vehicle.SOC.append(SOC[-1])

                elif isinstance(activity, model.Parked):
                    # Get the charging station if not already assigned
                    if activity.charging_station is None:
                        activity.charging_station = activity.location.assign_charging_station(indexA,
                                                                                              activity, vehicle)

                    # Compute the consumption at the charging station
                    SOC, _1 = activity.charging_station.charging(activity, vehicle,
                                                                 nb_interval, project.timestep,
                                                                 charging_option)
                    if len(SOC) != 0:
                        vehicle.SOC.append(SOC[-1])

            # Initiate Vehicle SOC last value to be the inital SOC next iteration
            vehicle.SOC = [vehicle.SOC[-1]]
            count += 1
            progress.update(count)

        # Update the convergence DataFrame
        convergence = pandas.concat([convergence, pandas.DataFrame(
            index=[indexI + 1],
            data={'mean': numpy.mean([v.SOC[0] for v in project.vehicles]),
                  'std': numpy.std([v.SOC[0] for v in project.vehicles])})],
            axis=0)
        convergence.loc[indexI + 1, 'mean_rate'] = (convergence.loc[indexI, 'mean'] -
                                                    convergence.loc[indexI + 1, 'mean'])
        convergence.loc[indexI + 1, 'std_rate'] = (convergence.loc[indexI, 'std'] -
                                                   convergence.loc[indexI + 1, 'std'])
    progress.finish()
    if verbose:
        print(convergence)
        print('')
    return convergence
