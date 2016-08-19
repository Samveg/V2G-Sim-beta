# -*- coding: utf-8 -*-
from __future__ import division
from pyomo.opt import SolverFactory
from pyomo.environ import *
import time
import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import v2gsim.model
import v2gsim.result


# Potentially add to requirement.txt
import math
import scipy.integrate as integrate
from cvxopt import matrix, solvers
from cStringIO import StringIO
import sys


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


def save_vehicle_state_for_decentralized_optimization(
    vehicle, timestep, date_from, date_to, activity=None, power_demand=None,
    SOC=None, detail=None, nb_interval=None, init=False, run=False, post=False):
    """Save results for individual vehicles. Power demand is positive when charging
    negative when driving. Energy consumption is positive when driving and negative
    when charging. Charging station that offer after simulation processing should
    have activity.charging_station.post_simulation True.
    """
    if run:
        if vehicle.result is not None:
            activity_index1, activity_index2, location_index1, location_index2, save = v2gsim.result._map_index(
                activity.start, activity.end, date_from, date_to, len(power_demand),
                len(vehicle.result['power_demand']), timestep)
            # Time frame are matching
            if save:
                # If driving pmin and pmax are equal to 0 since we are not plugged
                if isinstance(activity, v2gsim.model.Driving):
                    vehicle.result['p_max'][location_index1:location_index2] -= (
                        [0.0] * (activity_index2 - activity_index1))
                    vehicle.result['p_min'][location_index1:location_index2] -= (
                        [0.0] * (activity_index2 - activity_index1))
                    # Energy consumed is directly the power demand (sum later)
                    vehicle.result['energy'][location_index1:location_index2] += (
                        power_demand[activity_index1:activity_index2])
                    # Power demand on the grid is 0 since we are driving
                    vehicle.result['power_demand'][location_index1:location_index2] -= (
                        [0.0] * (activity_index2 - activity_index1))
                    # Boolean to set charging or not
                    vehicle.result['not_charging'][location_index1:location_index2] += (
                        [1.0] * (activity_index2 - activity_index1))

                # If parked pmin and pmax are not necessary the same
                if isinstance(activity, v2gsim.model.Parked):
                    # Save the positive power demand of this specific vehicle
                    vehicle.result['power_demand'][location_index1:location_index2] += (
                        power_demand[activity_index1:activity_index2])
                    if activity.charging_station.post_simulation:
                        # Find if vehicle or infra is limiting
                        pmax = min(activity.charging_station.maximum_power,
                                   vehicle.car_model.maximum_power)
                        pmin = max(activity.charging_station.minimum_power,
                                   vehicle.car_model.minimum_power)
                        vehicle.result['p_max'][location_index1:location_index2] += (
                            [pmax] * (activity_index2 - activity_index1))
                        vehicle.result['p_min'][location_index1:location_index2] += (
                            [pmin] * (activity_index2 - activity_index1))
                        # Energy consumed is 0 the optimization will decide
                        vehicle.result['energy'][location_index1:location_index2] -= (
                            [0.0] * (activity_index2 - activity_index1))
                    else:
                        vehicle.result['p_max'][location_index1:location_index2] += (
                            power_demand[activity_index1:activity_index2])
                        vehicle.result['p_min'][location_index1:location_index2] += (
                            power_demand[activity_index1:activity_index2])
                        # Energy is 0.0 because it's already accounted in power_demand
                        vehicle.result['energy'][location_index1:location_index2] -= (
                            [0.0] * (activity_index2 - activity_index1))
                        # Boolean to set not charging -- NOT used to force the power demand
                        # vehicle.result['not_charging'][location_index1:location_index2] += (
                        #     [1.0] * (activity_index2 - activity_index1))

    elif init:
        # Reset vehicle with no result and only 1 SOC value
        vehicle.SOC = [vehicle.SOC[0]]
        vehicle.result = None

        # Initialize result variable in a different manner if vehicle will be part of the optimization
        for activity in vehicle.activities:
            if isinstance(activity, v2gsim.model.Parked):
                if activity.charging_station.post_simulation:
                    # Initiate a dictionary of numpy array to hold result (faster than DataFrame)
                    vehicle.result = {'power_demand': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                                      'p_max': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                                      'p_min': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                                      'energy': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                                      'not_charging': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep))}
                    # Leave the init function
                    return
                else:
                    # Just save the power demand
                    vehicle.result = {'power_demand': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep))}
    
    elif post:
        if vehicle.result is not None:
            # Convert location result back into pandas DataFrame (faster that way)
            i = pandas.date_range(start=date_from, end=date_to,
                                  freq=str(timestep) + 's', closed='left')
            vehicle.result = pandas.DataFrame(index=i, data=vehicle.result)
            if 'energy' in vehicle.result.columns:
                vehicle.result['energy'] = vehicle.result['energy'].cumsum()

class DecentralizedOptimization(object):
    """Creates an object to perform optimization.
    The object contains some general parameters for the optimization
    """
    def __init__(self, project, optimization_timestep, date_from,
                 date_to, minimum_SOC=0.1, maximum_SOC=0.95):
        # All the variables are at the project timestep except for the model variables
        # optimization_timestep is in minutes
        self.optimization_timestep = optimization_timestep
        # Set minimum_SOC
        self.minimum_SOC = minimum_SOC
        self.maximum_SOC = maximum_SOC
        # Set date boundaries, should be same as the one used during the simulation
        self.date_from = date_from
        self.date_to = date_to
        self.SOC_index_from = int((date_from - project.date).total_seconds() / project.timestep)
        self.SOC_index_to = int((date_to - project.date).total_seconds() / project.timestep)
    
    def initialize_model(project):
        """
        """
        for vehicle in project.vehicles:
            if 'not_charging' in vehicle.result.columns:
                # Resample result at the right time step
                temp_vehicle_result = vehicle.result.resample(str(self.optimization_timestep) + 'T',
                                                              how='first')
                # Format arrays for optimization <-- give a description of their expected format
                # Initiate to be able to use numpy.vstack, remove later
                Aeq = numpy.array([0] * len(chargingOrNot)).reshape((1, len(chargingOrNot)))
                tempAeq = numpy.diag(chargingOrNot)
                deleteFirst = False
                for index, value in enumerate(chargingOrNot):
                    if value == 1:
                        Aeq = numpy.vstack((Aeq, tempAeq[index, :]))
                        deleteFirst = True
                if deleteFirst:
                    Aeq = numpy.delete(Aeq, (0), axis=0)  # Remove the fake first row
                beq = [0] * Aeq.shape[0]

                bineq1 = ([vehicle.carModel.batteryCap * (vehicle.carModel.SOCMax - vehicle.SOC[0]) / (timeInterval / 3600)] *
                          len(powerCons))  # time interval must be in hour since battery cap is in Wh
                for index, power in enumerate(powerCons):
                    bineq1[index] = bineq1[index - 1] + power

                bineq2 = ([-vehicle.carModel.batteryCap * (minimumSOC - vehicle.SOC[0]) / (timeInterval / 3600)] *
                          len(powerCons))  # time interval must be in hour since battery cap is in Wh
                for index, power in enumerate(powerCons):
                    bineq2[index] = bineq2[index - 1] - power
                bineq2[-1] = -(vehicle.carModel.batteryCap * (finalSOC - vehicle.SOC[0]) / (timeInterval / 3600) + sum(powerCons))

                bineq = bineq1 + bineq2

                # Append to optimization object
                vehicle.extra.Aeq = Aeq
                vehicle.extra.beq = beq
                vehicle.extra.bineq = bineq
                vehicle.extra.lb = minimumPower
                vehicle.extra.ub = maximumPower

class Optimization(object):
    """Optimization object for scheduling vehicle charging
    """

    def __init__(self):
        # self.H and self.f are not vehicle dependent
        self.Aeq = []  # eye matrix with size nbInterval, where "charging row" are cut off
        self.beq = []  # vector fill with 0 for the number of row in Aeq
        # self.Aineq is not needed since it's the same for everybody
        self.bineq = []  # vector fill with energy wise constraint
        self.lb = []  # vector fill with Pmin at all time
        self.ub = []  # vector fill with Pmax at all time
        self.powerDemand = []  # vector fill with the result from the optimization


def initialize_optimization(vehicleList, timeInterval=0.25, minimumSOC=0.1, finalSOC=0.4):
    """This function initialize an optimization object for each vehicle with the result of a previous simulation
    (note: we might have to add the timeHorizon as a parameter of the function)

    Args:
        vehicleList: list of vehicles
        timeInterval: time interval (in hour) to use in the optimization problem
        minimumSOC: minimum SOC at any time
        finalSOC: final SOC at the end of the time horizon

    Returns:
        vehicleList_forOptimization: list of vehicle ready for the optimization process

    """
    horizonTime = 24 * 3600
    timeInterval *= 3600
    minPowerRateNull = False

    for vehicle in vehicleList:
        length = int(horizonTime / vehicle.outputInterval)
        vehicle.extra = Optimization()
        energyCons = [0]
        chargingOrNot = []
        minimumPower = []
        maximumPower = []

        for activity in vehicle.itinerary[0].activity:
            if isinstance(activity, model.Driving):
                energyActivity = integrate.cumtrapz(y=activity.powerDemand, dx=vehicle.outputInterval, initial=0.0)
                energyActivity = [energyActivity[i] + energyCons[-1] for i in range(0, len(activity.powerDemand))]
                energyCons.extend(energyActivity)
                chargingOrNot.extend([1] * len(activity.powerDemand))

                if minPowerRateNull:
                    minimumPower.extend([0] * len(activity.powerDemand))
                else:
                    minimumPower.extend([-1440] * len(activity.powerDemand))
                maximumPower.extend([1440] * len(activity.powerDemand))

            elif isinstance(activity, model.Parked):
                energyActivity = [energyCons[-1] for i in range(0, len(activity.powerDemand))]
                energyCons.extend(energyActivity)
                if activity.pluggedIn:
                    chargingOrNot.extend([0] * len(activity.powerDemand))
                else:
                    chargingOrNot.extend([1] * len(activity.powerDemand))

                # <-- Replace by maximum of
                maxRate = vehicle.carModel.powerRateMax
                minRate = vehicle.carModel.powerRateMin
                if maxRate > activity.location.chargingInfra.powerRateMax:
                    maxRate = activity.location.chargingInfra.powerRateMax
                if minRate < -activity.location.chargingInfra.powerRateMax:
                    minRate = -activity.location.chargingInfra.powerRateMax

                if minPowerRateNull:
                    minRate = 0

                minimumPower.extend([minRate] * len(activity.powerDemand))
                maximumPower.extend([maxRate] * len(activity.powerDemand))
        del energyCons[0]

        # Interpolate arrays for the time interval of the optimization
        timeOfOpti = [index * timeInterval for index in range(0, int(horizonTime / timeInterval))]
        timeOfOutput = [index * vehicle.outputInterval for index in range(0, length)]

        energyCons = numpy.interp(timeOfOpti, timeOfOutput, energyCons[0:length])
        powerCons = [(energyCons[index] - energyCons[index - 1]) / timeInterval for index in range(1, len(energyCons))]
        powerCons.append(powerCons[-1])

        chargingOrNot = numpy.interp(timeOfOpti, timeOfOutput, chargingOrNot[0:length])
        chargingOrNot = [math.ceil(chargingOrNot[index]) for index in range(0, len(chargingOrNot))]  # Keep only 1 and 0
        minimumPower = numpy.interp(timeOfOpti, timeOfOutput, minimumPower[0:length])
        maximumPower = numpy.interp(timeOfOpti, timeOfOutput, maximumPower[0:length])

        if len(chargingOrNot) != int(horizonTime / timeInterval):
            raise TypeError("Not enough data for time horizon")

        # Format arrays for optimization <-- give a description of their expected format
        # Initiate to be able to use numpy.vstack, remove later
        Aeq = numpy.array([0] * len(chargingOrNot)).reshape((1, len(chargingOrNot)))
        tempAeq = numpy.diag(chargingOrNot)
        deleteFirst = False
        for index, value in enumerate(chargingOrNot):
            if value == 1:
                Aeq = numpy.vstack((Aeq, tempAeq[index, :]))
                deleteFirst = True
        if deleteFirst:
            Aeq = numpy.delete(Aeq, (0), axis=0)  # Remove the fake first row
        beq = [0] * Aeq.shape[0]

        bineq1 = ([vehicle.carModel.batteryCap * (vehicle.carModel.SOCMax - vehicle.SOC[0]) / (timeInterval / 3600)] *
                  len(powerCons))  # time interval must be in hour since battery cap is in Wh
        for index, power in enumerate(powerCons):
            bineq1[index] = bineq1[index - 1] + power

        bineq2 = ([-vehicle.carModel.batteryCap * (minimumSOC - vehicle.SOC[0]) / (timeInterval / 3600)] *
                  len(powerCons))  # time interval must be in hour since battery cap is in Wh
        for index, power in enumerate(powerCons):
            bineq2[index] = bineq2[index - 1] - power
        bineq2[-1] = -(vehicle.carModel.batteryCap * (finalSOC - vehicle.SOC[0]) / (timeInterval / 3600) + sum(powerCons))

        bineq = bineq1 + bineq2

        # Append to optimization object
        vehicle.extra.Aeq = Aeq
        vehicle.extra.beq = beq
        vehicle.extra.bineq = bineq
        vehicle.extra.lb = minimumPower
        vehicle.extra.ub = maximumPower

    return vehicleList


def select_participant(vehicleList, timeInterval, netLoadInput, netLoadSamplingRate=3600):
    """This function is putting aside all vehicle that could not be solved during a pre-optimization

    Args:
        vehicleList: list of vehicle
        timeInterval: timeInterval for the optimization (see initialize_optimization())
        netLoadInput: net load (kw every hour)
        netLoadSamplingRate: sampling rate of the netLoad (in seconds)
    Returns:
        vehicleList_forOptimization: list of vehicle that can participate in the optimization
        vehicleList_notParticipating: list of vehicle that are not sustaining enough SOC

    """
    print(colored.green('select_participant(vehicleList)'))

    vehicleList_forOptimization = []
    vehicleList_notParticipating = []
    sigma = 1

    # Get duck to be sample at timeInterval
    horizonTime = 24 * 3600
    timeInterval *= 3600
    timeOfOpti = [index * timeInterval for index in range(0, int(horizonTime / timeInterval))]
    timeOfNetLoad = [index * netLoadSamplingRate for index in range(0, len(netLoadInput))]
    D = numpy.interp(timeOfOpti, timeOfNetLoad, netLoadInput)
    lbd = numpy.array(D, ndmin=2)

    count = 0
    for vehicle in vehicleList:
        error = False
        try:
            with Capturing() as output:
                result = solve_local_optimal(vehicle.extra.Aeq, vehicle.extra.beq, vehicle.extra.bineq,
                                             vehicle.extra.lb, vehicle.extra.ub, lbd, sigma)
        except:
            # result = {'status': 'error'}
            error = True

        if error:
            count += 1
            vehicleList_notParticipating.append(vehicle)
        elif len(output) >= 1:
            if output[0] in 'Terminated (singular KKT matrix).':
                count += 1
                vehicleList_notParticipating.append(vehicle)
        else:
            vehicleList_forOptimization.append(vehicle)

    print("There is " + str(len(vehicleList_forOptimization) * 100 / len(vehicleList)) +
          "% vehicle participating in the optimization")
    print(str(count) + " vehicles could not be solved")
    print("")
    return vehicleList_forOptimization, vehicleList_notParticipating


def initialize_optimization(vehicleList, timeInterval=0.25, minimumSOC=0.1, finalSOC=0.4):
    """This function initialize an optimization object for each vehicle with the result of a previous simulation
    (note: we might have to add the timeHorizon as a parameter of the function)

    Args:
        vehicleList: list of vehicles
        timeInterval: time interval (in hour) to use in the optimization problem
        minimumSOC: minimum SOC at any time
        finalSOC: final SOC at the end of the time horizon

    Returns:
        vehicleList_forOptimization: list of vehicle ready for the optimization process

    """
    horizonTime = 24 * 3600
    timeInterval *= 3600
    minPowerRateNull = False

    for vehicle in vehicleList:
        length = int(horizonTime / vehicle.outputInterval)
        vehicle.extra = Optimization()
        energyCons = [0]
        chargingOrNot = []
        minimumPower = []
        maximumPower = []

        for activity in vehicle.itinerary[0].activity:
            if isinstance(activity, model.Driving):
                energyActivity = integrate.cumtrapz(y=activity.powerDemand, dx=vehicle.outputInterval, initial=0.0)
                energyActivity = [energyActivity[i] + energyCons[-1] for i in range(0, len(activity.powerDemand))]
                energyCons.extend(energyActivity)
                chargingOrNot.extend([1] * len(activity.powerDemand))

                if minPowerRateNull:
                    minimumPower.extend([0] * len(activity.powerDemand))
                else:
                    minimumPower.extend([-1440] * len(activity.powerDemand))
                maximumPower.extend([1440] * len(activity.powerDemand))

            elif isinstance(activity, model.Parked):
                energyActivity = [energyCons[-1] for i in range(0, len(activity.powerDemand))]
                energyCons.extend(energyActivity)
                if activity.pluggedIn:
                    chargingOrNot.extend([0] * len(activity.powerDemand))
                else:
                    chargingOrNot.extend([1] * len(activity.powerDemand))

                # <-- Replace by maximum of
                maxRate = vehicle.carModel.powerRateMax
                minRate = vehicle.carModel.powerRateMin
                if maxRate > activity.location.chargingInfra.powerRateMax:
                    maxRate = activity.location.chargingInfra.powerRateMax
                if minRate < -activity.location.chargingInfra.powerRateMax:
                    minRate = -activity.location.chargingInfra.powerRateMax

                if minPowerRateNull:
                    minRate = 0

                minimumPower.extend([minRate] * len(activity.powerDemand))
                maximumPower.extend([maxRate] * len(activity.powerDemand))
        del energyCons[0]

        # Interpolate arrays for the time interval of the optimization
        timeOfOpti = [index * timeInterval for index in range(0, int(horizonTime / timeInterval))]
        timeOfOutput = [index * vehicle.outputInterval for index in range(0, length)]

        energyCons = numpy.interp(timeOfOpti, timeOfOutput, energyCons[0:length])
        powerCons = [(energyCons[index] - energyCons[index - 1]) / timeInterval for index in range(1, len(energyCons))]
        powerCons.append(powerCons[-1])

        chargingOrNot = numpy.interp(timeOfOpti, timeOfOutput, chargingOrNot[0:length])
        chargingOrNot = [math.ceil(chargingOrNot[index]) for index in range(0, len(chargingOrNot))]  # Keep only 1 and 0
        minimumPower = numpy.interp(timeOfOpti, timeOfOutput, minimumPower[0:length])
        maximumPower = numpy.interp(timeOfOpti, timeOfOutput, maximumPower[0:length])

        if len(chargingOrNot) != int(horizonTime / timeInterval):
            raise TypeError("Not enough data for time horizon")

        # Format arrays for optimization <-- give a description of their expected format
        # Initiate to be able to use numpy.vstack, remove later
        Aeq = numpy.array([0] * len(chargingOrNot)).reshape((1, len(chargingOrNot)))
        tempAeq = numpy.diag(chargingOrNot)
        deleteFirst = False
        for index, value in enumerate(chargingOrNot):
            if value == 1:
                Aeq = numpy.vstack((Aeq, tempAeq[index, :]))
                deleteFirst = True
        if deleteFirst:
            Aeq = numpy.delete(Aeq, (0), axis=0)  # Remove the fake first row
        beq = [0] * Aeq.shape[0]

        bineq1 = ([vehicle.carModel.batteryCap * (vehicle.carModel.SOCMax - vehicle.SOC[0]) / (timeInterval / 3600)] *
                  len(powerCons))  # time interval must be in hour since battery cap is in Wh
        for index, power in enumerate(powerCons):
            bineq1[index] = bineq1[index - 1] + power

        bineq2 = ([-vehicle.carModel.batteryCap * (minimumSOC - vehicle.SOC[0]) / (timeInterval / 3600)] *
                  len(powerCons))  # time interval must be in hour since battery cap is in Wh
        for index, power in enumerate(powerCons):
            bineq2[index] = bineq2[index - 1] - power
        bineq2[-1] = -(vehicle.carModel.batteryCap * (finalSOC - vehicle.SOC[0]) / (timeInterval / 3600) + sum(powerCons))

        bineq = bineq1 + bineq2

        # Append to optimization object
        vehicle.extra.Aeq = Aeq
        vehicle.extra.beq = beq
        vehicle.extra.bineq = bineq
        vehicle.extra.lb = minimumPower
        vehicle.extra.ub = maximumPower

    return vehicleList


def optimization(vehicleList_forOptimization, timeInterval, netLoadInput, netLoadSamplingRate=3600):
    """This function compute the optimal power demand for each vehicle regarding their daily consumption and
    a net load curve. The function "initialize_optimization" must be run as a preliminary.

    Args:
        vehicleList_forOptimization: list of vehicle ready for optimization
        timeInterval: timeInterval for the optimization (see initialize_optimization())
        netLoadInput: net load (kw every hour)
        netLoadSamplingRate: sampling rate of the netLoad (in seconds)

    Returns:
        vehicleList_forOptimization: list of vehicle updated with their optimal power demand every outputInterval

    """
    print(colored.green('optimization(vehicleList, timeInterval, netLoadInput, netLoadSamplingRate=3600)'))

    horizonTime = 24 * 3600
    timeInterval *= 3600
    iteration = 1
    maxIteration = 25
    sigma = 1 + len(vehicleList_forOptimization) / 2  # Degradation (regularization term)
    beta = 0
    alpha = 2 * sigma / (sigma + len(vehicleList_forOptimization))
    epsilon = 0.0015

    # Get duck to be sample at timeInterval
    timeOfOpti = [index * timeInterval for index in range(0, int(horizonTime / timeInterval))]
    timeOfNetLoad = [index * netLoadSamplingRate for index in range(0, len(netLoadInput))]
    D = numpy.interp(timeOfOpti, timeOfNetLoad, netLoadInput)
    D = numpy.array(D, ndmin=2)

    # Optimization
    lbd = numpy.array(D)
    new_lbd = numpy.array(lbd)
    objp = 100
    objd = 0
    dualGap = []
    while iteration < maxIteration and abs(objp - objd) > epsilon * abs(objd):
        print("Iteration #" + str(iteration))
        vehicleLoad = [0] * len(timeOfOpti)
        norm_sum_u = 0
        lbd = numpy.array(new_lbd)
        objd = -0.25 * numpy.dot(lbd, numpy.transpose(lbd)) + numpy.dot(D, numpy.transpose(lbd))
        for vehicle in vehicleList_forOptimization:
            error = False
            try:
                result = solve_local_optimal(vehicle.extra.Aeq, vehicle.extra.beq, vehicle.extra.bineq, vehicle.extra.lb,
                                             vehicle.extra.ub, lbd, sigma)
            except:
                error = True
                # In case of error keep the previous version of the extra.powerDemand <-- pray it doesn't break at the
                # first iteration :)
                u = numpy.array(vehicle.extra.powerDemand).reshape((1, len(vehicle.extra.lb)))
            if not error:
                u = numpy.array([result['x'][index, 0] for index in range(0, len(vehicle.extra.lb))]).reshape((1, len(vehicle.extra.lb)))

            for index, value in enumerate(new_lbd):
                new_lbd[index] += alpha / iteration**beta * u[0, index]
            objd += numpy.dot(lbd, numpy.transpose(u)) + sigma * numpy.dot(u, numpy.transpose(u))
            norm_sum_u += sigma * numpy.dot(u, numpy.transpose(u))
            vehicle.extra.powerDemand = u.tolist()[0]
            vehicleLoad = [load + power for load, power in zip(vehicleLoad, vehicle.extra.powerDemand)]

        for index, value in enumerate(new_lbd):
            new_lbd[index] += alpha / iteration**beta * (-0.5 * lbd[0, index] + D[0, index])
        netLoad = numpy.array([D[0, i] + vehicleLoad[i] for i in range(0, len(timeOfOpti))])
        objp = numpy.dot(numpy.transpose(netLoad), netLoad) + norm_sum_u

        # Save the error to see the convergence rate toward epsilon=10e-5
        dualGap.append((abs(objp - objd))[0][0] / (abs(objd))[0][0])
        iteration += 1

    # Update each activity with a new calculated consumption
    for vehicle in vehicleList_forOptimization:
        hourlySteps = 3600 / vehicle.outputInterval
        timeOfOuput = [index * vehicle.outputInterval for index in range(0, int(24 * 3600 / vehicle.outputInterval))]
        powerDemand = numpy.interp(timeOfOuput, timeOfOpti, vehicle.extra.powerDemand)
        for activity in vehicle.itinerary[0].activity:
            if isinstance(activity, model.Parked):
                activity.powerDemand = powerDemand[int(activity.start * hourlySteps):int(activity.end * hourlySteps)]

    print("")
    return vehicleList_forOptimization, new_lbd, dualGap


def solve_local_optimal(Aeq, beq, bineq, lb, ub, lbd, sigma):
    # Get H, f and Aineq for optimization
    H = numpy.multiply(numpy.eye(len(lb)), 2 * sigma)
    f = lbd
    Aineq = numpy.vstack((numpy.tri(len(lb)), -numpy.tri(len(lb))))

    # Convert to cvxopt format
    n = H.shape[1]

    Q = H.astype(numpy.double)  # quadratic terms
    p = numpy.reshape(f.astype(numpy.double), (len(lb), 1))  # linear terms
    G = numpy.vstack((Aineq, -numpy.eye(n), numpy.eye(n))).astype(numpy.double)  # A inequalities
    h = numpy.hstack((bineq, -lb, ub)).astype(numpy.double)  # b inequalities
    h = numpy.reshape(h, (4 * len(lb), 1))
    A = Aeq.astype(numpy.double)  # A equalities
    b = numpy.array(beq).astype(numpy.double)  # b equalities
    b = numpy.reshape(b, (A.shape[0], 1))

    # Solve the problem
    solvers.options['show_progress'] = False
    if numpy.amax(Aeq) == 0:
        u = solvers.qp(matrix(Q), matrix(p), matrix(G), matrix(h))
    else:
        u = solvers.qp(matrix(Q), matrix(p), matrix(G), matrix(h), matrix(A), matrix(b))
    return u


def recompute_SOC_profile(vehicleList):
    """This function re-compute the SOC profile of a vehicle from the consumption of its individual activities.
    Initial SOC is conserved.

    Args:
        vehicleList: list of vehicle

    Returns:
        vehicleList: list of vehicle with SOC profile matching activity consumption

    """
    print(colored.green('recompute_SOC_profile(vehicleList)'))

    countRoof = 0
    for vehicle in vehicleList:
        vehicle.SOC = [vehicle.SOC[0]]
        for activity in vehicle.itinerary[0].activity:
            for power in activity.powerDemand:
                if isinstance(activity, model.Parked):
                    vehicle.SOC.append(_SOC_increase(vehicle.SOC[-1], power,
                                                     vehicle.outputInterval, vehicle.carModel.batteryCap))
                    if vehicle.SOC[-1] > vehicle.carModel.SOCMax:
                        vehicle.SOC[-1] = 0.95
                        countRoof += 1
                    if vehicle.SOC[-1] < 0:
                        vehicle.SOC[-1] = 0

                if isinstance(activity, model.Driving):
                    vehicle.SOC.append(_SOC_increase(vehicle.SOC[-1], -power,
                                                     vehicle.outputInterval, vehicle.carModel.batteryCap))
                    if vehicle.SOC[-1] > vehicle.carModel.SOCMax:
                        vehicle.SOC[-1] = 0.95
                        countRoof += 1
                    if vehicle.SOC[-1] < 0:
                        vehicle.SOC[-1] = 0

    print("SOC tried to go beyond maximum value " + str(countRoof) + " times")
    print("")
    return vehicleList


def _SOC_increase(lastSOC, powerRate, duration, batteryCap):
    # lastSOC (1<x<0), powerRate (Watt), duration (seconds), batteryCap (wh)
    return lastSOC + (powerRate * duration / (batteryCap * 3600))  # batteryCap from Wh to Joules
