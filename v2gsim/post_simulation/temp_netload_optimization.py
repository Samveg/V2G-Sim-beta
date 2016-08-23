# -*- coding: utf-8 -*-
from __future__ import division
from pyomo.opt import SolverFactory
from pyomo.environ import *
import time
import pandas
import numpy
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import v2gsim.model
import v2gsim.result
import sys
from cStringIO import StringIO
from cvxopt import matrix, solvers
import progressbar

# # Potentially add to requirement.txt
# import math
# import scipy.integrate as integrate

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


class InitDecentralizedOptimization(object):
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
        self.power_demand = []  # vector fill with the result from the optimization


def save_vehicle_state_for_decentralized_optimization(
    vehicle, timestep, date_from, date_to, activity=None, power_demand=None,
    SOC=None, detail=None, nb_interval=None, init=False, run=False, post=False):
    """Save results for individual vehicles. Power demand is positive when charging
    negative when driving. Energy consumption is positive when driving and negative
    when charging. Charging station that offer after simulation processing should
    have activity.charging_station.post_simulation True.
    """
    if run:
        activity_index1, activity_index2, location_index1, location_index2, save = v2gsim.result._map_index(
            activity.start, activity.end, date_from, date_to, len(power_demand),
            len(vehicle.result['power_demand_before']), timestep)
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
                vehicle.result['power_demand_before'][location_index1:location_index2] -= (
                    [0.0] * (activity_index2 - activity_index1))
                # Boolean to set charging or not
                vehicle.result['not_charging'][location_index1:location_index2] += (
                    [1.0] * (activity_index2 - activity_index1))

            # If parked pmin and pmax are not necessary the same
            if isinstance(activity, v2gsim.model.Parked):
                # Save the positive power demand of this specific vehicle
                vehicle.result['power_demand_before'][location_index1:location_index2] += (
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

        # Initiate a dictionary of numpy array to hold result (faster than DataFrame)
        vehicle.result = {'power_demand_before': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                          'p_max': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                          'p_min': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                          'energy': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                          'not_charging': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep))}
    
    elif post:
        # Convert location result back into pandas DataFrame (faster that way)
        i = pandas.date_range(start=date_from, end=date_to,
                              freq=str(timestep) + 's', closed='left')
        vehicle.result = pandas.DataFrame(index=i, data=vehicle.result)
        vehicle.result['energy'] = vehicle.result['energy'].cumsum()


class DecentralizedOptimization(object):
    """Creates an object to perform optimization.
    The object contains some general parameters for the optimization
    """
    def __init__(self, project, optimization_timestep, date_from=None, date_to=None, minimum_SOC=0.1):
        # All the variables are at the project timestep except for the model variables
        # optimization_timestep is in minutes
        self.optimization_timestep = optimization_timestep
        # Set minimum_SOC
        self.minimum_SOC = minimum_SOC
        # Set date boundaries, should be same as the one used during the simulation
        if date_from is None:
            self.date_from = project.date
        else:
            self.date_from = date_from

        if date_to is None:
            self.date_to = (project.date + datetime.timedelta(days=1) -
                            datetime.timedelta(seconds=project.timestep))
        else:
            self.date_to = date_to

        # Find out what is the last SOC index for the optimization window
        self.SOC_index_to = int((self.date_to - project.date).total_seconds() / project.timestep)

    def solve(self, project, net_load, real_number_of_vehicle):
        """
        Args:
            net_load (Pandas): net load in Watt
        """
        # Resample the netload
        new_net_load = self.initialize_net_load(net_load, real_number_of_vehicle, project)

        # Itinialize the model
        self.initialize_model(project)

        # Select feasible vehicles for the optimization
        vehicles_to_optimize, remaining_vehicles = self.select_feasible(project, new_net_load)

        # Launch the optimization
        vehicle_load, new_lbd, dualGap = self.process(vehicles_to_optimize, new_net_load)

        # post process
        total_vehicle_load, net_load_with_vehicles = self.postprocess(
            vehicles_to_optimize, remaining_vehicles, project, net_load, real_number_of_vehicle)

        return vehicle_load, total_vehicle_load, net_load_with_vehicles

    def initialize_net_load(self, net_load, real_number_of_vehicle, project):
        """Make sure that the net load has the right size and scale the net
        load for the optimization scale.

        Args:
            net_load (pandas.DataFrame): data frame with date index and a 'net_load' column in [W]
            net_load_pmax (int): maximum power on the scaled net load
        """
        # Make sure we are not touching the initial data
        new_net_load = net_load.copy()

        # Resample the net load
        new_net_load = new_net_load.resample(str(self.optimization_timestep) + 'T').first()
        temp_date_from = self.date_from.replace(year=new_net_load.head(1).index[0].year,
                                                month=new_net_load.head(1).index[0].month,
                                                day=new_net_load.head(1).index[0].day)
        temp_date_to = self.date_to.replace(year=new_net_load.head(1).index[0].year,
                                            month=new_net_load.head(1).index[0].month,
                                            day=new_net_load.head(1).index[0].day)
        new_net_load = new_net_load[temp_date_from:temp_date_to]

        if real_number_of_vehicle:
            # Set scaling factor
            scaling_factor = len(project.vehicles) / real_number_of_vehicle

            # Scale the temp net load
            new_net_load['netload'] *= scaling_factor

        return new_net_load

    def get_final_SOC(self, vehicle, SOC_margin, SOC_end=None):
        """Get final SOC that vehicle must reached at the end of the optimization
        """
        if SOC_end is not None:
            return SOC_end
        else:
            return vehicle.SOC[self.SOC_index_to] - SOC_margin
 
    def initialize_model(self, project, SOC_margin=0.02, SOC_end=None):
        """
        """
        for vehicle in project.vehicles:
            if 'not_charging' in vehicle.result.columns.tolist():
                # Resample result at the right time step
                temp_vehicle_result = vehicle.result.resample(str(self.optimization_timestep) + 'T').first()
                vehicle.init_opti = InitDecentralizedOptimization()

                # Set the final SOC to reach at the end of the day
                final_SOC = self.get_final_SOC(vehicle, SOC_margin, SOC_end)
                
                # Format arrays for optimization <-- give a description of their expected format
                # Create Aeq
                length = len(temp_vehicle_result)
                Aeq = numpy.array([0] * length).reshape((1, length))
                tempAeq = numpy.diag(temp_vehicle_result.not_charging.tolist())
                deleteFirst = False
                for index, value in enumerate(temp_vehicle_result.not_charging.tolist()):
                    if value == 1:
                        Aeq = numpy.vstack((Aeq, tempAeq[index, :]))
                        deleteFirst = True
                if deleteFirst:
                    Aeq = numpy.delete(Aeq, (0), axis=0)  # Remove the fake first row

                # Create beq
                beq = [0] * Aeq.shape[0]

                # Units !!!!!!
                bineq1 = ([vehicle.car_model.battery_capacity * (vehicle.car_model.maximum_SOC - vehicle.SOC[0]) /
                          (self.optimization_timestep / 60)] * length)  # time interval must be in hour since battery cap is in Wh
                for index, energy in enumerate(temp_vehicle_result.energy.tolist()):
                    bineq1[index] += (energy * project.timestep / 60) / self.optimization_timestep  # E --> Wmin --> W

                bineq2 = ([-vehicle.car_model.battery_capacity * (self.minimum_SOC - vehicle.SOC[0]) /  # minus sign to inverse inequality
                          (self.optimization_timestep / 60)] * length)  # time interval must be in hour since battery cap is in Wh
                for index, energy in enumerate(temp_vehicle_result.energy.tolist()):
                    bineq2[index] -= (energy * project.timestep / 60) / self.optimization_timestep  # minus sign to inverse inequality
                bineq2[-1] = -(vehicle.car_model.battery_capacity * (final_SOC - vehicle.SOC[0]) /
                               (self.optimization_timestep / 60) +
                               (temp_vehicle_result.energy.tolist()[-1] * project.timestep / 60) / self.optimization_timestep)

                # Append to optimization object
                vehicle.init_opti.Aeq = Aeq
                vehicle.init_opti.beq = beq
                vehicle.init_opti.bineq = bineq1 + bineq2  # Extend bineq1 with bineq2
                vehicle.init_opti.lb = temp_vehicle_result.p_min.values
                vehicle.init_opti.ub = temp_vehicle_result.p_max.values
            
            else:
                # Not paticipating in the optimization
                vehicle.init_opti = False

    def solve_local_optimal(self, Aeq, beq, bineq, lb, ub, lbd, sigma):
        """
        """
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

    def select_feasible(self, project, net_load, verbose=True):
        """This function is putting aside all vehicle that could not be solved during a pre-optimization

        Args:
            project (Project): a project
            net_load: net load [W]
        
        Returns:
            vehicles_to_optimize: list of vehicle that can participate in the optimization
            remaining_vehicles: list of vehicle that are not sustaining enough SOC
        """

        # Create the progress bar
        if verbose:
            progress = progressbar.ProgressBar(widgets=['select_feasible: ',
                                                        progressbar.Percentage(),
                                                        progressbar.Bar()],
                                               maxval=len(project.vehicles)).start()

        vehicles_to_optimize = []
        remaining_vehicles = []
        sigma = 1
        lbd = net_load['netload'].values

        count = 0
        for index, vehicle in enumerate(project.vehicles):
            if verbose:
                progress.update(index + 1)

            if vehicle.init_opti:
                error = False
                try:
                    with Capturing() as output:
                        result = self.solve_local_optimal(vehicle.init_opti.Aeq, vehicle.init_opti.beq, vehicle.init_opti.bineq,
                                                     vehicle.init_opti.lb, vehicle.init_opti.ub, lbd, sigma)
                except:
                    error = True

                if error:
                    count += 1
                    remaining_vehicles.append(vehicle)
                elif len(output) >= 1:
                    if output[0] in 'Terminated (singular KKT matrix).':
                        count += 1
                        remaining_vehicles.append(vehicle)
                else:
                    vehicles_to_optimize.append(vehicle)
            else:
                remaining_vehicles.append(vehicle)

        if verbose:
            progress.finish()
        print("There is " + str(len(vehicles_to_optimize) * 100 / len(project.vehicles)) +
              "% vehicle participating in the optimization")
        print(str(count) + " vehicles could not be solved")
        print("")
        return vehicles_to_optimize, remaining_vehicles

    def process(self, vehicles_to_optimize, net_load):
        """This function compute the optimal power demand for each vehicle regarding their daily consumption and
        a net load curve. The function "initialize_optimization" must be run as a preliminary.

        Args:
            vehicles_to_optimize: list of vehicle ready for optimization
            netLoadInput: net load (kw every hour)

        Returns:
            vehicles_to_optimize: list of vehicle updated with their optimal power demand every outputInterval

        """
        iteration = 1
        maxIteration = 10
        sigma = 1 + len(vehicles_to_optimize) / 2  # Degradation (regularization term)
        beta = 0
        alpha = 2 * sigma / (sigma + len(vehicles_to_optimize))
        epsilon = 0.0015

        # Optimization
        length = len(vehicles_to_optimize[0].init_opti.lb)
        D = net_load['netload'].values
        lbd = net_load['netload'].values
        new_lbd = net_load['netload'].values
        objp = 100
        objd = 0
        dualGap = []
        while iteration < maxIteration and abs(objp - objd) > epsilon * abs(objd):
            print('Iteration #' + str(iteration))
            vehicleLoad = [0] * length
            norm_sum_u = 0
            lbd = numpy.array(new_lbd)
            objd = -0.25 * numpy.dot(lbd, numpy.transpose(lbd)) + numpy.dot(D, numpy.transpose(lbd))
            for vehicle in vehicles_to_optimize:
                error = False
                try:
                    result = self.solve_local_optimal(vehicle.init_opti.Aeq, vehicle.init_opti.beq, vehicle.init_opti.bineq, vehicle.init_opti.lb,
                                                      vehicle.init_opti.ub, lbd, sigma)
                except:
                    error = True
                    # In case of error keep the previous version of the init_opti.powerDemand <-- pray it doesn't break at the
                    # first iteration :)
                    u = numpy.array(vehicle.init_opti.power_demand).reshape((1, len(vehicle.init_opti.lb)))
                if not error:
                    u = numpy.array([result['x'][index, 0] for index in range(0, len(vehicle.init_opti.lb))]).reshape((1, len(vehicle.init_opti.lb)))

                for index, value in enumerate(new_lbd):
                    new_lbd[index] += alpha / iteration**beta * u[0, index]
                objd += numpy.dot(lbd, numpy.transpose(u)) + sigma * numpy.dot(u, numpy.transpose(u))
                norm_sum_u += sigma * numpy.dot(u, numpy.transpose(u))
                vehicle.init_opti.power_demand = u.tolist()[0]
                vehicleLoad = [load + power for load, power in zip(vehicleLoad, vehicle.init_opti.power_demand)]

            for index, value in enumerate(new_lbd):
                new_lbd[index] += alpha / iteration**beta * (-0.5 * lbd[index] + D[index])
            netLoad = numpy.array([D[i] + vehicleLoad[i] for i in range(0, length)])
            objp = numpy.dot(numpy.transpose(netLoad), netLoad) + norm_sum_u

            # Save the error to see the convergence rate toward epsilon=10e-5
            dualGap.append((abs(objp - objd))[0][0] / (abs(objd))[0][0])
            iteration += 1

        print('')
        return vehicleLoad, new_lbd, dualGap


    def postprocess(self, vehicles_to_optimize, remaining_vehicles, project, net_load, real_number_of_vehicle):
        """
        """
        # Resample data and save it at the right place in vehicle.result
        i = pandas.date_range(start=self.date_from, end=self.date_to,
                              freq=str(self.optimization_timestep) + 'T', closed='left')
        for vehicle in vehicles_to_optimize:
            temp_demand = pandas.DataFrame(index=i, data=vehicle.init_opti.power_demand)
            vehicle.result['power_demand'] = temp_demand.resample(str(project.timestep) + 's').interpolate()
            vehicle.result['power_demand'] = vehicle.result['power_demand'].ffill()

        for vehicle in remaining_vehicles:
            if 'power_demand_before' in vehicle.result.columns.tolist():
                vehicle.result['power_demand'] = vehicle.result['power_demand_before']

        i = pandas.date_range(start=self.date_from, end=self.date_to,
                              freq=str(project.timestep) + 's', closed='left')
        vehicle_load = pandas.DataFrame(index=i, data={'power_demand': [0.0] * len(i)})
        # Get the total load <-- could make it clearer !
        for index, vehicle in enumerate(project.vehicles):
            vehicle_load['power_demand'] += vehicle.result.power_demand

        # Get the net load to sum with the vehicle load
        temp_net_load = net_load.copy()
        temp_net_load = temp_net_load.resample(str(project.timestep) + 's').interpolate()
        i = pandas.date_range(start=self.date_from, end=self.date_to,
                              freq=str(project.timestep) + 's', closed='left')
        temp_net_load = temp_net_load.head(len(i)).netload.tolist()
        temp_net_load = pandas.DataFrame(index=i, data={'power_demand': temp_net_load})

        if not real_number_of_vehicle:
            scaling_factor = 1
        else:
            scaling_factor =  real_number_of_vehicle / len(project.vehicles)

        return vehicle_load, vehicle_load * scaling_factor + temp_net_load
