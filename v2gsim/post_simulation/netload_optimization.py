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
import datetime


class CentralOptimization(object):
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

    def solve(self, project, net_load, real_number_of_vehicle, SOC_margin=0.02,
              SOC_offset=0.0, peak_shaving='peak_shaving', penalization=5, beta=None, plot=False):
        """Launch the optimization and the post_processing fucntion. Results
        and assumptions are appended to a data frame.

        Args:
            project (Project): project
            net_load (pandas.DataFrame): data frame with date index and a 'net_load' column in [W]
            real_number_of_vehicle (int): number of vehicle expected on the net load,
                False if number is the same as in project
            SOC_margin (float): SOC margin that can be used by the optimization at the end of the day [0, 1]
            SOC_offset (float): energy offset [0, 1]
            peak_shaving (bolean): if True ramping constraints are not taking in account within the objective else it is.
        """
        # Reset model
        self.times = []
        self.vehicles = []
        self.d = {}
        self.pmax = {}
        self.pmin = {}
        self.emin = {}
        self.emax = {}
        self.efinal = {}

        # Set the variables for the optimization
        new_net_load = self.initialize_net_load(net_load, real_number_of_vehicle, project)
        self.initialize_model(project, new_net_load, SOC_margin, SOC_offset)

        # Run the optimization
        timer = time.time()
        opti_model, result = self.process(self.times, self.vehicles, self.d, self.pmax,
                                          self.pmin, self.emin, self.emax,
                                          self.efinal, peak_shaving, penalization)
        timer2 = time.time()
        print('The optimization duration was ' + str((timer2 - timer) / 60) + ' minutes')
        print('')

        # Post process results
        return self.post_process(project, net_load, opti_model, result, plot)

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

        # Check against the actual lenght it should have
        diff = (len(new_net_load) -
                int((self.date_to - self.date_from).total_seconds() / (60 * self.optimization_timestep)))
        if diff > 0:
            # We should trim the net load with diff elements (normaly 1 because of slicing inclusion)
            new_net_load.drop(new_net_load.tail(diff).index, inplace=True)
        elif diff < 0:
            print('The net load does not contain enough data points')

        if real_number_of_vehicle:
            # Set scaling factor
            scaling_factor = len(project.vehicles) / real_number_of_vehicle

            # Scale the temp net load
            new_net_load['netload'] *= scaling_factor

        return new_net_load

    def check_energy_constraints_feasible(self, vehicle, SOC_init, SOC_final, SOC_offset, verbose=False):
        """Make sure that SOC final can be reached from SOC init under uncontrolled
        charging (best case scenario). Print details when conditions are not met.

        Args:
            vehicle (Vehicle): vehicle
            SOC_init (float): state of charge at the begining of the optimization [0, 1]
            SOC_final (float): state of charge at the end of the optimization [0, 1]
            SOC_offset (float): energy offset [0, 1]

        Return:
            (Boolean)
        """
        def print_status(SOC_final, SOC_init, diff):
                print('Set battery gain: ' + str((SOC_final - SOC_init) * 100) + '%')
                print('Simulation battery gain: ' + str(diff * 100))

        # Check if below minimum SOC at any time
        if (min(vehicle.SOC[self.SOC_index_from:self.SOC_index_to]) - SOC_offset) <= self.minimum_SOC:
            if verbose:
                print('Vehicle: ' + str(vehicle.id) + ' has a minimum SOC of ' +
                      str(min(vehicle.SOC[self.SOC_index_from:self.SOC_index_to]) * 100) + '%')
            return False

        # Check SOC difference between date_from and date_to ?
        # Diff represent the minimum loss or the maximum gain
        diff = vehicle.SOC[self.SOC_index_to] - vehicle.SOC[self.SOC_index_from]
        # Simulation show a battery gain
        if diff > 0:
            # Gain should be greater than the one we set up
            if SOC_final - SOC_init < diff:
                # Good to go
                return True
            else:
                # Set final SOC to be under diff
                print_status(SOC_final, SOC_init, diff)
                return False
        # Simulation show battery loss
        else:
            # energy balance should be negative
            if SOC_final - SOC_init > 0:
                print_status(SOC_final, SOC_init, diff)
                return False
            # Loss should be smaller than the one we set (less negative)
            if diff < SOC_init - SOC_final:
                return True
            else:
                # Set final SOC to include at least the lost
                print_status(SOC_final, SOC_init, diff)
                return False

    def initialize_time_index(self, net_load):
        """Replace date index by time ids

        Args:
            net_load (pandas.DataFrame): data frame with date index and a 'net_load' column in [W]

        Retrun:
            net_load as a dictionary with time ids, time ids
        """
        temp_index = pandas.DataFrame(range(0, len(net_load)), columns=['index'])
        # Set temp_index
        temp_net_load = net_load.copy()
        temp_net_load = temp_net_load.set_index(temp_index['index'])
        # Return a dictionary
        return temp_net_load.to_dict()['netload'], temp_index.index.values.tolist()

    def get_initial_SOC(self, vehicle, SOC_offset, SOC_init=None):
        """Get the initial SOC with which people start the optimization
        """
        if SOC_init is not None:
            return SOC_init
        else:
            return vehicle.SOC[self.SOC_index_from] - SOC_offset

    def get_final_SOC(self, vehicle, SOC_margin, SOC_offset, SOC_end=None):
        """Get final SOC that vehicle must reached at the end of the optimization
        """
        if SOC_end is not None:
            return SOC_end
        else:
            return vehicle.SOC[self.SOC_index_to] - SOC_offset - SOC_margin

    def initialize_model(self, project, net_load, SOC_margin, SOC_offset):
        """Select the vehicles that were plugged at controlled chargers and create
        the optimization variables (see inputs of optimization)

        Args:
            project (Project): project

        Return:
            times, vehicles, d, pmax, pmin, emin, emax, efinal
        """
        # Create a dict with the net load and get time index in a data frame
        self.d, self.times = self.initialize_time_index(net_load)
        vehicle_to_optimize = 0
        unfeasible_vehicle = 0

        for vehicle in project.vehicles:
            if vehicle.result is not None:
                # Get SOC init and SOC end
                SOC_init = self.get_initial_SOC(vehicle, SOC_offset)
                SOC_final = self.get_final_SOC(vehicle, SOC_margin, SOC_offset)

                # Find out if vehicle itinerary is feasible
                if not self.check_energy_constraints_feasible(vehicle, SOC_init, SOC_final, SOC_offset):
                    # Reset vehicle result to None
                    vehicle.result = None
                    unfeasible_vehicle += 1
                    continue

                # Add vehicle id to a list
                self.vehicles.append(vehicle.id)
                vehicle_to_optimize += 1

                # Resample vehicle result
                temp_vehicle_result = vehicle.result.resample(str(self.optimization_timestep) + 'T').first()

                # Set time_vehicle_index
                temp_vehicle_result = temp_vehicle_result.set_index(pandas.DataFrame(
                    index=[(time, vehicle.id) for time in self.times]).index)

                # Push pmax and pmin with vehicle and time key
                self.pmin.update(temp_vehicle_result.to_dict()['p_min'])
                self.pmax.update(temp_vehicle_result.to_dict()['p_max'])

                # Push emin and emax with vehicle and time key
                # Units! if project.timestep in seconds, self.timestep in minutes and battery in Wh
                # Units! Wproject.timestep --> Wself.timestep * (project.timestep / (60 * self.timestep))
                # Units! Wtimestep --> Wh * (60 / self.timestep)
                temp_vehicle_result['emin'] = (temp_vehicle_result.energy * (project.timestep / (60 * self.optimization_timestep)) +
                                               (self.minimum_SOC - SOC_init) * vehicle.car_model.battery_capacity *
                                               (60 / self.optimization_timestep))
                self.emin.update(temp_vehicle_result.to_dict()['emin'])
                temp_vehicle_result['emax'] = (temp_vehicle_result.energy * (project.timestep / (60 * self.optimization_timestep)) + 10000 +
                                               (self.maximum_SOC - SOC_init) * vehicle.car_model.battery_capacity *
                                               (60 / self.optimization_timestep))
                self.emax.update(temp_vehicle_result.to_dict()['emax'])

                # Push efinal with vehicle key
                self.efinal.update({vehicle.id: (temp_vehicle_result.tail(1).energy.values[0] * (project.timestep / (60 * self.optimization_timestep)) +
                                                 (SOC_final - SOC_init) * vehicle.car_model.battery_capacity *
                                                 (60 / self.optimization_timestep))})

        print('There is ' + str(vehicle_to_optimize) + ' vehicle participating in the optimization (' +
              str(vehicle_to_optimize * 100 / len(project.vehicles)) + '%)')
        print('There is ' + str(unfeasible_vehicle) + ' unfeasible vehicle.')
        print('')

    def process(self, times, vehicles, d, pmax, pmin, emin, emax,
                efinal, peak_shaving, penalization, solver="gurobi"):
        """The process function creates the pyomo model and solve it.
        Minimize sum( net_load(t) + sum(power_demand(t, v)))**2
        subject to:
        pmin(t, v) <= power_demand(t, v) <= pmax(t, v)
        emin(t, v) <= sum(power_demand(t, v)) <= emax(t, v)
        sum(power_demand(t, v)) >= efinal(v)
        rampmin(t) <= net_load_ramp(t) + power_demand_ramp(t, v) <= rampmax(t)

        Args:
            times (list): timestep list
            vehicles (list): unique list of vehicle ids
            d (dict): time - net load at t
            pmax (dict): (time, id) - power maximum at t for v
            pmin (dict): (time, id) - power minimum at t for v
            emin (dict): (time, id) - energy minimum at t for v
            emax (dict): (time, id) - energy maximum at t for v
            efinal (dict): id - final SOC
            solver (string): name of the solver to use (default is gurobi)

        Return:
            model (ConcreteModel), result
        """

        # Select gurobi solver
        with SolverFactory(solver) as opt:
            # Solver option see Gurobi website
            # opt.options['Method'] = 1

            # Creation of a Concrete Model
            model = ConcreteModel()

            # ###### Set
            model.t = Set(initialize=times, doc='Time', ordered=True)
            last_t = model.t.last()
            model.v = Set(initialize=vehicles, doc='Vehicles')

            # ###### Parameters
            # Net load
            model.d = Param(model.t, initialize=d, doc='Net load')

            # Power
            model.p_max = Param(model.t, model.v, initialize=pmax, doc='P max')
            model.p_min = Param(model.t, model.v, initialize=pmin, doc='P min')

            # Energy
            model.e_min = Param(model.t, model.v, initialize=emin, doc='E min')
            model.e_max = Param(model.t, model.v, initialize=emax, doc='E max')

            model.e_final = Param(model.v, initialize=efinal, doc='final energy balance')
            # model.beta = Param(initialize=beta, doc='beta')

            # ###### Variable
            model.u = Var(model.t, model.v, domain=Integers, doc='Power used')

            # ###### Rules
            def maximum_power_rule(model, t, v):
                return model.u[t, v] <= model.p_max[t, v]
            model.power_max_rule = Constraint(model.t, model.v, rule=maximum_power_rule, doc='P max rule')

            def minimum_power_rule(model, t, v):
                return model.u[t, v] >= model.p_min[t, v]
            model.power_min_rule = Constraint(model.t, model.v, rule=minimum_power_rule, doc='P min rule')

            def minimum_energy_rule(model, t, v):
                return sum(model.u[i, v] for i in range(0, t + 1)) >= model.e_min[t, v]
            model.minimum_energy_rule = Constraint(model.t, model.v, rule=minimum_energy_rule, doc='E min rule')

            def maximum_energy_rule(model, t, v):
                return sum(model.u[i, v] for i in range(0, t + 1)) <= model.e_max[t, v]
            model.maximum_energy_rule = Constraint(model.t, model.v, rule=maximum_energy_rule, doc='E max rule')

            def final_energy_balance(model, v):
                return sum(model.u[i, v] for i in model.t) >= model.e_final[v]
            model.final_energy_rule = Constraint(model.v, rule=final_energy_balance, doc='E final rule')

            # Set the objective to be either peak shaving or ramp mitigation
            if peak_shaving == 'peak_shaving':
                def objective_rule(model):
                    return sum([(model.d[t] + sum([model.u[t, v] for v in model.v]))**2 for t in model.t])
                model.objective = Objective(rule=objective_rule, sense=minimize, doc='Define objective function')

            elif peak_shaving == 'penalized_peak_shaving':
                def objective_rule(model):
                    return (sum([(model.d[t] + sum([model.u[t, v] for v in model.v]))**2 for t in model.t]) +
                            penalization * sum([sum([model.u[t, v]**2 for v in model.v]) for t in model.t]))
                model.objective = Objective(rule=objective_rule, sense=minimize, doc='Define objective function')

            elif peak_shaving == 'ramp_mitigation':
                def objective_rule(model):
                    return sum([(model.d[t + 1] - model.d[t] + sum([model.u[t + 1, v] - model.u[t, v] for v in model.v]))**2 for t in model.t if t != last_t])
                model.objective = Objective(rule=objective_rule, sense=minimize, doc='Define objective function')

            results = opt.solve(model)
            # results.write()

        return model, results

    def plot_result(self, model):
        """Create a plot showing the power constraints, the energy constraints and the ramp
        constraints as well as the final net load.
        """
        # Set the graph style
        sns.set_style("whitegrid")
        sns.despine()

        result = pandas.DataFrame()

        # Get the result
        df = pandas.DataFrame(index=['power'], data=model.u.get_values()).transpose().groupby(level=0).sum()
        # Ramp of the result
        mylist = [0]
        mylist.extend(list(numpy.diff(df['power'].values)))
        df['ramppower'] = mylist
        result = pandas.concat([result, df], axis=1)
        # cum sum of the result
        df = pandas.DataFrame(
            pandas.DataFrame(
                index=['anything'],
                data=model.u.get_values()).transpose().unstack().cumsum().sum(axis=1), columns=['powercum']) * (5 / 60)
        result = pandas.concat([result, df], axis=1)

        # Get pmax and pmin units of power [W]
        df = pandas.DataFrame(index=['pmax'], data=model.p_max.extract_values()).transpose().groupby(level=0).sum()
        result = pandas.concat([result, df], axis=1)
        df = pandas.DataFrame(index=['pmin'], data=model.p_min.extract_values()).transpose().groupby(level=0).sum()
        result = pandas.concat([result, df], axis=1)

        # Get emin and emax units of energy [Wh]
        df = pandas.DataFrame(index=['emax'], data=model.e_max.extract_values()).transpose().groupby(level=0).sum() * (5 / 60)
        result = pandas.concat([result, df], axis=1)
        df = pandas.DataFrame(index=['emin'], data=model.e_min.extract_values()).transpose().groupby(level=0).sum() * (5 / 60)
        result = pandas.concat([result, df], axis=1)

        # Get the minimum final energy quantity
        e_final = pandas.DataFrame(index=['efinal'], data=model.e_final.extract_values()).transpose().sum() * (5 / 60)

        # Get the actual ramp
        mylist = [0]
        df = pandas.DataFrame(index=['net_load'], data=model.d.extract_values()).transpose()
        mylist.extend(list(numpy.diff(df['net_load'].values)))
        df['rampnet_load'] = mylist
        result = pandas.concat([result, df], axis=1)

        # Plot power constraints
        plt.subplot(411)
        plt.plot(result.index.values, result.pmax.values, label='pmax')
        plt.plot(result.index.values, result.power.values, label='power')
        plt.plot(result.index.values, result.pmin.values, label='pmin')
        plt.legend(loc=0)

        # Plot energy constraints
        plt.subplot(412)
        plt.plot(result.index.values, result.emax.values, label='emax')
        plt.plot(result.index.values, result.powercum.values, label='powercum')
        plt.plot(result.index.values, result.emin.values, label='emin')
        plt.plot(result.index[-1], e_final, '*', markersize=15, label='efinal')
        plt.legend(loc=0)

        # Plot the result ramp
        plt.subplot(413)
        plt.plot(result.index.values, result.ramppower.values + result.rampnet_load.values, label='result ramp')
        plt.plot(result.index.values, result.rampnet_load.values, label='net_load ramp')
        plt.legend(loc=0)

        # Plot the power demand results
        plt.subplot(414)
        plt.plot(result.index.values, result.net_load.values, label='net_load')
        plt.plot(result.index.values, result.net_load.values + result.power.values, label='net_load + vehicle')
        plt.legend(loc=0)
        plt.show()

    def post_process(self, project, netload, model, result, plot):
        """Recompute SOC profiles and compute new total power demand

        Args:
            project (Project): project

        Note: Should check that 'vehicle before' and 'after' contain the same number of vehicles
        """
        if plot:
            self.plot_result(model)

        temp = pandas.DataFrame()
        first = True
        for vehicle in project.vehicles:
            if vehicle.result is not None:
                if first:
                    temp['vehicle_before'] = vehicle.result['power_demand']
                    first = False
                else:
                    temp['vehicle_before'] += vehicle.result['power_demand']

        temp2 = pandas.DataFrame(index=['vehicle_after'], data=model.u.get_values()).transpose().groupby(level=0).sum()
        i = pandas.date_range(start=self.date_from, end=self.date_to,
                              freq=str(self.optimization_timestep) + 'T', closed='left')
        temp2 = temp2.set_index(i)
        temp2 = temp2.resample(str(project.timestep) + 'S')
        temp2 = temp2.fillna(method='ffill').fillna(method='bfill')

        final_result = pandas.DataFrame()
        final_result = pandas.concat([temp['vehicle_before'], temp2['vehicle_after']], axis=1)
        final_result = final_result.fillna(method='ffill').fillna(method='bfill')

        return final_result


def save_vehicle_state_for_optimization(vehicle, timestep, date_from,
                                        date_to, activity=None, power_demand=None,
                                        SOC=None, detail=None, nb_interval=None, init=False,
                                        run=False, post=False):
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

    elif init:
        vehicle.SOC = [vehicle.SOC[0]]
        vehicle.result = None
        for activity in vehicle.activities:
            if isinstance(activity, v2gsim.model.Parked):
                if activity.charging_station.post_simulation:
                    # Initiate a dictionary of numpy array to hold result (faster than DataFrame)
                    vehicle.result = {'power_demand': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                                      'p_max': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                                      'p_min': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep)),
                                      'energy': numpy.array([0.0] * int((date_to - date_from).total_seconds() / timestep))}
                    # Leave the init function
                    return
    elif post:
        if vehicle.result is not None:
            # Convert location result back into pandas DataFrame (faster that way)
            i = pandas.date_range(start=date_from, end=date_to,
                                  freq=str(timestep) + 's', closed='left')
            vehicle.result = pandas.DataFrame(index=i, data=vehicle.result)
            vehicle.result['energy'] = vehicle.result['energy'].cumsum()
