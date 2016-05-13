# -*- coding: utf-8 -*-
from pyomo.opt import SolverFactory
from pyomo.environ import *
import pandas


class CentralOptimization(object):
    """
    """
    def __init__(self, project, initial_SOC, final_SOC, rampu_decrease,
                 rampd_decrease):
        self.result = []
        self.simulation_description = pandas.DataFrame(
            columns=['initial_SOC', 'initial_energy', 'final_SOC', 'final_energy',
                     'net_load_rampu', 'rampu_decrease', 'rampu_constraint', 'maximum_rampu',
                     'net_load_rampd', 'rampd_decrease', 'rampd_constraint', 'maximum_rampd',
                     'maximum_power_demand', 'maximum_net_load_power',
                     'minimum_power_demand', 'minimum_net_load_power',
                     'feasible'])

        # Data to be fed into pyomo model
        self.times = None
        self.vehicles = None
        self.d = None
        self.pmax = None
        self.pmin = None
        self.emin = None
        self.emax = None
        self.rampu = None
        self.rampd = None
        self.efinal = None

    def solve_with_maximum_ramp_constraint(self, ):
        """Launch the optimization for different ramping constraints and find
        the most constraining one. The intermediate results are saved if
        feasibles.
        """
        pass

    def solve(self, ):
        """Launch the optimization and the post_processing fucntion. Results
        and assumptions are appended to a data frame.
        """
        pass

    def check_energy_constraints_feasible(self, ):
        """Make sure that SOC final can be reached from SOC init under uncontrolled
        charging (best case scenario)
        """
        pass

    def scale_down_net_load(self, ):
        """Scale the net load to match the number of vehicles
        """
        pass

    def set_initial_SOC(self, ):
        """Set the initial SOC with which people start the optimization
        """
        pass

    def set_final_SOC(self, ):
        """Set final SOC that vehicle must reached at the end of the optimization
        """
        pass

    def initialize_model(self, project, net_load, rampu_decrease, rampd_decrease):
        """Select the vehicles that were plugged at controlled chargers and create
        the optimization variables (see inputs of optimization)

        Args:
            project (Project): project

        Return:
            times, vehicles, d, pmax, pmin, emin, emax, rampu, rampd, efinal
        """
        pass

    def set_ramp_constraints(rampu_decrease, rampd_decrease):
        """
        """
        pass

    def process(self, times, vehicles, d, pmax, pmin, emin, emax, rampu, rampd,
                efinal, solver="gurobi"):
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
            rampu (dict): time - maximum ramping up at t
            rampd (dict): time - maximum ramping down at t
            efinal (dict): id - final SOC
            solver (string): name of the solver to use (default is gurobi)

        Return:
            model (ConcreteModel), result
        """
        # Select gurobi solver
        opt = SolverFactory(solver)

        # Creation of a Concrete Model
        model = ConcreteModel()

        # ###### Set
        model.t = Set(initialize=range(0, len(net_load)), doc='Time')
        model.v = Set(initialize=['car1', 'car2'], doc='Vehicles')

        # ###### Parameters
        # Net load
        model.d = Param(model.t, initialize=dict_D, doc='Net load')

        # Power
        model.p_max = Param(model.t, model.v, initialize=dict_pmax, doc='P max')
        model.p_min = Param(model.t, model.v, initialize=dict_pmin, doc='P min')

        # Energy
        model.e_min = Param(model.t, model.v, initialize=dict_emin, doc='E min')
        model.e_max = Param(model.t, model.v, initialize=dict_emax, doc='E max')

        model.e_final = Param(initialize=0, doc='final energy balance')

        # Ramp
        model.ramp_up = Param(model.t, initialize=dict_rampu, doc='ramp up')
        model.ramp_down = Param(model.t, initialize=dict_rampd, doc='ramp up')

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

        def final_energy_balance(model, t, v):
            if t == 9:
                return sum(model.u[i, v] for i in model.t) >= model.e_final
            else:
                return Constraint.Skip
        model.final_energy_rule = Constraint(model.t, model.v, rule=final_energy_balance, doc='E final rule')

        def ramp_up_rule(model, t):
            if t == 0:
                return Constraint.Skip
            else:
                return (model.d[t] - model.d[t - 1] + sum([model.u[t, v] - model.u[t - 1, v] for v in model.v])) <= model.ramp_up[t]
        model.ramp_up_rule = Constraint(model.t, rule=ramp_up_rule, doc='limit ramping up')

        def ramp_down_rule(model, t):
            if t == 0:
                return Constraint.Skip
            else:
                return (model.d[t] - model.d[t - 1] + sum([model.u[t, v] - model.u[t - 1, v] for v in model.v])) >= model.ramp_down[t]
        model.ramp_down_rule = Constraint(model.t, rule=ramp_down_rule, doc='limit ramping down')

        def objective_rule(model):
            return sum([(model.d[t] + sum([model.u[t, v] for v in model.v]))**2 for t in model.t])
        model.objective = Objective(rule=objective_rule, sense=minimize, doc='Define objective function')

        results = opt.solve(model)
        results.write()

        return model, result

    def post_process(self, project, model):
        """Recompute SOC profiles and compute new total power demand

        Args:
            project (Project): project
        """
        pandas.DataFrame(columns=['SOC', 'power_demand', 'net_load'])
        pass

# df_pmax = pandas.DataFrame(index=index, data=[1000 for i in range(0, nb_car * len(net_load))], columns=['pmax'])
# dict_pmax = df_pmax.to_dict()['pmax']
