import v2gsim.driving.detailed.power_train


class DetailedCarModel(object):
    """
    """

    def __init__(self, name, maker=None,
                 driving=v2gsim.driving.detailed.power_train.consumption,
                 year=None, maximum_SOC=0.95, decay=0.95,
                 battery_capacity=23832, battery_efficiency_charging=1.0,
                 battery_efficiency_discharging=1.0, maximum_power=6600,
                 minimum_power=-6600):
        self.veh = Veh()
        self.env = Env()
        self.ess = Ess()
        self.mot = Mot()
        self.tc = Tc()
        self.fd = Fd()
        self.whl = Whl()
        self.chas = Chas()
        self.pc = Pc()
        self.accelec = Accelec()
        self.drv = Drv()
        self.vpa = Vpa()
        self.vpc = Vpc()

        # Add compatibility with simple powertrain model
        self.name = name
        self.maker = maker
        self.year = year
        self.battery_capacity = battery_capacity  # [Wh]
        self.battery_efficiency_charging = battery_efficiency_charging
        self.battery_efficiency_discharging = battery_efficiency_discharging
        self.maximum_SOC = maximum_SOC
        self.decay = decay
        self.minimum_power = minimum_power  # [W]
        self.maximum_power = maximum_power  # [W]
        self.driving = driving


class Veh(object):
    """
    """

    def __init__(self):
        self.mass = None  # kg


class Env(object):
    """
    """

    def __init__(self):
        self.air_cap = None
        self.dens_air = None
        self.gravity = None
        self.temp_amb = None


class Ess(object):
    """
    """

    def __init__(self):
        self.pack_energy_wh = None
        self.soc_min = None
        self.soc_max = None
        self.volt_nom = None
        self.volt_max = None
        self.volt_min = None
        self.element_per_module = None
        self.element_per_module_parallel = None
        self.num_per_module = None
        self.design_num_module_parallel = None
        self.cell_energy_density = None
        self.cell_to_module_weight_ration = None
        self.module_to_pack_weight_ratio = None
        self.temp_reg = None
        self.therm_cp_module = None
        self.dia = None
        self.length = None
        self.flow_air_mod = None
        self.case_thk = None
        self.mod_case_th_cond = None
        self.eff_coulomb = {}
        self.rint_dist = {}
        self.rint_chg = {}
        self.voc = {}
        self.num_cell = None
        self.num_cell_series = None
        self.cap_max = {}
        self.mass_module = None
        self.pwr_chg = {}
        self.pwr_dis = {}
        self.therm_flow_area_module = None
        self.speed_air = None
        self.therm_air_htcoef = None
        self.area_module = None
        self.therm_res_off = None
        self.therm_res_on = None


class Mot(object):
    """
    """

    def __init__(self):
        self.inertia = None
        self.t_max_trq = None
        self.trq_max = {}
        self.eff_trq = {}
        self.trq_cont = {}
        self.trq_neg_max = {}
        self.pwr_neg_max = {}
        self.trq_neg_cont = {}
        self.trq_pos_max = {}
        self.pwr_pos_max = {}
        self.trq_pos_cont = {}
        self.pwr_des = None
        self.eff_des = None
        self.eff_trq_max = None
        self.pwr_mech = {}
        self.pwr_elec_loss = {}
        self.pwr_elec = {}
        self.trq_pwr_elec = {}


class Tc(object):
    """
    """

    def __init__(self):
        self.ratio = None
        self.inertia = None
        self.eff_spec = None
        self.spd_thresh = None
        self.trq_loss = {}


class Fd(object):
    """
    """

    def __init__(self):
        self.ratio = None
        self.inertia = None
        self.eff_spec = None
        self.spd_thresh = None
        self.trq_loss = {}


class Whl(object):
    """
    """

    def __init__(self):
        self.inertia_per_wheel = None
        self.coeff_roll1 = None
        self.coeff_roll2 = None
        self.coeff_roll3 = None
        self.coeff_roll4 = None
        self.friction_coefficient = {}
        self.trq_brake_mech = {}
        self.number_wheels = None
        self.trq_brake_max = None
        self.theoretical_radius = None
        self.radius_correction = None
        self.radius = None
        self.spd_thresh = None
        self.rim_diameter = None
        self.tire_width = None
        self.profile = None
        self.inertia = None
        self.weight_fraction_effective = None
        self.brake_fraction = None


class Chas(object):
    """
    """

    def __init__(self):
        self.coeff_drag = None
        self.frontal_area = None
        self.ratio_weight_front = None


class Pc(object):
    """
    """

    def __init__(self):
        self.volt_out = None
        self.eff = None


class Accelec(object):
    """
    """

    def __init__(self):
        self.pwr = None


class Drv(object):
    """
    """

    def __init__(self):
        self.chas_spd_above_chas_started = None
        self.chas_spd_below_chas_stopped = None


class Vpa(object):
    """
    """

    def __init__(self):
        self.ratio_cum = None


class Vpc(object):
    """
    """

    def __init__(self):
        self.whl_trq_max = {}
        self.chas_spd_below_no_regen = None
        self.chas_spd_above_full_regen = None
        self.eff_accelec_to_mot = None
        self.eff_ess_to_mot = None
        self.ess_soc_above_regen_forbidden = None
        self.ess_soc_below_regen_allowed = None
        self.eff_to_whl = None
        self.ratio_cum = None
        self.trq_pwr_elec = {}
        self.brk_trq = {}
        self.ratio_ecu_brk_total_brk = {}
        self.whls_trq_brk_total = {}
