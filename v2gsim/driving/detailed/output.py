class Result(object):
    """
    """

    def __init__(self):
        self.drv = Drv()
        self.vpc = Vpc()
        self.ess = Ess()
        self.mot = Mot()
        self.tc = Tc()
        self.fd = Fd()
        self.whl = Whl()
        self.chas = Chas()
        self.pc = Pc()
        self.accelec = Accelec()


class Drv(object):
    """
    """

    def __init__(self):
        self.key_on = []
        self.cmd_brake = []
        self.cmd_accel = []
        self.V_dmd = []
        self.T_dmd = []


class Vpc(object):
    """
    """

    def __init__(self):
        self.T_mot_dmd = []
        self.T_mechbrake_dmd = []
        self.heat_integral_out = []


class Ess(object):
    """
    """

    def __init__(self):
        self.SOC = []
        self.V_pack = []
        self.i_out = []
        self.V_pack_OC = []
        self.T_cell = []
        self.Q_loss_case = []


class Mot(object):
    """
    """

    def __init__(self):
        self.i_out = []
        self.I_out = []
        self.spd = []


class Tc(object):
    """
    """

    def __init__(self):
        self.T_out = []
        self.I_out = []


class Fd(object):
    """
    """

    def __init__(self):
        self.spd_in = []
        self.T_out = []
        self.I_out = []


class Whl(object):
    """
    """

    def __init__(self):
        self.spd_rdl_in = []
        self.F_out = []
        self.M_inertia_equiv = []


class Chas(object):
    """
    """

    def __init__(self):
        self.V_chas = []
        self.d = []


class Pc(object):
    """
    """

    def __init__(self):
        self.V_out = []
        self.i_in = []


class Accelec(object):
    """
    """

    def __init__(self):
        self.i_in = []
