from __future__ import division
import output
import numpy


def consumption(driving, vehicle, nb_interval, project_timestep, timestep=0.1, verbose=False):
    """Calculate the consumption of a vehicle for a driving activity.

    Args:
        vehicle (Vehicle): a Vehicle object to update with the driving activity consumption
        driving (Driving): a driving activity
        timestep (float): the timestep used for the calculus (different from the output interval)

    Returns:
        SOC as a list
        powerDemand as a list
        stranded a boolean
    """

    # Create initial conditions
    V_chas = 0
    chas_d = 0
    vpc_heat_integral = -0.3
    ess_SOC = vehicle.SOC[-1]
    ess_T_cell = vehicle.car_model.env.temp_amb
    ess_Q_loss_case = 0
    mot_i_out = 0  # no need
    pc_i_in = 0
    ess_i_out_dmd = mot_i_out + pc_i_in
    mot_spd_in = 0
    fd_spd_in = 0
    whl_spd_rdl_in = 0
    accelec_i_in = 0

    # Initialize output signals
    outputSignal = output.Result()

    # Interpolate speed and terrain (m/s and grade)
    speed = numpy.interp(numpy.arange(0, len(driving.speed), timestep),
                         numpy.arange(0, len(driving.speed), 1), driving.speed)
    terrain = numpy.interp(numpy.arange(0, len(driving.speed), timestep),
                           numpy.arange(0, len(driving.terrain), 1), driving.terrain)

    # Begin running model
    for index in range(0, len(speed)):

        # Driver model
        drv_key_on, drv_cmd_brake, drv_cmd_accel, drv_V_dmd, drv_T_dmd = driver_main_nolookahead(
            V_chas, speed[index], terrain[index], timestep, vehicle.car_model)

        # Powertrain controller model
        vpc_T_mot_dmd, vpc_T_mechbrake_dmd, vpc_heat_integral = powertrain_control_main(
            drv_cmd_brake, drv_cmd_accel, drv_T_dmd, V_chas, mot_spd_in,
            ess_SOC, vpc_heat_integral, timestep, vehicle.car_model)

        # Battery
        ess_SOC, ess_V_pack_out, ess_V_pack_OC, ess_T_cell, ess_Q_loss_case = battery_plant_main(
            ess_i_out_dmd, ess_SOC, float(ess_T_cell), ess_Q_loss_case, timestep, vehicle.car_model)

        # Motor
        mot_i_out, mot_I_out = motor_plant_main(
            ess_V_pack_out, mot_spd_in, vpc_T_mot_dmd, vehicle.car_model)

        # Torque coupling
        mot_spd_in, tc_T_out, tc_I_out = torque_coupling_main(
            fd_spd_in, vpc_T_mot_dmd, mot_I_out, vehicle.car_model)

        # Final drive
        fd_spd_in, fd_T_out, fd_I_out = final_drive_main(
            whl_spd_rdl_in, tc_T_out, tc_I_out, vehicle.car_model)

        # Wheels
        whl_spd_rdl_in, whl_F_out, whl_M_inertia_equiv = wheel_plant_main(
            V_chas, fd_T_out, fd_I_out, drv_cmd_brake, vpc_T_mechbrake_dmd, terrain[index],
            vehicle.car_model)

        # Chassis
        V_chas, chas_d = chassis_main(
            whl_F_out, whl_M_inertia_equiv, terrain[index], V_chas, chas_d, timestep, vehicle.car_model)

        # Power converter
        pc_V_out, pc_i_in = power_converter_main(
            ess_V_pack_out, accelec_i_in, vehicle.car_model)

        # Electrical accessories
        accelec_i_in = elec_acc_main(
            pc_V_out, drv_key_on, vehicle.car_model)

        ess_i_out_dmd = mot_i_out + pc_i_in

        # Save results timestep=0.1 sec
        if ((index * timestep) % project_timestep) == 0:
            outputSignal.drv.key_on.append(drv_key_on)
            outputSignal.drv.cmd_brake.append(drv_cmd_brake)
            outputSignal.drv.cmd_accel.append(drv_cmd_accel)
            outputSignal.drv.V_dmd.append(drv_V_dmd)
            outputSignal.drv.T_dmd.append(drv_T_dmd)

            outputSignal.vpc.T_mot_dmd.append(vpc_T_mot_dmd)
            outputSignal.vpc.T_mechbrake_dmd.append(vpc_T_mechbrake_dmd)
            outputSignal.vpc.heat_integral_out.append(vpc_heat_integral)

            outputSignal.ess.SOC.append(ess_SOC)
            outputSignal.ess.V_pack.append(ess_V_pack_out)
            outputSignal.ess.i_out.append(ess_i_out_dmd)
            outputSignal.ess.V_pack_OC.append(ess_V_pack_OC)
            outputSignal.ess.T_cell.append(ess_T_cell)
            outputSignal.ess.Q_loss_case.append(ess_Q_loss_case)

            outputSignal.mot.i_out.append(mot_i_out)
            outputSignal.mot.I_out.append(mot_I_out)
            outputSignal.mot.spd.append(mot_spd_in)

            outputSignal.tc.T_out.append(tc_T_out)
            outputSignal.tc.I_out.append(tc_I_out)

            outputSignal.fd.spd_in.append(fd_spd_in)
            outputSignal.fd.T_out.append(fd_T_out)
            outputSignal.fd.I_out.append(fd_I_out)

            outputSignal.whl.spd_rdl_in.append(whl_spd_rdl_in)
            outputSignal.whl.F_out.append(whl_F_out)
            outputSignal.whl.M_inertia_equiv.append(whl_M_inertia_equiv)

            outputSignal.chas.V_chas.append(V_chas)
            outputSignal.chas.d.append(chas_d)

            outputSignal.pc.V_out.append(pc_V_out)
            outputSignal.pc.i_in.append(pc_i_in)

            outputSignal.accelec.i_in.append(accelec_i_in)

    # Export powerDemand to the value in each driving activity
    powerDemand = numpy.multiply(outputSignal.ess.V_pack, outputSignal.ess.i_out)

    # Check if not below stranding threshold
    stranded = False
    if outputSignal.ess.SOC[-1] < 0.1:
        stranded = True

    return outputSignal.ess.SOC, powerDemand, stranded, outputSignal


def driver_main_nolookahead(V_chas, speed, terrain, timestep, carModel):
    # Calculate
    r_wheel = carModel.whl.radius
    V_dmd = speed
    A_dmd = (V_dmd - V_chas) / timestep
    V_threshold_stopped = (carModel.drv.chas_spd_above_chas_started + carModel.drv.chas_spd_below_chas_stopped) / 2

    # Rolling and aero resitance
    B1 = drv_calc_rolling_aero_res(terrain, V_dmd, carModel)

    # Grade
    B2 = (carModel.veh.mass * carModel.env.gravity * numpy.sin(terrain)) * r_wheel

    # Dynamic torque demand
    B3 = carModel.veh.mass * A_dmd * carModel.whl.radius

    # Dynamic horizon torque demand
    B4 = 0

    # Torque demand
    T_dmd = B1 + B2 + B3 + B4

    if T_dmd < 0 and V_chas <= V_threshold_stopped:
        cmd_brake = -1
    else:
        cmd_brake = numpy.interp(-T_dmd, (carModel.vpc.whls_trq_brk_total['map_neg']).flatten(),
                                 (carModel.vpc.whls_trq_brk_total['idx1_brake_cmd']).flatten())
    cmd_brake = numpy.minimum(0, numpy.maximum(-1, cmd_brake))  # Saturation -1<= cmd_brake <=0

    # Determine accelerator command
    T_max_wheel = numpy.interp(V_chas, (carModel.vpc.whl_trq_max['idx1_chas_lin_spd']).flatten(),
                               (carModel.vpc.whl_trq_max['map']).flatten())
    cmd_accel = T_dmd / T_max_wheel
    cmd_accel = numpy.minimum(1, numpy.maximum(0, cmd_accel))

    return 1, cmd_brake, cmd_accel, V_dmd, T_dmd


def drv_calc_rolling_aero_res(terrain, V_dmd, carModel):
    r_wheel = carModel.whl.radius

    # Rolling resistance
    C1 = carModel.whl.coeff_roll1
    C2 = carModel.whl.coeff_roll2
    C3 = carModel.whl.coeff_roll3
    C4 = carModel.whl.coeff_roll4
    m_veh = carModel.veh.mass
    g = carModel.env.gravity

    # Aero resistance
    dens_air = carModel.env.dens_air
    Cd = carModel.chas.coeff_drag
    Af = carModel.chas.frontal_area

    # Calculating rolling losses
    V_temp = abs(V_dmd)
    Crr = numpy.cos(terrain) * (
        numpy.minimum(C1, 20 * C1 * V_temp) + (C2 * V_temp) + (C3 * (V_temp ** 2)) + (C4 * (V_temp ** 3))) * m_veh * g

    # Calculate aero losses
    A = (0.5 * dens_air * Cd * Af * (V_temp ** 2)) * numpy.sign(V_dmd)

    # Total aero and rolling resistance losses
    return ((Crr + A) * r_wheel) * numpy.sign(V_dmd)


def powertrain_control_main(cmd_brake, cmd_accel, T_dmd, V_chas, spd_mot, SOC, heat_integral_in, timestep, carModel):
    if T_dmd >= 0:
        # Propulsion model
        # Determine constraints
        P_ess_max_prop = numpy.interp(SOC, (carModel.ess.pwr_dis['idx1_soc']).flatten(),
                                      (carModel.ess.pwr_dis['map']).flatten()) * carModel.ess.num_cell
        T_max_mot_prop_peak = numpy.interp(spd_mot, (carModel.mot.trq_pos_max['idx1_speed']).flatten(),
                                           (carModel.mot.trq_pos_max['map']).flatten())
        T_max_mot_prop_cont = numpy.interp(spd_mot, (carModel.mot.trq_pos_cont['idx1_speed']).flatten(),
                                           (carModel.mot.trq_pos_cont['map']).flatten())

        # Motor heat index calculation
        heat_integral_out = (((numpy.absolute(
            T_dmd / T_max_mot_prop_cont) - 1) * 0.3 / carModel.mot.t_max_trq) * timestep) + heat_integral_in
        heat_integral_out = numpy.minimum(1, numpy.maximum(-0.3, heat_integral_out))
        heat_index = numpy.minimum(1, numpy.maximum(0, heat_integral_out))

        T_max_mot_prop = (T_max_mot_prop_cont * heat_index) + (T_max_mot_prop_peak * (1 - heat_index))

        # Calculate torque demands
        T_mot_dmd = vpc_propulsion(cmd_accel, cmd_brake, V_chas, spd_mot, P_ess_max_prop, T_max_mot_prop, carModel)
        T_mechbrake_dmd = 0

    else:
        # Braking model
        # Determine the constraints
        P_ess_max_regen = numpy.interp(SOC, (carModel.ess.pwr_chg['idx1_soc']).flatten(),
                                       (carModel.ess.pwr_chg['map']).flatten()) * carModel.ess.num_cell
        T_max_mot_regen_peak = numpy.interp(spd_mot, (carModel.mot.trq_neg_max['idx1_speed']).flatten(),
                                            (carModel.mot.trq_neg_max['map']).flatten())
        T_max_mot_regen_cont = numpy.interp(spd_mot, (carModel.mot.trq_neg_max['idx1_speed']).flatten(),
                                            (carModel.mot.trq_neg_max['map']).flatten())

        # Motor heat index calculation
        heat_integral_out = (((numpy.absolute(
            T_dmd / T_max_mot_regen_cont) - 1) * 0.3 / carModel.mot.t_max_trq) * timestep) + heat_integral_in
        heat_integral_out = numpy.minimum(1, numpy.maximum(-0.3, heat_integral_out))
        heat_index = numpy.minimum(1, numpy.maximum(0, heat_integral_out))

        T_max_mot_regen = (T_max_mot_regen_cont * heat_index) + (T_max_mot_regen_peak * (1 - heat_index))

        # Calculate torque demands
        T_mot_dmd, T_mechbrake_dmd = vpc_braking(SOC, V_chas, P_ess_max_regen, spd_mot, T_max_mot_regen, cmd_brake,
                                                 carModel)

    return T_mot_dmd, T_mechbrake_dmd, heat_integral_out


def vpc_propulsion(cmd_accel, cmd_brake, V_chas, spd_mot, P_ess_max_prop, T_max_mot_prop, carModel):
    # Calculate motor torque demand
    if cmd_accel > 0:
        T_whl_dmd = numpy.interp(V_chas, (carModel.vpc.whl_trq_max['idx1_chas_lin_spd']).flatten(),
                                 (carModel.vpc.whl_trq_max['map']).flatten()) * cmd_accel
    else:
        T_whl_dmd = numpy.interp(-cmd_brake, (carModel.whl.trq_brake_mech['idx1_brk_cmd']).flatten(),
                                 (carModel.whl.trq_brake_mech['map']).flatten())

    T_lim1 = T_whl_dmd / carModel.vpa.ratio_cum
    T_lim1 = numpy.maximum(0, T_lim1)  # value >= 0

    T_lim2 = interp_2d(spd_mot, (P_ess_max_prop - carModel.accelec.pwr),
                       (carModel.mot.trq_pwr_elec['idx1_speed']).flatten(),
                       (carModel.mot.trq_pwr_elec['idx2_pwr']).flatten(), carModel.mot.trq_pwr_elec['map'])

    T_lim3 = T_max_mot_prop  # Not used ?

    fric_tire = numpy.interp(V_chas, (carModel.whl.friction_coefficient['idx1_chas_lin_spd']).flatten(),
                             (carModel.whl.friction_coefficient['map']).flatten())
    T_lim4 = (
        carModel.veh.mass * carModel.env.gravity * carModel.chas.ratio_weight_front * fric_tire * carModel.whl.radius) / carModel.vpa.ratio_cum

    return numpy.amin([T_lim1, T_lim2, T_lim4])


def vpc_braking(SOC, V_chas, P_ess_max_regen, spd_mot, T_max_mot_regen, cmd_brake, carModel):
    # Total braking torque demand at wheels
    T_brake_dmd_temp = numpy.interp(cmd_brake, (carModel.vpc.brk_trq['idx1_brk_cmd']).flatten(),
                                    (carModel.vpc.brk_trq['map']).flatten())
    T_brake_dmd_temp = numpy.minimum(0, T_brake_dmd_temp)

    if T_brake_dmd_temp >= 0:
        T_brake_total = 0
    else:
        temp_val = (numpy.absolute(T_brake_dmd_temp) / carModel.whl.radius) / carModel.veh.mass
        temp_val2 = numpy.interp(temp_val, carModel.vpc.ratio_ecu_brk_total_brk['idx1_lin_accel'],
                                 carModel.vpc.ratio_ecu_brk_total_brk['map'])
        T_brake_total = T_brake_dmd_temp * temp_val2

    T_brake_total = numpy.minimum(0, T_brake_total)

    # Determine if regenerative braking is available
    SOC_regen_thresh = ((carModel.vpc.ess_soc_above_regen_forbidden + carModel.vpc.ess_soc_below_regen_allowed) / 2)
    V_regen_thresh = ((carModel.vpc.chas_spd_above_full_regen + carModel.vpc.chas_spd_below_no_regen) / 2)
    T_mot_regen_avail = 0
    if SOC <= SOC_regen_thresh and V_chas >= V_regen_thresh:
        T_mot_regen_avail = T_brake_total

    # Power available from battery for regen braking
    P_elec_regen_avail = (P_ess_max_regen / carModel.vpc.eff_ess_to_mot) - (
        carModel.accelec.pwr / carModel.vpc.eff_accelec_to_mot)

    # Maximum braking torque from motor
    temp_val3 = interp_2d(spd_mot, P_elec_regen_avail, (carModel.vpc.trq_pwr_elec['idx1_speed']).flatten(),
                          (carModel.vpc.trq_pwr_elec['idx2_pwr']).flatten(), carModel.vpc.trq_pwr_elec['map'])

    temp_val3 = numpy.minimum(0, temp_val3)
    temp_val4 = numpy.minimum(0, T_max_mot_regen)
    T_max_mot_brake = numpy.maximum(temp_val3, temp_val4)

    # Motor braking torque to wheels
    temp_val5 = (T_max_mot_brake * carModel.vpc.ratio_cum) / carModel.vpc.eff_to_whl
    T_mot_trq_dmd_wheels = numpy.maximum(temp_val5, T_mot_regen_avail)

    T_mot_dmd = (T_mot_trq_dmd_wheels / carModel.vpc.ratio_cum) / carModel.vpc.eff_to_whl
    T_mechbrake_dmd = T_brake_total - T_mot_trq_dmd_wheels
    T_mechbrake_dmd = numpy.minimum(0, T_mechbrake_dmd)

    return T_mot_dmd, T_mechbrake_dmd


def battery_plant_main(I_out_dmd, SOC_start, T_cell_start, Q_loss_case_prev, timestep, carModel):
    T_amb = carModel.env.temp_amb  # Ambient temperature
    ess_plant_cell_curr = I_out_dmd / carModel.ess.design_num_module_parallel  # Convert pack-level parameters to cell-level

    # SOC calculation
    SOC_end = SOC_calc(SOC_start, ess_plant_cell_curr, T_cell_start, timestep, carModel)

    # Cell voltage calculation
    V_cell_out_temp, V_cell_OC, R_cell_int = V_cell_out_calc(SOC_end, T_cell_start, ess_plant_cell_curr, carModel)
    V_cell_out = numpy.maximum(V_cell_out_temp,
                               carModel.ess.volt_min)  # Ensure output voltage does not fall below min allowable

    # Thermal model
    T_cell_end, Q_loss_case = module_therm(ess_plant_cell_curr, R_cell_int, V_cell_out, T_cell_start, T_amb,
                                           Q_loss_case_prev, timestep, carModel)

    # Convert cell-level to pack-level
    V_pack_out = V_cell_out * carModel.ess.num_cell_series
    V_pack_OC = V_cell_OC * carModel.ess.num_cell_series

    return SOC_end, V_pack_out, V_pack_OC, T_cell_end, Q_loss_case


def SOC_calc(SOC_start, I_cell_out, T, timestep, carModel):
    # Determine energy stored at start of timestep
    Ah_max = numpy.interp(T, (carModel.ess.cap_max['idx1_temp']).flatten(), (carModel.ess.cap_max['map']).flatten())
    Ah_start = SOC_start * Ah_max

    # Determine energy stored at end of timestep
    Ah_cell_out_total = I_cell_out * timestep / 3600  # Ah output per timestep
    Ah_end = Ah_start - Ah_cell_out_total  # Ah at the end of timestep

    # Calculate SOC at end of timestep
    SOC_end = Ah_end / Ah_max
    if SOC_end > 1:
        SOC_end = 1
    elif SOC_end < 0:
        SOC_end = 0

    return SOC_end


def V_cell_out_calc(SOC, T, I_cell_out, carModel):
    # Calculate open-circuit voltage
    V_OC = interp_2d(T, SOC, (carModel.ess.voc['idx1_temp']).flatten(), (carModel.ess.voc['idx2_soc']).flatten(),
                     carModel.ess.voc['map'])

    # Calculate internal resistance
    if I_cell_out >= 0:  # Discharging
        R_int = interp_2d(T, SOC, (carModel.ess.rint_dist['idx1_temp']).flatten(),
                          (carModel.ess.rint_dist['idx2_soc']).flatten(), carModel.ess.rint_dist['map'])
    elif I_cell_out < 0:
        R_int = interp_2d(T, SOC, (carModel.ess.rint_chg['idx1_temp']).flatten(),
                          (carModel.ess.rint_chg['idx2_soc']).flatten(), carModel.ess.rint_chg['map'])

    # Adjusting current using colombic effiency in charging case
    if I_cell_out >= 0:
        I_int_adj = I_cell_out
    elif I_cell_out < 0:
        eff_coul = numpy.interp(T, (carModel.ess.eff_coulomb['idx1_temp']).flatten(),
                                (carModel.ess.eff_coulomb['map']).flatten())
        I_int_adj = I_cell_out * eff_coul

    # Calculate output voltage
    V_cell_output = V_OC - (R_int * I_int_adj)

    return V_cell_output, V_OC, R_int


def module_therm(I_cell_out, R_cell_int, V_cell_out, T_cell_start, T_amb, Q_loss_case_prev, timestep, carModel):
    # Calculate heat generation from charging/discharging
    if I_cell_out >= 0:
        Q_gen_cell = (I_cell_out ** 2) * R_cell_int
    elif I_cell_out < 0:
        eff_coul = numpy.interp(T_cell_start, (carModel.ess.eff_coulomb['idx1_temp']).flatten(),
                                (carModel.ess.eff_coulomb['map']).flatten())
        Q_gen_cell = ((I_cell_out ** 2) * R_cell_int) - (I_cell_out * V_cell_out * (1 - eff_coul))
    Q_gen = carModel.ess.element_per_module * Q_gen_cell

    # Determine case/cooling heat loss
    if T_cell_start > carModel.ess.temp_reg:  # cooling fan is on
        T_air_ave = T_amb - ((0.5 * Q_loss_case_prev) / (carModel.ess.flow_air_mod * carModel.env.air_cap))
        Q_loss_case = (T_air_ave - T_cell_start) / carModel.ess.therm_res_on  # W
    elif T_cell_start <= carModel.ess.temp_reg:
        Q_loss_case = (T_amb - T_cell_start) / carModel.ess.therm_res_off  # W

    # Calculate overall cell temperature
    T_cell_end = T_cell_start + (
        ((Q_gen + Q_loss_case) * timestep) / (carModel.ess.mass_module * carModel.ess.therm_cp_module))  # deg Celsius

    return T_cell_end, Q_loss_case


def motor_plant_main(V_in, speed_in, T_mot_dmd, carModel):
    # Motor inertia
    I_out = carModel.mot.inertia

    # Motor current output
    P_elec = interp_2d(speed_in, T_mot_dmd, (carModel.mot.pwr_elec['idx1_speed']).flatten(),
                       (carModel.mot.pwr_elec['idx2_trq']).flatten(), carModel.mot.pwr_elec['map'])
    i_out = P_elec / V_in

    return i_out, I_out


def torque_coupling_main(spd_out, T_in, I_in, carModel):
    # Speed calculation
    spd_in = spd_out * carModel.tc.ratio

    # Torque calculation
    T_loss_temp = interp_2d(numpy.absolute(T_in), numpy.absolute(spd_in), (carModel.tc.trq_loss['idx1_trq']).flatten(),
                            (carModel.tc.trq_loss['idx2_speed']).flatten(), carModel.tc.trq_loss['map'])

    # Blend function
    u = spd_in
    xl = -carModel.tc.spd_thresh
    yl = -1
    xc = 0
    yc = 0
    xr = carModel.tc.spd_thresh
    yr = 1
    blend = (u < xl) * yl + (xl <= u) * (u < xc) * ((yc - yl) * (u - xl) / (xc - xl) + yl) + (xc <= u) * (u < xr) * (
        (yc - yr) * (u - xr) / (xc - xr) + yr) + (xr <= u) * yr

    T_loss = T_loss_temp * blend

    # Final torque calculation
    T_out = carModel.tc.ratio * (T_in - T_loss)

    # Inertia calculation
    I_out = (I_in * carModel.tc.ratio ** 2) + carModel.tc.inertia

    return spd_in, T_out, I_out


def final_drive_main(spd_out, T_in, I_in, carModel):
    # Speed calculation
    spd_in = spd_out * carModel.fd.ratio

    # Torque calculation
    T_loss_temp = interp_2d(numpy.absolute(T_in), numpy.absolute(spd_in), (carModel.fd.trq_loss['idx1_trq']).flatten(),
                            (carModel.fd.trq_loss['idx2_speed']).flatten(), carModel.fd.trq_loss['map'])

    # Blend function
    u = spd_in
    xl = -carModel.fd.spd_thresh
    yl = -1
    xc = 0
    yc = 0
    xr = carModel.fd.spd_thresh
    yr = 1
    blend = (u < xl) * yl + (xl <= u) * (u < xc) * ((yc - yl) * (u - xl) / (xc - xl) + yl) + (xc <= u) * (u < xr) * (
        (yc - yr) * (u - xr) / (xc - xr) + yr) + (xr <= u) * yr

    T_loss = T_loss_temp * blend

    # Final torque calculation
    T_out = carModel.fd.ratio * (T_in - T_loss)

    # Inertia calculation
    I_out = (I_in * carModel.fd.ratio ** 2) + carModel.fd.inertia

    return spd_in, T_out, I_out


def wheel_plant_main(V_chas_in, T_in, I_in, drv_cmd_brake, vpc_T_mech_brake_dmd, terrain, carModel):
    # Wheel brake control model
    T_brk_temp1 = numpy.interp(drv_cmd_brake, (carModel.vpc.brk_trq['idx1_brk_cmd']).flatten(),
                               (carModel.vpc.brk_trq['map']).flatten())
    T_brk_temp1 = numpy.minimum(0, numpy.maximum(-carModel.whl.trq_brake_max,
                                                 T_brk_temp1))  # Saturation -trq_brake_max<=T_brk_temp1<=0

    temp_val1 = (numpy.absolute(T_brk_temp1) / carModel.whl.radius) / carModel.veh.mass
    temp_val2 = numpy.interp(temp_val1, carModel.vpc.ratio_ecu_brk_total_brk['idx1_lin_accel'],
                             carModel.vpc.ratio_ecu_brk_total_brk['map'])

    if T_brk_temp1 < 0:
        temp_val3 = T_brk_temp1 * temp_val2
    elif T_brk_temp1 >= 0:
        temp_val3 = 0

    temp_val4 = T_brk_temp1 - temp_val3
    temp_val5 = vpc_T_mech_brake_dmd + temp_val4

    cmd_whl_ctrl_brk = numpy.absolute((temp_val5 * carModel.whl.brake_fraction) / carModel.whl.trq_brake_max)

    # Angular speed calculation
    spd_rdl_in = V_chas_in / carModel.whl.radius

    # Braking torque calculation
    T_brake_temp = -cmd_whl_ctrl_brk * carModel.whl.trq_brake_max

    xl = -0.1
    yl = -1
    xc = 0
    yc = 0
    xr = 0.1
    yr = 1
    u = spd_rdl_in
    blend = (u < xl) * yl + (xl <= u) * (u < xc) * ((yc - yl) * (u - xl) / (xc - xl) + yl) + (xc <= u) * (u < xr) * (
        (yc - yr) * (u - xr) / (xc - xr) + yr) + (xr <= u) * yr

    T_brake = numpy.minimum(T_brake_temp * blend, 0)

    # Torque from tire rolling resistance
    T_tireloss = calc_rolling_resistance(terrain, V_chas_in, carModel)

    # Net torque and force output
    T_out = T_in + T_brake - T_tireloss
    F_out = T_out / carModel.whl.radius

    # Inertia equivalent mass calculation
    M_inertia_equiv = (I_in + carModel.whl.inertia) / (carModel.whl.radius ** 2)

    return spd_rdl_in, F_out, M_inertia_equiv


def calc_rolling_resistance(terrain, V_chas_in, carModel):
    # Load relevant model parameters
    C1 = carModel.whl.coeff_roll1
    C2 = carModel.whl.coeff_roll2
    C3 = carModel.whl.coeff_roll3
    C4 = carModel.whl.coeff_roll4
    m_veh = carModel.veh.mass
    g = carModel.env.gravity
    weight_frac = carModel.whl.weight_fraction_effective
    V_spd_thresh = carModel.whl.spd_thresh
    r_wheel = carModel.whl.radius

    # Calculate torque from rolling resistance
    # Speed threshold term
    f_spd = ((V_chas_in * (V_chas_in <= V_spd_thresh)) / V_spd_thresh) + (V_chas_in > V_spd_thresh)

    # Torque Calculation
    return (numpy.cos(terrain) * (C1 + (C2 * V_chas_in) + (C3 * V_chas_in ** 2) + (
        C4 * V_chas_in ** 3)) * m_veh * g * weight_frac * f_spd * numpy.sign(V_chas_in)) * r_wheel


def chassis_main(F_in, m_inertia_equiv, terrain, V_in, d_in, timestep, carModel):
    # Vehicle losses
    # Grade force calculation
    F_grade = carModel.veh.mass * carModel.env.gravity * numpy.sin(terrain)

    # Drag force calculation
    F_drag = (0.5 * carModel.env.dens_air * carModel.chas.coeff_drag * carModel.chas.frontal_area * (
        V_in ** 2)) * numpy.sign(V_in)

    # Total vehicle losses
    F_loss = F_grade + F_drag

    # Acceleration, velocity and distance calculations
    # Net propulsion force
    F_net = F_in - F_loss

    # Total effective mass
    m_total_effective = carModel.veh.mass + m_inertia_equiv

    # Acceleration
    a = F_net / m_total_effective

    # Velocity
    V_chas = V_in + (a * timestep)
    if V_chas < 0:
        V_chas = 0

    # Distance
    d_out = d_in + (V_chas * timestep)

    return V_chas, d_out


def power_converter_main(V_in, I_out, carModel):
    # Perform calculation
    V_out = carModel.pc.volt_out

    if I_out >= 0:
        I_in = ((I_out * V_out) / carModel.pc.eff) / V_in
    elif I_out < 0:
        I_in = (I_out * V_out * carModel.pc.eff) / V_in

    return V_out, I_in


def elec_acc_main(V_in, key_on, carModel):
    # Current calculation
    return (carModel.accelec.pwr * key_on) / (V_in + numpy.finfo(float).eps)


def interp_2d(xq, yq, x, y, grid):
    # x and y must be sorted !

    x1_pos = None
    y1_pos = None

    # Determine indices of variables to load
    for i in range(0, numpy.size(x, 0) - 1):
        if xq >= x[i] and xq <= x[i + 1]:
            x1_pos = i
            x2_pos = i + 1
            break

    for j in range(0, numpy.size(y, 0) - 1):
        if yq >= y[j] and yq <= y[j + 1]:
            y1_pos = j
            y2_pos = j + 1
            break

    # In case we are out of bounds
    if x1_pos is None:
        if xq < x[0]:
            x1_pos = 0
            x2_pos = 1
        else:
            x1_pos = numpy.size(x, 0) - 2
            x2_pos = numpy.size(x, 0) - 1
    if y1_pos is None:
        if yq < y[0]:
            y1_pos = 0
            y2_pos = 1
        else:
            y1_pos = numpy.size(y, 0) - 2
            y2_pos = numpy.size(y, 0) - 1

    # Select variables
    x1 = x[x1_pos]
    x2 = x[x2_pos]
    y1 = y[y1_pos]
    y2 = y[y2_pos]
    zA = grid[x1_pos, y1_pos]
    zB = grid[x1_pos, y2_pos]
    zC = grid[x2_pos, y2_pos]
    zD = grid[x2_pos, y1_pos]

    # Interpolation accross x
    zAD = numpy.interp(xq, [x1, x2], [zA, zD])
    zBC = numpy.interp(xq, [x1, x2], [zB, zC])

    # Interpolation accross y
    return numpy.interp(yq, [y1, y2], [zAD, zBC])
