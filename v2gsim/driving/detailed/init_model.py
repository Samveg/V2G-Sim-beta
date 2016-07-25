from __future__ import division
import openpyxl
import car_model
import numpy
import math


def load_powertrain(excelFilename):
    """ Load a detailed car model with parameters written in an Excel file.

    Args:
        excelFilename (string): filename of the input file

    Returns:
        carModel: a detailed car model object
    """
    # Open the Excel sheet
    wb = openpyxl.load_workbook(filename=excelFilename)

    # Create an empty car model ready to be fill with right values
    model = car_model.DetailedCarModel(name='Leaf')

    # Launch every function one after the other
    vehicle_mass(wb, model)
    environment(wb, model)
    battery(wb, model)
    motor(wb, model)
    torque_coupling(wb, model)
    final_drive(wb, model)
    wheel(wb, model)
    chassis(wb, model)
    power_converter(wb, model)
    electrical_accessories(wb, model)
    model.drv.chas_spd_above_chas_started = 1  # speed above which we consider the vehicle moving in m/s
    model.drv.chas_spd_below_chas_stopped = 0.10  # speed below which we consider the vehicle stopped in m/s
    EV_vpc_vpa(model)

    return model


def vehicle_mass(excel, model):
    # Retrieve the data from the correct sheet
    data = excel.get_sheet_by_name('mass')

    # Set model according to excel data
    model.veh.mass = data['C2'].value


def environment(excel, model):
    # Retrieve the data from the correct sheet
    data = excel.get_sheet_by_name('environment')

    # Set model according to excel data
    model.env.air_cap = data['C5'].value
    model.env.dens_air = data['C4'].value
    model.env.gravity = data['C3'].value
    model.env.temp_amb = data['C2'].value


def battery(excel, model):
    # Retrieve the data from the correct sheet
    data = excel.get_sheet_by_name('battery')

    # Set model according to excel data
    model.ess.pack_energy_wh = data['C4'].value
    model.ess.soc_min = data['C5'].value
    model.ess.soc_max = data['C6'].value
    model.ess.volt_nom = data['C7'].value
    model.ess.volt_max = data['C8'].value
    model.ess.volt_min = data['C9'].value
    model.ess.element_per_module = data['C10'].value
    model.ess.element_per_module_parallel = data['C11'].value
    model.ess.num_per_module = data['C12'].value
    model.ess.design_num_module_parallel = data['C13'].value
    model.ess.design_num_module_parallel = model.ess.design_num_module_parallel * model.ess.element_per_module_parallel
    model.ess.cell_energy_density = data['C14'].value
    model.ess.cell_to_module_weight_ration = data['C15'].value
    model.ess.module_to_pack_weight_ratio = data['C16'].value
    model.ess.temp_reg = data['C18'].value  # Temperature above which cooling fan on
    model.ess.therm_cp_module = data['C17'].value  # Heat capacity of module
    model.ess.dia = data['C19'].value  # Cell diameter
    model.ess.length = data['C20'].value
    model.ess.flow_air_mod = data['C21'].value
    model.ess.case_thk = data['C22'].value
    model.ess.mod_case_th_cond = data['C23'].value

    model.ess.eff_coulomb['idx1_temp'] = map_input1D(data, 'C47', 'E47')
    model.ess.eff_coulomb['map'] = map_input1D(data, 'C48', 'E48')

    model.ess.rint_dist['idx1_temp'] = map_input1D(data, 'C26', 'E26')
    model.ess.rint_dist['idx2_soc'] = map_input1D(data, 'C27', 'M27')
    model.ess.rint_dist['map'] = map_input2D(data, 'C28', 'M30')

    model.ess.rint_chg['idx1_temp'] = map_input1D(data, 'C33', 'E33')
    model.ess.rint_chg['idx2_soc'] = map_input1D(data, 'C34', 'M34')
    model.ess.rint_chg['map'] = map_input2D(data, 'C35', 'M37')

    model.ess.voc['idx1_temp'] = map_input1D(data, 'C40', 'E40')
    model.ess.voc['idx2_soc'] = map_input1D(data, 'C41', 'M41')
    model.ess.voc['map'] = map_input2D(data, 'C42', 'M44')

    # Maximum charging and discharging current
    model.ess.pwr_chg['idx1_soc'] = numpy.array(model.ess.voc['idx2_soc'])
    model.ess.pwr_dis['idx1_soc'] = numpy.array(model.ess.voc['idx2_soc'])
    model.ess.pwr_chg['map'] = numpy.array(-numpy.amax(
        numpy.divide((model.ess.volt_max - model.ess.voc['map']) * model.ess.volt_max, model.ess.rint_chg['map']),
        axis=0), ndmin=2)
    numpy.transpose(model.ess.pwr_chg['map'])
    model.ess.pwr_chg['map'] = numpy.array(
        [model.ess.pwr_chg['map'][0, i] if model.ess.pwr_chg['idx1_soc'][0, i] <= model.ess.soc_max else 0 for i in
         range(0, numpy.size(model.ess.pwr_chg['map']), 1)])  # Check SOC condition
    model.ess.pwr_dis['map'] = numpy.array(numpy.amax(
        numpy.divide((model.ess.voc['map'] - model.ess.volt_min) * model.ess.volt_min, model.ess.rint_dist['map']),
        axis=0), ndmin=2)
    numpy.transpose(model.ess.pwr_dis['map'])
    model.ess.pwr_dis['map'] = numpy.array(
        [model.ess.pwr_dis['map'][0, i] if model.ess.pwr_dis['idx1_soc'][0, i] >= model.ess.soc_min else 0 for i in
         range(0, numpy.size(model.ess.pwr_dis['map']), 1)])

    # Ah capacity per cell
    temp_cap_max = model.ess.pack_energy_wh / (
    model.ess.volt_nom * model.ess.element_per_module * model.ess.num_per_module * model.ess.design_num_module_parallel)
    model.ess.cap_max['idx1_temp'] = numpy.array(model.ess.voc['idx1_temp'])
    model.ess.cap_max['map'] = numpy.array([temp_cap_max] * 3)  # numpy.size(model.ess.cap_max['idx1_temp'], 0)

    # Total number of cells
    model.ess.num_cell = model.ess.element_per_module * model.ess.num_per_module * model.ess.design_num_module_parallel
    model.ess.num_cell_series = model.ess.element_per_module * model.ess.num_per_module

    # Mass per module
    model.ess.mass_module = (numpy.mean(model.ess.cap_max[
                                            'map']) * model.ess.volt_nom / model.ess.cell_energy_density) * model.ess.element_per_module * model.ess.element_per_module_parallel * model.ess.cell_to_module_weight_ration

    # Thermal resistances for thermal/cooling model
    model.ess.therm_flow_area_module = model.ess.length * 2 * 0.00317  # m^2, Area across which cooling air flows per module
    model.ess.speed_air = model.ess.flow_air_mod / (1.16 * model.ess.therm_flow_area_module)
    model.ess.therm_air_htcoef = 30 * (model.ess.speed_air / 5) ** 0.8
    model.ess.area_module = math.pi * model.ess.dia * model.ess.length
    model.ess.therm_res_off = (0.25 + (model.ess.case_thk / model.ess.mod_case_th_cond)) / model.ess.area_module
    model.ess.therm_res_on = ((1 / model.ess.therm_air_htcoef) + (
    model.ess.case_thk / model.ess.mod_case_th_cond)) / model.ess.area_module
    model.ess.therm_res_on = min(model.ess.therm_res_on, model.ess.therm_res_off)


def motor(excel, model):
    # Retrieve the data from the correct sheet
    data = excel.get_sheet_by_name('motor')

    # Set model according to excel data
    model.mot.inertia = data['C5'].value
    model.mot.t_max_trq = data['C4'].value
    model.mot.pwr_des = data['C7'].value
    model.mot.eff_des = data['C8'].value

    model.mot.trq_max['idx1_speed'] = map_input1D(data, 'C13', 'N13')
    model.mot.trq_max['idx1_speed'] = numpy.array([model.mot.trq_max['idx1_speed'][i] * (2 * math.pi / 60) for i in
                                                   range(0, len(model.mot.trq_max['idx1_speed']))],
                                                  ndmin=2)  # RPM to rad/s conversion
    model.mot.trq_max['map'] = map_input1D(data, 'C14', 'N14')

    model.mot.eff_trq['idx1_speed'] = map_input1D(data, 'C17', 'N17')
    model.mot.eff_trq['idx1_speed'] = numpy.array([model.mot.eff_trq['idx1_speed'][i] * (2 * math.pi / 60) for i in
                                                   range(0, len(model.mot.eff_trq['idx1_speed']))],
                                                  ndmin=2)  # RPM to rad/s conversion
    model.mot.eff_trq['idx2_trq'] = map_input1D(data, 'C18', 'AE18')
    model.mot.eff_trq['map'] = map_input2D(data, 'C19', 'AE30')

    if data['C8'].value:  # If efficiency scaling value was specified
        model.mot.eff_trq_max = numpy.amax(model.mot.eff_trq['map'])  # Maximum effiency
        model.mot.eff_trq['map'] = (model.mot.eff_des / model.mot.eff_trq_max) * model.mot.eff_trq['map']
        model.mot.eff_trq_max = model.mot.eff_des

    # Maximum continious torque
    model.mot.trq_cont['idx1_speed'] = numpy.array(model.mot.trq_max['idx1_speed'])
    model.mot.trq_cont['map'] = numpy.multiply(0.5, model.mot.trq_max['map'])

    # Maximum negative peak torque
    model.mot.trq_neg_max['idx1_speed'] = numpy.hstack((-numpy.fliplr(
        numpy.atleast_2d(model.mot.trq_max['idx1_speed'][0, 1:])), numpy.array(
        [-numpy.finfo(float).eps, 0, numpy.finfo(float).eps], ndmin=2)))
    model.mot.trq_neg_max['idx1_speed'] = numpy.array(
        numpy.hstack((model.mot.trq_neg_max['idx1_speed'][0, :], model.mot.trq_max['idx1_speed'][0, 1:])), ndmin=2)
    model.mot.trq_neg_max['map'] = numpy.hstack((numpy.fliplr(numpy.atleast_2d(model.mot.trq_max['map'][0, 1:])),
                                                 numpy.array(
                                                     [model.mot.trq_max['map'][0, 1], -model.mot.trq_max['map'][0, 1],
                                                      -model.mot.trq_max['map'][0, 1]], ndmin=2)))
    model.mot.trq_neg_max['map'] = numpy.hstack((model.mot.trq_neg_max['map'][0, :], -model.mot.trq_max['map'][0, 1:]))
    model.mot.pwr_neg_max['map'] = numpy.multiply(model.mot.trq_neg_max['idx1_speed'], model.mot.trq_neg_max['map'])

    # Maximum negative continuous torque
    model.mot.trq_neg_cont['idx1_speed'] = numpy.hstack((-numpy.fliplr(
        numpy.atleast_2d(model.mot.trq_cont['idx1_speed'][0, 1:])), numpy.array(
        [-numpy.finfo(float).eps, 0, numpy.finfo(float).eps], ndmin=2)))
    model.mot.trq_neg_cont['idx1_speed'] = numpy.hstack(
        (model.mot.trq_neg_cont['idx1_speed'][0, :], model.mot.trq_neg_cont['idx1_speed'][0, 1:]))
    model.mot.trq_neg_cont['map'] = numpy.hstack((numpy.fliplr(numpy.atleast_2d(model.mot.trq_cont['map'][0, 1:])),
                                                  numpy.array([model.mot.trq_cont['map'][0, 1],
                                                               -model.mot.trq_cont['map'][0, 1],
                                                               -model.mot.trq_cont['map'][0, 1]], ndmin=2)))
    model.mot.trq_neg_cont['map'] = numpy.hstack(
        (model.mot.trq_neg_cont['map'][0, :], -model.mot.trq_neg_cont['map'][0, 1:]))

    # Maximum positive peak torque
    model.mot.trq_pos_max['idx1_speed'] = numpy.hstack((-numpy.fliplr(
        numpy.atleast_2d(model.mot.trq_max['idx1_speed'][0, 1:])), numpy.array(
        [-numpy.finfo(float).eps, 0, numpy.finfo(float).eps], ndmin=2)))
    model.mot.trq_pos_max['idx1_speed'] = numpy.array(
        numpy.hstack((model.mot.trq_pos_max['idx1_speed'][0, :], model.mot.trq_max['idx1_speed'][0, 1:])), ndmin=2)
    model.mot.trq_pos_max['map'] = numpy.hstack((-numpy.fliplr(numpy.atleast_2d(model.mot.trq_max['map'][0, 1:])),
                                                 numpy.array(
                                                     [-model.mot.trq_max['map'][0, 1], model.mot.trq_max['map'][0, 1],
                                                      model.mot.trq_max['map'][0, 1]], ndmin=2)))
    model.mot.trq_pos_max['map'] = numpy.hstack((model.mot.trq_pos_max['map'][0, :], model.mot.trq_max['map'][0, 1:]))
    model.mot.pwr_pos_max['map'] = numpy.multiply(model.mot.trq_pos_max['idx1_speed'], model.mot.trq_pos_max['map'])

    # Maximum positive continuous torque (multiply previous by 0.5)
    model.mot.trq_pos_cont['idx1_speed'] = numpy.hstack((-numpy.fliplr(
        numpy.atleast_2d(model.mot.trq_cont['idx1_speed'][0, 1:])), numpy.array(
        [-numpy.finfo(float).eps, 0, numpy.finfo(float).eps], ndmin=2)))
    model.mot.trq_pos_cont['idx1_speed'] = numpy.hstack(
        (model.mot.trq_pos_cont['idx1_speed'][0, :], model.mot.trq_pos_cont['idx1_speed'][0, 1:]))
    model.mot.trq_pos_cont['map'] = numpy.hstack((-numpy.fliplr(numpy.atleast_2d(model.mot.trq_cont['map'][0, 1:])),
                                                  numpy.array([-model.mot.trq_cont['map'][0, 1],
                                                               model.mot.trq_cont['map'][0, 1],
                                                               model.mot.trq_cont['map'][0, 1]], ndmin=2)))
    model.mot.trq_pos_cont['map'] = numpy.hstack(
        (model.mot.trq_pos_cont['map'][0, :], model.mot.trq_pos_cont['map'][0, 1:]))

    # Mechanical power map
    model.mot.pwr_mech['idx1_speed'] = numpy.array(model.mot.eff_trq['idx1_speed'], ndmin=2)
    model.mot.pwr_mech['idx2_trq'] = numpy.array(model.mot.eff_trq['idx2_trq'], ndmin=2)
    model.mot.pwr_mech['map'] = numpy.dot(numpy.transpose(model.mot.pwr_mech['idx1_speed']),
                                          model.mot.pwr_mech['idx2_trq'])  # Matrix product

    # Electrical losses map
    model.mot.pwr_elec_loss['idx1_speed'] = numpy.array(model.mot.eff_trq['idx1_speed'], ndmin=2)
    model.mot.pwr_elec_loss['idx2_trq'] = numpy.array(model.mot.eff_trq['idx2_trq'], ndmin=2)
    model.mot.pwr_elec_loss['map'] = numpy.multiply(model.mot.pwr_mech['map'],
                                                    1 - model.mot.eff_trq['map'])  # Element wise
    temp_pwr_elec_loss = numpy.array(model.mot.pwr_elec_loss['map'])
    temp_pos_mech_eff = numpy.array(model.mot.pwr_mech['map'])
    for i in range(0, numpy.size(model.mot.eff_trq['map'], 0)):
        for j in range(0, numpy.size(model.mot.eff_trq['map'], 1)):
            if model.mot.pwr_mech['map'][i, j] >= 0 and model.mot.eff_trq['map'][i, j] != 0:
                temp_pos_mech_eff[i, j] = 1
            else:
                temp_pos_mech_eff[i, j] = 0
    for i in range(0, numpy.size(model.mot.pwr_elec_loss['map'], 0)):
        for j in range(0, numpy.size(model.mot.pwr_elec_loss['map'], 1)):
            if temp_pos_mech_eff[i, j]:
                model.mot.pwr_elec_loss['map'][i, j] = numpy.divide(model.mot.pwr_elec_loss['map'][i, j],
                                                                    float(model.mot.eff_trq['map'][i, j]))

    # Expand maps to cover negative speeds and torques too
    # Motor efficiency
    model.mot.eff_trq['idx1_speed'] = numpy.hstack(
        (-numpy.fliplr(numpy.atleast_2d(model.mot.eff_trq['idx1_speed'][0, 1:])), model.mot.eff_trq['idx1_speed']))
    model.mot.eff_trq['idx2_trq'] = numpy.hstack(
        (-numpy.fliplr(numpy.atleast_2d(model.mot.eff_trq['idx2_trq'][0, 1:])), model.mot.eff_trq['idx2_trq']))
    model.mot.eff_trq['map'] = numpy.hstack((numpy.fliplr(model.mot.eff_trq['map'][:, 1:]), model.mot.eff_trq['map']))
    model.mot.eff_trq['map'] = numpy.vstack((numpy.flipud(model.mot.eff_trq['map'][1:, :]), model.mot.eff_trq['map']))

    # Mechanical power output
    model.mot.pwr_mech['idx1_speed'] = numpy.hstack((numpy.array([0, numpy.finfo(float).eps], ndmin=2),
                                                     numpy.array(model.mot.pwr_mech['idx1_speed'][0, 1:], ndmin=2)))
    model.mot.pwr_mech['idx1_speed'] = numpy.hstack(
        (-numpy.fliplr(numpy.atleast_2d(model.mot.pwr_mech['idx1_speed'][0, 1:])), model.mot.pwr_mech['idx1_speed']))
    model.mot.pwr_mech['idx2_trq'] = numpy.hstack(
        (-numpy.fliplr(numpy.atleast_2d(model.mot.pwr_mech['idx2_trq'][0, 1:])), model.mot.pwr_mech['idx2_trq']))
    model.mot.pwr_mech['map'] = numpy.hstack(
        (-numpy.fliplr(model.mot.pwr_mech['map'][:, 1:]), model.mot.pwr_mech['map']))
    temp = numpy.vstack((model.mot.pwr_mech['map'][0, :], model.mot.pwr_mech['map'][0, :]))
    model.mot.pwr_mech['map'] = numpy.vstack((temp, model.mot.pwr_mech['map'][1:, :]))
    model.mot.pwr_mech['map'] = numpy.vstack(
        (-numpy.flipud(model.mot.pwr_mech['map'][1:, :]), model.mot.pwr_mech['map']))

    # Electrical losses map
    model.mot.pwr_elec_loss['idx1_speed'] = numpy.hstack((numpy.array([0, numpy.finfo(float).eps], ndmin=2),
                                                          numpy.array(model.mot.pwr_elec_loss['idx1_speed'], ndmin=2)))
    model.mot.pwr_elec_loss['idx1_speed'] = numpy.hstack((-numpy.fliplr(
        numpy.atleast_2d(model.mot.pwr_elec_loss['idx1_speed'][0, 1:])), model.mot.pwr_elec_loss['idx1_speed']))
    model.mot.pwr_elec_loss['idx2_trq'] = numpy.hstack((-numpy.fliplr(
        numpy.atleast_2d(model.mot.pwr_elec_loss['idx2_trq'][0, 1:])), model.mot.pwr_elec_loss['idx2_trq']))
    temp = numpy.vstack((model.mot.pwr_elec_loss['map'][0, :], 0.05 * model.mot.pwr_elec_loss['map'][1, :]))
    model.mot.pwr_elec_loss['map'] = numpy.vstack((temp, model.mot.pwr_elec_loss['map'][1:, :]))
    temp = numpy.vstack((-numpy.flipud(temp_pwr_elec_loss[1:, :]), model.mot.pwr_elec_loss['map'][0, :]))
    model.mot.pwr_elec_loss['map'] = numpy.vstack((temp, model.mot.pwr_elec_loss['map']))
    temp = numpy.rot90(model.mot.pwr_elec_loss['map'][:, 1:], 2)
    model.mot.pwr_elec_loss['map'] = numpy.hstack((temp, model.mot.pwr_elec_loss['map']))

    # Motor electrical demand at each operating point
    model.mot.pwr_elec['idx1_speed'] = numpy.array(model.mot.pwr_mech['idx1_speed'])
    model.mot.pwr_elec['idx2_trq'] = numpy.array(model.mot.pwr_mech['idx2_trq'])
    model.mot.pwr_elec['map'] = model.mot.pwr_mech['map'] + numpy.absolute(model.mot.pwr_elec_loss['map'])

    # Mapping torque output from electrical power into motor
    model.mot.trq_pwr_elec['idx1_speed'] = numpy.array(model.mot.pwr_mech['idx1_speed'])
    model.mot.pwr_mech['max'] = numpy.maximum(numpy.amax(numpy.absolute(model.mot.pwr_pos_max['map'])),
                                              numpy.amax(numpy.absolute(model.mot.pwr_neg_max['map'])))
    model.mot.pwr_elec['max'] = 2 * model.mot.pwr_mech['max']  # AyR - approximation
    model.mot.trq_pwr_elec['idx2_pwr'] = numpy.array(
        numpy.arange(-model.mot.pwr_elec['max'], model.mot.pwr_elec['max'] + model.mot.pwr_elec['max'] / 20,
                     model.mot.pwr_elec['max'] / 20), ndmin=2)

    # Mapping torque output from electrical power input into motor
    model.mot.trq_pwr_elec['map'] = numpy.zeros(
        (numpy.size(model.mot.trq_pwr_elec['idx1_speed'], 1), numpy.size(model.mot.trq_pwr_elec['idx2_pwr'], 1)))
    for row_index in range(0, numpy.size(model.mot.trq_pwr_elec['idx1_speed'], 1)):
        # Skip row if pwr_elec.map row vector contains non-unique values
        if numpy.size(numpy.unique(model.mot.pwr_elec['map'][row_index, :]), 0) == numpy.size(
                model.mot.pwr_elec['map'][row_index, :], 0):
            skip_row = False
        else:
            skip_row = True

        if not skip_row:
            # Added to work around the fact that values need to be sorted
            if model.mot.pwr_elec['map'][row_index, 1] < 0:
                a = (model.mot.pwr_elec['map'][row_index, :]).flatten()
                b = (model.mot.eff_trq['idx2_trq']).flatten()
            else:
                a = (model.mot.pwr_elec['map'][row_index, :]).flatten()[::-1]
                b = (model.mot.eff_trq['idx2_trq']).flatten()[::-1]

            for col_index in range(0, numpy.size(model.mot.trq_pwr_elec['idx2_pwr'], 1)):
                temp_val1 = numpy.interp(model.mot.trq_pwr_elec['idx2_pwr'][0, col_index], a, b)
                if temp_val1 < 0:
                    temp_val2 = -numpy.absolute(
                        numpy.interp((model.mot.trq_pwr_elec['idx1_speed'][0, row_index]).flatten(),
                                     (model.mot.trq_pos_max['idx1_speed']).flatten(),
                                     (model.mot.trq_pos_max['map']).flatten()))
                    temp_val3 = numpy.maximum(temp_val2, temp_val1)
                elif temp_val1 >= 0:
                    temp_val2 = numpy.absolute(
                        numpy.interp((model.mot.trq_pwr_elec['idx1_speed'][0, row_index]).flatten(),
                                     (model.mot.trq_pos_max['idx1_speed']).flatten(),
                                     (model.mot.trq_pos_max['map']).flatten()))
                    temp_val3 = numpy.minimum(temp_val2, temp_val1)
                model.mot.trq_pwr_elec['map'][row_index, col_index] = temp_val3
        elif skip_row:
            if numpy.around(model.mot.trq_pwr_elec['idx1_speed'][0, row_index]) == 0:
                for col_index in range(0, numpy.size(model.mot.trq_pwr_elec['idx2_pwr'], 1)):
                    if numpy.around(model.mot.trq_pwr_elec['idx2_pwr'][0, col_index]) == 0:
                        model.mot.trq_pwr_elec['map'][row_index, col_index] = 0
                    elif model.mot.trq_pwr_elec['idx2_pwr'][0, col_index] != 0:
                        if model.mot.trq_pwr_elec['idx2_pwr'][0, col_index] >= 0:
                            model.mot.trq_pwr_elec['map'][row_index, col_index] = numpy.interp(
                                (model.mot.trq_pwr_elec['idx1_speed'][0, row_index]).flatten(),
                                (model.mot.trq_pos_max['idx1_speed']).flatten(),
                                (model.mot.trq_pos_max['map']).flatten())
                        elif model.mot.trq_pwr_elec['idx2_pwr'][0, col_index] < 0:
                            model.mot.trq_pwr_elec['map'][row_index, col_index] = numpy.interp(
                                (model.mot.trq_pwr_elec['idx1_speed'][0, row_index]).flatten(),
                                (model.mot.trq_neg_max['idx1_speed']).flatten(),
                                (model.mot.trq_neg_max['map']).flatten())
            else:
                print(
                'Portions of the electrical power input to torque output map are incomplete. This may result in erroneous simulation results.')

                # Apply power scaling
    if data['C7'].value:  # If power scaling value was specified
        print('Error in the Matlab code, please do not specify any value in motor - C7')


def torque_coupling(excel, model):
    # Retrieve the data from the correct sheet
    data = excel.get_sheet_by_name('torque coupling')

    # Set model according to excel data
    model.tc.ratio = data['C1'].value
    model.tc.inertia = data['C2'].value
    model.tc.eff_spec = data['C3'].value

    # Calculate remaining parameters    
    model.tc.spd_thresh = 10
    model.tc.trq_loss['idx1_trq'] = numpy.array([0, 5000], ndmin=2)
    model.tc.trq_loss['idx2_speed'] = numpy.array([0, 1000], ndmin=2)
    model.tc.trq_loss['map'] = numpy.tile(
        numpy.transpose(numpy.array((1 - model.tc.eff_spec) * model.tc.trq_loss['idx1_trq'])),
        (1, numpy.size(model.tc.trq_loss['idx2_speed'], 1)))


def final_drive(excel, model):
    # Retrieve the data from the correct sheet
    data = excel.get_sheet_by_name('final drive')

    # Set model according to excel data
    model.fd.ratio = data['C1'].value
    model.fd.inertia = data['C2'].value
    model.fd.eff_spec = data['C3'].value

    # Calculate remaining parameters    
    model.fd.spd_thresh = 10
    model.fd.trq_loss['idx1_trq'] = numpy.array([0, 5000], ndmin=2)
    model.fd.trq_loss['idx2_speed'] = numpy.array([0, 1000], ndmin=2)
    model.fd.trq_loss['map'] = numpy.tile(
        numpy.transpose(numpy.array((1 - model.fd.eff_spec) * model.fd.trq_loss['idx1_trq'])),
        (1, numpy.size(model.fd.trq_loss['idx2_speed'], 1)))


def wheel(excel, model):
    # Retrieve the data from the correct sheet
    data = excel.get_sheet_by_name('wheels')

    # Set model according to excel data
    tire_code = str(data['C3'].value)
    model.whl.inertia_per_wheel = data['C4'].value
    model.whl.coeff_roll1 = data['C6'].value
    model.whl.coeff_roll2 = data['C7'].value
    model.whl.coeff_roll3 = data['C8'].value
    model.whl.coeff_roll4 = data['C9'].value

    model.whl.friction_coefficient['idx1_chas_lin_spd'] = map_input1D(data, 'C12', 'L12')
    model.whl.friction_coefficient['idx1_chas_lin_spd'] = model.whl.friction_coefficient['idx1_chas_lin_spd'] * (
    1609 / 3600)  # Convert from mph to m/s
    model.whl.friction_coefficient['map'] = map_input1D(data, 'C13', 'L13')

    # Convert tire codes to the appropriate dimmesnions
    # For calculation details, see: http://www.tirerack.com/tires/tiretech/techpage.jsp?techid=7
    import re
    split_code = re.split('R|/', tire_code)
    model.whl.tire_width = int(split_code[0]) / float(1000)  # tire section with in m
    model.whl.profile = int(split_code[1]) / float(100)  # aspect ratio of profile, unitless
    model.whl.rim_diameter = int(split_code[2]) * float(float(25.4) / 1000)
    model.whl.theoretical_radius = (model.whl.rim_diameter + (2 * model.whl.tire_width * model.whl.profile)) / float(
        2)  # theoretical radius of tire in m

    model.whl.trq_brake_mech['idx1_brk_cmd'] = numpy.array([0, 0.05, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 1.0], ndmin=2)
    model.whl.trq_brake_mech['map'] = -numpy.array([0, 0.10, 0.70, 0.775, 0.825, 0.85, 0.88, 0.90, 1.0],
                                                   ndmin=2) * 2000  # Nm

    model.whl.number_wheels = 4
    model.whl.trq_brake_max = 2000  # N-m
    model.whl.radius_correction = 0.95
    model.whl.radius = model.whl.theoretical_radius * model.whl.radius_correction  # m
    model.whl.spd_thresh = 1
    model.whl.inertia = model.whl.inertia_per_wheel * model.whl.number_wheels
    model.whl.weight_fraction_effective = 1
    model.whl.brake_fraction = 1


def chassis(excel, model):
    # Retrieve the data from the correct sheet
    data = excel.get_sheet_by_name('chassis')

    # Set model according to excel data
    model.chas.coeff_drag = data['C2'].value
    model.chas.frontal_area = data['C3'].value
    model.chas.ratio_weight_front = data['C4'].value


def power_converter(excel, model):
    # Retrieve the data from the correct sheet
    data = excel.get_sheet_by_name('power converter')

    # Set model according to excel data
    model.pc.volt_out = data['C2'].value
    model.pc.eff = data['C3'].value


def electrical_accessories(excel, model):
    # Retrieve the data from the correct sheet
    data = excel.get_sheet_by_name('electrical accessories')

    # Set model according to excel data
    model.accelec.pwr = data['C2'].value


def EV_vpc_vpa(model):
    # Initiate VPA
    model.vpa.ratio_cum = model.fd.ratio * model.tc.ratio  # Overall torque ratio from motor to wheels

    # Initiate VPC propulsion
    # Map of maximum torque available at the wheels
    motor_chassis_speed = (model.mot.trq_pos_max['idx1_speed'] / model.vpa.ratio_cum) * model.whl.radius
    model.vpc.whl_trq_max['idx1_chas_lin_spd'] = numpy.arange(0, numpy.amax(motor_chassis_speed), 1)

    temp1 = model.mot.trq_pos_max['map'] * model.vpa.ratio_cum * model.fd.eff_spec
    max_motor_torque = numpy.interp(model.vpc.whl_trq_max['idx1_chas_lin_spd'], (motor_chassis_speed).flatten(),
                                    (temp1).flatten())
    model.vpc.whl_trq_max['map'] = numpy.array(
        [numpy.maximum(max_motor_torque[i], 1) for i in range(0, len(max_motor_torque))], ndmin=2)

    # Initiate VPC braking
    # Scalar values
    model.vpc.chas_spd_below_no_regen = 1.5  # m/s
    model.vpc.chas_spd_above_full_regen = 3  # m/s
    model.vpc.eff_accelec_to_mot = model.pc.eff  # electrical accessory to motor transfer efficiency
    model.vpc.eff_ess_to_mot = 1  # Battery to motor power transfer efficiency, 1 here because of direct link
    model.vpc.ess_soc_above_regen_forbidden = model.ess.soc_max - 0.03  # regen braking disabled above this battery SOC
    model.vpc.ess_soc_below_regen_allowed = model.ess.soc_max - 0.05  # regen braking enabled below this battery SOC
    model.vpc.eff_to_whl = model.fd.eff_spec
    model.vpc.ratio_cum = model.vpa.ratio_cum

    # Map data
    # Map of pedal position to braking torque    
    model.vpc.brk_trq['map'] = numpy.fliplr(model.whl.trq_brake_mech['map'])
    model.vpc.brk_trq['idx1_brk_cmd'] = numpy.fliplr(- model.whl.trq_brake_mech['idx1_brk_cmd'])

    # Map of traction motor torque vs speed and electrical power
    model.vpc.trq_pwr_elec['idx1_speed'] = model.mot.trq_pwr_elec['idx1_speed']  # rad/s
    model.vpc.trq_pwr_elec['idx2_pwr'] = model.mot.trq_pwr_elec['idx2_pwr']  # W
    model.vpc.trq_pwr_elec['map'] = model.mot.trq_pwr_elec['map']  # Nm

    # Map of regen vs mechanical braking
    model.vpc.ratio_ecu_brk_total_brk['idx1_lin_accel'] = [0, 2.5, 4,
                                                           5]  # Values in m/s^2 used to set acceleration values for regen vs mech brake blending
    model.vpc.ratio_ecu_brk_total_brk['map'] = [1, 1, 0, 0]

    # Map of braking command to braking torque at wheels
    model.vpc.whls_trq_brk_total['idx1_brake_cmd'] = -model.whl.trq_brake_mech['idx1_brk_cmd']
    model.vpc.whls_trq_brk_total['map_neg'] = -model.whl.trq_brake_mech['map']


def map_input1D(data, corner1, corner2):
    tempList = []
    for rowOfCellObjects in data[corner1:corner2]:
        for cellObj in rowOfCellObjects:
            tempList.append(cellObj.value)

    # Return a list of values
    return numpy.array(tempList, ndmin=2)


def map_input2D(data, corner1, corner2):
    tempList = []
    for index1, rowOfCellObjects in enumerate(data[corner1:corner2]):
        for index2, cellObj in enumerate(rowOfCellObjects):
            # Append list for the number of column in the map
            if index1 == 0:
                tempList.append([])
            tempList[index2].append(cellObj.value)

    # Return a list of lists
    return numpy.transpose(numpy.array(tempList, ndmin=2))

    # Usefull to copy list of lists without keeping references
    # def unshared_copy(inList):
    #    if isinstance(inList, list):
    #        return list(map(unshared_copy, inList))
    #    return inList

    # Usefull post mortem debuging
    #    except:
    #        import sys, traceback, pdb
    #        type, value, tb = sys.exc_info()
    #        traceback.print_exc()
    #        pdb.post_mortem(tb)
