"""Create parameters.py and load it"""

# Directory for .inp file:
path = '/home/yoda/Aron/parameter_studies/Bubble_dynamics_simulation/INP file examples/chem_Otomo2018_without_O.inp'

# import libraries:
import importlib   # For reloading your own files
from termcolor import colored   # for colored error messages
# my own files:
try:
    import inp_data_extractor as inp
except:
    try:
        import Bubble_dynamics_simulation.inp_data_extractor as inp
    except:
        print(colored(f'Error, \'inp_data_extractor.py\' not found', 'red'))
importlib.reload(inp)   # reload changes you made
inp.extract(path)

import parameters as par   # numeric constants and coefficents
importlib.reload(par)   # reload changes you made
print(par.model)


"""Libraries"""

# for plotting:
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import numpy as np   # matrices, math
import time   # runtime measurement
import random   # random number generator
from multiprocessing import Pool, cpu_count   # multithreading
import importlib   # reload changes you made
import json   # convert dictionary to string

# my own file:
already_imported = 'gd' in globals()
try:
    import gradient_descent as gd
except:
    try:
        import Bubble_dynamics_simulation.gradient_descent as gd
    except:
        print(colored(f'Error, \'gradient_descent.py\' not found', 'red'))
if already_imported: importlib.reload(gd)   # reload changes you made


"""Ranges"""

for P_amb in [x*par.atm2Pa for x in [0.2, 0.15, 0.10, 0.05]]:
    print('______________________________________________________________')
    print(f'PRESSURE: {P_amb / par.atm2Pa: .2f} [atm]')
    ranges = dict(
    # Initial conditions:
        # bubble equilibrium radius [m]
        R_E = [1.0e-6*x for x in [1.0, 500.0]], # [um --> m]
        # initial radius / equilibrium radius R_0/R_E [-]
        ratio = [1.0],
        # indexes of species in initial bubble
        gases=[[par.index['N2'], par.index['H2']]],
        # Molar fractions of species in the initial bubble (H2 and N2) [-]
        fractions=[[0.25, 0.75]],
    # Ambient parameters:
        # ambient pressure [Pa]
        P_amb = [P_amb],
        # ambient temperature [K]       
        T_inf = [par.absolute_zero + x for x in [30.0]], # [°C --> K]
    # Liquid parameters:
        # water accommodation coefficient [-]
        alfa_M = [0.35],
        #P_v = par.P_v,                    # vapour pressure [Pa]
        #mu_L = par.mu_L,                  # dynamic viscosity [Pa*s]
        # sound speed [m/s]
        c_L = [par.c_L],
        # surfactant (surface tension modfier) [-]
        surfactant = [1.0],
    # Excitation parameters: (excitation_type = no_excitation)
        p_A = [-x * par.atm2Pa for x in [3.0, 0.5]], # [atm --> Pa]
        logf = [4.0, 6.0],
        n = [1.0],
    )


    """First param study"""

    # create folder for parameter study results:
    pressure = P_amb / par.atm2Pa
    file = gd.de.Make_dir(f'/home/yoda/Aron/parameter_studies/2_plus_1_gradient_search/{pressure: .2f}_atm')
    to_optimize = 'energy_efficiency'   # key in data from de.get_data()
    searches = 50    # number os total searches
    trial_points = 2000  # number of trial start_points. best ones will be used for searches


    kwargs_list = [
        dict(
            point=gd.rand_point(ranges, ID=i, padding=0.01),
            to_optimize=to_optimize,
            t_int=[0.0, 1.0],
            LSODA_timeout=30,
            Radau_timeout=300,
            log10=False,
        )
    for i in range(trial_points)]
    best_energy_efficiency = 1e30
    start_points = []

    start = time.time()
    file.new_file()
    with Pool(processes=cpu_count()-2, maxtasksperchild=100) as pool:
        results = pool.imap_unordered(gd.evaluate_kwargs, kwargs_list)

        for result in results:
            data, point, success = result
            data = gd.de.dotdict(data)
            file.write_line(data)
            start_points.append(point)
        # print stuff:
            if data.energy_efficiency > 0 and data.energy_efficiency < best_energy_efficiency:
                best_energy_efficiency = data.energy_efficiency
            to_print = [f'{key}={data[key]: e};   ' for key in ranges if len(ranges[key]) == 2]
            to_print = ''.join(to_print)
            print(f'index: {data.ID: >8}/{len(kwargs_list)};   success: {success};   runtime: {data.elapsed_time: 6.2f} [s]   |   ' + to_print + '|   ' +
                f'{gd.de.target_specie} production: {data.energy_efficiency: e} [MJ/kg] (best: {best_energy_efficiency: .1f} [MJ/kg])'+
                '                                                 ', end='\r')
                
    file.close()
    end = time.time()
    elapsed = end - start
    print(f'\nDONE')
    print(f'total time: {elapsed: .2f} [s]   ({(elapsed / len(kwargs_list)): .2f} [s/run])')

    start_points.sort(key=lambda point: point['output'])
    outputs = [np.log10(point['output']) for point in start_points if point['output'] < 1e30]
    print(f'best energy_efficiency: {start_points[0]["output"]: e} [MJ/kg]')
    print(f'{searches}th energy_efficiency: {start_points[searches-1]["output"]: e} [MJ/kg]')


    """Kwargs and save"""

    kwargs_list = [dict(
        ranges=ranges,
        to_optimize=to_optimize,
        start_point=start_point,
        step_limit=200,
        max_step_until_decay=10,
        first_step=0.05, #between two parameter combinations
        min_step=1e-4, #between two parameter combinations
        decay=0.5,
        delta=1e-6,
        log10=False,
        verbose=False,
        t_int=[0.0, 1.0],
        LSODA_timeout=30,
        Radau_timeout=300,
        ) for start_point in start_points[:searches]]
    
    # save all settings (full_bubble_model.py, parameters.py, ranges) as txt:
    to_print = gd.de.copy(kwargs_list[0])
    del to_print['ranges']
    del to_print['start_point']
    ranges_str = f'''
    gradient descent general settings:
        to_optimize = '{to_optimize}'   # key in data from de.get_data()
        searches = {searches}    # number os total searches
        trial_points = {trial_points}  # number of trial start_points. best ones will be used for searches

    ranges = {json.dumps(ranges, indent=4)}

    settings = {json.dumps(to_print, indent=4)}

    start_points = [
    '''

    for kwargs in kwargs_list[:searches]:
        ranges_str += json.dumps(kwargs['start_point'], indent=4) + ',\n'
    ranges_str += ']'

    file.write_string(ranges_str, 'gradient_descent_settings')


    """Gradient method, multithread"""

    best_output = 1.0e30
    total_point_num = 0
    num = 0
    to_plot = []
    last_points = []
    start = time.time()

    with Pool(processes=cpu_count(), maxtasksperchild=1) as pool:
        results = pool.imap_unordered(gd.search, kwargs_list)
        for result in results:
            all_datas, best_outputs, elapsed = result
            point_num = sum([len(datas) for datas in all_datas])
            total_point_num += point_num
            num += 1
            to_plot.append(best_outputs)
            if len(all_datas) > 0 and len(all_datas[-1]) > 0:
                last_points.append(gd.de.copy(all_datas[-1][0]))
            if best_outputs[-1] < best_output and best_outputs[-1] > 0:
                best_output = best_outputs[-1]

            # save points
            file.new_file()
            for datas in all_datas:
                for data in datas:
                    file.write_line(data)
            file.close()
            del(all_datas)
            
            # print stuff:
            if point_num==0: point_num=1
            print(f'{num: >3}/{searches}: Total {len(best_outputs): <3} steps and {point_num: <4} points, finished in {elapsed: 8.2f} [s]   ({(elapsed / point_num): 4.2f} [s/run]).   '+
                f'Final {to_optimize}: {best_outputs[-1]: 8.1f} (best: {best_output: 6.1f})')
                
    file.close()
    end = time.time()
    elapsed = end - start
    print(f'\n\nDONE')
    print(f'total time: {((elapsed-elapsed % 3600) / 3600): .0f} hours {((elapsed % 3600) / 60): .0f} mins')
    print(f'            {elapsed: .2f} [s]   ({(elapsed / searches): .2f} [s/search])')
    print(f'            {total_point_num: .0f} total points   ({(total_point_num / searches): .0f} [points/search])')
    print(f'            best energy_efficiency: {best_output: .2f} [MJ/kg]')