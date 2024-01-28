if __name__ == '__main__':    
    """Create parameters.py and load it"""
    print('HERE')
    # Directory for .inp file:
    path = 'D:/parameter_studies/Bubble_dynamics_simulation/INP file examples/chem_Otomo2018_without_O.inp'

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
    #importlib.reload(inp)   # reload changes you made
    inp.extract(path)

    import parameters as par   # numeric constants and coefficents
    #importlib.reload(par)   # reload changes you made
    #print(par.model)



    """Libraries"""

    # for plotting:
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 18})

    import numpy as np   # matrices, math
    import time   # runtime measurement
    import random   # random number generator
    from multiprocessing import Pool, cpu_count   # multithreading
    #import importlib   # reload changes you made
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
    #if already_imported: importlib.reload(gd)   # reload changes you made

    for pressure_bar in [3.0, 5.0, 10.0]:
        """Control parameter ranges and division"""
        # a list for each control parameter, containing all the possible values

        ranges = dict(
        # Initial conditions:
            # bubble equilibrium radius [m]
            R_E = [1.0e-6*x for x in [5.0, 500.0]], # [um --> m]
            # initial radius / equilibrium radius R_0/R_E [-]
            ratio = [1.0],
            # indexes of species in initial bubble
            gases=[[par.index['N2'], par.index['H2']]],
            # Molar fractions of species in the initial bubble (H2 and N2) [-]
            fractions=[[0.25, 0.75]],
        # Ambient parameters:
            # ambient pressure [Pa]
            P_amb = [x * par.atm2Pa for x in [pressure_bar]], # [atm --> Pa]
            # ambient temperature [K]       
            T_inf = [par.absolute_zero + x for x in [20.0]], # [°C --> K]
        # Liquid parameters:
            # water accommodation coefficient [-]
            alfa_M = [par.alfa_M],
            # vapour pressure [Pa] - will be calculated from T_inf
            P_v = [par.P_v],
            # dynamic viscosity [Pa*s] - will be calculated from T_inf
            mu_L = [par.mu_L],
            # density [kg/m^3]
            rho_L = [par.rho_L],
            # sound speed [m/s]
            c_L = [par.c_L],
            # surfactant (surface tension modfier) [-]
            surfactant = [1.0],
        # Excitation parameters: (excitation_type = no_excitation)
            p_A = [-1e5*x for x in [1.5, 6.0]], # pressure amplitude [Pa]
            freq = [5000.0, 100000.0], # frequency [Hz]
            n = [1.0]
        )

        # create folder for parameter study results:
        save_path = f'D:/parameter_studies/2_plus_1_GD2/{pressure_bar:_>8}_bar'
        file = gd.de.Make_dir(save_path)
        to_optimize = 'energy_efficiency'   # key in data from de.get_data()
        searches = 12    # number os total searches
        trial_points = 1000  # number of trial start_points. best ones will be used for searches


        kwargs_list = [
            dict(
                point=gd.rand_point(ranges, ID=i, padding=0.001),
                to_optimize=to_optimize,
                t_int=[0.0, 1.0],
                LSODA_timeout=30,
                Radau_timeout=300,
            )
        for i in range(trial_points)]
        best_energy_efficiency = 1e30
        start_points = []

        start = time.time()
        file.new_file()
        with Pool(processes=cpu_count(), maxtasksperchild=100) as pool:
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
        ranges2 = gd.de.copy(ranges)
        for key in ranges:
            if len(ranges[key]) != 2:
                continue
            
            values = [point[key] for point in start_points[:searches]]
            ranges2[key][0] = min(values)
            ranges2[key][1] = max(values)
            print(f'old {key}: {ranges[key][1]-ranges[key][0]: e}')
            print(f'new {key}: {ranges2[key][1]-ranges2[key][0]: e}')

        kwargs_list = [dict(
            ranges=ranges2,
            path=save_path,
            to_optimize=to_optimize,
            start_point=start_point,
            step_limit=100,
            first_step=0.01,
            min_step=10e-5,
            delta=1e-6,
            verbose=False,
            t_int=[0.0, 1.0],
            LSODA_timeout=30,
            Radau_timeout=300,
            ) for start_point in start_points[:searches]]

        """Save settings as txt"""

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


        """Gradient descent, multithread"""

        best_output = 1.0e30
        num = 0
        to_plot = []
        last_points = []
        start = time.time()

        with Pool(processes=cpu_count(), maxtasksperchild=1) as pool:
            results = pool.imap_unordered(gd.search, kwargs_list)
            for result in results:
                last_data, best_outputs, elapsed = result
                num += 1
                to_plot.append(best_outputs)
                last_points.append(last_data)
                if best_outputs[-1] < best_output and best_outputs[-1] > 0:
                    best_output = best_outputs[-1]
                
                # print stuff:
                print(f'{num: >3}/{searches}: Total {len(best_outputs): <3} steps, finished in {elapsed: 8.2f} [s]   ({(elapsed / len(best_outputs)): 4.2f} [s/step]).   '+
                    f'Final {to_optimize}: {best_outputs[-1]: 8.1f} (best: {best_output: 6.1f})')
                    
        file.close()
        end = time.time()
        elapsed = end - start
        print(f'\n\nDONE')
        print(f'total time: {((elapsed-elapsed % 3600) / 3600): .0f} hours {((elapsed % 3600) / 60): .0f} mins')
        print(f'            {elapsed: .2f} [s]   ({(elapsed / searches): .2f} [s/search])')

        print(f'PRESSURE: {pressure_bar} bar')
        for x in to_plot:
            print(len(x), 'db;    ', x[-1], 'MJ/kg')

