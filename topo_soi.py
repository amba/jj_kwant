#!/usr/bin/env python3

import jj_kwant.spectrum
import scipy.constants as const
import numpy as np
import jj_kwant.data
import time
import multiprocessing
import psutil
import random

gap = 100e-6 * const.e
meVnm = 1e-3 * const.e * 1e-9
num_cores = 40

#rashba = 20e-3 * const.e * 1e-9 # 20 meV nm

#junction_island_width = 50e-9
#junction_island_spacing = 100e-9
args = {
    'mass': 0.02 * const.m_e,
    'gap': gap,
 #   'rashba': rashba,
    'width': 5e-6,
    'junction_length': 100e-9,
    'electrode_length': 2e-6,
    'a': 5e-9
}


data_file = jj_kwant.data.datafile(folder="topo_gap_jj", params=['phi', 'potential', 'disorder', 'B', 'mu', 's_xx', 's_xy', 's_yx', 's_yy', 'diff'], args=args)

def wait_for_mem():
    while True:
        mem_percent = psutil.virtual_memory().percent
        if mem_percent < 90:
            break;
        else:
            print("cannot start process. used memory is %.1f percent" % mem_percent)
            time.sleep(600 * random.random())

start_time = time.time()            
def calc(problem):
    mu = problem['mu']
    phi = problem['phi']
    B = problem['B']
    SOI = problem['soi'] # 2x2 matrix
    potential = problem['potential']
    disorder = problem['disorder']

    # do not start all processes at once
    if time.time() - start_time < 100:
        sleep_time = 600 * random.random()
        print("sleeping for ", sleep_time)
        time.sleep(sleep_time)
    
    wait_for_mem()
    
    ham = jj_kwant.spectrum.hamiltonian_jj_2d(
        a = args['a'],
        m = args['mass'],
        gap_potential = potential,
        gap_potential_shape='cosine',
        disorder=disorder,
        mu = mu,
        gap = args['gap'],
        SOI = SOI,
        width = args['width'],
        junction_length=args['junction_length'],
        electrode_length=args['electrode_length'],
        B = [0, B, 0],
        delta_phi = phi,
        salt=str(time.time())
            #junction_island_width = junction_island_width,
            #junction_island_spacing = junction_island_spacing,
            # debug=True
    );

    wait_for_mem()
    
    evs = jj_kwant.spectrum.positive_low_energy_spectrum(ham, 2)
    
    
    print("logging evs")
    diff = evs[0] - evs[1]
    print("diff: ", diff)
    data_file.log(evs, {'phi': phi, 'potential': potential, 'B': B, 'mu': mu, 's_xx': SOI[0,0], 's_xy': SOI[0,1], 's_yx': SOI[1,0], 's_yy': SOI[1,1], 'diff': diff})





if __name__ == '__main__':
    mu_vals = np.linspace(0, 20e-3 * const.e, 200)
    B = 0.1
    phi = np.pi
    potential_vals = (0,)
    disorder_vals = np.linspace(0, 10e-3*const.e, 100)
    soi_vals = (20 * meVnm,)
    problems = []
    for potential in potential_vals:
        for mu in mu_vals:
            for soi in soi_vals:
                for disorder in disorder_vals:
                    SOI = np.array([[0, soi], [-soi, 0]])
                    problems.append({
                        'mu': mu, 'phi': phi, 'B': B,
                        'soi': SOI,
                        'potential': potential,
                        'disorder': disorder,
                    })

                    
    with multiprocessing.Pool(num_cores) as p:
        p.map(calc, problems)
    

    

