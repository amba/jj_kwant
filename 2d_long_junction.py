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
num_cores = 1
alpha = 15* meVnm
a = 5e-9
mu = 50e-3 * const.e

mass = 0.04 * const.m_e
width = 3e-6
junction_length = 2e-6
electrode_length = 2e-6


k_fermi = 1/const.hbar * np.sqrt(2 * mass * mu)
l_fermi = 2*np.pi / k_fermi
print("l_fermi = %g nm" % (l_fermi * 1e9))
N_bound_states = int(4*width /l_fermi)
print("N_bound_states = ", N_bound_states)


data_file = jj_kwant.data.datafile(folder="long_junction", params=['phi', 'disorder', 'B', 'mu', 'alpha'])

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
    disorder = problem['disorder']

    ham = jj_kwant.spectrum.hamiltonian_jj_2d(
        a = a,
        m = mass,
        disorder=disorder,
        mu = mu,
        gap = gap,
        SOI = SOI,
        width = width,
        junction_length=junction_length,
        electrode_length=electrode_length,
        B = [0, B, 0],
        delta_phi = phi,
        salt=str(time.time())
        # debug=True
    );

    #wait_for_mem()
    
    evs = jj_kwant.spectrum.positive_low_energy_spectrum(ham, N_bound_states)
    
    print("logging evs")
    data_file.log(evs, {'phi': phi, 'disorder': disorder, 'B': B, 'mu': mu, 's_xx': SOI[0,0], 's_xy': SOI[0,1], 's_yx': SOI[1,0], 's_yy': SOI[1,1], 'diff': diff})





if __name__ == '__main__':
    B = 0
    phi = np.pi
    disorder = 0
    SOI = np.array([[0, alpha], [-alpha, 0]])

    phi_vals = np.linspace(-1,1,51) * np.pi
    problems = []
    
    for phi in phi_vals:
                    problems.append({
                        'mu': mu, 'phi': phi, 'B': B,
                        'soi': SOI,
                        'disorder': disorder,
                    })

                    
    with multiprocessing.Pool(num_cores) as p:
        p.map(calc, problems)
    

    

