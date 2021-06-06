#!/usr/bin/env python3

import jj_kwant.spectrum
import scipy.constants as const
import numpy as np
import jj_kwant.data
import time
import multiprocessing

gap = 100e-6 * const.e
mass =  0.03 * const.m_e
args = {
    'mass': mass,
    'gap': gap,
    'electrode_length': 4e-6,
    'junction_length': 100e-9,
    'a': 1e-9,
    'g': -10
    }

data_file = jj_kwant.data.datafile(folder="jj_1d", params=['ky', 'B' ], args=args)


mu = 100e-3 * const.e

phi = np.pi
B_max = 1
alpha = 20e-3 * const.e * 1e-9 # 20 meV nm
potential = 0
disorder = 0

def calc(problem):
    B = problem['B']
    print("B: ", B)

    kf_m = -mass * alpha / const.hbar**2 - \
        1/const.hbar * np.sqrt(mass**2 * alpha**2 / const.hbar**2 + 2 * mass * mu)
    kf_p = -mass * alpha / const.hbar**2 + \
        1/const.hbar * np.sqrt(mass**2 * alpha**2 / const.hbar**2 + 2 * mass * mu)
    n = 0
    for ky in np.linspace(-3e8, 0, 200):
        salt = str(time.time()) # new disorder for each disorder strength
        print("n: ", n)
        print("mu = %.2g meV" % (mu * 1e3 / const.e))
        n = n + 1
        ham = jj_kwant.spectrum.hamiltonian_jj_1d(
           # debug=True,
            ky=ky,
            a=args['a'],
            m=mass,
            junction_length=args['junction_length'],
            electrode_length=args['electrode_length'],
            gap=gap,
            B=[0,B,0],
            delta_phi = phi,
            g_factor=args['g'],
            disorder=disorder,
            gap_potential = potential,
            gap_potential_shape='cosine',
            mu=mu,
            alpha_rashba=alpha,
            salt=salt)
    
        evs = jj_kwant.spectrum.positive_low_energy_spectrum(ham, 3)
    
        data_file.log(evs, {'ky': ky, 'B': B})


num_cores = 100


if __name__ == '__main__':
    B_vals = np.linspace(0,1,100)
#    potential_vals = np.linspace(0,0.5*mu,20)
    problems = []
    for B in B_vals:
        problems.append({'B': B})
        
    with multiprocessing.Pool(num_cores) as p:
        p.map(calc, problems)





