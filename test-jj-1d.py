#!/usr/bin/env python3

import jj_kwant.spectrum
import scipy.constants as const
import numpy as np
import jj_kwant.data
import time

gap = 0
mass =  0.03 * const.m_e
args = {
    'mass': mass,
    'gap': gap,
    'electrode_length': 1e-6,
    'junction_length': -1e-9,
    'a': 1e-9,
    'g': -10
    }

data_file = jj_kwant.data.datafile(folder="test_1d", params=['ky', 'B' ], args=args)


mu = -10e-3 * const.e

phi = np.pi
B = 0
alpha = 20e-3 * const.e * 1e-9 # 20 meV nm
potential = 0
disorder = 0


#for B in np.linspace(0, B_max, 100):
print("disorder / mu: ", disorder / mu)
#kf_m = -mass * alpha / const.hbar**2 - \
#    1/const.hbar * np.sqrt(mass**2 * alpha**2 / const.hbar**2 + 2 * mass * mu)
#kf_p = -mass * alpha / const.hbar**2 + \
#    1/const.hbar * np.sqrt(mass**2 * alpha**2 / const.hbar**2 + 2 * mass * mu)
#print("\n\n--------------------------")
#print("kf_m: %g" % kf_m)
#print("kf_p: %g" % (-kf_p))
n = 0

for ky in np.linspace(-4e7, 4e7, 100):
    print("n: ", n)
    print("mu = %.2g meV" % (mu * 1e3 / const.e))
    n = n + 1
    ham = jj_kwant.spectrum.hamiltonian_jj_1d(
        #debug=True,
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
        salt='')
    
    evs = jj_kwant.spectrum.positive_low_energy_spectrum(ham, 50)
    
    data_file.log(evs, {'ky': ky, 'B': B})





