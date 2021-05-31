#!/usr/bin/env python3

import jj_kwant.spectrum
import scipy.constants as const
import numpy as np
import jj_kwant.data
import time

gap = 100e-6 * const.e
mass =  0.03 * const.m_e
alpha_max = 50e-3 * const.e * 1e-9 # 50 meV nm = 0.5 eV A
args = {
    'mass': mass,
    'gap': gap,
    'electrode_length': 15e-6,
    'junction_length': 100e-9,
    'a': 5e-9,
    'g': -10
    }

data_file = jj_kwant.data.datafile(folder="jj_1d", params=['ky', 'alpha', 'phi', 'B', 'mu'], args=args)


mu_max = 100e-3 * const.e
phi = np.pi
B = 0.2
#mu = 0.5 * mu_max
alpha = 10e-3 * const.e * 1e-9

for mu in np.linspace(0, mu_max, 100):
    kf_m = -mass * alpha / const.hbar**2 - \
        1/const.hbar * np.sqrt(mass**2 * alpha**2 / const.hbar**2 + 2 * mass * mu)
    kf_p = -mass * alpha / const.hbar**2 + \
        1/const.hbar * np.sqrt(mass**2 * alpha**2 / const.hbar**2 + 2 * mass * mu)
    print("\n\n--------------------------")
    print("kf_m: %g" % kf_m)
    print("kf_p: %g" % (-kf_p))
    n = 0
    for ky in np.linspace(1.1*kf_m, -0.9*kf_p, 1000):
        print("n: ", n)
        print("mu = %.2g meV" % (mu * 1e3 / const.e))
        n = n + 1
        ham = jj_kwant.spectrum.hamiltonian_jj_1d(
            ky=ky,
            a=args['a'],
            m=mass,
            junction_length=args['junction_length'],
            electrode_length=args['electrode_length'],
            gap=gap,
            B=[0,B,0],
            delta_phi = phi,
            g_factor=args['g'],
            mu=mu,
            alpha_rashba=alpha,
            salt='')
    
        evs = jj_kwant.spectrum.positive_low_energy_spectrum(ham, 1)
    
        data_file.log(evs, {'ky': ky, 'phi': phi, 'B': B, 'mu': mu, 'alpha': alpha})





