#!/usr/bin/env python3

import jj_kwant.spectrum
import scipy.constants as const
import numpy as np
import jj_kwant.data
import time

gap = 100e-6 * const.e
mass =  0.03 * const.m_e
mu = 100e-3 * const.e
args = {
    'mass': mass,
    'gap': gap,
    'mu': mu,
    'rashba': 20e-3 * const.e * 1e-9, # 50 meV nm = 0.5 eV A
    'electrode_length': 15e-6,
    'junction_length': 100e-9,
    'a': 1e-9,
    'g': -10
    }

data_file = jj_kwant.data.datafile(folder="jj_1d", params=['ky', 'phi', 'B', 'mu'], args=args)



mu = args['mu']
n = 0
k_fermi = 1/const.hbar * np.sqrt(2 * mass * mu)
phi = np.pi
for B in np.linspace(0, 2, 100):
    print("\n\n--------------------------")
    n = 0
    for ky in np.linspace(-1.05*k_fermi, -0.9*k_fermi, 1000):
        print("n: ", n)
        print("B = %.2g" % B)
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
            alpha_rashba=args['rashba'],
            salt='')
    
        evs = jj_kwant.spectrum.positive_low_energy_spectrum(ham, 1)
    
        data_file.log(evs, {'ky': ky, 'phi': phi, 'B': B, 'mu': mu})





