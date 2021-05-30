#!/usr/bin/env python3

import jj_kwant.spectrum
import scipy.constants as const
import numpy as np
import jj_kwant.data
import time


gap = 100e-6 * const.e

rashba = 20e-3 * const.e * 1e-9 # 20 meV nm
mu = 100e-3 * const.e

args = {
    'mass': 0.02 * const.m_e,
    'gap': gap,
    'mu': mu,
    'rashba': rashba,
    'width': 4e-6,
    'junction_length': 100e-9,
    'electrode_length': 600e-9,
    'a': 5e-9,
    'phi': np.pi,
    }


data_file = jj_kwant.data.datafile(folder="topo_gap_jj", params=['mod_amp', 'mod_length', 'phi', 'B'], args=args)

modulation_length = 200e-9

for modulation_amplitude in np.linspace(0, mu, 20):
#for B in np.linspace(0,1.6,200):
    print("mod_amp / μ = ", modulation_amplitude / mu)
    B = 0.2
    phi = args['phi']
    ham = jj_kwant.spectrum.hamiltonian_jj_2d(
        a = args['a'],
        m = args['mass'],
        mu = args['mu'],
        gap = args['gap'],
        alpha_rashba = args['rashba'],
        width = args['width'],
        junction_length=args['junction_length'],
        electrode_length=args['electrode_length'],
        B = [0, B, 0],
        delta_phi = phi,
        modulation_amplitude = modulation_amplitude,
        modulation_length = modulation_length,
#        debug=True
    );
 
    evs = jj_kwant.spectrum.positive_low_energy_spectrum(ham, 2)
    

    data_file.log(evs, {'mod_amp': modulation_amplitude, 'mod_length': modulation_length, 'phi': phi, 'B': B})

    

