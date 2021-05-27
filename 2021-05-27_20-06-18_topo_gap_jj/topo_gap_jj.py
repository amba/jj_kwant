#!/usr/bin/env python3

import jj_kwant.spectrum
import scipy.constants as const
import numpy as np
import jj_kwant.data
import time


gap = 100e-6 * const.e


args = {
    'mass': 0.02 * const.m_e,
    'gap': gap,
    'mu': 5e-3 * const.e,
    'rashba': 20e-3 * const.e * 1e-9, # 20 meV nm
    'width': 1.5e-6,
    'junction_length': 100e-9,
    'electrode_length': 1e-6,
    'disorder': 0 * const.e,
    'gap_disorder':  0 * const.e,
    'a': 5e-9,
    'phi': np.pi,
    }


data_file = jj_kwant.data.datafile(folder="topo_gap_jj", params=['phi', 'B', 'disorder'], args=args)


disorder = args['disorder']

for B in np.linspace(0,1.6,200):
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
        disorder = args['disorder'],     
        B = [0, B, 0],
        delta_phi = phi,
    );
 
    evs = jj_kwant.spectrum.positive_low_energy_spectrum(ham, 3)

    data_file.log(evs, {'phi': phi, 'B': B, 'disorder': disorder})

    

