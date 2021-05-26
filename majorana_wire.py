#!/usr/bin/env python3

import jj_kwant.spectrum
import scipy.constants as const
import numpy as np
import jj_kwant.data
import time

gap = 0.3e-3 * const.e

args = {
    'mass': 0.015 * const.m_e,
    'gap': gap,
    'mu': 1e-3 * const.e,
    'rashba': 50e-3 * const.e * 1e-9, # 50 meV nm = 0.5 eV A
    'length': 2e-6,
    'a': 1e-9
    }

data_file = jj_kwant.data.datafile(folder="majorana_1d", params=['phi', 'B', 'disorder', 'gap_disorder'], args=args)

disorder = 5 * args['mu']
gap_disorder =  0

B = 15

for B in np.linspace(0, B, 200):
    print("B: ", B)
    print("gap disorder: ", gap_disorder / gap)
    print("disorder: ", disorder / const.e)
    
    ham = jj_kwant.spectrum.hamiltonian_jj_1d(a=args['a'], m=args['mass'], junction_length=-1e-9, electrode_length=args['length'], gap=args['gap'], B=[B,0,0], delta_phi = 0, mu=args['mu'], alpha_rashba=args['rashba'], disorder=disorder, gap_disorder=gap_disorder, salt='')# str(time.time()))

    evs = jj_kwant.spectrum.positive_low_energy_spectrum(ham, 10)

    data_file.log(evs, {'phi': 0, 'B': B, 'disorder': disorder, 'gap_disorder': gap_disorder})





