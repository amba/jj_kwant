#!/usr/bin/env python3

import jj_kwant.spectrum
import scipy.constants as const
import numpy as np
import jj_kwant.data
import time

gap = 100e-6 * const.e

args = {
    'gap': gap,
    'mu': 0,
    'rashba': 20e-3 * const.e * 1e-9, # 10 meV nm
    'length': 2e-6,
    }

data_file = jj_kwant.data.datafile(folder="majorana_1d", params=['phi', 'B', 'disorder', 'gap_disorder'], args=args)

disorder = 2.5e-3 * const.e
gap_disorder =  0

B = 1.5

for B in np.linspace(0, B,100):
    print("B: ", B)
    print("gap disorder: ", gap_disorder / gap)
    print("disorder: ", disorder / const.e)
    
    ham = jj_kwant.spectrum.hamiltonian_jj_1d(junction_length=-1e-9, electrode_length=args['length'], gap=args['gap'], B=[B,0,0], delta_phi = 0, mu=args['mu'], alpha_rashba=args['rashba'], disorder=disorder, gap_disorder=gap_disorder, salt=str(time.time()))

    evs = jj_kwant.spectrum.low_energy_spectrum(ham, 6)

    data_file.log(evs, {'phi': 0, 'B': B, 'disorder': disorder, 'gap_disorder': gap_disorder})





