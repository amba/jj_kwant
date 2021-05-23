#!/usr/bin/env python3

import jj_kwant.spectrum
import scipy.constants as const

import jj_kwant.data

data_file = jj_kwant.data.datafile(params=['phi', 'B'])
print(data_file.__class__)



gap = 200e-6 * const.e
ham = jj_kwant.spectrum.hamiltonian_sparse_csc(width=50e-9, junction_length=100e-9, electrode_length=1e-6, gap=gap, B=[0,0.2, 0], delta_phi = 0, mu = 70e-3 * const.e)

print(ham.shape)



evs = jj_kwant.spectrum.low_energy_spectrum(ham, 10)
print("low energy evals / gap: ", evs / gap)

data_file.log(evs, {'phi': 0, 'B': 1})



