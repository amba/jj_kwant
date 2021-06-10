#!/usr/bin/env python3

import jj_kwant.spectrum
import scipy.constants as const
import numpy as np
import jj_kwant.data
import time
import multiprocessing

gap = 100e-6 * const.e

#rashba = 20e-3 * const.e * 1e-9 # 20 meV nm

#junction_island_width = 50e-9
#junction_island_spacing = 100e-9
args = {
    'mass': 0.02 * const.m_e,
    'gap': gap,
 #   'rashba': rashba,
    'width': 6e-6,
    'junction_length': 100e-9,
    'electrode_length': 1e-6,
    'a': 5e-9
}


data_file = jj_kwant.data.datafile(folder="topo_gap_jj", params=['phi', 'potential', 'B', 'mu', 'alpha'], args=args)


def calc(problem):
    mu = problem['mu']
    phi = problem['phi']
    B = problem['B']
    alpha = problem['alpha']
    potential = 0
    
    ham = jj_kwant.spectrum.hamiltonian_jj_2d(
        a = args['a'],
        m = args['mass'],
        gap_potential = potential,
        gap_potential_shape='cosine',
        mu = mu,
        gap = args['gap'],
        alpha_rashba = alpha,
        width = args['width'],
        junction_length=args['junction_length'],
        electrode_length=args['electrode_length'],
        B = [0, B, 0],
        delta_phi = phi,
            #junction_island_width = junction_island_width,
            #junction_island_spacing = junction_island_spacing,
            # debug=True
    );
     
    evs = jj_kwant.spectrum.positive_low_energy_spectrum(ham, 10)
        
    
    data_file.log(evs, {'phi': phi, 'potential': potential, 'B': B, 'mu': mu, 'alpha': alpha})


num_cores = 40


if __name__ == '__main__':
#    phi_vals = (np.pi, )
    B = 0.5
    phi = np.pi
    mu_vals = np.linspace(-1e-3, 15e-3, 100) * const.e
    alpha_vals = np.linspace(0, 100e-3* const.e * 1e-9, 10) # meV nm
#    potential_vals = np.linspace(0,0.5*mu,20)
    problems = []
    for mu in mu_vals:
        for alpha in alpha_vals:
            problems.append({'mu': mu, 'phi': phi, 'B': B, 'alpha': alpha})
        
    with multiprocessing.Pool(num_cores) as p:
        p.map(calc, problems)
    

    

