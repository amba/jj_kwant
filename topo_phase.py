#!/usr/bin/env python3

import jj_kwant.spectrum
import scipy.constants as const
import numpy as np
import jj_kwant.data
import time
import multiprocessing

gap = 100e-6 * const.e

rashba = 0# 20e-3 * const.e * 1e-9 # 20 meV nm
mu = 100e-3 * const.e # 100 meV 
#junction_island_width = 50e-9
#junction_island_spacing = 100e-9
args = {
    'mass': 0.02 * const.m_e,
    'gap': gap,
    'mu': mu,
    'rashba': rashba,
    'width': 1e-6,
    'junction_length': 100e-9,
    'electrode_length': 3e-6,
    'a': 5e-9
}


data_file = jj_kwant.data.datafile(folder="topo_gap_jj", params=['phi', 'potential', 'B'], args=args)


potential = 0.1 * mu

#B = 0.25




def calc(problem):
    phi = problem['phi']
    B = problem['B']
    
    ham = jj_kwant.spectrum.hamiltonian_jj_2d(
        a = args['a'],
        m = args['mass'],
        gap_potential = potential,
        gap_potential_shape='cosine',
        mu = args['mu'],
        gap = args['gap'],
        alpha_rashba = args['rashba'],
        width = args['width'],
        junction_length=args['junction_length'],
        electrode_length=args['electrode_length'],
        B = [0, B, 0],
        delta_phi = phi,
            #junction_island_width = junction_island_width,
            #junction_island_spacing = junction_island_spacing,
            # debug=True
    );
     
    evs = jj_kwant.spectrum.positive_low_energy_spectrum(ham, 2)
        
    
    data_file.log(evs, {'phi': phi, 'potential': potential, 'B': B})


num_cores = 50


if __name__ == '__main__':
    phi_vals = np.linspace(0,2*np.pi,100)
    B_vals = np.linspace(0,1,20)
#    potential_vals = np.linspace(0,0.5*mu,20)
    problems = []
    for B in B_vals:
        for phi in phi_vals:
            problems.append({'phi': phi, 'B': B})
        
    with multiprocessing.Pool(num_cores) as p:
        p.map(calc, problems)
    

    

