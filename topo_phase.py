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
    'width': 3e-6,
    'junction_length': 100e-9,
    'electrode_length': 100e-9,
    'a': 5e-9
}


data_file = jj_kwant.data.datafile(folder="topo_gap_jj", params=['phi', 'potential', 'B', 'mu', 'theta'], args=args)

alpha = 20e-3 * const.e * 1e-9

def calc(problem):
    mu = problem['mu']
    phi = problem['phi']
    B = problem['B']
    theta = problem['theta']
    
    potential = 0
    SOI = np.array([[0, np.cos(theta)], [np.sin(theta), 0]]) * alpha    
    ham = jj_kwant.spectrum.hamiltonian_jj_2d(
        a = args['a'],
        m = args['mass'],
        gap_potential = potential,
        gap_potential_shape='cosine',
        mu = mu,
        gap = args['gap'],
        SOI = SOI,
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
    print("evs: ", evs)
    print("logging evs")    
    data_file.log(evs, {'phi': phi, 'potential': potential, 'B': B, 'mu': mu, 'theta': theta})


num_cores = 10


if __name__ == '__main__':
#    phi_vals = (np.pi, )
    B = 0.5
    phi = np.pi
    mu_vals = np.linspace(-1e-3, 15e-3, 100) * const.e
    theta_vals = np.linspace(0,np.pi, 30)
    alpha = 20e-3 *const.e * 1e-9
    
#    potential_vals = np.linspace(0,0.5*mu,20)
    problems = []
    for mu in mu_vals:
        for theta  in theta_vals:

            problems.append({'mu': mu, 'phi': phi, 'B': B, 'theta': theta})
        
    with multiprocessing.Pool(num_cores) as p:
        p.map(calc, problems)
    

    

