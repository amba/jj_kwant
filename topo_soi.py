#!/usr/bin/env python3

import jj_kwant.spectrum
import scipy.constants as const
import numpy as np
import jj_kwant.data
import time
import multiprocessing

gap = 100e-6 * const.e
meVnm = 1e-3 * const.e * 1e-9

#rashba = 20e-3 * const.e * 1e-9 # 20 meV nm

#junction_island_width = 50e-9
#junction_island_spacing = 100e-9
args = {
    'mass': 0.02 * const.m_e,
    'gap': gap,
 #   'rashba': rashba,
    'width': 4e-6,
    'junction_length': 100e-9,
    'electrode_length': 1e-6,
    'a': 5e-9
}


data_file = jj_kwant.data.datafile(folder="topo_gap_jj", params=['phi', 'potential', 'B', 'mu', 'soi', 'alpha', 'free_energy'], args=args)



def calc(problem):
    mu = problem['mu']
    phi = problem['phi']
    B = problem['B']
    soi = problem['soi']
    alpha = problem['alpha']
    
    potential = 0
    SOI = np.array([[0, soi], [alpha*soi, 0]])
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
     
    evs = jj_kwant.spectrum.positive_low_energy_spectrum(ham, 20)
    
    
    print("logging evs")
    free_energy = np.sum(evs)
    data_file.log(evs, {'phi': phi, 'potential': potential, 'B': B, 'mu': mu, 'soi': soi, 'free_energy': free_energy, 'alpha': alpha})


num_cores = 50


if __name__ == '__main__':
#    phi_vals = (np.pi, )
    
    
    mu_vals = (100e-3 * const.e, )

    B_vals = np.linspace(0,1.5,30)
    soi_vals = (20* meVnm,)
    alpha_vals = np.linspace(0, -1, 20)
    phi_vals = (0,)    
    problems = []
    for mu in mu_vals:
        for soi  in soi_vals:
            for B in B_vals:
                for phi in phi_vals:
                    for alpha in alpha_vals:
                        problems.append({
                            'mu': mu, 'phi': phi, 'B': B, 'soi': soi,
                            'alpha': alpha})
        
    with multiprocessing.Pool(num_cores) as p:
        p.map(calc, problems)
    

    

