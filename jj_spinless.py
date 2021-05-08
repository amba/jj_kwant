#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import kwant

import scipy.sparse.linalg
import scipy.linalg

import tinyarray
from datetime import datetime
import scipy.constants as const
import pathlib
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--lattice-constant', default=5e-9, help="lattice constant", type=float)
parser.add_argument('--width', help="width of structure in m", required=True, type=float)
parser.add_argument('--junction-length', help="junction length in m", required=True, type=float)
parser.add_argument('--electrode-length', help="electrode length in m", required=True, type=float)
parser.add_argument('--carrier-density', help="carrier density / m^2 (default: 1e16)", default=1e16, type=float)
parser.add_argument('--mass', help="effective mass / m_e(default: 0.03)", default=0.03, type=float)
parser.add_argument('--gap', help="superconducting gap / eV (default: 200e-6)", default=200e-6, type=float)
parser.add_argument('--n-phi', help="number of values of phase difference (default: 20)", default=20, type=int)

args = parser.parse_args()
a = args.lattice_constant

# width of josephson junction
width = args.width


electrode_length = args.electrode_length

junction_length = args.junction_length

gap = args.gap * const.e
n_s = args.carrier_density

m_eff = args.mass * const.m_e

n_phi = args.n_phi


E_fermi = const.hbar**2 * 2 * np.pi * n_s / (2 * m_eff)
k_fermi = np.sqrt(2 * m_eff * E_fermi) / const.hbar
lambda_fermi = 2 * np.pi / k_fermi
mu = E_fermi
n_bound_states = 2 * width / lambda_fermi # with spin: 4 * width / lambda_fermi

sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j,0]])
sigma_z = tinyarray.array([[1, 0], [0,-1]])

date_string = datetime.now()
date_string = date_string.strftime("%Y-%m-%d_%H-%M-%S")
data_folder = date_string + "_JJ_width=%g_junction_length=%g_electrode_length=%g" % (width, junction_length, electrode_length)

print("data folder: ", data_folder)
pathlib.Path(data_folder).mkdir()

    

def make_syst(m = 0.03 * const.m_e, a=5e-9, width=3e-6,
              electrode_length = 3e-6, junction_length=100e-9,
              mu=0, gap=0, delta_phi=0):
    t = const.hbar**2 / (2 * m * a**2)
    print("m = %g, a = %g, width = %g, electrode_length = %g, junction_length = %g, t = %g" % (m, a, width, electrode_length, junction_length, t))
    W = int(width/a)
    L = int((2*electrode_length + junction_length) / a)
    L_junction = int(junction_length/a)
    print("L = %d, W = %d, L_junction = %d" % (L, W, L_junction))
    lat = kwant.lattice.square(1)
    

    syst = kwant.Builder()

    #
    # H_0
    #
    
    # On-site
    def onsite(site):
        (x, y) = site.pos
        pot = sigma_z * (4*t - mu)
        dphi = delta_phi
        
        if x > L/2:
            dphi = -dphi

        start_junction = int((L - L_junction) / 2)
        if x < start_junction or x >= start_junction + L_junction:
            pot = pot + gap * (sigma_x * np.cos(dphi/2) - sigma_y * np.sin(dphi/2))
        return pot
    
    syst[(lat(x,y) for x in range(L) for y in range(W))] = onsite

    
    # Hoppings
    syst[lat.neighbors()] = -t * sigma_z
    
    # # Need periodic boundaries x=L-1 <-> x = 0
    # for j in range(W):
    #     syst[lat(0,j), lat(L-1,j)] = -t * sigma_z
    def onsite_00(site):
        return np.abs(onsite(site)[0,0])
    def onsite_01(site):
        return np.abs(onsite(site)[1, 0])
    
#    kwant.plot(syst,site_color=onsite_00)
 #   kwant.plot(syst,site_color=onsite_01)
    
    return syst.finalized()

def make_hamiltonian_sparse_csc(*args, **kwargs):
    syst = make_syst(*args, **kwargs)
    ham_mat = syst.hamiltonian_submatrix(sparse=True)
    print("Hamiltonian shape: ", ham_mat.shape)
    ham_mat = ham_mat.tocsc()
    return ham_mat

print("E_fermi = %.3g meV, λ_fermi = %.3g nm, N0 = %.2g" % (1000 * mu / const.e, lambda_fermi * 1e9, n_bound_states))



phi_vals = np.linspace(-0.1*np.pi, 1.1*np.pi, n_phi)
energies = []
free_energies = []
for phi in phi_vals:
    print("phi = %.3g π" % (phi/np.pi))
    print("make syst")
    t0 = time.time()

    ham_mat = make_hamiltonian_sparse_csc(a=5e-9, width=width, electrode_length=electrode_length, junction_length=junction_length, mu=mu, gap=gap, delta_phi=phi)
    t1 = time.time()
    print("time for make_hamiltonian: ", t1 - t0)
    
    k= 2 * int(n_bound_states)
    print("calculating %d eigenvalues" % k)
    evs = scipy.sparse.linalg.eigsh(ham_mat, k=k, sigma=0, which='LA', return_eigenvectors=False)
    evs = evs
    energies.append(evs)
    free_energies.append(-np.sum(evs))
    print("execution time: %.1f s" % (time.time() - t1))

free_energies = 2 * np.array(free_energies) # include 2 for spin
energies = np.array(energies)

current = 2 * const.e / const.hbar * np.gradient(free_energies)

current_modes = current * const.hbar / (const.e * gap)

    
plt.grid()
plt.xlabel('phi/π')
plt.plot(phi_vals/np.pi, energies/gap)
plt.savefig(data_folder + '/energies.pdf')

plt.clf()

plt.xlabel('phi/π')
plt.grid()
plt.ylabel('current (mu A)')

plt.plot(phi_vals/np.pi, current*1e6, label="width = %g, junction_length = %g, Δ = %.3g meV, fermi_energy = %.3g meV" % (width, junction_length, 1000 * gap / const.e, 1000 * mu / const.e))
plt.legend()
plt.savefig(data_folder + '/current-phase.pdf')

data_block = np.array([phi_vals, free_energies, current, current_modes]).T
data_block_header = "phi\t\tfree-energy\t\tcurrent\t\tmodes"
np.savetxt(data_folder + '/current-phase.txt', data_block, fmt="%.17g",
           header=data_block_header, delimiter="\t\t", footer="\n")


# plt.ylabel('E/Δ')
# plt.legend()

# plt.plot(phi_vals/np.pi, energies)
# plt.show(block=False)
# dense_ham = ham_mat.todense()

# print(dense_ham.__class__, dense_ham.shape)
# dense_ev = scipy.linalg.eigvalsh(dense_ham)

# n_dense_ev = dense_ev.shape[0]
# print(np.sort(dense_ev)[int(n_dense_ev/2-10):int(n_dense_ev/2 + 10)])

# plt.plot(np.zeros((n_ev)), ev, 'x', markersize=1)
# plt.grid()
# plt.show(block=False)
# import code
# code.interact()
