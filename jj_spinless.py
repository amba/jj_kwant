#!/usr/bin/env python3

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

# lattice constant for discretization
a = 5e-9

# width of josephson junction
width = 3e-6


electrode_length = 3e-6

junction_length = 100e-9


print("width = %.3g μm")

Gap = 200e-6 * const.e

n_phi = 20
n_s = 1e16
m_eff = 0.03 * const.m_e
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
              mu=0, Gap=0, delta_phi=0):
    t = const.hbar**2 / (2 * m * a**2)
    print(f"m = {m}, a = {a}, width = {width}, electrode_length = {electrode_length}, junction_length = {junction_length}, t = {t}")
    W = int(width/a)
    L = int((2*electrode_length + junction_length) / a)
    L_junction = int(junction_length/a)
    print(f"L = {L}, W = {W}, L_junction = {L_junction}")
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

        start_junction = int(L_junction - junction_length) / 2
        if x < start_junction or x >= start_junction + junction_length:
            pot = pot + Gap * (sigma_x * np.cos(dphi/2) - sigma_y * np.sin(dphi/2))
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
    
    # kwant.plot(syst,site_color=onsite_00)
    # kwant.plot(syst,site_color=onsite_01)
    
    return syst.finalized()


print("E_fermi = %.3g meV, λ_fermi = %.3g nm, N0 = %.2g" % (1000 * mu / const.e, lambda_fermi * 1e9, n_bound_states))



phi_vals = np.linspace(-0.1*np.pi, 1.1*np.pi, n_phi)
energies = []
free_energies = []
gaps = np.linspace(0, Gap, 5)
for Gap in gaps:
    phi = 0
    print("phi = %.3g π" % (phi/np.pi))
    print("make syst")
    t0 = time.time()
    syst = make_syst(a=5e-9, width=width, electrode_length=electrode_length, junction_length=junction_length, mu=mu, Gap=Gap, delta_phi=phi)
    t1 = time.time()
    print("time for make_syst: %.3g" % (t1 - t0))
    ham_mat = syst.hamiltonian_submatrix(sparse=True)
    ham_mat = ham_mat.tocsc()

    k= int(n_bound_states)
    evs = scipy.sparse.linalg.eigsh(ham_mat, k=k, sigma=0, which='LA', return_eigenvectors=False)
    evs = evs
    energies.append(evs)
    free_energies.append(-np.sum(evs))
    print("execution time: %.1f s" % (time.time() - t1))

free_energies = 2 * np.array(free_energies) # include 2 for spin
energies = np.array(energies)

plt.plot(gaps / Gap , energies/Gap)
plt.show()

current = 2 * const.e / const.hbar * np.gradient(free_energies)

current_modes = current * const.hbar / (const.e * Gap)

    
plt.grid()
plt.xlabel('φ/π')
plt.plot(phi_vals/np.pi, energies/Gap)
plt.savefig(data_folder + '/energies.pdf')

plt.clf()

plt.xlabel('φ/π')
plt.grid()
plt.ylabel('current (μA)')



plt.plot(phi_vals/np.pi, current*1e6, label="W = %d, L = %d, Δ = %.3g meV, μ = %.3g meV" % (W, L, 1000 * Gap / const.e, 1000 * mu / const.e))
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
