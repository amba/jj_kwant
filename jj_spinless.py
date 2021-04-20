#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import kwant

import scipy.sparse.linalg
import scipy.linalg

import tinyarray

import scipy.constants as const


sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j,0]])
sigma_z = tinyarray.array([[1, 0], [0,-1]])


def make_syst(m = 0.03 * const.m_e, a=10e-9, W=10, L=10, junction_length=-0.1e-9):
    t = const.hbar**2 / (2 * m * a**2)
    print(t)
    
    lat = kwant.lattice.square(a, norbs=2)
    

    syst = kwant.Builder()

    #
    # H_0
    #
    
    # On-site
    def onsite(site, mu, Gap, delta_phi):
        (x, y) = site.pos
        pot = sigma_z * (4*t - mu)
        if x > (a*L)/2:
            delta_phi = -delta_phi
        if x < (a*L - junction_length) / 2 or x > (a*L + junction_length)/2:
            pot = pot + Gap * (sigma_x * np.cos(delta_phi/2) - sigma_y * np.sin(delta_phi/2))
        return pot

    
    syst[(lat(x,y) for x in range(L) for y in range(W))] = onsite
    # Hoppings
    
    syst[lat.neighbors()] = -t * sigma_z
    
    #  syst[kwant.builder.HoppingKind((0,1), lat, lat)] = -t * sigma_0
    # kwant.plot(syst)
    return syst.finalized()

W = 10
n_phi = 50
n_s = 1e16
m_eff = 0.03 * const.m_e
E_fermi = const.hbar**2 * 2 * np.pi * n_s / (2 * m_eff)

#for L in (100, 200, 300, 400):
for L in (100,):
    L = L+1
    mu = E_fermi
    print("E_fermi = %.3g (meV)" % (1000 * mu / const.e))
    Gap = 0.2e-3 * const.e
    junction_length = 100e-9

    syst = make_syst(L=L, W=W, junction_length=junction_length)

    phi_vals = np.linspace(-0.1*np.pi, 1.1*np.pi, n_phi)
    energies = []
    free_energies = []
    for phi in phi_vals:
        print(phi/np.pi)
        ham_mat = syst.hamiltonian_submatrix(params=dict(Gap=Gap, mu=mu, delta_phi=phi), sparse=True)
        ham_mat = ham_mat.tocsc()

        k=50
        evs = scipy.sparse.linalg.eigsh(ham_mat, k=k, sigma=0, which='LA', return_eigenvectors=False)
        evs = evs / Gap
        energies.append(evs)
        free_energies.append(-np.sum(evs))
    
    free_energies = np.array(free_energies)
    current = np.gradient(free_energies)
    plt.grid()
    plt.xlabel('φ/π')
    plt.plot(phi_vals/np.pi, energies)
    plt.show()
    plt.plot(phi_vals/np.pi, current, label="W = %d, L = %d, Δ = %.3g meV, μ = %.3g meV" % (W, L, 1000 * Gap / const.e, 1000 * mu / const.e))
    plt.legend()
    plt.show()

# plt.ylabel('E/Δ')
# plt.xlabel('φ/π')
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
