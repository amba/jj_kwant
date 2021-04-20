#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import kwant
import scipy.sparse.linalg as sla

import tinyarray

sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j,0]])
sigma_z = tinyarray.array([[1, 0], [0,-1]])


def make_syst(t=1, a=1, r=10, e_z=0):
    lat = kwant.lattice.square(a, norbs=1)

    syst = kwant.Builder()

    def circle(pos):
        (x, y) = pos
        rsq = x**2 + y**2
        return rsq < r ** 2

    def hopx(site1, site2, B, e_z):
        y = site1.pos[1]
        return -t * np.exp(-1j * B * y) * sigma_0
    def onsite(site, B, e_z):
        return 4 * t * sigma_0 + sigma_z * e_z
    syst[lat.shape(circle, (0,0))] = onsite
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopx
    syst[kwant.builder.HoppingKind((0,1), lat, lat)] = -t * sigma_0
    kwant.plot(syst)
    return syst.finalized()


def plot_spectrum(syst, Bfields):
    energies = []
    for B in Bfields:
        ham_mat = syst.hamiltonian_submatrix(params=dict(B=B, e_z=0.05*B), sparse='True')
        print("B = %.3g" % B)
        ev = sla.eigsh(ham_mat.tocsc(), k=24, sigma=0, return_eigenvectors=False)
        energies.append(ev)

    plt.plot(Bfields, energies, 'x', markersize=1)
    plt.savefig("quantum_dot.pdf")
    plt.show(block=False)
    import code
    code.interact()

Bfields = np.linspace(0, 0.02, 500)
syst = make_syst(r=10)
plot_spectrum(syst, Bfields)
