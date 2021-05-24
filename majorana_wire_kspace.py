#!/usr/bin/env python3

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import tinyarray

sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j,0]])
sigma_z = tinyarray.array([[1, 0], [0,-1]])

tau_0 = tinyarray.array([[1, 0], [0, 1]])
tau_x = tinyarray.array([[0, 1], [1, 0]])
tau_y = tinyarray.array([[0, -1j], [1j,0]])
tau_z = tinyarray.array([[1, 0], [0,-1]])


# parameters:

# H = (p^2 + u * p * σ_x - μ)τ_z -Bσ_z + Δτ_x

def hamiltonian(p, u, μ, B, Δ):
    h_0 = (p**2 - μ) * sigma_0 + u * p * sigma_x
    h_zeeman = - B * sigma_z

    h = np.kron(tau_z, h_0)
    h = h + np.kron(tau_0, h_zeeman)
    h = h + np.kron(tau_x, Δ * sigma_0)
    return h


u = 0.5
μ = 0
Δ = 0.1
B = 0.3

pvals = np.linspace(-1,1,200)

    
for μ in np.linspace(-0.1,0.1,20):
    print("B = ", B)
    evs = []
    for p in pvals:
        h = hamiltonian(p, u, μ, B, Δ)
        evs.append(scipy.linalg.eigvalsh(h))
    plt.plot(pvals, evs)# label="B = %.2g" % B)

# p_f = np.sqrt(B + μ)
# delta_eff = p_f * u * Δ  / B
# print("effective gap: ", delta_eff)
plt.legend()
plt.grid()
plt.show(block=False)

import code
code.interact()
