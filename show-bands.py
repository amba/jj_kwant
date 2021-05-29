#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import tinyarray
import scipy.linalg

const_hbar = 1.0545718176461565e-34
const_e = 1.602176634e-19
const_m_e = 9.1093837015e-31
const_bohr_magneton = 9.274009e-24 # J/T


sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j,0]])
sigma_z = tinyarray.array([[1, 0], [0,-1]])

def H_rashba(kx=None, ky=None, m=None, μ=None, alpha=None, g=None, B=[0,0,0]):
    H = 1 / (2*m) * const_hbar**2 * (kx**2 + ky**2) * sigma_0 +\
        alpha * (ky * sigma_x - kx * sigma_y) + \
        1/2 * g * const_bohr_magneton * (\
        B[0]*sigma_x +\
        B[1]*sigma_y + \
        B[2]*sigma_z)
    return H


m = 0.03 * const_m_e
μ = 100e-3 * const_e
g = -10
alpha = 20e-3 * const_e * 1e-9 # 10 meV nm
B_x = 1

k_max = 1 / const_hbar * np.sqrt(2 * m * μ)
print("lambda_min = %.2g" % (2*np.pi / k_max))

kyvals = np.linspace(-k_max, k_max, 10000)

kxvals = np.linspace(-0.5 * k_max, 0.5*k_max, 5)
for kx in kxvals:
    evals_p = []
    evals_m = []
    for ky in (kyvals):
        H = H_rashba(kx=kx, ky=ky,m=m, μ=μ, alpha=alpha, g=g, B=[B_x,0,0])
        evals = scipy.linalg.eigvalsh(H)
        evals_p.append(evals[0])
        evals_m.append(evals[1])
    
    evals_p = np.array(evals_p)
    evals_m = np.array(evals_m)
    
    plt.plot(kyvals*1e-9, evals_p / const_e * 1e3, label="kx=%.2g" % kx)
    plt.plot(kyvals*1e-9, evals_m / const_e * 1e3)
plt.ylabel('E (meV)')
plt.xlabel('k (1/nm)')
plt.grid()



plt.show(block=False)
import code
code.interact()
