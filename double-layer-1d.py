#!/usr/bin/env python3
import kwant
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
import tinyarray
import numpy as np



const_hbar = 1.0545718176461565e-34
const_e = 1.602176634e-19
const_m_e = 9.1093837015e-31
const_bohr_magneton = 9.274009e-24 # J/T

a = 5e-9
m = 0.04 * const_m_e
t = const_hbar**2 / (2 * m * a**2)
delta = 250e-6 * const_e
t_prime = 1000e-6* const_e
print("t / t_prime = ", t/t_prime)
W = 2
L = 10000
L_junction = 20
mu = 50e-3 * const_e

sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j,0]])
sigma_z = tinyarray.array([[1, 0], [0,-1]])

tau_0 = tinyarray.array([[1, 0], [0, 1]])
tau_x = tinyarray.array([[0, 1], [1, 0]])
tau_y = tinyarray.array([[0, -1j], [1j,0]])
tau_z = tinyarray.array([[1, 0], [0,-1]])

# Define the scattering region


def make_system(
        L=1000,
        delta_phi=None):
    syst = kwant.Builder()
    lat = kwant.lattice.square(1)

    def onsite(site):
        x, y = site.pos
        pairing = 0
        h0 = 2*t -mu
        dphi = delta_phi
        if x > L/2:
            dphi = -dphi
            #print("mu = ", mu)
        if y == 1:
            pairing = tau_x * np.cos(dphi/2) - tau_y * np.sin(dphi/2)
            pairing *= delta
        
        return tau_z * h0 + pairing

        
        
    syst[(lat(x,y) for x in range(L) for y in range(W))] = onsite

    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = -t * tau_z 

    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = -t_prime * tau_z#



    def in_jj(site):
        x, y = site.pos
        return abs(x-L/2) < L_junction/2 and y == 1

    for site in filter(in_jj, list(syst.sites())):
        del syst[site]
    sites = syst.sites()
    syst = syst.finalized()
    ham_mat = syst.hamiltonian_submatrix(sparse=True)
    ham_mat = ham_mat.tocsc()
    return ham_mat


# sites, ham_mat = make_system(delta_phi=0*np.pi)
# positions = [site.pos for site in sites]
# x_positions = np.array([pos[0] for pos in positions])

# print(positions)
# evals,evec = ssl.eigsh(ham_mat, k=1, sigma=0, which='LA', return_eigenvectors=True)
# evec = evec[:,0]
# print(evec.shape)
# wf = np.abs(evec[0::2]) + np.abs(evec[1::2])

# # print(wf.shape)
# #plt.yscale('log')
# plt.plot(x_positions, wf)
# plt.show()
delta_phi_vals = np.linspace(0, 2*np.pi, 100)
spectrum = []
for delta_phi in delta_phi_vals:
    print("delta_phi = ", delta_phi / np.pi)
    ham_mat = make_system(delta_phi=delta_phi)
    evals = ssl.eigsh(ham_mat, k=4, sigma=0, which='LA', return_eigenvectors=False)
    spectrum.append(evals)

spectrum = np.array(spectrum)
print(spectrum.shape)
plt.plot(delta_phi_vals/np.pi, spectrum/const_e * 1e6)
plt.xlabel('φ (π)')
plt.ylabel('E (μeV)*')
plt.show()

