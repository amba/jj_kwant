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
t_prime = 200e-6* const_e
print("t / t_prime = ", t/t_prime)
W = 2
L = 500
L_junction = 20
mu = 50e-3 * const_e
kf = np.sqrt(2*mu*m)/const_hbar
print("l_F = 2π/kf = ", 2*np.pi/kf * 1e9, " nm")

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
        delta_phi=None,
        g_factor = -10,
        B = 0,
        alpha_rashba = 15e-3 * const_e * 1e-9
):
    syst = kwant.Builder()
    lat = kwant.lattice.square(1)

    def onsite(site):
        x, y = site.pos
        pairing = 0
        h0 = 2*t -mu
        dphi = delta_phi
        zeeman = 0
        if x > L/2:
            dphi = -dphi
            #print("mu = ", mu)
        if y == 1:
            pairing = np.kron(tau_x * np.cos(dphi/2) - tau_y * np.sin(dphi/2), sigma_0)
            pairing *= delta
            zeeman = 0 * sigma_y
        else:
            zeeman =  0.5 * g_factor * const_bohr_magneton * B * sigma_y
            
        
        h0 = h0 * sigma_0
        return np.kron(tau_z, h0) + np.kron(tau_0,zeeman) + pairing

        
        
    syst[(lat(x,y) for x in range(L) for y in range(W))] = onsite

    # intra-layer hopping with rashba term
    print("alpha_rashba = ", alpha_rashba / (1e-3 * const_e) / 1e-9)
    print("a = ", a)
    for i in range(L):
        if i > 0:
            # 2DEG: add Rashba term
            syst[lat(i, 0), lat(i-1, 0)] = np.kron(tau_z, -t*sigma_0 - 1j/(2*a) * alpha_rashba * sigma_y)
            # SC
            syst[lat(i, 1), lat(i-1, 1)] =  np.kron(tau_z, -t*sigma_0 - 1j/(2*a) * alpha_rashba * sigma_y)# np.kron(tau_z, -t*sigma_0)
        # inter-layer hopping (delta*)
        syst[lat(i, 1), lat(i, 0)] = np.kron(tau_z, -t_prime * sigma_0) 


    
 #    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = \
 # np.kron(tau_z, -t*sigma_0 - 1j/(2*a) * alpha_rashba * sigma_y)
 #    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = np.kron(tau_z, -t_prime * sigma_0) 



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
delta_phi_vals = np.linspace(-1*np.pi, 1*np.pi, 31)

for alpha_rashba in np.linspace(0, 5e-3 * const_e * 1e-9, 5):
    print("α = %g meV nm" % (alpha_rashba / (1e-3*const_e) / 1e-9))
    F_vals = []
    spectrum = []
    for delta_phi in delta_phi_vals:
        print("delta_phi = ", delta_phi / np.pi)
    
        ham_mat = make_system(delta_phi=delta_phi,B=0.1,alpha_rashba=alpha_rashba)
        evals = ssl.eigsh(ham_mat, k=2, sigma=0, which='LA', return_eigenvectors=False)
        spectrum.append(evals)
        F = -np.sum(evals)
        F_vals.append(F)

    spectrum = np.array(spectrum)
    F_vals = np.array(F_vals)
    plt.plot(delta_phi_vals/np.pi, spectrum/const_e * 1e6, label="α = %g" % alpha_rashba)
plt.grid()
plt.legend()
plt.xlabel('φ (π)')
plt.ylabel('E (μeV)*')
plt.show()
plt.grid()
plt.plot(delta_phi_vals/np.pi, F_vals)
plt.show()

