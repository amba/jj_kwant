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
delta = 150e-6 * const_e
L = 500
L_junction = int(1e-6/a)
print("L_junction = ", L_junction)
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
    lat = kwant.lattice.chain(1)

    def onsite(site):
        (x,) = site.pos
        pairing = 0
        h0 = 2*t -mu
        dphi = delta_phi
        zeeman = 0
        if x > L/2:
            dphi = -dphi
            #print("mu = ", mu)

        if abs(x-L/2) > L_junction/2:
            pairing = np.kron(tau_x * np.cos(dphi/2) - tau_y * np.sin(dphi/2), sigma_0)
            pairing *= delta
        zeeman =  0.5 * g_factor * const_bohr_magneton * B * sigma_y
            
        
        h0 = h0 * sigma_0
        return np.kron(tau_z, h0) + np.kron(tau_0,zeeman) + pairing

        
        
    syst[(lat(x) for x in range(L))] = onsite

    # intra-layer hopping with rashba term
    print("a = ", a)
    syst[kwant.builder.HoppingKind((1,), lat, lat)] = \
         np.kron(tau_z, -t * sigma_0 -1j/(2*a) * alpha_rashba * sigma_y)

    syst = syst.finalized()
    ham_mat = syst.hamiltonian_submatrix(sparse=True)
    ham_mat = ham_mat.tocsc()
    return ham_mat

delta_phi_vals = np.linspace(-0.1*np.pi, 0.1*np.pi, 31)

for alpha_rashba in np.linspace(0, 20e-3 * const_e * 1e-9, 3):
    print("α = %g meV nm" % (alpha_rashba / (1e-3*const_e) / 1e-9))
    F_vals = []
    spectrum = []
    for delta_phi in delta_phi_vals:
        print("delta_phi = ", delta_phi / np.pi)
    
        ham_mat = make_system(delta_phi=delta_phi,B=0.1,alpha_rashba=alpha_rashba)
        evals = ssl.eigsh(ham_mat, k=20, sigma=0, which='LA', return_eigenvectors=False)
        spectrum.append(evals)
        F = -np.sum(evals)
        F_vals.append(F)

    spectrum = np.array(spectrum)
    F_vals = np.array(F_vals)
    plt.plot(delta_phi_vals/np.pi, F_vals, label="α = %g" % alpha_rashba)
    #plt.plot(delta_phi_vals/np.pi, spectrum/const_e * 1e6, label="α = %g" % alpha_rashba)
plt.grid()
plt.legend()
plt.xlabel('φ (π)')
plt.ylabel('E (μeV)*')
plt.show()
#plt.grid()
#plt.plot(delta_phi_vals/np.pi, F_vals)
#plt.show()

