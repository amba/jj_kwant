#!/usr/bin/env python3


import kwant
import kwant.digest
import numpy as np

import scipy.sparse.linalg

import tinyarray
import matplotlib.pyplot as plt

import time

const_hbar = 1.0545718176461565e-34
const_e = 1.602176634e-19
const_m_e = 9.1093837015e-31
const_bohr_magneton = 9.274009e-24 # J/T


sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j,0]])
sigma_z = tinyarray.array([[1, 0], [0,-1]])

tau_0 = tinyarray.array([[1, 0], [0, 1]])
tau_x = tinyarray.array([[0, 1], [1, 0]])
tau_y = tinyarray.array([[0, -1j], [1j,0]])
tau_z = tinyarray.array([[1, 0], [0,-1]])



###############################
#           #     #           #
#           #     #           #  
#           #     #           #  |
# <- l_e -> # <l> #  <- l_e ->#  width
#           #     #           #  |
#           #     #           #  
#           #     #           #
###############################

# internal function
def _make_syst_jj_2d(
        m = 0.03 * const_m_e,
        a=5e-9,
        width=3e-6,
        electrode_length = 3e-6,
        junction_length=100e-9,
        mu=None,
        gap_potential=0,
        gap_potential_shape=None,
        disorder=0,
        gap=None,
        delta_phi=None,
        alpha_rashba=0,
        B=[0,0,0],
        g_factor=-10,
        debug=False,
        salt=''
):
    print("making 2d JJ system:\n", locals())
    t = const_hbar**2 / (2 * m * a**2)
#    print("m = %g, a = %g, width = %g, electrode_length = %g, junction_length = %g, t = %g" % (m, a, width, electrode_length, junction_length, t))
    W = int(width/a)
    L = int((2*electrode_length + junction_length) / a)
    L_junction = int(junction_length/a)
    print("L = %d, W = %d, L_junction = %d" % (L, W, L_junction))
    lat = kwant.lattice.square(1)
    

    syst = kwant.Builder()

    # On-site
    def onsite(site):
        (x, y) = site.pos

        # p^2 / (2m) - μ + U(r)
        
        h0 = 4*t - mu
        # variance of h0 is disorder
        h0 = h0 + disorder * kwant.digest.gauss(site.pos, salt=salt)
            
        dphi = delta_phi
        
        if x > L/2:
            dphi = -dphi

        start_junction = int((L - L_junction) / 2)
        pairing = 0
        if x < start_junction or x >= start_junction + L_junction:
            # in electrodes: add pairing term
            pairing =  np.kron(tau_x * np.cos(dphi/2) - tau_y * np.sin(dphi/2), gap * sigma_0)
        else:
            # in junction
            if gap_potential_shape == 'cosine_half':
                h0 = h0 + gap_potential*np.cos(2*np.pi * (x - L/2) / (2*L_junction))
            elif gap_potential_shape == 'cosine':
                h0 = h0 + gap_potential * 1/2 * (1 + np.cos(2*np.pi * (x - L/2) / L_junction))
            else:
                h0 = h0 + gap_potential
        #    mod = (y*a) % junction_island_spacing
        #    if mod < junction_island_width and\
        #       abs((x-L/2)*a) < junction_island_width/2:
        #        pairing = np.kron(tau_x, gap*sigma_0)
        
            
        
        # from "a josephson supercurrent diode" paper supplement
        zeeman = 0.5 * g_factor * const_bohr_magneton * (B[0] * sigma_x + B[1] * sigma_y + B[2] * sigma_z)
        
        return np.kron(tau_z, h0 * sigma_0) + np.kron(tau_0, zeeman) + pairing
    
    syst[(lat(x,y) for x in range(L) for y in range(W))] = onsite

    
    # Hoppings

    # Rashba term in hamiltonian: -iα(∂_x σ_y - ∂_y σ_x) = α(k_xσ_y - k_yσ_x)
    # ∂_x -> 1/(2a) (f_{n+1} - f_{n-1})
    # k_x -> -i/(2a)(f_{n+1} - f_{n-1})
    
    # x direction
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = \
        np.kron(tau_z, -t * sigma_0 -1j/(2*a) * alpha_rashba * sigma_y)
    # y direction
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = \
        np.kron(tau_z, -t * sigma_0 + 1j/(2*a) * alpha_rashba * sigma_x)
    
    # debug functions

    def onsite_00(site):
        # onsite electron
        return np.abs(onsite(site)[0,0])
    def onsite_01(site):
        return np.abs(onsite(site)[2, 0])

    if debug:
        kwant.plot(syst,site_color=onsite_00)
        kwant.plot(syst,site_color=onsite_01)
    
    return syst.finalized()

# internal function
def _make_syst_jj_1d(
        m = 0.03 * const_m_e,
        a=5e-9,
        electrode_length = 3e-6,
        junction_length=100e-9,
        mu=None,
        gap_potential=0,
        gap_potential_shape=None,
        gap=None,
        delta_phi=None,
        alpha_rashba=0,
        B=[0,0,0],
        g_factor=-10,
        debug=False,
        disorder=0,
        gap_disorder=0,
        ky=0,
        salt='',
):
    print("making 1d JJ system:\n", locals())
    t = const_hbar**2 / (2 * m * a**2)
    print("phi / π = %g" % (delta_phi / np.pi))
    L = int((2*electrode_length + junction_length) / a)
    L_junction = int(junction_length/a)
    print("L = %d, L_junction = %d" % (L, L_junction))
    lat = kwant.lattice.chain(1)
    

    syst = kwant.Builder()

    # On-site
    def onsite(site):
        (x,) = site.pos

        # p^2 / (2m) - μ + U(r)
        
        h0 = const_hbar**2 * ky**2 / (2*m) + \
            2*t - mu # 1D wire: 2t, not 4t
        h0 = h0 + disorder * kwant.digest.gauss(site.pos, salt=salt)
            
        dphi = delta_phi
        
        if x > L/2:
            dphi = -dphi

        start_junction = int((L - L_junction) / 2)
        pairing = 0
        real_gap = gap + gap_disorder * kwant.digest.gauss(site.pos, salt=salt)
        if x < start_junction or x >= start_junction + L_junction:
            pairing =  np.kron(tau_x * np.cos(dphi/2) - tau_y * np.sin(dphi/2), real_gap * sigma_0)
        else:
            if gap_potential_shape == 'cosine_half':
                h0 = h0 + gap_potential*np.cos(2*np.pi * (x - L/2) / (2*L_junction))
            elif gap_potential_shape == 'cosine':
                h0 = h0 + gap_potential*1/2 * (1 + np.cos(2*np.pi * (x - L/2) / L_junction))
            else:
                h0 = h0 + gap_potential
        
        # from "a josephson supercurrent diode" paper supplement
        zeeman = 0.5 * g_factor * const_bohr_magneton * (B[0] * sigma_x + B[1] * sigma_y + B[2] * sigma_z)

        # Rashba term in hamiltonian: -iα(∂_x σ_y - ∂_y σ_x) = α(k_xσ_y - k_yσ_x)
        # add constant -α k_y σ_x term to onsite hamiltonian
        h0 = h0 * sigma_0 - alpha_rashba * ky * sigma_x
        return np.kron(tau_z, h0) + np.kron(tau_0, zeeman) + pairing
    
    syst[(lat(x) for x in range(L))] = onsite

    
    # Hoppings

    # Rashba term in hamiltonian: -iα(∂_x σ_y - ∂_y σ_x) = α(k_xσ_y - k_yσ_x)
    # ∂_x -> 1/(2a) (f_{n+1} - f_{n-1})
    # k_x -> -i/(2a)(f_{n+1} - f_{n-1})
    
    # x direction
    # add hopping for α k_x σ_y = -i/(2a) |n+1><n| σ_y + h.c.
    syst[kwant.builder.HoppingKind((1,), lat, lat)] = \
        np.kron(tau_z, -t * sigma_0 -1j/(2*a) * alpha_rashba * sigma_y)
    
    # debug functions

    def onsite_00(site):
        return np.abs(onsite(site)[0,0])
    def onsite_01(site):
        return np.abs(onsite(site)[1, 0])

    if debug:
        sites_list = kwant.plotter.sys_leads_sites(syst)
                
        sites = [x[0] for x in sites_list[0]]
                
        onsite_list = [onsite_00(site) for site in sites]
        pos_list = [site.pos for site in sites]
        plt.plot(pos_list, onsite_list)
        plt.show()
        #        kwant.plot(syst,site_color=onsite_00)
 #       kwant.plot(syst,site_color=onsite_01)
        sys.exit(1)
    return syst.finalized()

# API function

def _hamiltonian(func, timing=True, *args, **kwargs):
    t1 = time.time()
    syst = func(*args, **kwargs)
    ham_mat = syst.hamiltonian_submatrix(sparse=True)
    print("Hamiltonian shape: ", ham_mat.shape)
    ham_mat = ham_mat.tocsc()
    print("time to generate hamiltonian: %.2f s" % (time.time() - t1))
    return ham_mat

def hamiltonian_jj_2d(timing=True, *args, **kwargs):
    return _hamiltonian(func=_make_syst_jj_2d, *args, **kwargs)

def hamiltonian_jj_1d(timing=True, *args, **kwargs):
    return _hamiltonian(func=_make_syst_jj_1d, *args, **kwargs)

def mumps_eigsh(matrix, k=None, tol=None, which=None):
    class LuInv(scipy.sparse.linalg.LinearOperator):

        def __init__(self, matrix):
            instance = kwant.linalg.mumps.MUMPSContext()
            instance.analyze(matrix, ordering='pord')
            instance.factor(matrix)
            self.solve = instance.solve
            scipy.sparse.linalg.LinearOperator.__init__(self, matrix.dtype, matrix.shape)

        def _matvec(self, x):
            return self.solve(x.astype(self.dtype))

    opinv = LuInv(matrix)
    return scipy.sparse.linalg.eigsh(matrix, k, sigma=0, OPinv=opinv,
                                     which=which, return_eigenvectors=False, tol=tol)


# API function
def low_energy_spectrum(ham_mat, n_evs, tol=1e-3, which='LM', timing=True):
    print("calculating %d eigenvalues of hamiltionian with shape" % n_evs, ham_mat.shape)
    t_start = time.time()
    evs = mumps_eigsh(ham_mat, k=n_evs, tol=tol, which=which)
    print("execution time: %.2f s" % (time.time() - t_start))
    return evs

# API function
def positive_low_energy_spectrum(*args, **kwargs):
    return low_energy_spectrum(*args, **kwargs, which='LA')






 
