#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import kwant
import kwant.digest

import json
import scipy.sparse.linalg
import scipy.linalg
import shutil
import tinyarray
from datetime import datetime
import scipy.constants as const
import pathlib
import os.path
import os

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--lattice-constant', default=5e-9, help="lattice constant", type=float)
parser.add_argument('-w', '--width', help="width of structure in m", required=True, type=float)
parser.add_argument('-l', '--junction-length', help="junction length in m", required=True, type=float)
parser.add_argument('-e', '--electrode-length', help="electrode length in m", required=True, type=float)
parser.add_argument('--carrier-density', help="carrier density / m^2 (default: 1e16)", default=1e16, type=float)
parser.add_argument('--mass', help="effective mass / m_e(default: 0.03)", default=0.03, type=float)
parser.add_argument('--disorder', help="random disorder potential (in units of fermi energy E_F)", type=float, default=0)
parser.add_argument('--gap', help="superconducting gap / eV (default: 200e-6)", default=200e-6, type=float)
parser.add_argument('--by', help="in-plane magnetic field in y-direction (Tesla)", default=0, type=float)



parser.add_argument('--n-phi', help="number of values of phase difference (default: 20)", default=20, type=int)
parser.add_argument('--tol', help="stopping accuracy of eigenvalues (default: 1e-3)", type=float, default=1e-3)

args = parser.parse_args()
a = args.lattice_constant

# width of josephson junction
width = args.width


electrode_length = args.electrode_length

junction_length = args.junction_length

gap = args.gap * const.e

B_y = args.by

n_s = args.carrier_density

m_eff = args.mass * const.m_e

n_phi = args.n_phi

tolerance = args.tol

E_fermi = const.hbar**2 * 2 * np.pi * n_s / (2 * m_eff)
k_fermi = np.sqrt(2 * m_eff * E_fermi) / const.hbar
lambda_fermi = 2 * np.pi / k_fermi
mu = E_fermi
n_bound_states = 2 * width / lambda_fermi # with spin: 4 * width / lambda_fermi

sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j,0]])
sigma_z = tinyarray.array([[1, 0], [0,-1]])

tau_0 = tinyarray.array([[1, 0], [0, 1]])
tau_x = tinyarray.array([[0, 1], [1, 0]])
tau_y = tinyarray.array([[0, -1j], [1j,0]])
tau_z = tinyarray.array([[1, 0], [0,-1]])

bohr_magneton = 9.274009e-24 # J/T

date_string = datetime.now()
date_string = date_string.strftime("%Y-%m-%d_%H-%M-%S")
data_folder = date_string + "_JJ_spin_width=%g_junction_length=%g_electrode_length=%g_By=%g_disorder=%g" % (width, junction_length, electrode_length, args.by, args.disorder)

print("data folder: ", data_folder)
pathlib.Path(data_folder).mkdir()

# save args to file
with open(data_folder + '/args.json', 'w') as outfile:
    json.dump(vars(args), outfile)

# copy script into output folder
shutil.copyfile(__file__, data_folder + '/' + os.path.basename(__file__))

# create output data files
fh_free_energy = open(data_folder + '/free_energy.dat', 'w')
fh_free_energy.write("# phi\t\tfree-energy\n")

fh_spectrum = open(data_folder + '/spectrum.dat', 'w')
fh_spectrum.write("# phi\t\tenergy\n")

def make_syst(m = 0.03 * const.m_e, a=5e-9, width=3e-6,
              electrode_length = 3e-6, junction_length=100e-9,
              mu=0, disorder=0, gap=0, delta_phi=0, alpha_rashba=0, B=[0,0,0], g_factor=-10):
    t = const.hbar**2 / (2 * m * a**2)
    print("m = %g, a = %g, width = %g, electrode_length = %g, junction_length = %g, t = %g" % (m, a, width, electrode_length, junction_length, t))
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
        if (disorder > 1e-9):
            h0 = h0 + disorder * mu * (kwant.digest.uniform(site.pos) - 0.5)
            
        dphi = delta_phi
        
        if x > L/2:
            dphi = -dphi

        start_junction = int((L - L_junction) / 2)
        pairing = 0
        if x < start_junction or x >= start_junction + L_junction:
            pairing =  np.kron(tau_x * np.cos(dphi/2) - tau_y * np.sin(dphi/2), gap * sigma_0)
            
        
        # from "a josephson supercurrent diode" paper supplement
        zeeman = 0.5 * g_factor * bohr_magneton * (B[0] * sigma_x + B[1] * sigma_y + B[2] * sigma_z)
        
        return np.kron(tau_z, h0 * sigma_0) + np.kron(tau_0, zeeman) + pairing
    
    syst[(lat(x,y) for x in range(L) for y in range(W))] = onsite

    
    # Hoppings

    # Rashba term in hamiltonian: -iα(∂_x σ_y - ∂_y σ_x)
    
    # x direction
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = \
        np.kron(tau_z, -t * sigma_0 + 1j * alpha_rashba * sigma_y / 2)
    # y direction
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = \
        np.kron(tau_z, -t * sigma_0 - 1j * alpha_rashba * sigma_x / 2)
    
    
    
    def onsite_0000(site):
        return np.abs(onsite(site)[0,0,0,0])
    def onsite_01(site):
        return np.abs(onsite(site)[1, 0])
    
    # kwant.plot(syst,site_color=onsite_00)
    # kwant.plot(syst,site_color=onsite_01)
    
    return syst.finalized()

def make_hamiltonian_sparse_csc(*args, **kwargs):
    syst = make_syst(*args, **kwargs)
    ham_mat = syst.hamiltonian_submatrix(sparse=True)
    print("Hamiltonian shape: ", ham_mat.shape)
    ham_mat = ham_mat.tocsc()
    return ham_mat

print("E_fermi = %.3g meV, lambda_fermi = %.3g nm, N0 = %.2g" % (1000 * mu / const.e, lambda_fermi * 1e9, n_bound_states))



phi_vals = np.linspace(-0.2*np.pi, 1.2*np.pi, n_phi)

for phi in phi_vals:
    print("phi = %.3g pi" % (phi/np.pi))
    print("make syst")
    t0 = time.time()

    ham_mat = make_hamiltonian_sparse_csc(a=5e-9, width=width, electrode_length=electrode_length, junction_length=junction_length, mu=mu, gap=gap, delta_phi=phi, disorder=args.disorder, B=[0, B_y, 0])
    t1 = time.time()
    print("time for make_hamiltonian: ", t1 - t0)

    k= int(2 * n_bound_states + 6)     # factor 2 for spin

    print("calculating %d eigenvalues" % k)
    evs = scipy.sparse.linalg.eigsh(ham_mat, k=k, sigma=0, which='LA', return_eigenvectors=False, tol=tolerance)
    
    print("execution time: %.1f s" % (time.time() - t1))

    free_energy = -np.sum(evs)
    np.savetxt(fh_free_energy, [[phi, free_energy]], fmt="%.10g", delimiter="\t\t")
    for ev in (evs):
        np.savetxt(fh_spectrum, [[phi, ev]], fmt="%.10g", delimiter="\t\t")

    fh_spectrum.write("\n") # new block in datafile
    fh_free_energy.flush()
    fh_spectrum.flush()
    os.fsync(fh_free_energy)
    os.fsync(fh_spectrum)
    
