#!/usr/bin/env python3
from datetime import datetime
import jj_kwant.spectrum
import scipy.constants as const
import numpy as np
import time
import pathlib
import sys
import shutil
import os.path
import scipy.sparse.linalg
import matplotlib.pyplot as plt
gap = 100e-6 * const.e
mass =  0.04 * const.m_e
g_factor_N = -10
g_factor_S = -10

args = {
    'mass': mass,
    'gap': gap,
    'electrode_length': 1e-6,
    'junction_length': 100e-9,
    'a': 5e-9,
    }

phi_vals = np.linspace(-0.2*np.pi, 0.2*np.pi, 21)
B = 0.3
mu = 50e-3 * const.e
k_fermi = 1/const.hbar * np.sqrt(2 * mass * mu)

# α m / (hbar**2 k_f) = 0.6
alpha = 10e-3 * const.e * 1e-9
#alpha = 0.6 * const.hbar**2 * k_fermi / mass
print("alpha = %g meV nm" % (alpha / (const.e * 1e-3 * 1e-9)))


kf_vals = np.linspace(-0.9* k_fermi, 0.9*k_fermi, 5)
print("k_F = ", k_fermi)

# # make new datafolder
# date_string = datetime.now()
# data_folder = date_string.strftime("%Y-%m-%d_%H-%M-%S_1d_jj")

# print("creating new datafolder: ", data_folder)
# pathlib.Path(data_folder).mkdir()
# script = sys.argv[0]
# # copy script into output folder
# shutil.copyfile(script, data_folder + '/' + os.path.basename(script))





def calc(ky=None, phi=None, B=None):
    ham = jj_kwant.spectrum.hamiltonian_jj_1d(
        # debug=True,
            ky=ky,
        a=args['a'],
        m=mass,
        junction_length=args['junction_length'],
        electrode_length=args['electrode_length'],
        gap=gap,
        B=[0,B,0],
        delta_phi = phi,
        g_factor_N = g_factor_N,
        g_factor_S = g_factor_S,
        mu=mu,
        alpha_rashba=alpha,
        salt='')
    evs = jj_kwant.spectrum.positive_low_energy_spectrum(ham, 8)
    return evs

F_vals = []
for phi in phi_vals:
    ev_vals = []
    for ky in kf_vals:
        print("phi = %g π" % (phi/np.pi)) 
        evs = calc(ky=ky, phi=phi, B=B)
        evs = evs / gap
        print("evs = ", evs)
        ev_vals.append(evs)
    F_vals.append(-np.sum(ev_vals))
    ev_vals = np.array(ev_vals)
    
plt.plot(phi_vals, F_vals)
plt.grid()
plt.legend()
plt.show()
        
# for B in Bvals:
#     data_file = "data_B=%.2g.dat" % B
#     fh = open(data_folder + "/" + data_file, "w")
#     fh.write("#\t\tky\t\tphi\t\tB\t\tE\t\ttime\n")
#     t_start = time.time()
    
#     for phi in phi_vals:
#         for ky in kf_vals:
#             evs = calc(ky=ky, phi=phi, B=B)
#             evs = evs / gap
#             print("evs: ", ev)
#             fh.write("%g\t\t%g\t\t%g\t\t%g\t\t%g\n" %(ky, phi, B, ev, time.time() - t_start))
#             fh.flush()
#             os.fsync(fh)
#         fh.write("\n")
#         fh.flush()
#         os.fsync(fh)
        
                





