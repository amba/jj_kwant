#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import io

import scipy.optimize

gap = 200e-6 * const.e

def open_3d_file(file):
    fh = open(file, 'r')
    header = fh.readline().rstrip()
    contents = fh.read().rstrip()
    
    list_of_blocks = contents.split("\n\n")
    arrays = []
    for block in (list_of_blocks):
        arrays.append(np.genfromtxt(io.StringIO(block)))

    return arrays, header


data, header = open_3d_file('current-phase.txt')

col_legends = header.split()[1:]
col_dict = {}
for col_num , col_name in enumerate(col_legends):
    col_dict[col_name] = col_num
    print(f"{col_name} -> {col_num}")

    
data = data[0]
phi_col = col_dict['phi']
energy_col = col_dict['free-energy']

phi_data = data[:,phi_col]
energy_data = data[:,energy_col]/gap


def fit_func(phi, tau, E0):
    return E0 * (-np.sqrt(1 - tau*np.sin(phi/2)**2) + 1)

print(data.shape)
offset = energy_data[np.argmin(energy_data)]
print("offset = %g" % offset)
energy_data = energy_data - offset
E0 = np.amax(energy_data)
print("E0 = %g" % E0)
tau = 0.9

result = scipy.optimize.curve_fit(fit_func, phi_data, energy_data, p0 = [tau, E0])
fit_data = fit_func(phi_data, *result[0])
fit_tau = result[0][0]
fit_E0 = result[0][1]

print(result)
plt.plot(phi_data / np.pi, energy_data, label='data')
plt.plot(phi_data / np.pi, fit_data, label="fit τ = %g" % fit_tau)
plt.grid()
plt.legend()
plt.show(block=False)
plt.savefig('fit-free-energy.pdf')
import code
code.interact()
