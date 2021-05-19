#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import io
import json
import scipy.optimize


import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('energy_file', help='file with free energy values')

#args = parser.parse_args()

energy_file = 'free_energy.dat'

with open('args.json','r') as infile:
    parameters = json.load(infile)

print("simulation parameters:\n", parameters)

gap = parameters['gap'] * const.e


def open_3d_file(file):
    fh = open(file, 'r')
    header = fh.readline().rstrip()
    contents = fh.read().rstrip()
    
    list_of_blocks = contents.split("\n\n")
    arrays = []
    for block in (list_of_blocks):
        arrays.append(np.genfromtxt(io.StringIO(block)))

    return arrays, header


data, header = open_3d_file(energy_file)

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


print(data.shape)
offset = energy_data[np.argmin(energy_data)]
print("offset = %g" % offset)
energy_data = energy_data - offset

current = np.gradient(energy_data)

plt.xlabel('phase / π')
plt.ylabel('current (a.u.)')
plt.plot(phi_data / np.pi, current, label='data')

plt.grid()
plt.legend()
plt.show(block=False)
plt.savefig('fit-free-energy.pdf')
import code
code.interact()
