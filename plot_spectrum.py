#!/usr/bin/env python3

import numpy as np
import io
import json
import matplotlib.pyplot as plt
import scipy.constants as const


if np.__version__ < '1.14.1':
    sys.exit("numpy version " + np.__version__ + " is too old")
    
def open_3d_file(file):
    fh = open(file, 'r')
    header = fh.readline().rstrip()
    contents = fh.read().rstrip()
    
    list_of_blocks = contents.split("\n\n")
    arrays = []
    for block in (list_of_blocks):
        arrays.append(np.genfromtxt(io.StringIO(block)))
    first_shape = arrays[0].shape
    for i in range(len(arrays)-1, -1, -1):
        shape = arrays[i].shape
        if shape != first_shape:
            print("block ", i, " with first line", arrays[i][0], " does not match :", shape, " != ", first_shape)
            del arrays[i]
    return np.stack(arrays), header


with open('args.json','r') as infile:
    parameters = json.load(infile)

print("simulation parameters:\n", parameters)

gap = parameters['gap'] * const.e

data, header = open_3d_file('spectrum.dat')
phi_vals = data[:,0,0]
ev_data = data[:,:,1] / gap
print(ev_data.shape)

plt.plot(phi_vals / np.pi, ev_data, '.')
plt.xlabel('φ / π')
plt.ylabel('ε / Δ')
plt.grid()

plt.show(block=False)


import code
code.interact()


