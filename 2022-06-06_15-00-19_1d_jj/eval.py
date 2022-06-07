#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import io
import os.path

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



for file in ('data_B=0.dat', 'data_B=0.2.dat', 'data_B=0.4.dat', 'data_B=0.5.dat'):
    data, header = open_3d_file(file)
    # axis 0: phi
    # axis 1: ky
    # axis 2: param
    print("header: ", header)
    print(data.shape)
    B = data[0,0,2]
    # Get E_J(φ)
    phi_vals = []
    E_vals = []
    for i in range(data.shape[0]):
        phi = data[i][0][1]
        print("phi: ", phi)
        # sum of transverse momentum ky
        E_J = np.sum(data[i,:,3])
        E_J = -E_J / data.shape[1]
        phi_vals.append(phi)
        E_vals.append(E_J)
    phi_vals = np.array(phi_vals)
    E_vals = np.array(E_vals)
    plt.plot(phi_vals/np.pi, E_vals, label="B = %g" % B)

plt.legend()


plt.grid()

plt.show(block=False)
import code
code.interact()
