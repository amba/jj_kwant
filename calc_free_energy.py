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


data, header = open_3d_file('data.dat')


output_fh = open('free_energy.dat', 'w')

print("shape: ", data.shape)

output_fh.write(header + "\n")

for block in data:
    energy = -np.sum(block[:,-1])
    values = block[0,:-1].tolist()
    values.append(energy)
    values = ["%.7g" % x for x in values]
    line = "\t\t".join(values) + "\n"
    output_fh.write(line)


                     
