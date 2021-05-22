#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import io
import json
import scipy.constants as const
import sys

font = {'family' : 'sans',
        'weight' : 'normal',
        'size'   : 5}

matplotlib.rc('font', **font)


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



argv = sys.argv
data_folders = argv[1:]
if not data_folders:
    print("need data_folders as arguments")
    sys.exit(1)
    
print(data_folders)


def get_params(json_file):
    with open(json_file, 'r') as jsonfile:
        return json.load(jsonfile)
    
def eval_data_folder(data_folder):
    parameters = get_params(data_folder + '/args.json')
    print(parameters)
    gap = parameters['gap'] * const.e
    spectrum_data, header = open_3d_file(data_folder + '/spectrum.dat')
    phi_vals = spectrum_data[:,0,0]
    ev_data = spectrum_data[:,:,1]

    # plot spectrum
    plt.clf()
    plt.close()
    plt.xlabel('φ / π')
    plt.ylabel('ε / Δ')
    plt.grid()
    plt.title(data_folder)
    plt.plot(phi_vals / np.pi, ev_data / gap, '.')
    plt.savefig(data_folder + '/spectrum-plot.pdf')


    # get transmission histogram
    pi_index = np.argmin(np.abs(phi_vals - np.pi))
    print("pi_index: ", pi_index)
    pi_bound_states = ev_data[pi_index, :] / gap
    transmissions = 1 - pi_bound_states**2

    plt.clf()
    plt.close()
    plt.xlabel('τ')
    plt.ylabel('count')
    plt.hist(transmissions, bins=20)
    plt.title(data_folder)
    plt.savefig(data_folder + '/transmission_histogram.pdf', dpi=600)
    
    
for folder in data_folders:
    eval_data_folder(folder)




