#!/usr/bin/env python3

import numpy as np
import io
# from collections import defaultdict

file = "2022-05-19_07-59-11_jj_1d/data.dat"
#file = "2022-05-18_21-54-43_jj_1d/data.dat"
#file = '2022-05-12_19-13-43_jj_1d/data.dat'

fh = open(file, 'r')
header = fh.readline().rstrip()
contents = fh.read().rstrip()
data = np.genfromtxt(io.StringIO(contents))
data_length = data.shape[0]

# data_dict = defaultdict(dict)

bvals = data[:,1]
bvals = np.unique(bvals)


ky_vals = np.unique(data[:,0])

for b in (bvals):
    b_vals = []
    for i in range(data_length):
        if data[i,1] == b:
            b_vals.append(data[i])
    b_vals = np.array(b_vals)
    output_file = file + "_B=%g-sorted.dat" % b
    fh_out = open(output_file, 'w')
    for ky in (ky_vals):
        vals = []
        for i in range(b_vals.shape[0]):
            if b_vals[i,0] == ky:
                vals.append(b_vals[i])
        vals = np.array(vals)
        # sort vals for phi value
        vals = vals[vals[:,2].argsort()]
        for d in vals:
            print(d[0], d[1], d[2], d[3], file=fh_out)
        print(file=fh_out)
# for i in range(data.shape[0]):
#     ky = data[i,0]
#     B = data[i,1]
#     phi = data[i,2]
#     e = data[i,3]
#     print(ky, B, phi, e)
#     data_dict[ky][B][phi] = e



