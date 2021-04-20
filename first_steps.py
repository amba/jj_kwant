#!/usr/bin/env python3

import kwant
import matplotlib.pyplot as plt

syst = kwant.Builder()

a = 1
lat = kwant.lattice.square(a)

t = 1
W = 10
L = 30

for i in range(L):
    for j in range(W):
        syst[lat(i,j)] = 4*t
        if j > 0:
            syst[lat(i,j),lat(i,j-1)] = -t

        if i > 0:
            syst[lat(i,j),lat(i-1,j)] = -t
    


sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
left_lead = kwant.Builder(sym_left_lead)

for j in range(W):
    left_lead[lat(0,j)] = 4*t
    if j > 0:
        left_lead[lat(0,j), lat(0,j-1)] = -t
    left_lead[lat(1,j),lat(0,j)] = -t

syst.attach_lead(left_lead)


sym_right_lead = kwant.TranslationalSymmetry((a, 0))
right_lead = kwant.Builder(sym_right_lead)

for j in range(W):
    right_lead[lat(0,j)] = 4 * t
    if j > 0:
        right_lead[lat(0,j), lat(0,j-1)] = -t
    right_lead[lat(1,j), lat(0,j)] = -t

syst.attach_lead(right_lead)

#kwant.plot(syst)


syst = syst.finalized()

energies = []
data = []

for ie in range(100):
    energy = ie * 0.01
    smatrix = kwant.smatrix(syst, energy)
    energies.append(energy)
    data.append(smatrix.transmission(1,0))

plt.figure()

plt.plot(energies, data)
plt.xlabel('E')
plt.ylabel("conductance (e^2/h)")

plt.show(block=False)
import code
code.interact()
