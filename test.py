#!/usr/bin/env python3

import kwant

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

kwant.plot(syst)
