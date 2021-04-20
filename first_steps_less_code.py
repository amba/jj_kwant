#!/usr/bin/env python3

import kwant
import matplotlib.pyplot as plt
import numpy as np

def make_system(a=1, t=1.0, W=10, L=30):
    lat = kwant.lattice.square(a)
    syst= kwant.Builder()
    syst[(lat(x,y) for x in range(L) for y in range(W))] = 4 * t
    syst[lat.neighbors()] = -t
    lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
    lead[(lat(0,j) for j in range(W))] = 4 * t
    lead[lat.neighbors()] = -t
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    kwant.plot(syst)
    syst = syst.finalized()
    return syst

def plot_conductance(syst ,energies):
    data  = []
    for energy in energies:
        print("energy: ", energy)
        print("syst: ", syst)
        smatrix = kwant.smatrix(syst, energy)
        data.append(smatrix. transmission(1, 0))
        
    plt.plot(energies, data)
    plt.xlabel('E')
    plt.ylabel("conductance (e^2/h)")

    plt.show(block=False)
    import code
    code.interact()



syst = make_system()

energies = np.linspace(0,3,100)

plot_conductance(syst, energies)
