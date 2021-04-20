#!/usr/bin/env python3

import kwant
import matplotlib.pyplot as plt
import numpy as np

import tinyarray

sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j,0]])
sigma_z = tinyarray.array([[1, 0], [0,-1]])


    
def make_system(a=1, t=1.0, e_z=0, e_y = 0, alpha=0, W=10, L=30, pot=10):
    def potential(site):
        (x, y) = site.pos
        x0 = a * L / 2
        y0 = a * W / 2
        s_V = 1 / (1 + (x - x0)**2) * (y - y0)**2
#        s_V = 1/np.sqrt(1 + (x - x0)**2 + y**2) + 1/np.sqrt((x - x0)**2 + (y - y0)**2 + 1)
        return pot * s_V
    
    def onsite(site):
        return (4 * t + potential(site))* sigma_0 + e_z * sigma_z + e_y * sigma_y

    lat = kwant.lattice.square(a)
    syst= kwant.Builder()
    syst[(lat(x,y) for x in range(L) for y in range(W))] = onsite
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = \
        -t * sigma_0 + 1j * alpha * sigma_y / 2
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = \
        -t * sigma_0 - 1j * alpha * sigma_x / 2
    
    lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
    lead[(lat(0,j) for j in range(W))] = onsite
    lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = \
        -t * sigma_0 + 1j* alpha * sigma_y / 2
    lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = \
        -t * sigma_0 - 1j* alpha * sigma_x / 2
    
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
#     kwant.plot(syst, site_color=potential)
    syst = syst.finalized()
    return syst


e_z = 0
α = 0
for e_y in (0,):
    for pot in np.linspace(0, 1, 4):
        syst = make_system(L = 60, W = 20, alpha = α,e_z = e_z, e_y = e_y, pot = pot)
        energies = np.linspace(0,5,100)
        data  = []
        for energy in energies:
            print("energy: %.3g, pot: %.3g" % (energy, pot))
            smatrix = kwant.smatrix(syst, energy)
            t = smatrix.transmission(1, 0)
            print("t: %.3g" % t)
            data.append(t)
        
        plt.plot(energies, data, label="e_z = %.3g, e_y = %.3g, α = %.3g, pot = %.3g" % (e_z, e_y, α, pot))

plt.legend()
plt.xlabel('E')
plt.grid()
plt.ylabel("conductance (e^2/h)")
plt.savefig('conductance.pdf')
plt.show()
