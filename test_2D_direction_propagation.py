#!/usr/bin/env python3

#-------------------------------------------------
# Select systematically different directions for
# 2D and check that the propagation works.
# Assume 1 source in center.
#-------------------------------------------------

from mcrt_grid import *
from constants import kpc, MSol
from photon_packet import photon_packet
import numpy as np

npackets = 1000
boxlen = 10 * kpc
my_grid = mcrt_grid(boxlen, extent = 64, dimension = 2)

my_grid.init_density("const", const_dens_val = 13.) # works
my_grid.init_internal_energy("const", const_u_val = 1e16) # works
my_grid.init_mass_fractions("const", const_mass_fractions_val = [1, 0])
my_grid.dump(0)


my_grid.dump(0)
for p in range(npackets):

    # make new packet
    packet = photon_packet(0.5 * boxlen, 0.5 * boxlen, 0., 0.)

    # select next phi
    phi = p/npackets * 2 * np.pi 
    packet.direction = np.array([phi, 0.])

    # set initial cell packet is in
    if phi < 0.5 * np.pi :
        packet.cell_index_i = int(my_grid.extent // 2)
        packet.cell_index_j = int(my_grid.extent // 2)
    elif phi < np.pi:
        packet.cell_index_i = int(my_grid.extent // 2) - 1
        packet.cell_index_j = int(my_grid.extent // 2)
    elif phi < 1.5 * np.pi:
        packet.cell_index_i = int(my_grid.extent // 2) - 1
        packet.cell_index_j = int(my_grid.extent // 2) - 1
    else:
        packet.cell_index_i = int(my_grid.extent // 2)
        packet.cell_index_j = int(my_grid.extent // 2) - 1

    is_in_box = True
    it = 0
    while is_in_box:
        # propagate packet
        it += 1
        packet.propagate(my_grid)
        is_in_box = packet.is_in_box(my_grid)
        print("packet = {0:9d}, iter = {1:6d}".format(p, it))

my_grid.dump(1)

