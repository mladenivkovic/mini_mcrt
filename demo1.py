#!/usr/bin/env python3

# -----------------------------------
# Run a demo without proper physics
# -----------------------------------

from mcrt_grid import *
from constants import kpc, MSol
from photon_packet import photon_packet
import numpy as np

npackets = 10000
boxlen = 401
dt = 1.
photon_packet_energy = 1.

my_grid = mcrt_grid(boxlen, extent=401, dimension=2)
my_grid.init_number_density("const", const_number_dens_val=1.)
my_grid.init_internal_energy("const", const_u_val=1.)  # works
my_grid.init_step()

my_grid.dump(0)
for p in range(npackets):

    # generate initial values
    packet = photon_packet(0.5 * boxlen, 0.5 * boxlen, 0.0, photon_packet_energy, dt)
    packet.generate_random_direction()
    packet.sample_optical_depth()

    # deposit packet from initial position to cell wall
    packet.first_propagation(my_grid)

    it = 0
    is_in_box = packet.is_in_box(my_grid)
    while is_in_box:
        it += 1
        absorbed = packet.propagate(my_grid)
        if absorbed: 
            break
        is_in_box = packet.is_in_box(my_grid)

        if it % 100 == 0:
            print("packet = {0:9d}, iter = {1:6d}".format(p, it))
    if p % 100 == 0:
        print("Finished packet {0:0d}".format(p), it)

my_grid.finalise_step()

my_grid.dump(1)
