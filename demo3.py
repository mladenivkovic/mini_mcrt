#!/usr/bin/env python3

# -----------------------------------
# Run a demo without proper physics
# Put a wall above and below the center
# -----------------------------------

from mcrt_grid import *
from constants import kpc, MSol
from photon_packet import photon_packet
import numpy as np

npackets = 50000
boxlen = 201
dt = 1.
photon_packet_energy = 1.


my_grid = mcrt_grid(boxlen, extent=201, dimension=2)

n = np.ones((201, 201), dtype=float)
for i in range(70, 131):
    for j in range(145, 155):
        n[i, j] = 20
    for j in range(45, 55):
        n[i, j] = 10

my_grid.init_number_density("manual", manual_number_dens_array = n)
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
