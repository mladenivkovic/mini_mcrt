#!/usr/bin/env python3

# -----------------------------------
# Run tests.
# -----------------------------------

from mcrt_grid import *
from constants import kpc, MSol
from photon_packet import photon_packet
import numpy as np

npackets = 8
boxlen = 10 * kpc
#  boxlen = 65
# Note: Stick with uneven number of cells for now
my_grid = mcrt_grid(boxlen, extent=129, dimension=2)

my_grid.init_density("const", const_dens_val=13.0)  # works
my_grid.init_internal_energy("const", const_u_val=1e16)  # works
#  my_grid.init_mass_fractions("const", const_mass_fractions_val = [1, 0]) works
my_grid.init_mass_fractions("equilibrium", XH=1.0)
#  my_grid.dump(0) # works

#  test_array = np.ones((12, 12), dtype=float) * 17
#  test_mass_fractions_array = np.ones((12, 12, 2), dtype=float) * 18.
#  my_grid.init_density("manual", manual_dens_array = test_array) # works
#  my_grid.init_internal_energy("manual", manual_u_array = test_array) # works
#  my_grid.init_mass_fractions("manual", manual_mass_fractions_array = test_mass_fractions_array) # works

#  print(my_grid.density)
#  print(my_grid.internal_energy)
#  print("Mass fractions final", my_grid.mass_fractions)

my_grid.dump(0)
npackets = 1
for p in range(npackets):

    #  phi = p / npackets * 2 * np.pi
    #  phi = (3. + 4 * p)/8. * 2 * np.pi + 0.18 * np.pi
    #  phi = 7./8. * 2 * np.pi + 0.18 * np.pi
    #  phi = 1.75 * np.pi
    #  phi = 0.75 * np.pi
    #  packet.direction = np.array([0.0, phi])

    # generate initial values
    packet = photon_packet(0.5 * boxlen, 0.5 * boxlen, 0.0, 0.0)
    packet.generate_random_direction()
    packet.sample_optical_depth()

    # deposit packet from initial position to cell wall
    packet.first_propagation(my_grid)

    it = 0
    is_in_box = packet.is_in_box(my_grid)
    while is_in_box:
        it += 1
        packet.propagate(my_grid)
        is_in_box = packet.is_in_box(my_grid)
        #  if it % 100 == 0:
        #  print("packet = {0:9d}, iter = {1:6d}".format(p, it))
    #  if p % 100 == 0:
    #  print("Finished packet {0:0d}".format(p), it)
my_grid.dump(1)
