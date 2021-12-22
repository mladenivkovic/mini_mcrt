#!/usr/bin/env python3

# -----------------------------------
# Run tests.
# -----------------------------------

from mcrt_grid import *
from constants import kpc, MSol
from photon_packet import photon_packet
import numpy as np

npackets = 1000
boxlen = 10 * kpc
my_grid = mcrt_grid(boxlen, extent=64, dimension=2)

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
for p in range(npackets):

    packet = photon_packet(0.5 * boxlen, 0.5 * boxlen, 0.0, 0.0)
    phi = p / npackets * 2 * np.pi
    packet.direction = np.array([phi, 0.0])
    if phi < 0.5 * np.pi:
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
        it += 1
        packet.propagate(my_grid)
        is_in_box = packet.is_in_box(my_grid)
        #  if it % 100 == 0:
        print("packet = {0:9d}, iter = {1:6d}".format(p, it))
    #  if p % 100 == 0:
    #  print("Finished packet {0:0d}".format(p), it)
    my_grid.dump(1)
