#!/usr/bin/env python3

#-----------------------------------
# Run tests.
#-----------------------------------

from mcrt_grid import *
import numpy as np

my_grid = mcrt_grid(extent = 2, dimension = 2)

my_grid.init_density("const", const_dens_val = 13.) # works
my_grid.init_internal_energy("const", const_u_val = 1e16) # works
#  my_grid.init_mass_fractions("const", const_mass_fractions_val = [1, 0]) works
my_grid.init_mass_fractions("equilibrium", XH = 1.)
#  my_grid.dump(0) # works

#  test_array = np.ones((12, 12), dtype=float) * 17
#  test_mass_fractions_array = np.ones((12, 12, 2), dtype=float) * 18.
#  my_grid.init_density("manual", manual_dens_array = test_array) # works
#  my_grid.init_internal_energy("manual", manual_u_array = test_array) # works
#  my_grid.init_mass_fractions("manual", manual_mass_fractions_array = test_mass_fractions_array) # works

#  print(my_grid.density)
#  print(my_grid.internal_energy)
print("Mass fractions final", my_grid.mass_fractions)
