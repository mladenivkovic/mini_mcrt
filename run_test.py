#!/usr/bin/env python3

#-----------------------------------
# Run tests.
#-----------------------------------

from mcrt_grid import *

my_grid = mcrt_grid(extent = 12, dimension = 2)
#  my_grid.dump(0)
my_grid.init_density("const", const_dens_val = 13.)
print(my_grid.density)
