#!/usr/bin/env python3

#----------------------------------------
# Plot solution of a 2D problem
# Usage:
# ./plotSolution2D.py output_0000.pkl
#----------------------------------------

import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import SymLogNorm, LogNorm
import sys
from config import *




fname = sys.argv[1]

f = open(fname, 'rb')
my_grid = pickle.load(f)

if my_grid.dimension != 2:
   error("Output dimension != 2; got", my_grid.dimension) 

def set_colorbar(ax, im):
    """
    Adapt the colorbar a bit for axis object <ax> and
    imshow instance <im>
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    return


intensity = my_grid.mean_specific_intensity

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(intensity.T, origin="lower")
set_colorbar(ax, im)
plt.show()
