#!/usr/bin/env python3

# ----------------------------------------
# Plot solution of a 2D problem
# Usage:
# ./plotSolution2D.py output_0000.pkl
# ----------------------------------------

import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import SymLogNorm, LogNorm
import sys
from config import *


fname = sys.argv[1]

f = open(fname, "rb")
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
path_length_estimator = my_grid.path_length_estimator
absorbed_energy = my_grid.absorbed_energy

fig = plt.figure(figsize=(8, 4))

ax1 = fig.add_subplot(121)
im1 = ax1.imshow(intensity.T, origin="lower", norm=SymLogNorm(linthresh=1e-4, base=10))
set_colorbar(ax1, im1)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("Mean Specific Intensity")

#  ax2 = fig.add_subplot(132)
#  im2 = ax2.imshow(path_length_estimator.T, origin="lower", norm=SymLogNorm(linthresh=1e-4, base=10))
#  set_colorbar(ax2, im2)
#  ax2.set_xlabel("x")
#  ax2.set_ylabel("y")
#  ax2.set_title("Path Length Estimator")

ax3 = fig.add_subplot(122)
im3 = ax3.imshow(absorbed_energy.T, origin="lower", norm=SymLogNorm(linthresh=1e-4, base=10))
set_colorbar(ax3, im3)
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_title("Absorbed Energy")



plt.tight_layout()
#  plt.show()

plt.savefig(fname[:-4]+"-full.png", dpi=300)
