#!/usr/bin/env python3

#----------------------------------------
# This module defines the grid for the
# MCRT programs
#----------------------------------------

from config import *

import numpy as np
import pickle


class mcrt_grid:
    """
    the grid to use
    """

    dimension = None
    extent = 0

    density = None
    internal_energy = None
    composition = None
    #  temperature = None

    def __init__(self, extent = 64, dimension=3):
        """
        extent: grid size in each dimension
        dimension: how many dims to work in 
        """

        self.dimension = dimension
        self.extent = extent

        shape = None
        if dimension == 1:
            error("1D not implemented")
        elif dimension == 2:
            shape = (extent, extent)
            shape_comp = (extent, extent, NSPECIES)
        elif dimension == 3:
            shape = (extent, extent, extent)
            shape_comp = (extent, extent, extent, NSPECIES)
        else:
            error("unknown dimension", dimension)
            

        self.density = np.zeros(shape, dtype=float)
        self.internal_energy = np.zeros(shape, dtype=float)
        self.composition = np.zeros(shape_comp, dtype=float)
        #  self.temperature = np.zeros(shape, dtype=float)

    
    def dump(self, number, basename="output_"):
        """
        Dump the entire struct as a snapshot
        """
        fname = basename + str(number).zfill(4) + ".pkl"
        f = open(fname, 'wb')
        pickle.dump(self, f)
        f.close()
        return

    def init_density(self, method, const_dens_val=None):
        """
        initialize the density fields.   

        method: 
            "const" : constant density everywhere
            you need to provide "const_dens_val"
        """

        if method == "const":
            if const_dens_val is None or const_dens_val < 0.:
                error("Invalid density", const_dens_val)
            
            if self.dimension == 2:
                self.density[:, :] = const_dens_val
            if self.dimension == 3:
                self.density[:, :, :] = const_dens_val

        return
