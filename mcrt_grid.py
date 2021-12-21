#!/usr/bin/env python3

#----------------------------------------
# This module defines the grid for the
# MCRT programs
#----------------------------------------

from config import *
from ionization_equilibrium import get_temperature_from_internal_energy

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
    mass_fractions = None
    temperature = None

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
        self.mass_fractions = np.zeros(shape_comp, dtype=float)
        self.temperature = np.zeros(shape, dtype=float)

    
    def dump(self, number, basename="output_"):
        """
        Dump the entire struct as a snapshot
        """
        fname = basename + str(number).zfill(4) + ".pkl"
        f = open(fname, 'wb')
        pickle.dump(self, f)
        f.close()
        return

    def init_density(self, method, const_dens_val=None, manual_dens_array = None):
        """
        initialize the density fields.   

        method: 
            "const" :   constant density everywhere
                        you need to provide "const_dens_val"
            "manual":   set manual array.
                        you need to provide "manual_dens_array"
        """

        if method == "const":
            if const_dens_val is None or const_dens_val < 0.:
                error("Invalid density", const_dens_val)
            
            if self.dimension == 2:
                self.density[:, :] = const_dens_val
            if self.dimension == 3:
                self.density[:, :, :] = const_dens_val
        elif method == "manual":
            if manual_dens_array is None or manual_dens_array.shape != self.density.shape:
                error("Invalid density array. Have shape", manual_dens_array.shape, "need shape", self.density.shape)
            self.density = manual_dens_array

        else:
            error("init_density: unknown method", method)

        return

    def init_internal_energy(self, method, const_u_val=None, manual_u_array = None):
        """
        initialize the specific internal energy fields.   

        method: 
            "const" :   constant specific internal energy everywhere
                        you need to provide "const_u_val"
            "manual":   set manual array.
                        you need to provide "manual_u_array"
        """

        if method == "const":
            if const_u_val is None or const_u_val < 0.:
                error("Invalid internal energy", const_u_val)
            
            if self.dimension == 2:
                self.internal_energy[:, :] = const_u_val
            if self.dimension == 3:
                self.internal_energy[:, :, :] = const_u_val

        elif method == "manual":
            if manual_u_array is None or manual_u_array.shape != self.internal_energy.shape:
                error("Invalid internal energy array. Have shape", manual_u_array.shape, "need shape", self.internal_energy.shape)
            self.internal_energy = manual_u_array

        else:
            error("init_internal_energy: unknown method", method)

        return


    def init_mass_fractions(self, method, const_mass_fractions_val=None, manual_mass_fractions_array = None, XH = -1.):
        """
        initialize the mass_fractions fields.

        method: 
            "const" :   constant mass_fractions energy everywhere
                        you need to provide "const_mass_fractions_val"
            "manual":   set manual array.
                        you need to provide "manual_mass_fractions_array"
            "equilibrium": Assume ionization equilibrium. You need to provide
                        total hydrogen mass fraction XH
        """

        if method == "const":
            if const_mass_fractions_val is None:
                error("Invalid mass_fractions", const_mass_fractions_val)
            const_mass_fractions_val = np.array(const_mass_fractions_val)
            if const_mass_fractions_val.shape[0] != NSPECIES:
                error("Invalid species count. Got", const_mass_fractions_val.shape[0], "need", NSPECIES)
            
            if self.dimension == 2:
                self.mass_fractions[:, :] = const_mass_fractions_val
            if self.dimension == 3:
                self.mass_fractions[:, :, :] = const_mass_fractions_val

        elif method == "manual":
            if manual_mass_fractions_array is None or manual_mass_fractions_array.shape != self.mass_fractions.shape:
                error("Invalid mass_fractions array. Have shape", manual_mass_fractions_array.shape, "need shape", self.mass_fractions.shape)
            self.mass_fractions = manual_mass_fractions_array

        elif method == "equilibrium":
            XH_arr = np.ones(self.internal_energy.shape) * XH
            XHe_arr = 1. - XH_arr
            T, mu, XH0, XHp, XHe0, XHep, XHepp = get_temperature_from_internal_energy(self.internal_energy, XH_arr, XHe_arr)
            self.temperature = T
            if self.dimension == 2:
                self.mass_fractions[:, :, 0] = XH0
                self.mass_fractions[:, :, 1] = XHp
                if NSPECIES > 2:
                    self.mass_fractions[:, :, 2] = XHe0
                    self.mass_fractions[:, :, 3] = XHep
                    self.mass_fractions[:, :, 4] = XHepp
            elif self.dimension == 3:
                self.mass_fractions[:, :, :, 0] = XH0
                self.mass_fractions[:, :, :, 1] = XHp
                if NSPECIES > 2:
                    self.mass_fractions[:, :, :, 2] = XHe0
                    self.mass_fractions[:, :, :, 3] = XHep
                    self.mass_fractions[:, :, :, 4] = XHepp

        else:
            error("init_mass_fractions: unknown method", method)

        self.check_mass_fractions()

        return


    def check_mass_fractions(self):
        """
        check that the mass fractions sum up to ~1
        """
        # in case of roundoff errors
        mask_neg = self.mass_fractions < 0.
        self.mass_fractions[mask_neg] = 0.

        comp_tot = self.mass_fractions.sum(axis=-1)
        diff = np.abs(comp_tot - 1.)
        mask = diff > 1e-3
        if mask.any():
            error("Got wrong mass fractions", comp_tot[mask])


    def update_temperature(self):
        """
        Given the internal energy, density, and ionization mass fractions,
        compute the temperature of the entire grid
        """
