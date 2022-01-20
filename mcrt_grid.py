#!/usr/bin/env python3

# ----------------------------------------
# This module defines the grid for the
# MCRT programs
# ----------------------------------------

from config import *
from ionization_equilibrium import get_temperature_from_internal_energy

import numpy as np
import pickle


# Note:
# I prepared for a full physics module, but the only working way
# is to specify the number densities of the grid manually


class mcrt_grid:
    """
    the grid to use
    """

    dimension = 2
    extent = 0
    boxlen = 0  # size of box in each dimension
    dx = 0  # size of cell in each dimension

    density = None
    number_density = None
    internal_energy = None
    mass_fractions = None
    temperature = None

    mean_specific_intensity = None
    path_length_estimator = None
    absorbed_energy = None

    cell_index = None

    def __init__(self, boxlen, extent=64, dimension=2):
        """
        extent: grid size in each dimension
        dimension: how many dims to work in 
        """

        self.boxlen = boxlen
        self.dx = boxlen / extent
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
        self.number_density = np.zeros(shape, dtype=float)
        self.internal_energy = np.zeros(shape, dtype=float)
        self.mass_fractions = np.zeros(shape_comp, dtype=float)
        self.temperature = np.zeros(shape, dtype=float)
        self.mean_specific_intensity = np.zeros(shape, dtype=float)
        self.path_length_estimator = np.zeros(shape, dtype=float)
        self.absorbed_energy = np.zeros(shape, dtype=float)
        self.cell_index = np.zeros(shape, dtype=int)

        self.assign_cell_index()

        return

    def assign_cell_index(self):
        """
        Generate cell indices
        """

        if self.dimension == 2:
            for i in range(self.extent):
                for j in range(self.extent):
                    ind = i * self.extent + j
                    self.cell_index[i, j] = ind

        elif self.dimension == 3:
            for i in range(self.extent):
                for j in range(self.extent):
                    for k in range(self.extent):
                        ind = i * self.extent ** 2 + j * self.extent + k
                        self.cell_index[i, j, k] = ind
        return

    def get_cell_center(self, i, j, k):
        """
        Get the cell center with all 3 cell indices given
        """
        x = ((i + 0.5) / self.extent) * self.boxlen
        y = ((j + 0.5) / self.extent) * self.boxlen
        z = ((k + 0.5) / self.extent) * self.boxlen
        return np.array([x, y, z])

    def dump(self, number, basename="output_"):
        """
        Dump the entire struct as a snapshot
        """
        fname = basename + str(number).zfill(4) + ".pkl"
        f = open(fname, "wb")
        pickle.dump(self, f)
        f.close()
        return

    def init_density(self, method, const_dens_val=None, manual_dens_array=None):
        """
        initialize the density fields.   

        method: 
            "const" :   constant density everywhere
                        you need to provide "const_dens_val"
            "manual":   set manual array.
                        you need to provide "manual_dens_array"
        """

        if method == "const":
            if const_dens_val is None or const_dens_val < 0.0:
                error("Invalid density", const_dens_val)

            if self.dimension == 2:
                self.density[:, :] = const_dens_val
            if self.dimension == 3:
                self.density[:, :, :] = const_dens_val
        elif method == "manual":
            if (
                manual_dens_array is None
                or manual_dens_array.shape != self.density.shape
            ):
                error(
                    "Invalid density array. Have shape",
                    manual_dens_array.shape,
                    "need shape",
                    self.density.shape,
                )
            self.density = manual_dens_array

        else:
            error("init_density: unknown method", method)

        return

    def init_number_density(
        self, method, const_number_dens_val=None, manual_number_dens_array=None
    ):
        """
        initialize the density fields.   

        method: 
            "const" :   constant density everywhere
                        you need to provide "const_number_dens_val"
            "manual":   set manual array.
                        you need to provide "manual_number_dens_array"
        """

        if method == "const":
            if const_number_dens_val is None or const_number_dens_val < 0.0:
                error("Invalid density", const_number_dens_val)

            if self.dimension == 2:
                self.number_density[:, :] = const_number_dens_val
            if self.dimension == 3:
                self.number_density[:, :, :] = const_number_dens_val
        elif method == "manual":
            if (
                manual_number_dens_array is None
                or manual_number_dens_array.shape != self.number_density.shape
            ):
                error(
                    "Invalid number_density array. Have shape",
                    manual_number_dens_array.shape,
                    "need shape",
                    self.number_density.shape,
                )
            self.number_density = manual_number_dens_array

        return

    def init_internal_energy(self, method, const_u_val=None, manual_u_array=None):
        """
        initialize the specific internal energy fields.   

        method: 
            "const" :   constant specific internal energy everywhere
                        you need to provide "const_u_val"
            "manual":   set manual array.
                        you need to provide "manual_u_array"
        """

        if method == "const":
            if const_u_val is None or const_u_val < 0.0:
                error("Invalid internal energy", const_u_val)

            if self.dimension == 2:
                self.internal_energy[:, :] = const_u_val
            if self.dimension == 3:
                self.internal_energy[:, :, :] = const_u_val

        elif method == "manual":
            if (
                manual_u_array is None
                or manual_u_array.shape != self.internal_energy.shape
            ):
                error(
                    "Invalid internal energy array. Have shape",
                    manual_u_array.shape,
                    "need shape",
                    self.internal_energy.shape,
                )
            self.internal_energy = manual_u_array

        else:
            error("init_internal_energy: unknown method", method)

        return

    def init_mass_fractions(
        self,
        method,
        const_mass_fractions_val=None,
        manual_mass_fractions_array=None,
        XH=-1.0,
    ):
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
                error(
                    "Invalid species count. Got",
                    const_mass_fractions_val.shape[0],
                    "need",
                    NSPECIES,
                )

            if self.dimension == 2:
                self.mass_fractions[:, :] = const_mass_fractions_val
            if self.dimension == 3:
                self.mass_fractions[:, :, :] = const_mass_fractions_val

        elif method == "manual":
            if (
                manual_mass_fractions_array is None
                or manual_mass_fractions_array.shape != self.mass_fractions.shape
            ):
                error(
                    "Invalid mass_fractions array. Have shape",
                    manual_mass_fractions_array.shape,
                    "need shape",
                    self.mass_fractions.shape,
                )
            self.mass_fractions = manual_mass_fractions_array

        elif method == "equilibrium":
            XH_arr = np.ones(self.internal_energy.shape) * XH
            XHe_arr = 1.0 - XH_arr
            T, mu, XH0, XHp, XHe0, XHep, XHepp = get_temperature_from_internal_energy(
                self.internal_energy, XH_arr, XHe_arr
            )
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
        mask_neg = self.mass_fractions < 0.0
        self.mass_fractions[mask_neg] = 0.0

        comp_tot = self.mass_fractions.sum(axis=-1)
        diff = np.abs(comp_tot - 1.0)
        mask = diff > 1e-3
        if mask.any():
            error("Got wrong mass fractions", comp_tot[mask])

    def update_temperature(self):
        """
        Given the internal energy, density, and ionization mass fractions,
        compute the temperature of the entire grid
        """

        if self.dimension == 2:
            XH0 = self.mass_fractions[:, :, 0]
            XHp = self.mass_fractions[:, :, 1]
            if NSPECIES > 2:
                XHe = self.mass_fractions[:, :, 2]
                XHep = self.mass_fractions[:, :, 3]
                XHepp = self.mass_fractions[:, :, 4]
            else:
                XHe = np.zeros(self.mass_fractions[:, :, 0].shape)
                XHep = np.zeros(self.mass_fractions[:, :, 0].shape)
                XHepp = np.zeros(self.mass_fractions[:, :, 0].shape)
        elif self.dimension == 3:
            XH0 = self.mass_fractions[:, :, :, 0]
            XHp = self.mass_fractions[:, :, :, 1]
            if NSPECIES > 2:
                XHe = self.mass_fractions[:, :, :, 2]
                XHep = self.mass_fractions[:, :, :, 3]
                XHepp = self.mass_fractions[:, :, :, 4]
            else:
                XHe = np.zeros(self.mass_fractions[:, :, :, 0].shape)
                XHep = np.zeros(self.mass_fractions[:, :, :, 0].shape)
                XHepp = np.zeros(self.mass_fractions[:, :, :, 0].shape)

        mu = mean_molecular_weight(XH0, XHp, XHe0, XHep, XHepp)
        self.temperature = gas_temperature(self.internal_energy, mu)

        return

    def update_cell_radiation(self, i, j, k, l, energy, dt, tau):
        """
        i, j, k: cell indices
        l : length passed through this cell
        energy: photon packet energy
        dt: current time step
        tau: optical depth
        """

        if self.dimension == 2:
            volume = self.dx * self.dx
            self.mean_specific_intensity[i, j] += energy / dt / 4 / np.pi / volume * l
            self.path_length_estimator[i, j] += l
            self.absorbed_energy[i, j] += tau * energy / volume
        elif self.dimension == 3:
            error("TODO: grid update cell radiation 3D")
            self.mean_specific_intensity[i, j, k] += l
            self.path_length_estimator[i, j, k] += l
            self.absorbed_energy[i, j] += l
        return

    def get_cross_section(self, i, j, k, E):
        """
        get the interaction cross section of current cell 
        i, j, k: cell indexes
        E: photon packet energy
        """

        # note: this has no physical basis.
        # just try to get a couple of scatterings for demo purposes.
        # assume number density of unity to be used as default to
        # compute optical depth

        sigma = 0.5 / np.sqrt(self.extent)

        return sigma

    def get_number_density(self, i, j, k):
        """
        return current number densit of cell i, j, k
        """

        if self.dimension == 2:
            return self.number_density[i, j]
        elif self.dimension == 3:
            return self.number_density[i, j, k]
        else:
            error("wtf?")
        return

    def get_optical_depth(self, i, j, k, l, energy):
        """
        Get the optical depth of the current cell
        """

        sigma = self.get_cross_section(i, j, k, energy)
        n = self.get_number_density(i, j, k)
        tau = sigma * n * l

        return tau

    def init_step(self):
        """
        Initialize the grid for a new step
        """

        if self.dimension == 2:
            #  for i in range(self.extent):
            #      for j in range(self.extent):
            self.path_length_estimator[:, :] = 0.0
            self.absorbed_energy[:, :] = 0.0

        else:
            error("ToDo grid.init_step for 3D")
        return

    def finalise_step(self):
        """
        Finish up whatever needs finishing
        """

        #  if self.dimension == 2:
        #      for i in range(self.extent):
        #          for j in range(self.extent):
        #  else:
        #      error("ToDo grid.finalise_step for 3D")
        #
        return
