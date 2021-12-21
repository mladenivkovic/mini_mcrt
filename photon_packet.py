#!/usr/bin/env python3

#--------------------------------------
# Photon packet related class
#--------------------------------------

import numpy as np
from config import *

class photon_packet():

    x = 0.
    y = 0.
    z = 0.
    direction = np.zeros(2)
    energy = 0.
    cell_index_i = -1 # first index of cell to transverse next
    cell_index_j = -1 # second index of cell to transverse next
    cell_index_k = -1 # third index of cell to transverse next
    optical_depth_transversed = 0
    optical_depth_to_reach = -1
    
    def __init__(self, x, y, z, energy):
        self.x = x
        self.y = y
        self.z = z
        self.energy = energy
        return

    def generate_random_direction(self):
        error("TODO")
        self.direction = np.array((0., 0.))
        return

    def sample_optical_depth(self):
        error("TODO")
        self.optical_depth_to_reach = 1.
        return

    def propagate(self, grid):
        """
        Propagate through a single cell
        """

        i = self.cell_index_i
        j = self.cell_index_j
        k = self.cell_index_k

        center = grid.get_cell_center(i, j, k)

        xc = center[0]
        yc = center[1]
        zc = center[2]

        xp = self.x
        yp = self.y
        zp = self.z

        dx_half = grid.dx * 0.5
        xup = xc + dx_half
        xdown = xc - dx_half
        yup = yc + dx_half
        ydown = yc - dx_half
        zup = zc + dx_half
        zdown = zc - dx_half

        phi = self.direction[0]
        
        xnew = None
        ynew = None
        inew = None
        jnew = None

        #  print("xp", xp, yp, zp)
        #  print("xc", xc, yc, zc)
        #  print("xlims", xdown, xup)
        #  print("ylims", ydown, yup)
        #  print("xp - xlim", abs(xp - xdown) / xdown, abs(xup - xp) / xup, abs(yup - yp)/yup, abs(yp - ydown)/ydown)

        # on which wall are we currently?
        if abs(xp - xdown) < 1e-5 * xdown:
            # left wall

            if phi > 0.5 * np.pi and phi < 1.5 * np.pi:
                error("wrong wall for given angle v1")

            if phi <= 0.5 * np.pi:
                # get phi max to hit upper wall before right wall
                dwall = yup - yp
                phi_max = np.arctan(dwall / grid.dx)

                if phi <= phi_max:
                    # we're hitting the right wall
                    xnew = xup
                    ynew = yp + np.tan(phi) * grid.dx
                    inew = i + 1
                    jnew = j
                else:
                    # we're hitting the upper wall
                    xnew = xp + dwall / np.tan(phi)
                    ynew = yup
                    inew = i
                    jnew = j + 1

            elif phi >= 1.5 * np.pi:
                # get phi max to hit lower wall before right wall
                dwall = ydown - yp
                phi_max = np.arctan(dwall / grid.dx)

                if phi <= phi_max:
                    # we're hitting the lower wall
                    xnew = xp + np.tan(phi - 1.5 * np.pi) * abs(dwall)
                    ynew = ydown
                    inew = i
                    jnew = j - 1

                else:
                    # we're hitting the right wall
                    xnew = xup
                    ynew = yp - grid.dx * np.tan(2 * np.pi - phi) 
                    inew = i + 1
                    jnew = j

        elif abs(xp - xup) < 1e-5 * xup:
            # right wall
            if phi < 0.5 * np.pi and phi > 1.5 * np.pi:
                error("wrong wall for given angle v2")

            # max phi to hit top wall
            dwall_top = yup - yp
            phi_max_top = np.pi - np.arctan(dwall_top/grid.dx)
            # max phi to hit left wall
            dwall_bottom = yp - ydown
            phi_max_left = np.pi + np.arctan(dwall_bottom / grid.dx)

            if phi < phi_max_top:
                # we're hitting the top wall
                xnew = xp - dwall_top * np.tan(phi - 0.5 * phi)
                ynew = yup
                inew = i
                jnew = j + 1

            elif phi < phi_max_left:
                # we're hitting the left wall
                xnew = xdown
                ynew = yp + np.tan(np.pi - phi) * grid.dx
                inew = i - 1
                jnew = j

            else:
                # we're hitting the bottom wall
                xnew = xp - np.tan(2 * np.pi - phi) * dwall_bottom
                ynew = ydown
                inew = i
                jnew = j - 1

        elif abs(yp - ydown) < 1e-5 * ydown:
            # bottom wall
            if phi > np.pi:
                error("wrong wall for given angle v3")

            # max phi to hit right wall
            dwall_right = xup - xp
            phi_max_right = np.arctan(grid.dx / dwall_right)
            # max phi to hit top wall
            dwall_left = xp - xdown
            phi_max_top = np.pi - np.arctan(grid.dx / dwall_left)

            if phi < phi_max_right:
                # we're hitting the right wall
                xnew = xup
                ynew = yp + dwall_right * np.arctan(phi)
                inew = i + 1
                jnew = j

            elif phi < phi_max_top:
                # we're hitting the top wall
                xnew = xp + grid.dx / np.arctan(phi)
                ynew = yup
                inew = i
                jnew = j + 1

            else:
                # we're hitting the left wall
                xnew = xdown
                ynew = yp + dwall_left * np.arctan(np.pi - phi)
                inew = i - 1
                jnew = j


        elif abs(yp - yup) < 1e-5 * yup:
            # top wall
            if phi < np.pi:
                error("wrong wall for given angle v4")

            # max phi to hit left wall
            dwall_left = xp - xdown
            phi_max_left = np.arctan(grid.dx / dwall_left) + np.pi
            # max phi to hit bottom wall
            dwall_right = xup - xp
            phi_max_bottom = 2 * np.pi - np.arctan(grid.dx/dwall_right)

            if phi < phi_max_left:
                # we're hitting the left wall
                xnew = xdown
                ynew = yup - np.arctan(phi - np.pi) * dwall_left 
                inew = i - 1
                jnew = j

            elif phi < phi_max_bottom:
                # we're hitting the bottom wall
                xnew = xp + np.arctan(phi - 1.5 * np.pi) * grid.dx
                ynew = ydown
                inew = i
                jnew = j - 1

            else:
                # we're hitting the right wall
                xnew = xup
                ynew = yup - dwall * np.tan(2 * np.pi - phi) 
                inew = i + 1
                jnew = j

        if xnew is None or ynew is None:
            error("No xnew, ynew", xnew, ynew)
        if inew is None or jnew is None:
            error("No inew, jnew", inew, jnew)


        # get length through the cell that has been passed
        l = np.sqrt((xnew - xp)**2 + (ynew - yp)**2)

        grid.update_cell_radiation(i, j, k, l)
        self.cell_index_i = inew
        self.cell_index_j = jnew
        self.cell_index_k = 0
        self.x = xnew
        self.y = ynew
        self.z = 0.

        return

    def is_in_box(self, grid):
        """
        Is this packet still inside the box?
        """
        if self.cell_index_i < 0 or self.cell_index_i >= grid.extent:
            return False
        if self.cell_index_j < 0 or self.cell_index_j >= grid.extent:
            return False
        if self.cell_index_k < 0 or self.cell_index_k >= grid.extent:
            return False
        return True

