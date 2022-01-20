#!/usr/bin/env python3

# --------------------------------------
# Photon packet related class
# --------------------------------------

import numpy as np
from config import *

rng = np.random.default_rng(627)

class photon_packet:

    x = 0.0
    y = 0.0
    z = 0.0
    direction = np.zeros(2)
    energy = 0.0
    cell_index_i = -1  # first index of cell to transverse next
    cell_index_j = -1  # second index of cell to transverse next
    cell_index_k = -1  # third index of cell to transverse next

    cell_wall_index = -1 # Wall index definition: 
                         #     2
                         #   |---|
                         # 3 |   | 1
                         #   |---|
                         #     4
    optical_depth_transversed = 0
    optical_depth_to_reach = -1
    dt = -1. # current time step

    def __init__(self, x, y, z, energy, dt):
        self.x = x
        self.y = y
        self.z = z
        self.energy = energy
        self.dt = dt
        return

    def generate_random_direction(self):
        theta = np.arccos(2. * rng.random() - 1.)
        phi = rng.random() * 2. * np.pi
        self.direction = np.array((theta, phi))
        return

    def sample_optical_depth(self):
        tau = - np.log(rng.random())
        self.optical_depth_to_reach = tau
        return

    def check_direction(self):
        """
        Make sure 0 <= theta <= pi and 0 <= phi <= 2*pi

        This is called during first propagation.
        """
        theta = self.direction[0]
        while theta > np.pi:
            theta -= np.pi
        while theta < 0.:
            theta += np.pi

        if theta != self.direction[0]:
            debugging_msg("Corrected theta from", self.direction[0], "to", theta)

        phi = self.direction[1]
        while phi > 2 * np.pi:
            phi -= 2 * np.pi
        while phi < 0.:
            phi += 2 * np.pi

        if phi != self.direction[1]:
            debugging_msg("Corrected phi from", self.direction[1], "to", phi)

        self.direction[0] = theta
        self.direction[1] = phi

    def propagate(self, grid):
        """
        Propagate through a single cell from a 
        wall to the next wall.

        returns: absorbed
        true if photon packet is absorbed and shouldn't propagate any further
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

        phi = self.direction[1]

        xnew = None
        ynew = None
        inew = None
        jnew = None
        next_wall_index = None

        # on which wall are we currently?
        #-----------------------------------

        if self.cell_wall_index == 3:
            # left wall
            #-------------
            debugging_msg("left wall")

            if phi > 0.5 * np.pi and phi < 1.5 * np.pi:
                error("wrong wall for given angle v1")

            if phi <= 0.5 * np.pi:
                # get phi max to hit upper wall before right wall
                dely = max(yup - yp, 1e-6 * grid.dx)
                phi_max_top = np.arctan(dely / grid.dx)

                if phi <= phi_max_top:
                    # we're hitting the right wall
                    xnew = xup
                    ynew = yp + np.tan(phi) * grid.dx
                    inew = i + 1
                    jnew = j
                    next_wall_index = 3 # if we hit right wall, next cell's wall will be left
                else:
                    # we're hitting the upper wall
                    xnew = xp + dely / np.tan(phi)
                    ynew = yup
                    inew = i
                    jnew = j + 1
                    next_wall_index = 4 # if we hit top wall, next cell's wall will be bottom

            elif phi >= 1.5 * np.pi:
                # get phi max to hit lower wall before right wall
                dely = max(yp - ydown, 1e-6 * grid.dx)
                phi_max_bottom = np.arctan(grid.dx / dely) + 1.5 * np.pi

                if phi <= phi_max_bottom:
                    # we're hitting the lower wall
                    xnew = xp + np.tan(phi - 1.5 * np.pi) * dely
                    ynew = ydown
                    inew = i
                    jnew = j - 1
                    next_wall_index = 2 # if we hit bottom wall, next cell's wall will be top

                else:
                    # we're hitting the right wall
                    xnew = xup
                    ynew = yp - grid.dx * np.tan(2 * np.pi - phi)
                    inew = i + 1
                    jnew = j
                    next_wall_index = 3 # if we hit right wall, next cell's wall will be left

        elif self.cell_wall_index == 1:
            # right wall
            #------------
            debugging_msg("right wall")
            if phi < 0.5 * np.pi or phi > 1.5 * np.pi:
                error("wrong wall for given angle v2")

            # max phi to hit top wall
            dely_top = max(yup - yp, 1e-6 * grid.dx)
            phi_max_top = np.pi - np.arctan(dely_top / grid.dx)
            # max phi to hit left wall
            dely_bottom = max(yp - ydown, 1e-6 * grid.dx)
            phi_max_left = 1.5 * np.pi - np.arctan(grid.dx / dely_bottom)

            if phi < phi_max_top:
                # we're hitting the top wall
                xnew = xp - dely_top / np.tan(np.pi - phi)
                ynew = yup
                inew = i
                jnew = j + 1
                next_wall_index = 4 # if we hit top wall, next cell's wall will be bottom

            elif phi < phi_max_left:
                # we're hitting the left wall
                if phi <= np.pi:
                    ynew = yp + np.tan(np.pi - phi) * grid.dx
                else:
                    ynew = yp - grid.dx / np.tan(1.5 * np.pi - phi)
                xnew = xdown
                inew = i - 1
                jnew = j
                next_wall_index = 1 # if we hit top left, next cell's wall will be right

            else:
                # we're hitting the bottom wall
                xnew = xp - np.tan(1.5 * np.pi - phi) * dely_bottom
                ynew = ydown
                inew = i
                jnew = j - 1
                next_wall_index = 2 # if we hit bottom wall, next cell's wall will be top

        elif self.cell_wall_index == 4:
            # bottom wall
            #------------------
            debugging_msg("bottom wall")
            if phi > np.pi:
                error("wrong wall for given angle v3")

            # max phi to hit right wall
            delx_right = max(xup - xp, 1e-6 * grid.dx)
            phi_max_right = np.arctan(grid.dx / delx_right)
            # max phi to hit top wall
            delx_left = max(xp - xdown, 1e-6 * grid.dx)
            phi_max_top = np.pi - np.arctan(grid.dx / delx_left)

            if phi < phi_max_right:
                # we're hitting the right wall
                xnew = xup
                ynew = yp + delx_right * np.arctan(phi)
                inew = i + 1
                jnew = j
                next_wall_index = 3 # if we hit right wall, next cell's wall will be left

            elif phi < phi_max_top:
                # we're hitting the top wall
                if phi < 0.5 * np.pi:
                    xnew = xp + grid.dx / np.arctan(phi)
                else:
                    xnew = xp - grid.dx / np.arctan(np.pi - phi)
                ynew = yup
                inew = i
                jnew = j + 1
                next_wall_index = 4 # if we hit top wall, next cell's wall will be bottom

            else:
                # we're hitting the left wall
                xnew = xdown
                ynew = yp + delx_left * np.arctan(np.pi - phi)
                inew = i - 1
                jnew = j
                next_wall_index = 1 # if we hit left wall, next cell's wall will be right

        elif self.cell_wall_index == 2:
            # top wall
            #---------------
            debugging_msg("top wall")
            if phi < np.pi:
                error("wrong wall for given angle v4")

            # max phi to hit left wall
            delx_left = max(xp - xdown, 1e-6 * grid.dx)
            phi_max_left = np.pi + np.arctan(grid.dx / delx_left)
            # max phi to hit bottom wall
            delx_right = max(xup - xp, 1e-6 * grid.dx)
            phi_max_bottom = 2 * np.pi - np.arctan(grid.dx / delx_right)

            if phi < phi_max_left:
                # we're hitting the left wall
                xnew = xdown
                ynew = yp - np.arctan(phi - np.pi) * delx_left
                inew = i - 1
                jnew = j
                next_wall_index = 1 # if we hit left wall, next cell's wall will be right

            elif phi < phi_max_bottom:
                # we're hitting the bottom wall
                if phi < 1.5 * np.pi:
                    xnew = xp - grid.dx * np.arctan(1.5 * np.pi - phi)
                else:
                    xnew = xp + grid.dx * np.arctan(phi - 1.5 * np.pi)
                ynew = ydown
                inew = i
                jnew = j - 1
                next_wall_index = 2 # if we hit bottom wall, next cell's wall will be top

            else:
                # we're hitting the right wall
                xnew = xup
                ynew = yp - delx_right / np.tan(phi - 1.5 * np.pi)
                inew = i + 1
                jnew = j
                next_wall_index = 3 # if we hit right wall, next cell's wall will be left
        else:
            error("Invalid wall index", self.cell_wall_index)

        if xnew is None or ynew is None:
            error("No xnew, ynew", xnew, ynew)
        if inew is None or jnew is None:
            error("No inew, jnew", inew, jnew)
        if next_wall_index is None:
            error("No next_wall_index")

        # get length through the cell that has been passed
        l = np.sqrt((xnew - xp) ** 2 + (ynew - yp) ** 2)
        tau = grid.get_optical_depth(i, j, k, l, self.energy)
        grid.update_cell_radiation(i, j, k, l, self.energy, self.dt, tau)
        self.optical_depth_transversed += tau

        if self.optical_depth_transversed >= self.optical_depth_to_reach:
            # check whether we're done
            scatter = self.is_scattered(grid, i, j, k)
            if scatter:
                # re-emit
                self.cell_index_i = i
                self.cell_index_j = j
                self.cell_index_k = k
                self.x = xc
                self.y = yc
                self.z = 0.
                self.generate_random_direction()
                self.sample_optical_depth()
                self.optical_depth_transversed = 0.
                self.first_propagation(grid, scatter=True)
                return False
            else:
                return True

        else:
            # update and continue

            self.cell_index_i = inew
            self.cell_index_j = jnew
            self.cell_index_k = 0
            self.cell_wall_index = next_wall_index
            self.x = xnew
            self.y = ynew
            self.z = 0.0
            return False

        return False

    def first_propagation(self, grid, scatter=False):
        """
        First propagation: From arbitrary 
        position to a cell wall.

        scatter: if True, we're re-emitting a photon from a cell.
        Don't update the grid radiation fields then, as it is already
        updated during the normal propagation
        """

        self.check_direction()
        self.optical_depth_transversed = 0.

        self.cell_index_i = int(self.x / grid.boxlen * grid.extent)
        self.cell_index_j = int(self.y / grid.boxlen * grid.extent)
        self.cell_index_k = int(self.z / grid.boxlen * grid.extent)

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

        phi = self.direction[1]

        xnew = None
        ynew = None
        inew = None
        jnew = None
        next_wall_index = None

        if phi < 0.5 * np.pi:
            # we can hit right or top wall
            delx = xup - xp
            dely = yup - yp
            # max angle to hit right wall
            phi_max_right = np.arctan(dely / delx)
            if phi < phi_max_right:
                # hitting right wall
                inew = i + 1
                jnew = j
                xnew = xup
                ynew = yp + np.cos(phi) * delx
                next_wall_index = 3 # if we hit right wall, next cell's wall will be left
            else:
                # hitting top wall
                inew = i
                jnew = j + 1
                xnew = xp + dely / np.tan(phi)
                ynew = yup
                next_wall_index = 4 # if we hit top wall, next cell's wall will be bottom

        elif phi < np.pi:
            # we can hit top or left wall
            delx = max(xp - xdown, 1e-6 * grid.dx)
            dely = max(yup - yp, 1e-6 * grid.dx)
            phi_new = np.pi - phi
            phi_max_top = np.arctan(dely / delx)
            if phi_new > phi_max_top:
                # hitting top wall
                inew = i
                jnew = j + 1
                xnew = xp - dely / np.tan(phi_new)
                ynew = yup
                next_wall_index = 4 # if we hit top wall, next cell's wall will be bottom
            else:
                # hitting left wall
                inew = i - 1
                jnew = j
                xnew = xdown
                ynew = yp + delx * np.tan(phi_new)
                next_wall_index = 1 # if we hit left wall, next cell's wall will be right

        elif phi < 1.5 * np.pi:
            # we can hit left or lower wall
            delx = xp - xdown
            dely = yp - ydown
            phi_new = 1.5 * np.pi - phi
            phi_max_bottom = np.arctan(delx / dely)
            if phi_new < phi_max_bottom:
                # we hit the lower wall
                inew = i
                jnew = j - 1
                xnew = xp - dely * np.tan(phi_new)
                ynew = ydown
                next_wall_index = 2 # if we hit bottom wall, next cell's wall will be top
            else:
                # we hit the left wall
                inew = i - 1
                jnew = j
                xnew = xdown
                ynew = yp - delx * np.tan(phi - np.pi)
                next_wall_index = 1 # if we hit left wall, next cell's wall will be right
        else:
            # we hit lower or right wall
            delx = max(xup - xp, 1e-6 * grid.dx)
            dely = max(yp - ydown, 1e-6 * grid.dx)
            phi_new = phi - 1.5 * np.pi
            phi_max_bottom = np.arctan(delx / dely)
            if phi_new <= phi_max_bottom:
                # we're hitting bottom wall
                inew = i
                jnew = j - 1
                xnew = xp + dely * np.tan(phi_new)
                ynew = ydown
                next_wall_index = 2 # if we hit bottom wall, next cell's wall will be top
            else:
                # we're hitting right wall
                inew = i + 1
                jnew = j
                xnew = xup
                ynew = yp - delx / np.tan(phi_new)
                next_wall_index = 3 # if we hit right wall, next cell's wall will be left

        if xnew is None or ynew is None:
            error("Error in propagate: No xnew, ynew", xnew, ynew)
        if inew is None or jnew is None:
            error("Error in propagate: No inew, jnew", inew, jnew)
        if next_wall_index is None:
            error("Error in propagate: No next wall index")

        # get length through the cell that has been passed
        l = np.sqrt((xnew - xp) ** 2 + (ynew - yp) ** 2)
        if not scatter:
            tau = grid.get_optical_depth(i, j, k, l, self.energy)
            grid.update_cell_radiation(i, j, k, l, self.energy, self.dt, tau)
            self.optical_depth_transversed += tau


        self.cell_index_i = inew
        self.cell_index_j = jnew
        self.cell_index_k = 0
        self.x = xnew
        self.y = ynew
        self.z = 0.0
        self.cell_wall_index = next_wall_index

        debugging_msg("phi", phi / np.pi, "pi; going from", i, j, "to", inew, jnew)

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

    def is_scattered(self, grid, i, j, k):
        """
        When the packet reaches the predestined optical depth,
        is it scattered? If not, it's absorbed
        """

        # note: this has no physical basis, it's just
        # for demo purposes
        scattered = rng.random() <= 0.9
        return scattered
