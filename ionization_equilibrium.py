#!/usr/bin/env python3

# ----------------------------------------------------
# Find the ionization equilibrium for a given
# internal energy of the gas.
#
# The issue is that the ionization state determines
# the mean molecular weight of the gas, while it
# depends on the gas temperature. Conversely, the
# temperature determines the ionization state of the
# gas if it is in ionization equilibrium. So we need
# to iterate!
# NOTE: all units in CGS!
# ----------------------------------------------------

import numpy as np

# globally defined constants
import constants
from gas_functions import *

# Set some initial conditions
# --------------------------------

epsilon_iter = 1e-3 # convergence criterion
iter_max = 100

def newton_raphson_iteration(u_expect, u_guess, T_guess, mu_guess):
    """
    One iteration of the Newton-Raphson root 
    finding algorithm to find the correct values for
    temperature and ionization equilibrium by checking
    whether the obtained internal energy given by the
    temperature guess is close enough to the expected
    internal energy.

    u_expect: The correct (given) internal energy of the gas
    u_guess: The current guess for internal energy of the gas
    T_guess: current guess for temperature

    returns: T_next
        next temperature to iterate over
    """

    negatives = T_guess < 0
    if negatives.any():
        T_guess[negatives] = 0.1
        print(" Warning: Got negative temperature, resetting.")

    # find next temperature guess by solving linear equation
    # m * T_next + n = u_expect - u_guess_new ~ 0
    # NOTE: we pretend that the function that we're looking the
    # root of is f(T) = u_expect - u_guess(T), with u_expect = const.
    # so df/dT = - du_guess/dT, therefore we add a minus sign here.
    m = -internal_energy_derivative(T_guess, mu_guess)
    n = u_expect - u_guess - m * T_guess
    T_next = -n / m

    return T_next


def get_temperature_from_internal_energy(u_expect, XH, XHe):
    """
    Find the gas temperature for a given
    internal energy u_expect.
    XH: total mass fraction of hydrogen
    XHe: total mass fraction of helium
    """

    # get first estimate for temperature.
    # First assume we're fully neutral.
    XH0 = XH.copy()
    XHp = np.zeros(XH.shape, dtype=float)
    XHe0 = XHe.copy()
    XHep = np.zeros(XHe.shape, dtype=float)
    XHepp = np.zeros(XHe.shape, dtype=float)
    mu_guess = mean_molecular_weight(XH0, XHp, XHe0, XHep, XHepp)
    T_guess = gas_temperature(u_expect, mu_guess)

    # If we're above the temperature threshold with this guess,
    # assume we're fully ionized as first guess instead.
    above_ion_thresh = T_guess > constants.T_ion_thresh
    if above_ion_thresh.any():
        XHp[above_ion_thresh] = XH[above_ion_thresh].copy()
        XH0[above_ion_thresh] = 0. # do Hp first so you don't overwrite stuff
        XHepp[above_ion_thresh] = XHe[above_ion_thresh].copy()
        XHe0[above_ion_thresh] = 0. # do XHepp  first so you don't overwrite stuff
        XHep[above_ion_thresh] = 0.
        mu_guess[above_ion_thresh] = mean_molecular_weight(XH0[above_ion_thresh], XHp[above_ion_thresh], XHe0[above_ion_thresh], XHep[above_ion_thresh], XHepp[above_ion_thresh])
        T_guess[above_ion_thresh] = gas_temperature(u_expect[above_ion_thresh], mu_guess[above_ion_thresh])

    # get updated mean molecular weight
    XH0, XHp, XHe0, XHep, XHepp = get_mass_fractions(T_guess, XH, XHe)
    mu_guess = mean_molecular_weight(XH0, XHp, XHe0, XHep, XHepp)

    # get first internal energy guess
    u_guess = internal_energy(T_guess, mu_guess)

    niter = 0
    du = u_expect - u_guess
    du_old = u_expect - u_guess

    repeat = np.abs(du) >= epsilon_iter * u_expect

    # start iteration
    while repeat.any():
        niter += 1

        if niter > iter_max:
            print("Error: Iteration didn't converge")
            print("     u              = ", u_guess[repeat])
            print("     T              = ", T_guess[repeat])
            print("     u_expect - u   = ", du[repeat])
            print("     1 - u/u_expect = ", 1.0 - u_guess[repeat] / u_expect[repeat])
            return T_guess, mu_guess, XH0, XHp, XHe0, XHep, XHepp

        # prepare arrays
        T_next = T_guess.copy()
        mu_next = mu_guess.copy()
        u_next = u_guess.copy()

        # do a Newton-Raphson iteration
        T_next[repeat] = newton_raphson_iteration(u_expect[repeat], u_guess[repeat], T_guess[repeat], mu_guess[repeat])

        # Given the new temperature guess, compute the
        # expected mean molecular weight
        XH0[repeat], XHp[repeat], XHe0[repeat], XHep[repeat], XHepp[repeat] = get_mass_fractions(T_next[repeat], XH[repeat], XHe[repeat])

        mu_next[repeat] = mean_molecular_weight(XH0[repeat], XHp[repeat], XHe0[repeat], XHep[repeat], XHepp[repeat])

        # now given the new temperature and mass fraction guess, update the
        # expected gas internal energy
        u_next[repeat] = internal_energy(T_next[repeat], mu_next[repeat])

        # save the old internal energy, and get the new one
        du_old = du.copy()
        du[repeat] = u_expect[repeat] - u_next[repeat]

        # if we're oscillating between positive and negative values,
        # try a bisection to help out
        oscillate = du_old * du < 0.
        if oscillate.any():

            T_next[oscillate] = 0.5 * (T_guess[oscillate] + T_next[oscillate])
            XH0[oscillate], XHp[oscillate], XHe0[oscillate], XHep[oscillate], XHepp[oscillate] = get_mass_fractions(T_next[oscillate], XH[oscillate], XHe[oscillate])
            mu_next[oscillate] = mean_molecular_weight(XH0[oscillate], XHp[oscillate], XHe0[oscillate], XHep[oscillate], XHepp[oscillate])
            u_next[oscillate] = internal_energy(T_next[oscillate], mu_next[oscillate])

        # reset what "current values" are and iterate again
        T_guess = T_next.copy()
        mu_guess = mu_next.copy()
        u_guess = u_next.copy()
        du = u_expect - u_next
        repeat = np.abs(du) >= epsilon_iter * u_expect

    return T_guess, mu_guess, XH0, XHp, XHe0, XHep, XHepp


if __name__ == "__main__":

    T, mu, XH0, XHp, XHe0, XHep, XHepp = get_temperature_from_internal_energy(u)
