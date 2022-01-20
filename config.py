#!/usr/bin/env python3

# ----------------------------
# Some global configurations
# ----------------------------

debug_verbose = False


def error(*args):
    """
    Print an error and exit.
    """
    print(*args)
    quit()


def debugging_msg(*args):
    """
    Print a debugging message, if code is set to verbose.
    """
    if debug_verbose:
        print("---", *args)


# use HI and HII only for now
NSPECIES = 2
