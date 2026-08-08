"""
Binary_RK45 orbit physics module.

This module encodes the Newtonian two-body gravitational
equations of motion and associated energies.

Only physical constants appear as literals here.
"""

import numpy as np

# Newton's gravitational constant (SI)
G = 6.6726e-11

def acceleration_binary(t, state, MA, MB):
    """
    Computes accelerations for a Newtonian two-body system.
    State vector:
        [xA, yA, xB, yB, vA, uA, vB, uB]
    """

    xA, yA, xB, yB, vA, uA, vB, uB = state

    # Displacement of A from B (Schutz uses xAB = xA - xB)
    xAB = xA - xB
    yAB = yA - yB

    rAB = np.sqrt(xAB*xAB + yAB*yAB)
    rAB3 = rAB**3

    # Gravitational parameters (GM)
    kGravityA = G * MA
    kGravityB = G * MB

    # Acceleration of A (toward B)
    axA = -kGravityB * xAB / rAB3
    ayA = -kGravityB * yAB / rAB3

    # Acceleration of B (toward A)
    axB =  kGravityA * xAB / rAB3
    ayB =  kGravityA * yAB / rAB3

    return np.array([vA, uA, vB, uB, axA, ayA, axB, ayB])
