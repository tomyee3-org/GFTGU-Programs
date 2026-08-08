"""
Binary orbit physics module.

This module encodes the Newtonian two-body gravitational
equations of motion and associated energies.

Only physical constants appear as literals here.
"""

from dataclasses import dataclass
from math import sqrt

# Newton's gravitational constant (SI units)
G: float = 6.6726e-11


@dataclass
class BinaryState:
    t: float
    xA: float
    yA: float
    vA: float
    uA: float
    xB: float
    yB: float
    vB: float
    uB: float


def relative_displacement(xA: float, yA: float, xB: float, yB: float):
    """
    Compute displacement of A from B and its magnitude and cube.

    Corresponds to Schutz's xAB, yAB, rAB, rAB3.
    """
    xAB = xA - xB
    yAB = yA - yB
    rAB2 = xAB * xAB + yAB * yAB
    rAB = sqrt(rAB2)
    rAB3 = rAB2 * rAB
    return xAB, yAB, rAB, rAB3


def accelerations(MA: float, MB: float,
                  xA: float, yA: float,
                  xB: float, yB: float):
    """
    Compute accelerations of bodies A and B due to mutual gravity.

    Java analogue:
        kGravityA = G * MA
        kGravityB = G * MB
        axA = -kGravityB * xAB / rAB3
        ayA = -kGravityB * yAB / rAB3
        axB =  kGravityA * xAB / rAB3
        ayB =  kGravityA * yAB / rAB3
    """
    xAB, yAB, rAB, rAB3 = relative_displacement(xA, yA, xB, yB)

    if rAB3 == 0.0:
        # Avoid division by zero; physically this would be a collision.
        return 0.0, 0.0, 0.0, 0.0

    kGravityA = G * MA
    kGravityB = G * MB

    axA = -kGravityB * xAB / rAB3
    ayA = -kGravityB * yAB / rAB3
    axB =  kGravityA * xAB / rAB3
    ayB =  kGravityA * yAB / rAB3

    return axA, ayA, axB, ayB


def energies(MA: float, MB: float,
             xA: float, yA: float,
             vA: float, uA: float,
             xB: float, yB: float,
             vB: float, uB: float):
    """
    Compute potential, kinetic, and total energy of the binary system.

    Potential energy:
        U = - G * MA * MB / rAB

    Kinetic energy:
        K = 0.5 * MA * (vA^2 + uA^2) + 0.5 * MB * (vB^2 + uB^2)
    """
    xAB, yAB, rAB, _ = relative_displacement(xA, yA, xB, yB)

    if rAB == 0.0:
        U = 0.0
    else:
        U = -G * MA * MB / rAB

    KA = 0.5 * MA * (vA * vA + uA * uA)
    KB = 0.5 * MB * (vB * vB + uB * uB)
    K = KA + KB

    E = K + U
    return U, K, E
