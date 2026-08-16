"""
Newtonian two-body physics for Binary.
"""

from dataclasses import dataclass
from math import hypot

G: float = 6.67430e-11  # m^3 kg^-1 s^-2


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
    """Return A-minus-B displacement, separation, and separation cubed."""
    xAB = xA - xB
    yAB = yA - yB
    rAB = hypot(xAB, yAB)

    if rAB == 0.0:
        raise ValueError(
            "The two point masses have zero separation. "
            "Newtonian point-mass gravity is singular at r = 0."
        )

    return xAB, yAB, rAB, rAB ** 3


def accelerations(
    MA: float, MB: float,
    xA: float, yA: float,
    xB: float, yB: float,
):
    """Return the Newtonian acceleration components of bodies A and B."""
    xAB, yAB, _, rAB3 = relative_displacement(xA, yA, xB, yB)

    axA = -G * MB * xAB / rAB3
    ayA = -G * MB * yAB / rAB3
    axB =  G * MA * xAB / rAB3
    ayB =  G * MA * yAB / rAB3
    return axA, ayA, axB, ayB


def energies(
    MA: float, MB: float,
    xA: float, yA: float, vA: float, uA: float,
    xB: float, yB: float, vB: float, uB: float,
):
    """Return gravitational potential, kinetic, and total system energy."""
    _, _, rAB, _ = relative_displacement(xA, yA, xB, yB)

    U = -G * MA * MB / rAB
    KA = 0.5 * MA * (vA * vA + uA * uA)
    KB = 0.5 * MB * (vB * vB + uB * uB)
    K = KA + KB
    return U, K, K + U
