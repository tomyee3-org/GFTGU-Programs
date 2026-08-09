"""
This module encodes the Schwarzschild test-particle acceleration
used in RelativisticOrbit, with only physical constants as literals.
"""

import math

# Physical constants (SI units)
GMSOLAR = 1.327e20          # GM_sun
C2 = 8.98755e16             # c^2
HORIZON_RADIUS = 2.0 * GMSOLAR / C2  # Schwarzschild radius for 1 M_sun


def kepler_constants(x_init: float, u_init: float) -> tuple[float, float]:
    """
    Compute Kepler's constant K and Q = 12 K^2 / c^2 for the orbit.

    K = 0.5 * u * x0
    Q = 12 * K^2 / c^2
    """
    K = 0.5 * u_init * x_init
    Q = 12.0 * K * K / C2
    return K, Q


def schwarzschild_acceleration(x: float, y: float, K: float, Q: float) -> tuple[float, float]:
    """
    Relativistic (Schwarzschild) acceleration components for a test particle.

    ax = -kGravity * x / r^3 * (1 + Q / r^2)
    ay = -kGravity * y / r^3 * (1 + Q / r^2)

    where r^2 = x^2 + y^2, r^3 = r * r^2.
    """
    r2 = x * x + y * y
    r = math.sqrt(r2)
    r3 = r * r2

    factor = -GMSOLAR / r3 * (1.0 + Q / r2)
    ax = factor * x
    ay = factor * y
    return ax, ay
