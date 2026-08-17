"""
Physics for RelativisticOrbit.

The Schwarzschild mode integrates the Newtonian-looking Cartesian form of the
equatorial timelike Schwarzschild geodesic when the independent variable is the
test particle's proper time.  A Newtonian mode is also provided for direct
comparison.
"""

from __future__ import annotations

import math

# IAU 2015 nominal solar mass parameter and exact SI speed of light.
GM_SUN = 1.3271244e20          # m^3 s^-2
C = 299_792_458.0              # m s^-1, exact
C2 = C * C

HORIZON_RADIUS = 2.0 * GM_SUN / C2
ISCO_RADIUS = 6.0 * GM_SUN / C2
PHOTON_ORBIT_RADIUS = 3.0 * GM_SUN / C2


def orbital_constants(x_init: float, u_init: float) -> tuple[float, float]:
    """
    Return the conserved specific angular momentum h and Schutz's Q.

    The particle starts at (x_init, 0) with (dx/dtau, dy/dtau)=(0, u_init),
    where tau is the particle's proper time.

      h = x_init * u_init
      K = h/2
      Q = 12 K^2/c^2 = 3 h^2/c^2
    """
    h = x_init * u_init
    q = 3.0 * h * h / C2
    return h, q


def central_acceleration(
    x: float,
    y: float,
    h: float,
    model: str = "schwarzschild",
) -> tuple[float, float]:
    """
    Return d^2(x,y)/dtau^2 for the selected orbital model.

    schwarzschild:
        a = -GM/r^3 * (1 + 3 h^2/(c^2 r^2)) * r_vector

    newtonian:
        a = -GM/r^3 * r_vector

    In Schwarzschild mode this is an exact rewriting of the equatorial timelike
    Schwarzschild geodesic for the spatial orbit r(tau), phi(tau).  The
    "acceleration" here is a coordinate second derivative; a freely falling
    particle's physical proper acceleration is zero.
    """
    r2 = x * x + y * y
    if r2 <= 0.0:
        raise ValueError("The test particle reached r=0, where this coordinate equation is singular.")

    r = math.sqrt(r2)
    r3 = r * r2

    model_key = model.lower()
    if model_key == "schwarzschild":
        correction = 1.0 + 3.0 * h * h / (C2 * r2)
    elif model_key == "newtonian":
        correction = 1.0
    else:
        raise ValueError('model must be "schwarzschild" or "newtonian".')

    factor = -GM_SUN * correction / r3
    return factor * x, factor * y


def specific_angular_momentum(
    x: float,
    y: float,
    vx: float,
    vy: float,
) -> float:
    """Return h = x*dy/dtau - y*dx/dtau."""
    return x * vy - y * vx


def effective_specific_energy(
    x: float,
    y: float,
    vx: float,
    vy: float,
    h_constant: float,
    model: str = "schwarzschild",
) -> float:
    """
    Conserved energy integral for the Newtonian-like proper-time orbit equation.

    This is a numerical diagnostic for the equation being integrated, not the
    locally measured kinetic-plus-potential energy of a relativistic observer.
    """
    r = math.hypot(x, y)
    if r <= 0.0:
        raise ValueError("Effective energy is undefined at r=0.")

    kinetic = 0.5 * (vx * vx + vy * vy)
    potential = -GM_SUN / r

    if model.lower() == "schwarzschild":
        potential -= GM_SUN * h_constant * h_constant / (C2 * r**3)
    elif model.lower() != "newtonian":
        raise ValueError('model must be "schwarzschild" or "newtonian".')

    return kinetic + potential


def circular_proper_time_speed(radius: float) -> float:
    """
    Return dy/dtau for a timelike circular Schwarzschild geodesic.

    Valid only for radius > 3 GM/c^2.  Circular timelike geodesics between
    3 GM/c^2 and 6 GM/c^2 are unstable; those above 6 GM/c^2 are stable.
    """
    denominator = radius - 3.0 * GM_SUN / C2
    if denominator <= 0.0:
        raise ValueError("No timelike circular Schwarzschild geodesic exists at or below 3GM/c^2.")
    return math.sqrt(GM_SUN / denominator)
