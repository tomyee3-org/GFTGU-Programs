"""
Physics routines for Orbit.

State vector
------------
x, y   : position (m)
vx, vy : velocity (m/s)

Gravity
-------
For a fixed central point mass with gravitational parameter mu = GM,

    a = -mu * r_vector / r^3

The central body remains fixed, so this is a test-particle model.
"""

from __future__ import annotations

import math


# IAU 2015 nominal solar mass parameter (m^3 s^-2).
GM_SUN = 1.3271244e20


def compute_acceleration(x: float, y: float, mu: float) -> tuple[float, float]:
    """Return Newtonian gravitational acceleration components."""
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(mu)):
        raise ValueError("x, y, and mu must be finite.")
    if mu <= 0.0:
        raise ValueError("mu=GM must be positive.")

    r2 = x * x + y * y
    if r2 <= 0.0:
        raise ValueError("The point-mass gravitational field is singular at r=0.")

    r = math.sqrt(r2)
    factor = -mu / (r2 * r)
    return factor * x, factor * y


def specific_energy(
    x: float,
    y: float,
    vx: float,
    vy: float,
    mu: float,
) -> float:
    """Return specific mechanical energy, v^2/2 - mu/r (J/kg)."""
    if not all(math.isfinite(value) for value in (x, y, vx, vy, mu)):
        raise ValueError("Position, velocity, and mu must be finite.")
    if mu <= 0.0:
        raise ValueError("mu=GM must be positive.")

    r = math.hypot(x, y)
    if r <= 0.0:
        raise ValueError("Specific energy is undefined at r=0.")
    return 0.5 * (vx * vx + vy * vy) - mu / r


def specific_angular_momentum(
    x: float,
    y: float,
    vx: float,
    vy: float,
) -> float:
    """Return signed specific angular momentum h_z = x*vy - y*vx."""
    if not all(math.isfinite(value) for value in (x, y, vx, vy)):
        raise ValueError("Position and velocity components must be finite.")
    return x * vy - y * vx
