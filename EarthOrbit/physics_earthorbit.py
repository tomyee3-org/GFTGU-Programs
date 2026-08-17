"""
physics_earthorbit.py

Physical parameters and gravitational acceleration laws for the EarthOrbit
simulation (Newton's Cannon thought experiment).

Two force laws are available through the `force_law` argument to
compute_acceleration().

"simplified" (default)
    In this simplified model, the magnitude of the
    gravitational acceleration is held at the surface value g while its
    direction always points toward Earth's centre:

        ax = -g * x / r
        ay = -g * y / r

    This is useful close to Earth's surface but is not a global model of
    terrestrial gravity.

"inverse_square"
    Uses

        |a| = g * (R_Earth / r)^2

    so the acceleration falls with distance and produces the usual
    Keplerian inverse-square dynamics.
"""

import math
from typing import Literal

# Physical constants used by the simulation.
G_SURFACE = 9.8
R_EARTH = 6_378_200.0  # m; reference radius, close to Earth's equatorial radius

# Approximate gravitational parameter implied by the rounded values above.
# It is close to, but not exactly equal to, the accepted GM_Earth.
K_APPROX = G_SURFACE * R_EARTH * R_EARTH

ForceLaw = Literal["simplified", "inverse_square"]


def compute_acceleration(
    x: float, y: float, force_law: ForceLaw = "simplified"
) -> tuple[float, float]:
    """
    Compute gravitational acceleration components at position (x, y).

    Parameters
    ----------
    x, y : float
        Position of the projectile in metres (origin = Earth's centre).
    force_law : "simplified" or "inverse_square"
        "simplified" uses the constant-magnitude-g model.
        "inverse_square" uses the inverse-square extension.

    Returns
    -------
    ax, ay : float
        Acceleration components in m/s^2.
    """
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("x and y must be finite positions in metres")

    r2 = x * x + y * y
    r = math.sqrt(r2)
    if r == 0.0:
        raise ValueError("gravitational acceleration is undefined at Earth's centre")

    if force_law == "simplified":
        ax = -G_SURFACE * x / r
        ay = -G_SURFACE * y / r
    elif force_law == "inverse_square":
        r3 = r * r2
        ax = -K_APPROX * x / r3
        ay = -K_APPROX * y / r3
    else:
        raise ValueError(f"Unknown force_law: {force_law!r}")

    return ax, ay
