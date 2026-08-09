"""
physics_earthorbit.py

Physical constants and gravitational acceleration law for the EarthOrbit
simulation (Newton's Cannon thought experiment).

The only literal numbers in this module are universal physical constants
and Earth's measured surface parameters.

Force law: two options are available, selected via the `force_law`
argument to compute_acceleration().

"simplified" (the default): does NOT use the
full inverse-square law. Instead it takes the MAGNITUDE of the
gravitational acceleration to be the constant surface value g, always
redirected toward Earth's center as the projectile moves:

    ax = -g * x / r
    ay = -g * y / r

so that |a| = g everywhere, regardless of altitude -- only the
direction changes. This is explicitly described in the text as
"a good approximation as long as the initial launch location is not
too high (less than a few kilometers)".

It is intentionally simpler than the true inverse-square law,
so that this exercise can show
where the approximation holds and where it visibly breaks down at
higher launch altitudes (e.g. as a precessing, non-closing rosette
instead of a clean ellipse).

"inverse_square": the physically accurate law, matching the one used
in the Orbit program:

    |a| = g * (R_Earth / r)^2

which falls off correctly with altitude and produces closed Keplerian
ellipses. Useful for contrasting against the simplified law.
"""

import math
from typing import Literal

# Physical constants / Earth parameters
G_SURFACE = 9.8            # gravitational acceleration at Earth's surface (m/s^2)
R_EARTH   = 6_378_200.0   # mean radius of the Earth (m)

# Composite constant for the inverse-square option: k = g * R^2
# (analogous to GM in the Orbit program). Numerically equals
# GM_Earth ~ 3.987e14 m^3/s^2.
_K = G_SURFACE * R_EARTH * R_EARTH

ForceLaw = Literal["simplified", "inverse_square"]


def compute_acceleration(
    x: float, y: float, force_law: ForceLaw = "simplified"
) -> tuple[float, float]:
    """
    Compute the gravitational acceleration components at position (x, y).

    Parameters
    ----------
    x, y : float
        Position of the projectile in metres (origin = Earth's centre).
    force_law : "simplified" or "inverse_square"
        "simplified" (default) uses Schutz's constant-magnitude
        approximation (ax=-g*x/r, ay=-g*y/r), so |a| = g always and
        only the direction changes with position -- accurate near the
        surface but increasingly wrong at higher altitude, by design.
        "inverse_square" uses the physically accurate |a| = g*(R_Earth/r)^2
        law, which falls off correctly with altitude.

    Returns
    -------
    ax, ay : float
        Acceleration components in m/s^2.
    """
    r2 = x * x + y * y
    r = math.sqrt(r2)

    if force_law == "simplified":
        ax = -G_SURFACE * x / r
        ay = -G_SURFACE * y / r
    elif force_law == "inverse_square":
        r3 = r * r2
        ax = -_K * x / r3
        ay = -_K * y / r3
    else:
        raise ValueError(f"Unknown force_law: {force_law!r}")

    return ax, ay
