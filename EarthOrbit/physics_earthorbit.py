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

MODEL_VERSION = "1.0.0"


#: The exact source files this build identifier covers: a documentation-only
#: change, a sample-output file, or an edit to the test suite does not change
#: this value -- only the four core program modules listed here do.  Exposed
#: so callers can determine precisely what BUILD_ID covers without duplicating
#: this list.
BUILD_ID_COVERS = (
    "physics_earthorbit.py",
    "driver_earthorbit.py",
    "main.py",
    "plot_earthorbit.py",
)


def _compute_build_id():
    """Return a short identifier derived from the core source files.

    MODEL_VERSION records the program's declared release version.  BUILD_ID
    additionally distinguishes source revisions that retain the same declared
    version.  The hash is independent of LF versus CRLF line endings and
    frames each file with its name and length so file-boundary changes cannot
    collide with an unchanged concatenated byte stream.

    Return ``"unknown"`` rather than preventing the program from running if
    the source files cannot be located or decoded, as can happen in some
    frozen or zipped distributions.
    """
    import hashlib
    import os

    try:
        here = os.path.dirname(os.path.abspath(__file__))
        digest = hashlib.sha256()
        for name in BUILD_ID_COVERS:
            with open(os.path.join(here, name), "r", encoding="utf-8",
                      newline=None) as source:
                content = source.read().encode("utf-8")
            digest.update(name.encode("utf-8"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        return digest.hexdigest()[:12]
    except (OSError, UnicodeDecodeError):
        return "unknown"


BUILD_ID = _compute_build_id()


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
