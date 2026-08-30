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

        |a| = MU_EARTH / r^2

    with the modern JPL DE440 value of Earth's gravitational parameter, so
    the acceleration falls with distance and produces the usual Keplerian
    inverse-square dynamics.
"""

import math
import numbers
from typing import Literal

MODEL_VERSION = "1.1.0"


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
#
# G_SURFACE and R_EARTH deliberately retain the rounded textbook values used
# by the near-surface EarthOrbit exercise.  MU_EARTH is the modern Earth GM
# from JPL DE440 and is used by the optional inverse-square extension.
G_SURFACE = 9.8
R_EARTH = 6_378_200.0  # m; textbook reference radius, close to the equatorial radius
MU_EARTH = 3.986_004_355_07e14  # m^3/s^2; JPL DE440 Earth GM

# Kept for comparison with the textbook-derived value and for compatibility
# with earlier student calculations.  The inverse-square force uses MU_EARTH.
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
        "inverse_square" uses MU_EARTH/r**2.

    Returns
    -------
    ax, ay : float
        Acceleration components in m/s^2.
    """
    for name, value in (("x", x), ("y", y)):
        if (
            not isinstance(value, numbers.Real)
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite real position in metres")

    # hypot() avoids the avoidable overflow/underflow of sqrt(x*x + y*y)
    # for extreme but finite direct calls.
    r = math.hypot(x, y)
    if r == 0.0:
        raise ValueError("gravitational acceleration is undefined at Earth's centre")

    unit_x = x / r
    unit_y = y / r

    if force_law == "simplified":
        magnitude = G_SURFACE
    elif force_law == "inverse_square":
        # Dividing twice avoids forming r**2 or r**3, either of which may
        # overflow even when the physically correct acceleration is finite.
        magnitude = MU_EARTH / r / r
    else:
        raise ValueError(f"Unknown force_law: {force_law!r}")

    if not math.isfinite(magnitude):
        raise ValueError(
            "gravitational acceleration is too large to represent at this radius"
        )

    return -magnitude * unit_x, -magnitude * unit_y
