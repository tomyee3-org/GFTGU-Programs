"""
Newtonian two-body physics for Binary.
"""

from dataclasses import dataclass
from math import hypot

MODEL_VERSION = "1.0.0"


#: The exact source files this build identifier covers: a documentation-only
#: change, a sample-output file, or an edit to the test suite does not change
#: this value -- only the four core program modules listed here do.  Exposed
#: so callers can determine precisely what BUILD_ID covers without duplicating
#: this list.
BUILD_ID_COVERS = (
    "physics_binary.py",
    "driver_binary.py",
    "main.py",
    "plot_binary.py",
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
