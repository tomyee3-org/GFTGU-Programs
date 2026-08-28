"""
Newtonian two-body physics for Binary.
"""

from dataclasses import dataclass
from math import hypot, isfinite
from numbers import Real

MODEL_VERSION = "1.1.0"


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


def _finite_real(name: str, value: float) -> None:
    """Reject values that cannot represent a finite physical input."""
    if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")


def _positive_mass(name: str, value: float) -> None:
    """Validate a point mass used by a public physics calculation."""
    _finite_real(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")


def relative_displacement(xA: float, yA: float, xB: float, yB: float):
    """Return A-minus-B displacement components and scalar separation."""
    for name, value in (("xA", xA), ("yA", yA), ("xB", xB), ("yB", yB)):
        _finite_real(name, value)

    xAB = xA - xB
    yAB = yA - yB
    if not (isfinite(xAB) and isfinite(yAB)):
        raise ValueError(
            "The relative displacement is outside the numerical range of "
            "double-precision arithmetic."
        )

    rAB = hypot(xAB, yAB)

    if rAB == 0.0:
        raise ValueError(
            "The two point masses have zero separation. "
            "Newtonian point-mass gravity is singular at r = 0."
        )

    return xAB, yAB, rAB


def accelerations(
    MA: float, MB: float,
    xA: float, yA: float,
    xB: float, yB: float,
):
    """Return the Newtonian acceleration components of bodies A and B."""
    _positive_mass("MA", MA)
    _positive_mass("MB", MB)
    xAB, yAB, rAB = relative_displacement(xA, yA, xB, yB)

    # Normalize first and divide sequentially by r. This is algebraically
    # equivalent to displacement/r^3, but avoids constructing r^3 and thereby
    # supports the full useful floating-point separation range.
    direction_x = xAB / rAB
    direction_y = yAB / rAB
    acceleration_a = G * MB / rAB / rAB
    acceleration_b = G * MA / rAB / rAB
    axA = -acceleration_a * direction_x
    ayA = -acceleration_a * direction_y
    axB = acceleration_b * direction_x
    ayB = acceleration_b * direction_y
    values = (axA, ayA, axB, ayB)
    if not all(isfinite(value) for value in values):
        raise ValueError(
            "The calculated acceleration is outside the numerical range of "
            "double-precision arithmetic."
        )
    return values


def energies(
    MA: float, MB: float,
    xA: float, yA: float, vA: float, uA: float,
    xB: float, yB: float, vB: float, uB: float,
):
    """Return gravitational potential, kinetic, and total system energy."""
    _positive_mass("MA", MA)
    _positive_mass("MB", MB)
    for name, value in (("vA", vA), ("uA", uA), ("vB", vB), ("uB", uB)):
        _finite_real(name, value)
    _, _, rAB = relative_displacement(xA, yA, xB, yB)

    # Divide before the final mass multiplication to avoid an avoidable
    # intermediate overflow when the representable final value is finite.
    U = -(G * MA / rAB) * MB
    KA = 0.5 * MA * (vA * vA + uA * uA)
    KB = 0.5 * MB * (vB * vB + uB * uB)
    K = KA + KB
    E = K + U
    values = (U, K, E)
    if not all(isfinite(value) for value in values):
        raise ValueError(
            "The calculated energy is outside the numerical range of "
            "double-precision arithmetic."
        )
    return values
