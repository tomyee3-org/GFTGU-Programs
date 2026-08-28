"""
Physics for MercPert, a planar circular restricted three-body simulation.
"""

from dataclasses import dataclass
import math

MODEL_VERSION = "1.0.1"


#: The exact source files this build identifier covers: a documentation-only
#: change, a sample-output file, or an edit to the test suite does not change
#: this value -- only the four core program modules listed here do.  Exposed
#: so callers can determine precisely what BUILD_ID covers without duplicating
#: this list.
BUILD_ID_COVERS = (
    "physics_mercpert.py",
    "driver_mercpert.py",
    "main.py",
    "plot_mercpert.py",
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


# IAU 2015 nominal solar mass parameter and nominal solar radius.
# These are exact nominal conversion constants, not measurements of G or M_sun.
GM_SUN = 1.3271244e20       # m^3 s^-2
R_SUN = 6.957e8             # m
AU = 1.495978707e11         # m


def _require_finite(name: str, *values: float) -> None:
    """Reject booleans and non-finite numeric inputs with a clear message."""
    for value in values:
        try:
            valid = not isinstance(value, bool) and math.isfinite(value)
        except TypeError:
            valid = False
        if not valid:
            raise ValueError(f"{name} must be finite real number(s).")


@dataclass(frozen=True)
class BinarySystemParams:
    m_sun_solar: float
    m_planet_solar: float
    binary_separation: float


@dataclass(frozen=True)
class MercuryInitialConditions:
    """
    Mercury's initial position and velocity RELATIVE TO THE SUN.

    The driver converts these values to barycentric inertial coordinates before
    beginning the integration, matching the physical meaning of the original
    MercPert inputs.
    """
    x_init: float
    y_init: float
    vx_init: float
    vy_init: float


def validate_binary_params(params: BinarySystemParams) -> None:
    values = (
        params.m_sun_solar,
        params.m_planet_solar,
        params.binary_separation,
    )
    _require_finite("Binary-system parameters", *values)
    if params.m_sun_solar <= 0.0:
        raise ValueError("m_sun_solar must be positive.")
    if params.m_planet_solar <= 0.0:
        raise ValueError("m_planet_solar must be positive.")
    if params.binary_separation <= 0.0:
        raise ValueError("binary_separation must be positive.")
    if not math.isfinite(params.m_sun_solar + params.m_planet_solar):
        raise ValueError("The total binary mass is too large to represent.")


def validate_mercury_ic(ic: MercuryInitialConditions) -> None:
    _require_finite(
        "Mercury initial position and velocity",
        ic.x_init,
        ic.y_init,
        ic.vx_init,
        ic.vy_init,
    )


def compute_binary_radii(params: BinarySystemParams) -> tuple[float, float]:
    """Return the Sun and companion radii about the binary barycentre."""
    validate_binary_params(params)
    total_mass = params.m_sun_solar + params.m_planet_solar
    r_sun = params.m_planet_solar / total_mass * params.binary_separation
    r_planet = params.m_sun_solar / total_mass * params.binary_separation
    return r_sun, r_planet


def compute_binary_angular_velocity(params: BinarySystemParams) -> float:
    """Angular velocity of the prescribed circular binary orbit."""
    validate_binary_params(params)
    total_mu = GM_SUN * (params.m_sun_solar + params.m_planet_solar)
    # sqrt(mu / d) / d is algebraically identical to sqrt(mu / d**3),
    # but avoids overflowing d**3 for large (yet finite) separations.
    omega = math.sqrt(total_mu / params.binary_separation) \
        / params.binary_separation
    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError(
            "The masses and separation produce an angular velocity that "
            "cannot be represented by a finite positive float."
        )
    return omega


def binary_positions(
    t: float,
    params: BinarySystemParams,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Return barycentric inertial positions of the Sun and companion at time t.

    At t=0 the Sun is on the negative x-axis and the companion is on the
    positive x-axis. The orbit is counter-clockwise.
    """
    _require_finite("t", t)
    r_sun, r_planet = compute_binary_radii(params)
    omega = compute_binary_angular_velocity(params)
    phase = omega * t
    if not math.isfinite(phase):
        raise ValueError("omega * t is too large to represent.")
    c = math.cos(phase)
    s = math.sin(phase)

    sun = (-r_sun * c, -r_sun * s)
    planet = (r_planet * c, r_planet * s)
    return sun, planet


def binary_velocities(
    t: float,
    params: BinarySystemParams,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return barycentric inertial velocities of the Sun and companion."""
    _require_finite("t", t)
    r_sun, r_planet = compute_binary_radii(params)
    omega = compute_binary_angular_velocity(params)
    phase = omega * t
    if not math.isfinite(phase):
        raise ValueError("omega * t is too large to represent.")
    c = math.cos(phase)
    s = math.sin(phase)

    sun = (r_sun * omega * s, -r_sun * omega * c)
    planet = (-r_planet * omega * s, r_planet * omega * c)
    return sun, planet


def mercury_initial_barycentric_state(
    params: BinarySystemParams,
    ic: MercuryInitialConditions,
) -> tuple[float, float, float, float]:
    """
    Convert Sun-relative user input to barycentric inertial coordinates.

    This keeps the user-facing meaning of x_init, y_init, vx_init, vy_init
    tied to the Sun while the integration itself is performed in an inertial
    frame centred on the binary barycentre.
    """
    validate_binary_params(params)
    validate_mercury_ic(ic)
    (x_sun, y_sun), _ = binary_positions(0.0, params)
    (vx_sun, vy_sun), _ = binary_velocities(0.0, params)
    return (
        x_sun + ic.x_init,
        y_sun + ic.y_init,
        vx_sun + ic.vx_init,
        vy_sun + ic.vy_init,
    )


def _primary_displacements(
    t: float,
    x_merc: float,
    y_merc: float,
    params: BinarySystemParams,
):
    _require_finite("t, x_merc, and y_merc", t, x_merc, y_merc)
    (x_sun, y_sun), (x_planet, y_planet) = binary_positions(t, params)

    dx_sun = x_merc - x_sun
    dy_sun = y_merc - y_sun
    dx_planet = x_merc - x_planet
    dy_planet = y_merc - y_planet

    r_sun = math.hypot(dx_sun, dy_sun)
    r_planet = math.hypot(dx_planet, dy_planet)

    if not all(math.isfinite(value) for value in (
        dx_sun, dy_sun, dx_planet, dy_planet, r_sun, r_planet
    )):
        raise ValueError(
            "The requested coordinates are too large to form finite "
            "displacements from the primaries."
        )

    if r_sun == 0.0:
        raise ValueError(
            "Mercury has reached the Sun's point-mass position; "
            "the Newtonian point-mass force is singular there."
        )
    if r_planet == 0.0:
        raise ValueError(
            "Mercury has reached the companion's point-mass position; "
            "the Newtonian point-mass force is singular there."
        )

    return dx_sun, dy_sun, r_sun, dx_planet, dy_planet, r_planet


def distances_to_primaries(
    t: float,
    x_merc: float,
    y_merc: float,
    params: BinarySystemParams,
) -> tuple[float, float]:
    """Return Mercury's distances from the Sun and companion."""
    _, _, r_sun, _, _, r_planet = _primary_displacements(
        t, x_merc, y_merc, params
    )
    return r_sun, r_planet


def mercury_acceleration(
    t: float,
    x_merc: float,
    y_merc: float,
    params: BinarySystemParams,
) -> tuple[float, float]:
    """Newtonian acceleration of Mercury due to both massive bodies."""
    (
        dx_sun, dy_sun, r_sun,
        dx_planet, dy_planet, r_planet,
    ) = _primary_displacements(t, x_merc, y_merc, params)

    # Compute each inverse-cube vector as (component / r) / r / r.
    # This is algebraically equivalent to component / r**3 but does not
    # overflow merely because a perfectly valid distance has a large cube.
    sun_factor = GM_SUN * params.m_sun_solar / r_sun / r_sun
    planet_factor = GM_SUN * params.m_planet_solar / r_planet / r_planet
    ax = -(sun_factor * (dx_sun / r_sun)
           + planet_factor * (dx_planet / r_planet))
    ay = -(sun_factor * (dy_sun / r_sun)
           + planet_factor * (dy_planet / r_planet))
    if not math.isfinite(ax) or not math.isfinite(ay):
        raise ValueError(
            "The requested state produces an acceleration that cannot be "
            "represented by finite floats."
        )
    return ax, ay


def jacobi_constant(
    t: float,
    x: float,
    y: float,
    vx: float,
    vy: float,
    params: BinarySystemParams,
) -> float:
    """
    Return the Jacobi constant in SI units (m^2/s^2).

    Coordinates are barycentric inertial coordinates. The inertial velocity is
    transformed to the frame rotating with the circular binary before applying
    the usual CR3BP Jacobi integral.
    """
    _require_finite("t, position, and velocity", t, x, y, vx, vy)
    omega = compute_binary_angular_velocity(params)
    r_sun, r_planet = distances_to_primaries(t, x, y, params)

    # v_rot = v_inertial - omega x r
    vx_rot = vx + omega * y
    vy_rot = vy - omega * x

    potential_term = 2.0 * GM_SUN * (
        params.m_sun_solar / r_sun
        + params.m_planet_solar / r_planet
    )
    centrifugal_term = omega * omega * (x * x + y * y)
    speed2_rot = vx_rot * vx_rot + vy_rot * vy_rot

    result = centrifugal_term + potential_term - speed2_rot
    if not math.isfinite(result):
        raise ValueError(
            "The requested state produces a Jacobi constant that cannot be "
            "represented by a finite float."
        )
    return result
