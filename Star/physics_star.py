"""
Physical relations for the Star program.

The model is a spherically symmetric Newtonian polytrope.  Central density
is fixed from the ideal-gas relation, the pressure-density relation is then
held to one polytropic law throughout the star, and hydrostatic equilibrium
and enclosed mass are integrated outward.
"""

from math import frexp, isfinite, ldexp, pi, sqrt
from numbers import Real
from typing import NamedTuple

# Public release metadata. MODEL_VERSION changes when the model's documented
# behaviour changes; BUILD_ID changes whenever one of the core source files
# changes.
MODEL_VERSION = "1.2.0"
BUILD_ID_COVERS = (
    "physics_star.py",
    "driver_star.py",
    "main.py",
    "plot_star.py",
)


def _read_build_source(path):
    """Read one Build-ID source with normalized newlines."""
    with open(path, "r", encoding="utf-8", newline=None) as source:
        return source.read().encode("utf-8")


def _compute_build_id() -> str:
    """Return a short, reproducible identifier for the core source files.

    Files are read as UTF-8 text with universal-newline conversion, so merely
    switching between LF and CRLF line endings does not create a new build.
    Filename and byte-length framing prevents ambiguous concatenations.
    """
    import hashlib
    import os

    try:
        here = os.path.dirname(os.path.abspath(__file__))
        digest = hashlib.sha256()
        for name in BUILD_ID_COVERS:
            path = os.path.join(here, name)
            content = _read_build_source(path)
            digest.update(name.encode("utf-8"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        return digest.hexdigest()[:12]
    except (OSError, UnicodeDecodeError):
        return "unknown"


BUILD_ID = _compute_build_id()

# Physical constants in SI units.  These values are retained from the
# educational model so that its default results remain consistent with the
# accompanying material.
k_BOLTZMANN = 1.38e-23      # J/K
MPROTON = 1.67e-27          # kg
G_NEWTON = 6.672e-11        # m^3 kg^-1 s^-2


def _validate_finite_real(name, value):
    """Require a finite real scalar, excluding bool values."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")


def _require_positive_finite_result(name, value):
    """Reject floating-point overflow or underflow in a positive result."""
    if not isfinite(value) or value <= 0.0:
        raise OverflowError(
            f"{name} is outside the positive finite floating-point range."
        )
    return value


def _multiply_divide_positive(a, b, c, name):
    """Combine three positive finite operands as a*b/c without range loss.

    The operands must already be representable positive finite floats; this
    helper does not range-protect any upstream calculations that produced them.
    The two ordinary groupings are tried first, preserving their normal
    arithmetic for teaching-scale inputs.  If each loses range prematurely,
    binary mantissas and exponents are combined separately.  A result is
    rejected only when a*b/c itself lies outside the positive finite
    floating-point range.
    """
    result = (a / c) * b
    if isfinite(result) and result > 0.0:
        return result

    result = (a * b) / c
    if isfinite(result) and result > 0.0:
        return result

    a_mantissa, a_exponent = frexp(a)
    b_mantissa, b_exponent = frexp(b)
    c_mantissa, c_exponent = frexp(c)
    mantissa, adjustment = frexp(a_mantissa * b_mantissa / c_mantissa)
    exponent = a_exponent + b_exponent - c_exponent + adjustment
    try:
        result = ldexp(mantissa, exponent)
    except OverflowError:
        result = float("inf")
    return _require_positive_finite_result(name, result)


def q_factor(mu: float) -> float:
    """Return m_p * mu / k_B for the ideal-gas relation."""
    _validate_finite_real("mu", mu)
    if mu <= 0.0:
        raise ValueError("mu must be positive.")
    return _require_positive_finite_result(
        "q_factor", MPROTON * mu / k_BOLTZMANN
    )


def central_density(p_c: float, T_c: float, mu: float) -> float:
    """Compute central density from central pressure and temperature."""
    _validate_finite_real("p_c", p_c)
    _validate_finite_real("T_c", T_c)
    _validate_finite_real("mu", mu)
    if p_c <= 0.0:
        raise ValueError("p_c must be positive.")
    if T_c <= 0.0:
        raise ValueError("T_c must be positive.")
    # Range-aware regrouping avoids both p_c*q overflow and p_c/T_c
    # underflow when the complete density remains representable.
    return _multiply_divide_positive(
        p_c, q_factor(mu), T_c, "central density"
    )


def polytropic_D(rho_c: float, p_c: float, gamma: float) -> float:
    """Return D in rho = D * p**(1/gamma), fixed by central conditions."""
    _validate_finite_real("rho_c", rho_c)
    _validate_finite_real("p_c", p_c)
    _validate_finite_real("gamma", gamma)
    if rho_c <= 0.0:
        raise ValueError("rho_c must be positive.")
    if p_c <= 0.0:
        raise ValueError("p_c must be positive.")
    if gamma <= 1.2:
        raise ValueError("gamma must be greater than 1.2 for a finite-radius polytrope.")
    return _require_positive_finite_result(
        "polytropic D", rho_c / (p_c ** (1.0 / gamma))
    )


def radial_scale(p_c: float, rho_c: float) -> float:
    """
    Return the characteristic radial scale used to choose the integration step:

        scale = sqrt(p_c / G) / rho_c

    This is a dimensional scale for the model, not a literal local pressure
    scale height at the stellar center (where dp/dr tends to zero).
    """
    _validate_finite_real("p_c", p_c)
    _validate_finite_real("rho_c", rho_c)
    if p_c <= 0.0:
        raise ValueError("p_c must be positive.")
    if rho_c <= 0.0:
        raise ValueError("rho_c must be positive.")
    # This algebraically equivalent form avoids overflow in p_c/G for very
    # large, but still finite, pressures.  If sqrt(G)*rho_c underflows, the
    # two divisions are made one at a time instead.
    denominator = sqrt(G_NEWTON) * rho_c
    if denominator > 0.0:
        scale = sqrt(p_c) / denominator
    else:
        scale = (sqrt(p_c) / sqrt(G_NEWTON)) / rho_c
    return _require_positive_finite_result("radial scale", scale)


# Backward-compatible name retained for callers of the earlier Python version.
def scale_height(p_c: float, rho_c: float) -> float:
    return radial_scale(p_c, rho_c)


def hydrostatic_step(p_prev: float, rho_prev: float,
                     mass_prev: float, r_prev: float, dr: float) -> float:
    """One forward-Euler step of dp/dr = -G rho m(r) / r^2."""
    for name, value in (("p_prev", p_prev), ("rho_prev", rho_prev),
                        ("mass_prev", mass_prev), ("r_prev", r_prev), ("dr", dr)):
        _validate_finite_real(name, value)
    if p_prev < 0.0:
        raise ValueError("p_prev must not be negative.")
    if rho_prev < 0.0:
        raise ValueError("rho_prev must not be negative.")
    if mass_prev < 0.0:
        raise ValueError("mass_prev must not be negative.")
    if r_prev <= 0.0:
        raise ValueError("r_prev must be positive in hydrostatic_step().")
    if dr <= 0.0:
        raise ValueError("dr must be positive.")
    r_squared = r_prev * r_prev
    if r_squared == 0.0:
        raise OverflowError(
            "r_prev**2 is below the positive floating-point range, so the "
            "gravitational pressure gradient cannot be evaluated."
        )
    result = p_prev - G_NEWTON * rho_prev * mass_prev * dr / r_squared
    if not isfinite(result):
        raise OverflowError("hydrostatic pressure step is not finite.")
    return result


def mass_step(mass_prev: float, r_prev: float, rho_prev: float, dr: float) -> float:
    """One forward-Euler step of dm/dr = 4*pi*r^2*rho."""
    for name, value in (("mass_prev", mass_prev), ("r_prev", r_prev),
                        ("rho_prev", rho_prev), ("dr", dr)):
        _validate_finite_real(name, value)
    if mass_prev < 0.0:
        raise ValueError("mass_prev must not be negative.")
    if r_prev < 0.0:
        raise ValueError("r_prev must not be negative.")
    if rho_prev < 0.0:
        raise ValueError("rho_prev must not be negative.")
    if dr <= 0.0:
        raise ValueError("dr must be positive.")
    result = mass_prev + 4.0 * pi * r_prev * r_prev * rho_prev * dr
    if not isfinite(result):
        raise OverflowError("enclosed-mass step is not finite.")
    return result


def density_from_pressure(p: float, D: float, gamma: float) -> float:
    """Polytropic equation of state rho = D * p**(1/gamma)."""
    _validate_finite_real("p", p)
    _validate_finite_real("D", D)
    _validate_finite_real("gamma", gamma)
    if p < 0.0:
        raise ValueError("p must not be negative when computing density.")
    if D <= 0.0:
        raise ValueError("D must be positive.")
    if gamma <= 1.2:
        raise ValueError("gamma must be greater than 1.2 for a finite-radius polytrope.")
    if p == 0.0:
        return 0.0
    return _require_positive_finite_result(
        "density", D * (p ** (1.0 / gamma))
    )


def temperature_from_prho(p: float, rho: float, mu: float) -> float:
    """Recover temperature from the ideal-gas relation T = q*p/rho."""
    _validate_finite_real("p", p)
    _validate_finite_real("rho", rho)
    _validate_finite_real("mu", mu)
    if p < 0.0:
        raise ValueError("p must not be negative when computing temperature.")
    q = q_factor(mu)
    if rho <= 0.0:
        if p == 0.0 and rho == 0.0:
            return 0.0
        raise ValueError("rho must be positive when p is positive.")
    if p == 0.0:
        return 0.0
    return _require_positive_finite_result("temperature", q * p / rho)


# ---------------------------------------------------------------------------
# The Lane-Emden solution: the exact solution of the same three equations
# ---------------------------------------------------------------------------

# Beyond this dimensionless radius the Lane-Emden surface is not sought.  It
# is reached only for polytropic indices within about 2e-11 of n = 5, that is
# for gamma within about 1e-12 of 6/5.
LANE_EMDEN_MAX_XI = 1.0e12
# Relative step of the Runge-Kutta integration in xi; it gives about ten
# significant figures, far more than the five that are printed.
LANE_EMDEN_RELATIVE_STEP = 1.0e-3


class LaneEmdenSolution(NamedTuple):
    """Surface of the Lane-Emden function theta_n and the physical star."""

    n: float
    xi_1: float
    mass_factor: float
    length_scale: float
    radius: float
    mass: float


def polytropic_index(gamma: float) -> float:
    """Return the polytropic index n = 1/(gamma - 1) for gamma > 6/5."""
    _validate_finite_real("gamma", gamma)
    if gamma <= 1.2:
        raise ValueError("gamma must be greater than 1.2 for a finite-radius polytrope.")
    return 1.0 / (gamma - 1.0)


def lane_emden_surface(n: float):
    """Return (xi_1, -xi_1**2 theta'(xi_1)) for the Lane-Emden equation.

    The equation (1/xi^2) d/dxi (xi^2 dtheta/dxi) = -theta**n, with
    theta(0) = 1 and theta'(0) = 0, is started from its series at a small xi
    and integrated with classical fourth-order Runge-Kutta steps of 0.1% of
    xi (or of 0.001 near the centre), shortened where theta is small when
    n < 1, and refined a thousandfold in three stages once a step crosses the
    zero.  The first zero of theta is located on a cubic Hermite interpolant
    of the final step, and the mass factor is completed with the integral of
    xi**2 theta**n over the last short interval.  A ValueError is raised if
    n is not in [0, 5) or the surface lies beyond LANE_EMDEN_MAX_XI.
    """
    _validate_finite_real("n", n)
    if not 0.0 <= n < 5.0:
        raise ValueError("the polytropic index n must satisfy 0 <= n < 5.")

    def slope(xi, theta, dtheta):
        return dtheta, -(max(theta, 0.0) ** n) - 2.0 * dtheta / xi

    xi = 1.0e-3
    theta = 1.0 - xi * xi / 6.0 + n * xi**4 / 120.0
    dtheta = -xi / 3.0 + n * xi**3 / 30.0
    refine = 1.0
    while True:
        h = LANE_EMDEN_RELATIVE_STEP * max(1.0, xi) * refine
        if n < 1.0:
            # theta**n has an unbounded slope at theta = 0 when n < 1, so take
            # shorter steps as theta approaches zero.
            h *= max(1.0e-3, min(1.0, theta / 0.01))
        k1 = slope(xi, theta, dtheta)
        k2 = slope(xi + h / 2, theta + h / 2 * k1[0], dtheta + h / 2 * k1[1])
        k3 = slope(xi + h / 2, theta + h / 2 * k2[0], dtheta + h / 2 * k2[1])
        k4 = slope(xi + h, theta + h * k3[0], dtheta + h * k3[1])
        theta_next = theta + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        dtheta_next = dtheta + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        if theta_next <= 0.0:
            # theta**n is not smooth where theta reaches zero, so approach the
            # zero with steps a thousand times smaller before locating it.
            if refine > 1.0e-3:
                refine /= 10.0
                continue
            break
        xi, theta, dtheta = xi + h, theta_next, dtheta_next
        if xi > LANE_EMDEN_MAX_XI:
            raise ValueError(
                f"the Lane-Emden surface lies beyond xi = {LANE_EMDEN_MAX_XI:g}; "
                "gamma is too close to 6/5."
            )

    def hermite(s):
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        return h00 * theta + h10 * h * dtheta + h01 * theta_next + h11 * h * dtheta_next

    low, high = 0.0, 1.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if hermite(middle) > 0.0:
            low = middle
        else:
            high = middle
    s = 0.5 * (low + high)
    xi_1 = xi + s * h
    # The mass factor -xi**2 theta' changes over the last short interval by
    # the integral of xi**2 theta**n, taken with theta falling linearly to
    # zero; this avoids differentiating across the zero, where theta**n is
    # not smooth for n < 1.
    last = s * h
    mass_factor = -xi * xi * dtheta + xi * xi * theta**n * last / (n + 1.0)
    return xi_1, mass_factor


def lane_emden_solution(p_c: float, rho_c: float, gamma: float) -> LaneEmdenSolution:
    """Return the exact radius and mass of the polytrope with these central values.

    With n = 1/(gamma - 1) and a = sqrt((n + 1)/(4 pi)) * radial_scale(p_c,
    rho_c), the radius is xi_1 a and the mass 4 pi a**3 rho_c (-xi_1**2
    theta'(xi_1)).  These are the exact solution of the equations that
    integrate_star() steps through, so the difference between the two is the
    error of the numerical integration.
    """
    n = polytropic_index(gamma)
    xi_1, mass_factor = lane_emden_surface(n)
    length = sqrt((n + 1.0) / (4.0 * pi)) * radial_scale(p_c, rho_c)
    radius = _require_positive_finite_result("Lane-Emden radius", xi_1 * length)
    try:
        mass = 4.0 * pi * mass_factor * rho_c * length**3
    except OverflowError:
        mass = float("inf")
    mass = _require_positive_finite_result("Lane-Emden mass", mass)
    return LaneEmdenSolution(n, xi_1, mass_factor, length, radius, mass)
