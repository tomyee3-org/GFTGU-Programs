"""
Physical relations for the Star program.

The model is a spherically symmetric Newtonian polytrope.  Central density
is fixed from the ideal-gas relation, the pressure-density relation is then
held to one polytropic law throughout the star, and hydrostatic equilibrium
and enclosed mass are integrated outward.
"""

from math import frexp, isfinite, ldexp, pi, sqrt
from numbers import Real

# Public release metadata. MODEL_VERSION changes when the model's documented
# behaviour changes; BUILD_ID changes whenever one of the core source files
# changes.
MODEL_VERSION = "1.0.3"
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
    # large, but still finite, pressures.
    return _require_positive_finite_result(
        "radial scale", sqrt(p_c) / (sqrt(G_NEWTON) * rho_c)
    )


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
    result = p_prev - G_NEWTON * rho_prev * mass_prev * dr / (r_prev * r_prev)
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
