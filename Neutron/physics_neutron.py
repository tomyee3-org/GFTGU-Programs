"""
Physics for the Neutron stellar-structure model.

The model is a static, spherically symmetric perfect fluid governed by the
Tolman-Oppenheimer-Volkoff (TOV) equation and a simple polytropic equation of
state p = K rho**gamma.

This is an educational polytropic model, not a modern tabulated neutron-star
equation of state.
"""

from __future__ import annotations

import math

# Public release metadata. MODEL_VERSION changes when the model's documented
# behaviour changes; BUILD_ID changes whenever one of the core source files
# changes.
MODEL_VERSION = "1.2.0"
BUILD_ID_COVERS = (
    "physics_neutron.py",
    "driver_neutron.py",
    "main.py",
    "plot_neutron.py",
)


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
            with open(path, "r", encoding="utf-8", newline=None) as source:
                content = source.read().encode("utf-8")
            digest.update(name.encode("utf-8"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        return digest.hexdigest()[:12]
    except (OSError, UnicodeDecodeError):
        return "unknown"


BUILD_ID = _compute_build_id()

# SI constants
G = 6.67430e-11               # m^3 kg^-1 s^-2
C = 299_792_458.0             # m s^-1, exact
C2 = C * C
M_SUN = 1.98847e30            # kg, convenient solar-mass conversion


class SurfaceCrossingError(ValueError):
    """Signal that an RK4 trial stage crossed the zero-pressure surface."""


def _require_finite_number(value: float, name: str) -> None:
    """Raise ValueError unless value is a finite real number (not bool)."""
    try:
        finite = math.isfinite(value)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number.") from exc
    if isinstance(value, bool) or not finite:
        raise ValueError(f"{name} must be a finite real number.")


def eos_density(p: float, K: float, gamma: float) -> float:
    """Return mass density rho from the polytropic EOS p = K rho**gamma."""
    _require_finite_number(p, "Pressure")
    _require_finite_number(K, "K")
    _require_finite_number(gamma, "gamma")
    if p < 0.0:
        raise ValueError("Pressure must be non-negative when evaluating the EOS.")
    if K <= 0.0:
        raise ValueError("K must be positive when evaluating the EOS.")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than 1 when evaluating the EOS.")
    if p == 0.0:
        return 0.0

    # The logarithmic form avoids an avoidable overflow in p / K when both
    # inputs are individually finite but have very different magnitudes.
    try:
        rho = math.exp((math.log(p) - math.log(K)) / gamma)
    except OverflowError as exc:
        raise ValueError(
            "The EOS parameters produce a density outside the finite "
            "floating-point range."
        ) from exc
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError(
            "The EOS parameters produce a density outside the positive "
            "finite floating-point range."
        )
    return rho


def structure_derivatives(
    r: float,
    p: float,
    m: float,
    K: float,
    gamma: float,
) -> tuple[float, float]:
    """
    Return (dp/dr, dm/dr) for the TOV equations.

    rho is mass density, so energy density is rho*c^2:

      dp/dr = -G (rho + p/c^2) (m + 4*pi*r^3*p/c^2)
              / [r (r - 2Gm/c^2)]

      dm/dr = 4*pi*r^2*rho
    """
    _require_finite_number(r, "r")
    _require_finite_number(p, "Pressure")
    _require_finite_number(m, "Enclosed mass")
    if r <= 0.0:
        raise ValueError("TOV derivatives require r > 0.")
    if p < 0.0:
        raise ValueError("TOV derivatives require non-negative pressure.")
    if m < 0.0:
        raise ValueError("TOV derivatives require non-negative enclosed mass.")

    rho = eos_density(p, K, gamma)
    try:
        compactness_denominator = r - 2.0 * G * m / C2
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(
            "The TOV compactness calculation is outside the finite "
            "floating-point range."
        ) from exc
    if not math.isfinite(compactness_denominator):
        raise ValueError(
            "The TOV compactness denominator is not finite."
        )
    if compactness_denominator <= 0.0:
        raise RuntimeError(
            "The attempted static model reached r <= 2Gm/c^2; "
            "the TOV denominator is singular and no regular static solution "
            "can be continued by this integration."
        )

    try:
        full_denominator = r * compactness_denominator
        pressure_term = 4.0 * math.pi * r * r * r * (p / C2)
        pressure_source = m + pressure_term
        inertial_density = rho + p / C2
        pressure_numerator = G * inertial_density * pressure_source
        dmdr = 4.0 * math.pi * r * r * rho
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(
            "The TOV derivative calculation is outside the finite "
            "floating-point range."
        ) from exc

    if full_denominator <= 0.0 or not math.isfinite(full_denominator):
        raise ValueError(
            "The full TOV denominator is outside the positive finite "
            "floating-point range."
        )
    if not all(
        math.isfinite(value)
        for value in (
            pressure_term,
            pressure_source,
            inertial_density,
            pressure_numerator,
            dmdr,
        )
    ):
        raise ValueError(
            "The TOV derivative calculation is outside the finite "
            "floating-point range."
        )

    dpdr = -pressure_numerator / full_denominator
    if not math.isfinite(dpdr):
        raise ValueError(
            "The TOV pressure derivative is outside the finite "
            "floating-point range."
        )
    return dpdr, dmdr


def central_state(
    p_c: float,
    K: float,
    gamma: float,
    r0: float,
) -> tuple[float, float, float]:
    """
    Regular small-r starting state (p, rho, m) at r=r0.

    Near the center:
      m(r) = 4*pi*rho_c*r^3/3 + O(r^5)

    and the TOV pressure gradient is linear in r, giving
      p(r) = p_c
             - 2*pi*G*(rho_c+p_c/c^2)*(rho_c/3+p_c/c^2)*r^2
             + O(r^4).
    """
    _require_finite_number(r0, "r0")
    _require_finite_number(p_c, "p_c")
    if r0 <= 0.0:
        raise ValueError("The central expansion requires r0 > 0.")
    if p_c <= 0.0:
        raise ValueError("The central expansion requires p_c > 0.")

    rho_c = eos_density(p_c, K, gamma)
    try:
        r0_squared = r0 * r0
        r0_cubed = r0_squared * r0
        m0 = 4.0 * math.pi * rho_c * r0_cubed / 3.0
        coeff = (
            2.0
            * math.pi
            * G
            * (rho_c + p_c / C2)
            * (rho_c / 3.0 + p_c / C2)
        )
        pressure_correction = coeff * r0_squared
        p0 = p_c - pressure_correction
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(
            "The central expansion overflowed for the requested parameters."
        ) from exc
    if (
        m0 <= 0.0
        or not all(
            math.isfinite(value)
            for value in (
                r0_squared,
                r0_cubed,
                m0,
                coeff,
                pressure_correction,
                p0,
            )
        )
    ):
        raise ValueError(
            "The central expansion is outside the positive finite "
            "floating-point range for the requested parameters."
        )
    if p0 <= 0.0:
        raise ValueError(
            "The initial radial step is too large: the central expansion "
            "already reaches non-positive pressure. Increase steps_per_scale."
        )
    rho0 = eos_density(p0, K, gamma)
    return p0, rho0, m0


def rk4_step(
    r: float,
    p: float,
    m: float,
    h: float,
    K: float,
    gamma: float,
) -> tuple[float, float]:
    """Advance pressure and enclosed mass by one classical RK4 radial step."""

    _require_finite_number(h, "RK4 step h")
    if h <= 0.0:
        raise ValueError("RK4 step h must be positive.")

    def f(rr: float, pp: float, mm: float) -> tuple[float, float]:
        # A negative intermediate pressure means the proposed step crosses the
        # stellar surface. The driver will retry with a shorter step.
        if pp <= 0.0:
            raise SurfaceCrossingError(
                "RK4 trial stage crossed the stellar surface."
            )
        return structure_derivatives(rr, pp, mm, K, gamma)

    k1p, k1m = f(r, p, m)
    k2p, k2m = f(
        r + 0.5 * h,
        p + 0.5 * h * k1p,
        m + 0.5 * h * k1m,
    )
    k3p, k3m = f(
        r + 0.5 * h,
        p + 0.5 * h * k2p,
        m + 0.5 * h * k2m,
    )
    k4p, k4m = f(
        r + h,
        p + h * k3p,
        m + h * k3m,
    )

    p_new = p + h * (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0
    m_new = m + h * (k1m + 2.0 * k2m + 2.0 * k3m + k4m) / 6.0
    if not math.isfinite(p_new) or not math.isfinite(m_new):
        raise ValueError(
            "The RK4 result is outside the finite floating-point range."
        )
    return p_new, m_new
