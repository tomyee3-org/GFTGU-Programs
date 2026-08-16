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

# SI constants
G = 6.67430e-11               # m^3 kg^-1 s^-2
C = 299_792_458.0             # m s^-1, exact
C2 = C * C
M_SUN = 1.98847e30            # kg, convenient solar-mass conversion


def eos_density(p: float, K: float, gamma: float) -> float:
    """Return mass density rho from the polytropic EOS p = K rho**gamma."""
    if p < 0.0:
        raise ValueError("Pressure must be non-negative when evaluating the EOS.")
    if p == 0.0:
        return 0.0
    return (p / K) ** (1.0 / gamma)


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
    if r <= 0.0:
        raise ValueError("TOV derivatives require r > 0.")
    if p < 0.0:
        raise ValueError("TOV derivatives require non-negative pressure.")

    rho = eos_density(p, K, gamma)
    compactness_denominator = r - 2.0 * G * m / C2
    if compactness_denominator <= 0.0:
        raise RuntimeError(
            "The attempted static model reached r <= 2Gm/c^2; "
            "the TOV denominator is singular and no regular static solution "
            "can be continued by this integration."
        )

    pressure_source = m + 4.0 * math.pi * r**3 * p / C2
    dpdr = (
        -G
        * (rho + p / C2)
        * pressure_source
        / (r * compactness_denominator)
    )
    dmdr = 4.0 * math.pi * r * r * rho
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
    rho_c = eos_density(p_c, K, gamma)
    m0 = 4.0 * math.pi * rho_c * r0**3 / 3.0
    coeff = (
        2.0
        * math.pi
        * G
        * (rho_c + p_c / C2)
        * (rho_c / 3.0 + p_c / C2)
    )
    p0 = p_c - coeff * r0 * r0
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

    def f(rr: float, pp: float, mm: float) -> tuple[float, float]:
        # A negative intermediate pressure means the proposed step crosses the
        # stellar surface. The driver will retry with a shorter step.
        if pp <= 0.0:
            raise ValueError("RK4 trial stage crossed the stellar surface.")
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
    return p_new, m_new
