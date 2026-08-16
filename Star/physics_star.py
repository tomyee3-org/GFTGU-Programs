"""
Physical relations for the Star program.

The model is a spherically symmetric Newtonian polytrope.  Central density
is fixed from the ideal-gas relation, the pressure-density relation is then
held to one polytropic law throughout the star, and hydrostatic equilibrium
and enclosed mass are integrated outward.
"""

from math import pi, sqrt

# Physical constants in SI units.  These values are retained from the
# educational model so that its default results remain consistent with the
# accompanying material.
k_BOLTZMANN = 1.38e-23      # J/K
MPROTON = 1.67e-27          # kg
G_NEWTON = 6.672e-11        # m^3 kg^-1 s^-2


def q_factor(mu: float) -> float:
    """Return m_p * mu / k_B for the ideal-gas relation."""
    if mu <= 0.0:
        raise ValueError("mu must be positive.")
    return MPROTON * mu / k_BOLTZMANN


def central_density(p_c: float, T_c: float, mu: float) -> float:
    """Compute central density from central pressure and temperature."""
    if p_c <= 0.0:
        raise ValueError("p_c must be positive.")
    if T_c <= 0.0:
        raise ValueError("T_c must be positive.")
    return p_c * q_factor(mu) / T_c


def polytropic_D(rho_c: float, p_c: float, gamma: float) -> float:
    """Return D in rho = D * p**(1/gamma), fixed by central conditions."""
    if rho_c <= 0.0:
        raise ValueError("rho_c must be positive.")
    if p_c <= 0.0:
        raise ValueError("p_c must be positive.")
    if gamma <= 1.2:
        raise ValueError("gamma must be greater than 1.2 for a finite-radius polytrope.")
    return rho_c / (p_c ** (1.0 / gamma))


def radial_scale(p_c: float, rho_c: float) -> float:
    """
    Return the characteristic radial scale used to choose the integration step:

        scale = sqrt(p_c / G) / rho_c

    This is a dimensional scale for the model, not a literal local pressure
    scale height at the stellar center (where dp/dr tends to zero).
    """
    if p_c <= 0.0:
        raise ValueError("p_c must be positive.")
    if rho_c <= 0.0:
        raise ValueError("rho_c must be positive.")
    return sqrt(p_c / G_NEWTON) / rho_c


# Backward-compatible name retained for callers of the earlier Python version.
def scale_height(p_c: float, rho_c: float) -> float:
    return radial_scale(p_c, rho_c)


def hydrostatic_step(p_prev: float, rho_prev: float,
                     mass_prev: float, r_prev: float, dr: float) -> float:
    """One forward-Euler step of dp/dr = -G rho m(r) / r^2."""
    if r_prev <= 0.0:
        raise ValueError("r_prev must be positive in hydrostatic_step().")
    if dr <= 0.0:
        raise ValueError("dr must be positive.")
    return p_prev - G_NEWTON * rho_prev * mass_prev * dr / (r_prev * r_prev)


def mass_step(mass_prev: float, r_prev: float, rho_prev: float, dr: float) -> float:
    """One forward-Euler step of dm/dr = 4*pi*r^2*rho."""
    if r_prev < 0.0:
        raise ValueError("r_prev must not be negative.")
    if rho_prev < 0.0:
        raise ValueError("rho_prev must not be negative.")
    if dr <= 0.0:
        raise ValueError("dr must be positive.")
    return mass_prev + 4.0 * pi * r_prev * r_prev * rho_prev * dr


def density_from_pressure(p: float, D: float, gamma: float) -> float:
    """Polytropic equation of state rho = D * p**(1/gamma)."""
    if p < 0.0:
        raise ValueError("p must not be negative when computing density.")
    if D <= 0.0:
        raise ValueError("D must be positive.")
    if gamma <= 1.2:
        raise ValueError("gamma must be greater than 1.2 for a finite-radius polytrope.")
    return D * (p ** (1.0 / gamma))


def temperature_from_prho(p: float, rho: float, mu: float) -> float:
    """Recover temperature from the ideal-gas relation T = q*p/rho."""
    if p < 0.0:
        raise ValueError("p must not be negative when computing temperature.")
    if rho <= 0.0:
        if p == 0.0 and rho == 0.0:
            return 0.0
        raise ValueError("rho must be positive when p is positive.")
    return q_factor(mu) * p / rho
