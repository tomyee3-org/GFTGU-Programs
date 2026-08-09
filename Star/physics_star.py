"""
Physical laws for the Star program (polytropic stellar structure)

This module encodes the physical relations used to model a spherically
symmetric star with a polytropic equation of state. The only literal
numbers here represent physical constants.
"""

# Physical constants in SI units
k_BOLTZMANN = 1.38e-23      # Boltzmann constant [J/K]
MPROTON = 1.67e-27          # Proton mass [kg]
G_NEWTON = 6.672e-11        # Newton's gravitational constant [m^3 kg^-1 s^-2]


def q_factor(mu: float) -> float:
    """
    Combination of constants in the ideal gas law:
        p = rho * k_BOLTZMANN * T / (MPROTON * mu)
    so q = MPROTON * mu / k_BOLTZMANN.
    """
    return MPROTON * mu / k_BOLTZMANN


def central_density(p_c: float, T_c: float, mu: float) -> float:
    """
    Compute central density from central pressure and temperature
    using the ideal gas law.
    """
    q = q_factor(mu)
    return p_c * q / T_c


def polytropic_D(rho_c: float, p_c: float, gamma: float) -> float:
    """
    Proportionality factor in the polytropic equation of state
    written as:
        rho = D * p^(1/gamma)
    determined by demanding that the polytropic law reproduce
    the central density.
    """
    gamma_recip = 1.0 / gamma
    return rho_c / (p_c ** gamma_recip)


def scale_height(p_c: float, rho_c: float) -> float:
    """
    Approximate pressure scale height:
        scale ~ sqrt(p_c / G_NEWTON) / rho_c
    roughly the distance over which the pressure falls by a factor of ~2.
    """
    from math import sqrt
    return sqrt(p_c / G_NEWTON) / rho_c


def hydrostatic_step(p_prev: float, rho_prev: float,
                     mass_prev: float, r_prev: float, dr: float) -> float:
    """
    One step of the hydrostatic equilibrium equation:
        dp/dr = - G_NEWTON * rho * m(r) / r^2
    implemented in finite-difference form.
    """
    return p_prev - G_NEWTON * rho_prev * mass_prev * dr / (r_prev * r_prev)


def mass_step(mass_prev: float, r_prev: float, rho_prev: float, dr: float) -> float:
    """
    Update the mass interior to radius r by adding the mass of a shell:
        dm = 4 * pi * r^2 * rho * dr
    """
    from math import pi
    return mass_prev + 4.0 * pi * r_prev * r_prev * rho_prev * dr


def density_from_pressure(p: float, D: float, gamma: float) -> float:
    """
    Polytropic equation of state:
        rho = D * p^(1/gamma)
    """
    from math import pow
    gamma_recip = 1.0 / gamma
    return D * pow(p, gamma_recip)


def temperature_from_prho(p: float, rho: float, mu: float) -> float:
    """
    Ideal gas law written as:
        T = p / (rho * k_BOLTZMANN / (MPROTON * mu))
          = p / (rho / q_factor(mu))
    """
    q = q_factor(mu)
    return q * p / rho
