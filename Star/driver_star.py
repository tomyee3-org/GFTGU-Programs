"""
driver_star.py — Numerical integration of stellar structure for the Star program.

This module steps outward from the center of the star, enforcing hydrostatic
equilibrium and a polytropic equation of state, adjusting the radial
step size so that the surface (where p ~ 0) is reached within a fixed
maximum number of steps.
"""

from dataclasses import dataclass
from typing import List, Literal
from math import pi

from physics_star import (
    central_density,
    polytropic_D,
    scale_height,
    hydrostatic_step,
    mass_step,
    density_from_pressure,
    temperature_from_prho,
)

OutputType = Literal["pressure", "density", "temperature", "mass"]


@dataclass
class StarResult:
    """
    Structured, uniformly-typed replacement for the old
    Dict[str, List[float]] result: radius/pressure/density/temperature/
    mass are the plotted profiles, last_index and output_type are not
    themselves List[float] and don't belong in that same mapping.
    """
    radius: List[float]
    pressure: List[float]
    density: List[float]
    temperature: List[float]
    mass: List[float]
    last_index: int
    output_type: OutputType


def integrate_star(
    p_c: float,
    T_c: float,
    mu: float,
    gamma: float,
    max_points: int = 2000,
    output_type: OutputType = "pressure",
) -> StarResult:
    """
    Integrate the stellar structure outward from the center.

    Parameters
    ----------
    p_c : float
        Central pressure [Pa].
    T_c : float
        Central temperature [K].
    mu : float
        Mean molecular weight (dimensionless).
    gamma : float
        Polytropic exponent in the equation of state.
    max_points : int
        Maximum number of radial grid points.
    output_type : "pressure", "density", "temperature", or "mass"
        Which profile the plot module should display.

    Returns
    -------
    result : StarResult
        radius, pressure, density, temperature, mass: profile arrays.
        last_index: index of the surface (where p crosses zero).
        output_type: echoed back for the plot module.
    """
    # Central quantities
    rho_c = central_density(p_c, T_c, mu)
    D = polytropic_D(rho_c, p_c, gamma)
    scale = scale_height(p_c, rho_c)

    # Initial guess for radial step
    dr = scale / 400.0

    # Allocate arrays
    radius = [0.0] * max_points
    pressure = [0.0] * max_points
    density = [0.0] * max_points
    temperature = [0.0] * max_points
    mass = [0.0] * max_points

    # Central values
    radius[0] = 0.0
    pressure[0] = p_c
    temperature[0] = T_c
    density[0] = rho_c
    mass[0] = 0.0

    last_step = 0

    # Outer loop: adjust dr until the surface is reached within max_points
    while last_step == 0:
        # First non-zero radial step (approximation at r = dr)
        radius[1] = dr
        pressure[1] = p_c
        density[1] = rho_c
        mass[1] = 4.0 * pi * dr * dr * dr * rho_c / 3.0
        temperature[1] = temperature_from_prho(p_c, rho_c, mu)

        # Step-by-step integration
        for j in range(2, max_points):
            radius[j] = radius[j - 1] + dr
            pressure[j] = hydrostatic_step(
                pressure[j - 1],
                density[j - 1],
                mass[j - 1],
                radius[j - 1],
                dr,
            )

            # Stop when pressure goes negative: we've passed the surface
            if pressure[j] < 0.0:
                last_step = j
                break

            mass[j] = mass_step(mass[j - 1], radius[j - 1], density[j - 1], dr)
            density[j] = density_from_pressure(pressure[j], D, gamma)
            temperature[j] = temperature_from_prho(pressure[j], density[j], mu)

        # If we used all points and still haven't reached the surface,
        # increase dr and try again.
        if last_step == 0:
            dr *= 2.0

    # Truncate arrays at the surface
    radius = radius[:last_step]
    pressure = pressure[:last_step]
    density = density[:last_step]
    temperature = temperature[:last_step]
    mass = mass[:last_step]

    return StarResult(
        radius=radius,
        pressure=pressure,
        density=density,
        temperature=temperature,
        mass=mass,
        last_index=last_step,
        output_type=output_type,
    )
