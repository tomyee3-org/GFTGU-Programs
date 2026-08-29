"""
driver_star.py — Numerical integration of polytropic stellar structure.

The calculation uses forward Euler steps in radius.  If the initially chosen
step does not span the star within max_points, the driver doubles the step and
restarts.  This restart is a radial-range mechanism, not an error-controlled
adaptive integrator.
"""

from dataclasses import dataclass
from typing import get_args, List, Literal
from math import isfinite, pi
from numbers import Real

import physics_star as phys
from physics_star import (
    central_density,
    polytropic_D,
    radial_scale,
    hydrostatic_step,
    mass_step,
    density_from_pressure,
    temperature_from_prho,
)

OutputType = Literal["pressure", "density", "temperature", "mass"]
OUTPUT_TYPES = get_args(OutputType)


@dataclass
class StarResult:
    """Computed stellar profiles and numerical information for one run."""
    model_version: str
    build_id: str
    radius: List[float]
    pressure: List[float]
    density: List[float]
    temperature: List[float]
    mass: List[float]
    surface_index: int
    output_type: OutputType
    radial_step: float
    restart_count: int

    @property
    def last_index(self) -> int:
        """Backward-compatible alias for the surface array index."""
        return self.surface_index


def _validate_finite_real(name, value):
    """Require a finite real scalar, excluding bool values."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")


def _validate_inputs(p_c, T_c, mu, gamma, max_points, steps_per_scale, output_type):
    for name, value in (("p_c", p_c), ("T_c", T_c), ("mu", mu), ("gamma", gamma)):
        _validate_finite_real(name, value)

    if p_c <= 0.0:
        raise ValueError("p_c must be positive.")
    if T_c <= 0.0:
        raise ValueError("T_c must be positive.")
    if mu <= 0.0:
        raise ValueError("mu must be positive.")
    if gamma <= 1.2:
        raise ValueError(
            "gamma must be greater than 1.2. Polytropes with gamma <= 6/5 "
            "do not have an ordinary finite-radius zero-pressure surface."
        )
    if not isinstance(max_points, int) or isinstance(max_points, bool) or max_points < 3:
        raise ValueError("max_points must be an integer of at least 3.")
    if (not isinstance(steps_per_scale, int) or isinstance(steps_per_scale, bool)
            or steps_per_scale <= 0):
        raise ValueError("steps_per_scale must be a positive integer.")
    if output_type not in OUTPUT_TYPES:
        choices = ", ".join(f'"{name}"' for name in OUTPUT_TYPES)
        raise ValueError(
            f"output_type must be one of {choices}."
        )


def integrate_star(
    p_c: float,
    T_c: float,
    mu: float,
    gamma: float,
    max_points: int = 2000,
    steps_per_scale: int = 400,
    output_type: OutputType = "pressure",
) -> StarResult:
    """Integrate the stellar structure outward from the center."""
    _validate_inputs(p_c, T_c, mu, gamma, max_points, steps_per_scale, output_type)

    rho_c = central_density(p_c, T_c, mu)
    D = polytropic_D(rho_c, p_c, gamma)
    scale = radial_scale(p_c, rho_c)
    dr = scale / steps_per_scale
    if not isfinite(dr) or dr <= 0.0:
        raise OverflowError(
            "The requested radial step is outside the positive finite "
            "floating-point range."
        )
    restart_count = 0
    max_restarts = 64

    while True:
        radius = [0.0] * max_points
        pressure = [0.0] * max_points
        density = [0.0] * max_points
        temperature = [0.0] * max_points
        mass = [0.0] * max_points

        # Central values.
        radius[0] = 0.0
        pressure[0] = p_c
        temperature[0] = T_c
        density[0] = rho_c
        mass[0] = 0.0

        # First non-zero radius.  The central quantities are used as the
        # leading-order approximation over this small sphere.
        radius[1] = dr
        pressure[1] = p_c
        density[1] = rho_c
        mass[1] = 4.0 * pi * dr**3 * rho_c / 3.0
        temperature[1] = T_c

        surface_found = False

        for j in range(2, max_points):
            r_prev = radius[j - 1]
            p_prev = pressure[j - 1]
            rho_prev = density[j - 1]
            m_prev = mass[j - 1]

            r_trial = r_prev + dr
            p_trial = hydrostatic_step(p_prev, rho_prev, m_prev, r_prev, dr)

            if p_trial <= 0.0:
                # Linear interpolation of the final Euler pressure segment gives
                # a cleaner estimate of the p=0 surface than simply discarding
                # the negative-pressure trial point.
                frac = p_prev / (p_prev - p_trial) if p_prev != p_trial else 1.0
                frac = min(1.0, max(0.0, frac))
                dr_surface = frac * dr

                radius[j] = r_prev + dr_surface
                pressure[j] = 0.0
                density[j] = 0.0
                temperature[j] = temperature_from_prho(0.0, 0.0, mu)
                # Retain the left-endpoint density for the fractional shell,
                # consistently with forward Euler elsewhere.  Because density
                # falls to zero at the surface, this first-order update slightly
                # overestimates the final shell's mass.
                mass[j] = mass_step(m_prev, r_prev, rho_prev, dr_surface)

                surface_index = j
                surface_found = True
                break

            radius[j] = r_trial
            pressure[j] = p_trial
            mass[j] = mass_step(m_prev, r_prev, rho_prev, dr)
            density[j] = density_from_pressure(p_trial, D, gamma)
            temperature[j] = temperature_from_prho(p_trial, density[j], mu)

        if surface_found:
            end = surface_index + 1
            return StarResult(
                model_version=phys.MODEL_VERSION,
                build_id=phys.BUILD_ID,
                radius=radius[:end],
                pressure=pressure[:end],
                density=density[:end],
                temperature=temperature[:end],
                mass=mass[:end],
                surface_index=surface_index,
                output_type=output_type,
                radial_step=dr,
                restart_count=restart_count,
            )

        # The initial radial spacing did not span the whole object.  Double it
        # and restart; this sacrifices resolution to obtain sufficient range.
        dr *= 2.0
        restart_count += 1
        if restart_count > max_restarts or not isfinite(dr):
            raise RuntimeError(
                "Unable to reach the zero-pressure surface with finite radial steps."
            )
