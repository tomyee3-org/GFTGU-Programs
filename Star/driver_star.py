"""
driver_star.py — Numerical integration of polytropic stellar structure.

The calculation uses forward Euler steps in radius.  If the initially chosen
step does not span the star within max_points, the driver doubles the step and
restarts.  This restart is a radial-range mechanism, not an error-controlled
adaptive integrator.
"""

from dataclasses import dataclass
from bisect import bisect_left
from typing import get_args, List, Literal, Optional, Sequence
from math import inf, isfinite, nextafter, pi
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

# Largest accepted grid-point capacity.  Each attempt allocates five lists of
# this length, so the limit keeps the memory of one run to tens of megabytes.
MAX_POINTS_LIMIT = 1_000_000


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
    gamma: Optional[float] = None

    @property
    def last_index(self) -> int:
        """Backward-compatible alias for the surface array index."""
        return self.surface_index


@dataclass(frozen=True)
class ProfileCheckpoint:
    """Linearly interpolated stellar quantities at one radius fraction."""
    radius_fraction: float
    radius: float
    pressure: float
    density: float
    temperature: float
    mass: float


DEFAULT_CHECKPOINT_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 0.90)


def interpolate_profile_checkpoints(
    result: StarResult,
    fractions: Sequence[float] = DEFAULT_CHECKPOINT_FRACTIONS,
) -> List[ProfileCheckpoint]:
    """Return linearly interpolated profiles at fractions of surface radius."""
    arrays = (
        result.radius,
        result.pressure,
        result.density,
        result.temperature,
        result.mass,
    )
    if not result.radius or any(len(values) != len(result.radius) for values in arrays):
        raise ValueError("result profiles must be nonempty and have equal lengths.")
    if any(
        not isfinite(value)
        for values in arrays
        for value in values
    ):
        raise ValueError("result profiles must contain only finite values.")
    if result.radius[0] != 0.0 or any(
        right <= left for left, right in zip(result.radius, result.radius[1:])
    ):
        raise ValueError("result.radius must start at zero and strictly increase.")

    surface_radius = result.radius[-1]
    checkpoints = []
    for fraction in fractions:
        _validate_finite_real("radius fraction", fraction)
        fraction = float(fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("radius fractions must lie between zero and one.")

        target = fraction * surface_radius
        upper = bisect_left(result.radius, target)
        if upper == 0:
            lower = upper
            weight = 0.0
        elif upper == len(result.radius):
            lower = upper = len(result.radius) - 1
            weight = 0.0
        elif result.radius[upper] == target:
            lower = upper
            weight = 0.0
        else:
            lower = upper - 1
            weight = (
                (target - result.radius[lower])
                / (result.radius[upper] - result.radius[lower])
            )

        def interpolate(values):
            if lower == upper:
                return float(values[lower])
            return float(values[lower] + weight * (values[upper] - values[lower]))

        checkpoints.append(
            ProfileCheckpoint(
                radius_fraction=fraction,
                radius=target,
                pressure=interpolate(result.pressure),
                density=interpolate(result.density),
                temperature=interpolate(result.temperature),
                mass=interpolate(result.mass),
            )
        )
    return checkpoints


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
    if (not isinstance(max_points, int) or isinstance(max_points, bool)
            or not 3 <= max_points <= MAX_POINTS_LIMIT):
        raise ValueError(
            f"max_points must be an integer from 3 to {MAX_POINTS_LIMIT}."
        )
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
        try:
            mass[1] = 4.0 * pi * dr**3 * rho_c / 3.0
        except OverflowError:
            mass[1] = float("inf")
        if not isfinite(mass[1]):
            raise OverflowError(
                "The mass of the first central sphere is outside the finite "
                "floating-point range."
            )
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
                if radius[j] <= r_prev:
                    # The crossing lies within rounding of the last point; keep
                    # the radii strictly increasing by the smallest amount.
                    radius[j] = nextafter(r_prev, inf)
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
                gamma=float(gamma),
            )

        # The initial radial spacing did not span the whole object.  Double it
        # and restart; this sacrifices resolution to obtain sufficient range.
        dr *= 2.0
        restart_count += 1
        if restart_count > max_restarts or not isfinite(dr):
            raise RuntimeError(
                "Unable to reach the zero-pressure surface with finite radial steps."
            )
