"""
Driver for Neutron star structure.

The solver integrates the coupled TOV pressure equation and mass equation with
classical fourth-order Runge-Kutta (RK4). The radial grid grows dynamically;
there is no 2000-point array limit and no resolution-degrading step doubling.

The stellar surface is located by extrapolating the regular surface variable
p**((gamma-1)/gamma) to zero on the final interval.
"""

from __future__ import annotations

import math

import physics_neutron as phys
from physics_neutron import (
    C2,
    G,
    M_SUN,
    central_state,
    eos_density,
    rk4_step,
    structure_derivatives,
)


def _validate_inputs(
    gamma: float,
    pC: float,
    K: float,
    steps_per_scale: int,
    max_steps: int,
) -> None:
    def is_finite_number(value: float) -> bool:
        try:
            return not isinstance(value, bool) and math.isfinite(value)
        except TypeError:
            return False

    if not is_finite_number(gamma) or gamma <= 1.0:
        raise ValueError("gamma must be finite and greater than 1.")
    if not is_finite_number(pC) or pC <= 0.0:
        raise ValueError("pC must be a positive finite pressure.")
    if not is_finite_number(K) or K <= 0.0:
        raise ValueError("K must be positive and finite.")
    if (
        not isinstance(steps_per_scale, int)
        or isinstance(steps_per_scale, bool)
        or steps_per_scale < 50
    ):
        raise ValueError("steps_per_scale must be an integer of at least 50.")
    if (
        not isinstance(max_steps, int)
        or isinstance(max_steps, bool)
        or max_steps < 100
    ):
        raise ValueError("max_steps must be an integer of at least 100.")


def compute_neutron_star(
    gamma: float,
    pC: float,
    K: float,
    *,
    steps_per_scale: int = 400,
    max_steps: int = 200_000,
) -> dict:
    """
    Compute a static polytropic TOV model.

    Returns pressure, density, enclosed gravitational mass, and radius arrays,
    plus a compact summary of the resulting model.
    """
    _validate_inputs(gamma, pC, K, steps_per_scale, max_steps)

    rhoC = eos_density(pC, K, gamma)
    # Writing sqrt(pC/G) as sqrt(pC)/sqrt(G) avoids an intermediate overflow
    # for otherwise finite inputs.
    scale = math.sqrt(pC) / (math.sqrt(G) * rhoC)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "The requested parameters produce a radial scale outside the "
            "positive finite floating-point range."
        )
    dr_nominal = scale / float(steps_per_scale)

    # Start slightly away from the coordinate singularity at r=0 using the
    # regular central series.
    r = dr_nominal
    p, rho, m = central_state(pC, K, gamma, r)

    radius = [0.0, r]
    pressure = [pC, p]
    density = [rhoC, rho]
    mass = [0.0, m]

    min_step = dr_nominal / (2.0 ** 24)

    for _ in range(max_steps):
        # Near a polytropic surface, q = p**((gamma-1)/gamma) is linear in
        # radius to leading order. Extrapolating q, rather than p itself,
        # removes the systematic early-surface bias of a pressure tangent.
        dpdr, dmdr = structure_derivatives(r, p, m, K, gamma)
        if dpdr >= 0.0:
            raise RuntimeError(
                "Pressure stopped decreasing outward; the requested polytropic "
                "model is outside the regime expected by this solver."
            )

        pressure_scale_distance = -p / dpdr
        surface_distance = (
            gamma / (gamma - 1.0) * pressure_scale_distance
        )
        if 0.0 < surface_distance <= dr_nominal:
            r_surface = r + surface_distance

            # Integrating the leading polytropic density falloff over the
            # remaining interval gives dmdr * (-p/dpdr).
            m_surface = m + dmdr * pressure_scale_distance

            radius.append(r_surface)
            pressure.append(0.0)
            density.append(0.0)
            mass.append(m_surface)
            break

        # Normal RK4 step. If an intermediate stage crosses p=0, shorten the
        # trial step until all stages remain within the fluid.
        h = dr_nominal
        while True:
            if h < min_step:
                raise RuntimeError(
                    "Could not resolve the stellar surface with a positive "
                    "pressure RK4 step."
                )
            try:
                p_new, m_new = rk4_step(r, p, m, h, K, gamma)
            except ValueError:
                h *= 0.5
                continue
            break

        if p_new <= 0.0:
            # This should be rare because the local surface estimate above
            # usually catches the crossing. Interpolate conservatively.
            fraction = p / (p - p_new)
            r_surface = r + fraction * h
            m_surface = m + fraction * (m_new - m)
            radius.append(r_surface)
            pressure.append(0.0)
            density.append(0.0)
            mass.append(m_surface)
            break

        r += h
        p = p_new
        m = m_new
        rho = eos_density(p, K, gamma)

        radius.append(r)
        pressure.append(p)
        density.append(rho)
        mass.append(m)

        compactness = 2.0 * G * m / (r * C2)
        if compactness >= 1.0:
            raise RuntimeError(
                "The integration reached 2Gm/(rc^2) >= 1 before p=0. "
                "No regular static stellar surface was found."
            )
    else:
        raise RuntimeError(
            f"The stellar surface was not reached within {max_steps} radial "
            "steps. Increase max_steps or reconsider the model parameters."
        )

    R = radius[-1]
    M = mass[-1]
    compactness = 2.0 * G * M / (R * C2)
    buchdahl_ratio = compactness  # Buchdahl requires 2GM/(Rc^2) <= 8/9.
    central_sound_speed_squared_over_c2 = gamma * pC / (rhoC * C2)

    return {
        "model_version": phys.MODEL_VERSION,
        "build_id": phys.BUILD_ID,
        "radius": radius,
        "pressure": pressure,
        "density": density,
        "mass": mass,
        "last_step": len(radius) - 1,
        "gamma": gamma,
        "pC": pC,
        "K": K,
        "rhoC": rhoC,
        "scale": scale,
        "dr_nominal": dr_nominal,
        "surface_radius_m": R,
        "surface_radius_km": R / 1000.0,
        "total_mass_kg": M,
        "total_mass_solar": M / M_SUN,
        "compactness": compactness,
        "buchdahl_satisfied": buchdahl_ratio <= (8.0 / 9.0),
        "central_sound_speed_squared_over_c2": (
            central_sound_speed_squared_over_c2
        ),
        "central_sound_speed_over_c": math.sqrt(
            central_sound_speed_squared_over_c2
        ),
        "causality_satisfied": central_sound_speed_squared_over_c2 <= 1.0,
    }
