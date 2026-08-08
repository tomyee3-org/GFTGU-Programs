"""
MercPert driver module (non-RK45 version)

Implements an explicit integrator with
time-step halving and a simple predictor-corrector
"""

from dataclasses import dataclass
from typing import List, Dict

from physics_mercpert import (
    BinarySystemParams,
    MercuryInitialConditions,
    mercury_acceleration,
    binary_positions,
)


@dataclass
class MercPertRunParams:
    dt: float          # initial time-step (s)
    max_steps: int     # maximum number of steps
    eps1: float        # accuracy parameter for time-step halving
    eps2: float        # accuracy parameter for predictor-corrector


@dataclass
class MercPertOutput:
    times: List[float]
    sun_x: List[float]
    sun_y: List[float]
    planet_x: List[float]
    planet_y: List[float]
    merc_x: List[float]
    merc_y: List[float]


def run_mercpert(binary_params: BinarySystemParams,
                 merc_ic: MercuryInitialConditions,
                 run_params: MercPertRunParams) -> MercPertOutput:
    """
    Main driver for MercPert.

    - Uses dt1 as working time-step, halved when fractional changes
      exceed eps1.
    - Uses a simple predictor-corrector iteration controlled by eps2.
    - Does NOT stop on completion of an orbit; runs until max_steps.
    """

    dt1 = run_params.dt
    max_steps = run_params.max_steps
    eps1 = run_params.eps1
    eps2 = run_params.eps2

    # Initial state of Mercury
    t = 0.0
    x_merc = merc_ic.x_init
    y_merc = merc_ic.y_init
    vx_merc = merc_ic.vx_init
    vy_merc = merc_ic.vy_init

    times: List[float] = []
    sun_x: List[float] = []
    sun_y: List[float] = []
    planet_x: List[float] = []
    planet_y: List[float] = []
    merc_x: List[float] = []
    merc_y: List[float] = []

    for step in range(max_steps):
        # Record current positions. This happens exactly once per
        # accepted step -- rejected attempts inside the retry loop below
        # never reach this point again, so they are never logged and
        # never consume the max_steps budget, mirroring the
        # `dt1 /= 2; j--;` retry in the original MercPert.java.
        (x_sun, y_sun), (x_planet, y_planet) = binary_positions(t, binary_params)

        times.append(t)
        sun_x.append(x_sun)
        sun_y.append(y_sun)
        planet_x.append(x_planet)
        planet_y.append(y_planet)
        merc_x.append(x_merc)
        merc_y.append(y_merc)

        # Compute acceleration at current state
        ax0, ay0 = mercury_acceleration(t, x_merc, y_merc, binary_params)

        while True:
            # Predictor step: simple Euler
            x_pred = x_merc + vx_merc * dt1
            y_pred = y_merc + vy_merc * dt1
            vx_pred = vx_merc + ax0 * dt1
            vy_pred = vy_merc + ay0 * dt1

            # Corrector iteration: average accelerations over the step
            # until fractional change is below eps2
            x_new = x_pred
            y_new = y_pred
            vx_new = vx_pred
            vy_new = vy_pred

            for _ in range(10):  # modest cap on iterations
                ax1, ay1 = mercury_acceleration(t + dt1, x_new, y_new, binary_params)

                vx_corr = vx_merc + 0.5 * (ax0 + ax1) * dt1
                vy_corr = vy_merc + 0.5 * (ay0 + ay1) * dt1
                x_corr = x_merc + 0.5 * (vx_merc + vx_corr) * dt1
                y_corr = y_merc + 0.5 * (vy_merc + vy_corr) * dt1

                # Check fractional changes
                dvx_frac = abs(vx_corr - vx_new) / max(abs(vx_corr), 1e-30)
                dvy_frac = abs(vy_corr - vy_new) / max(abs(vy_corr), 1e-30)
                dx_frac = abs(x_corr - x_new) / max(abs(x_corr), 1e-30)
                dy_frac = abs(y_corr - y_new) / max(abs(y_corr), 1e-30)

                x_new, y_new = x_corr, y_corr
                vx_new, vy_new = vx_corr, vy_corr

                if max(dvx_frac, dvy_frac, dx_frac, dy_frac) < eps2:
                    break

            # Time-step halving if changes are too large. A rejection
            # here only shrinks dt1 and loops back to retry the SAME
            # step; it does not touch the output lists and does not
            # advance `step`.
            dvx_frac0 = abs(vx_new - vx_merc) / max(abs(vx_new), 1e-30)
            dvy_frac0 = abs(vy_new - vy_merc) / max(abs(vy_new), 1e-30)
            dx_frac0 = abs(x_new - x_merc) / max(abs(x_new), 1e-30)
            dy_frac0 = abs(y_new - y_merc) / max(abs(y_new), 1e-30)

            if max(dvx_frac0, dvy_frac0, dx_frac0, dy_frac0) > eps1:
                dt1 *= 0.5
                continue  # retry the SAME step with a smaller dt1

            break  # step accepted; leave the retry loop

        # Accept step
        t += dt1
        x_merc, y_merc = x_new, y_new
        vx_merc, vy_merc = vx_new, vy_new

        # Let the step size grow back towards the user's requested dt
        # now that we know the last step was well-behaved.
        dt1 = min(dt1 * 1.1, run_params.dt)

    return MercPertOutput(
        times=times,
        sun_x=sun_x,
        sun_y=sun_y,
        planet_x=planet_x,
        planet_y=planet_y,
        merc_x=merc_x,
        merc_y=merc_y,
    )