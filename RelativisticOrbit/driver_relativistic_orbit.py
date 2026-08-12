"""
RelativisticOrbit driver module

This module performs the time integration of the relativistic orbit,
using a variable time-step predictor-corrector scheme and orbit counting.
"""

import math
from dataclasses import dataclass

from physics_relativistic_orbit import (
    kepler_constants,
    schwarzschild_acceleration,
    HORIZON_RADIUS,
)


@dataclass
class RelativisticOrbitParams:
    x_init: float      # initial x position (m)
    u_init: float      # initial y velocity (m/s)
    dt: float          # initial time step (s)
    max_steps: int     # maximum number of integration steps
    max_orbits: int    # maximum number of completed orbits
    eps1: float        # time-step accuracy control
    eps2: float        # predictor–corrector accuracy control


@dataclass
class RelativisticOrbitResult:
    x: list[float]
    y: list[float]
    horizon_x: list[float]
    horizon_y: list[float]
    n_orbits: float
    final_step: int
    fell_into_hole: bool


def _build_horizon_circle() -> tuple[list[float], list[float]]:
    """Build 101 points on the Schwarzschild horizon circle."""
    horizon_x = []
    horizon_y = []
    angle_step = math.pi / 50.0  # 0..2π with 101 points
    for i in range(101):
        angle = angle_step * i
        horizon_x.append(HORIZON_RADIUS * math.cos(angle))
        horizon_y.append(HORIZON_RADIUS * math.sin(angle))
    return horizon_x, horizon_y


def integrate_relativistic_orbit(params: RelativisticOrbitParams) -> RelativisticOrbitResult:
    """
    Perform the relativistic orbit integration using
    predictor–corrector and orbit counting.
    """

    # Initial conditions
    dt1 = params.dt
    v = 0.0
    u = params.u_init
    x0 = params.x_init
    y0 = 0.0

    # Kepler constants (fixed along the orbit)
    K, Q = kepler_constants(x0, u)

    # Initial acceleration
    ax0, ay0 = schwarzschild_acceleration(x0, y0, K, Q)

    # Storage for trajectory
    x_coordinate = [0.0] * params.max_steps
    y_coordinate = [0.0] * params.max_steps
    x_coordinate[0] = x0
    y_coordinate[0] = y0

    # Horizon circle
    horizon_x, horizon_y = _build_horizon_circle()

    # Orbit counting
    n_orbits = 0.0
    counterclockwise = (params.u_init > 0.0)
    full_orbit = False
    half_orbit = False

    # Helper to compute current radius and angle
    def current_radius_angle(x: float, y: float) -> tuple[float, float]:
        r = math.sqrt(x * x + y * y)
        angle = math.atan2(y, x)
        return r, angle

    # Initial radius
    r, angle_now = current_radius_angle(x0, y0)

    fell_into_hole = False
    final_step = 0

    # Main integration loop
    j = 1
    while (r > HORIZON_RADIUS) and (n_orbits < params.max_orbits) and (j < params.max_steps):
        # --- Predictor step (Euler + kinematics) ---
        dv = ax0 * dt1
        du = ay0 * dt1

        dx = v * dt1
        dy = u * dt1

        # extra displacement due to changing velocity (average velocity)
        ddx0 = 0.5 * dv * dt1
        ddy0 = 0.5 * du * dt1

        # initial guess for new position
        x1 = x0 + dx + ddx0
        y1 = y0 + dy + ddy0

        # acceleration at predicted position
        ax1, ay1 = schwarzschild_acceleration(x1, y1, K, Q)

        # --- Time-step accuracy control (eps1) ---
        # Time-step accuracy control Checks against the
        # cheap predictor's acceleration, BEFORE spending effort on the
        # corrector iterations below, using the sum-of-absolute-values
        # test (not a Euclidean-norm ratio, and not a component-wise
        # test divided by a single component -- the real algorithm sums
        # |ax0|+|ay0| together, so it is never at risk of a lone-axis
        # division by zero in the first place).
        accel_change = abs(ax1 - ax0) + abs(ay1 - ay0)
        accel_ref = abs(ax0) + abs(ay0)
        if accel_ref > 0.0 and accel_change > params.eps1 * accel_ref:
            dt1 *= 0.5
            # recompute with smaller step on next iteration
            continue

        # --- Predictor–corrector iteration ---
        # We iteratively refine ddx1, ddy1 using averaged acceleration.
        ddx1 = ddx0
        ddy1 = ddy0

        # Simple iteration: average old and new acceleration until change small
        test_prediction = abs(ddx0) + abs(ddy0)
        for _ in range(10):  # cap iterations
            dv1 = ax1 * dt1
            du1 = ay1 * dt1

            new_ddx1 = 0.5 * (dv + dv1) * dt1 / 2.0
            new_ddy1 = 0.5 * (du + du1) * dt1 / 2.0

            # relative change test: sum of absolute changes compared
            # against a threshold fixed once, before the loop, as
            # eps2 * (abs(ddx0)+abs(ddy0))
            change = abs(new_ddx1 - ddx1) + abs(new_ddy1 - ddy1)

            ddx1 = new_ddx1
            ddy1 = new_ddy1

            # update predicted position with refined ddx1, ddy1
            x1 = x0 + dx + ddx1
            y1 = y0 + dy + ddy1
            ax1, ay1 = schwarzschild_acceleration(x1, y1, K, Q)

            if change < params.eps2 * test_prediction:
                break

        # Accept step: update velocities and positions, using the
        # averaged acceleration for the velocity update which is
        # computed from (ax0+ax1)/2*dt1 inside
        # the corrector loop and then applied as v+=dv; u+=du;).
        v += 0.5 * (ax0 + ax1) * dt1
        u += 0.5 * (ay0 + ay1) * dt1
        x0 = x1
        y0 = y1
        ax0, ay0 = ax1, ay1

        x_coordinate[j] = x0
        y_coordinate[j] = y0

        # Update radius and angle
        r, angle_now = current_radius_angle(x0, y0)

        # --- Orbit counting ---
        # We detect crossings of the x-axis to count half and full orbits.
        if counterclockwise:
            if (y0 > 0.0) and (not half_orbit):
                half_orbit = True
            if (y0 < 0.0) and half_orbit and (not full_orbit):
                full_orbit = True
                n_orbits += 1.0
                half_orbit = False
        else:
            if (y0 < 0.0) and (not half_orbit):
                half_orbit = True
            if (y0 > 0.0) and half_orbit and (not full_orbit):
                full_orbit = True
                n_orbits += 1.0
                half_orbit = False

        full_orbit = False  # reset flag each step

        final_step = j
        j += 1

        if r <= HORIZON_RADIUS:
            fell_into_hole = True
            break

    # Trim arrays to actual length
    x_out = x_coordinate[: final_step + 1]
    y_out = y_coordinate[: final_step + 1]

    return RelativisticOrbitResult(
        x=x_out,
        y=y_out,
        horizon_x=horizon_x,
        horizon_y=horizon_y,
        n_orbits=n_orbits,
        final_step=final_step,
        fell_into_hole=fell_into_hole,
    )
