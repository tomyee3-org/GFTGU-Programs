"""
Integration driver for MercPert.
"""

from dataclasses import dataclass
from math import hypot, isfinite
import math
from typing import List, Optional

from physics_mercpert import (
    BinarySystemParams,
    MercuryInitialConditions,
    binary_positions,
    distances_to_primaries,
    jacobi_constant,
    mercury_acceleration,
    mercury_initial_barycentric_state,
    validate_binary_params,
    validate_mercury_ic,
)


@dataclass(frozen=True)
class MercPertRunParams:
    dt: float
    max_steps: int
    eps1: float
    eps2: float

    # Optional finite-radius stopping surfaces. A value of 0 disables the
    # corresponding collision test.
    sun_collision_radius: float = 0.0
    companion_collision_radius: float = 0.0


@dataclass
class MercPertOutput:
    times: List[float]
    sun_x: List[float]
    sun_y: List[float]
    planet_x: List[float]
    planet_y: List[float]
    merc_x: List[float]
    merc_y: List[float]
    merc_vx: List[float]
    merc_vy: List[float]
    jacobi: List[float]
    dt_used: List[float]
    accepted_steps: int
    termination_reason: str
    collision_body: Optional[str] = None


def _vector_relative_change(
    old_x: float,
    old_y: float,
    new_x: float,
    new_y: float,
) -> float:
    """Dimensionless relative change of a two-dimensional vector."""
    change = hypot(new_x - old_x, new_y - old_y)
    scale = max(hypot(old_x, old_y), hypot(new_x, new_y))
    return 0.0 if scale == 0.0 else change / scale


def _moving_circle_crossing_fraction(
    particle_start: tuple[float, float],
    particle_end: tuple[float, float],
    primary_start: tuple[float, float],
    primary_end: tuple[float, float],
    radius: float,
) -> Optional[float]:
    """
    Return the first fraction s in [0, 1] at which an accepted particle
    segment crosses a moving circular collision boundary.

    Over one accepted timestep, both the particle and the primary are
    represented by the straight segment joining their endpoint positions.
    Equivalently, the relative displacement is linear in s.  Near a close
    encounter the adaptive timestep is already reduced, so this is a much
    stronger collision test than checking accepted endpoints alone.
    """
    if radius <= 0.0:
        return None

    rx0 = particle_start[0] - primary_start[0]
    ry0 = particle_start[1] - primary_start[1]
    rx1 = particle_end[0] - primary_end[0]
    ry1 = particle_end[1] - primary_end[1]

    # Already inside/on the boundary.
    c = rx0 * rx0 + ry0 * ry0 - radius * radius
    if c <= 0.0:
        return 0.0

    dx = rx1 - rx0
    dy = ry1 - ry0
    a = dx * dx + dy * dy
    if a == 0.0:
        return None

    b = 2.0 * (rx0 * dx + ry0 * dy)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None

    root = math.sqrt(disc)
    roots = (
        (-b - root) / (2.0 * a),
        (-b + root) / (2.0 * a),
    )
    valid = [s for s in roots if 0.0 <= s <= 1.0]
    return min(valid) if valid else None


def _validate_run_params(run: MercPertRunParams) -> None:
    if not isfinite(run.dt) or run.dt <= 0.0:
        raise ValueError("dt must be a positive finite number.")
    if not isinstance(run.max_steps, int) or isinstance(run.max_steps, bool) or run.max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")
    if not isfinite(run.eps1) or not (0.0 < run.eps1 < 1.0):
        raise ValueError("eps1 must satisfy 0 < eps1 < 1.")
    if not isfinite(run.eps2) or not (0.0 < run.eps2 < 1.0):
        raise ValueError("eps2 must satisfy 0 < eps2 < 1.")
    if run.eps2 >= run.eps1:
        raise ValueError("eps2 should be smaller than eps1.")
    for name, value in (
        ("sun_collision_radius", run.sun_collision_radius),
        ("companion_collision_radius", run.companion_collision_radius),
    ):
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")


def run_mercpert(
    binary_params: BinarySystemParams,
    merc_ic: MercuryInitialConditions,
    run_params: MercPertRunParams,
) -> MercPertOutput:
    """
    Integrate Mercury's motion in the planar circular restricted three-body problem.

    User-supplied Mercury initial conditions are Sun-relative; they are converted
    once to barycentric inertial coordinates before integration.

    The calculation always stops after max_steps accepted steps unless a configured
    finite-radius collision surface is reached first.
    """
    validate_binary_params(binary_params)
    validate_mercury_ic(merc_ic)
    _validate_run_params(run_params)

    t = 0.0
    x_merc, y_merc, vx_merc, vy_merc = mercury_initial_barycentric_state(
        binary_params, merc_ic
    )

    dt_work = run_params.dt
    max_corrector_iterations = 10
    max_retries_per_step = 80

    times: List[float] = []
    sun_x: List[float] = []
    sun_y: List[float] = []
    planet_x: List[float] = []
    planet_y: List[float] = []
    merc_x: List[float] = []
    merc_y: List[float] = []
    merc_vx: List[float] = []
    merc_vy: List[float] = []
    jacobi: List[float] = []
    dt_used: List[float] = []

    def record_state(step_dt: float) -> None:
        (xs, ys), (xp, yp) = binary_positions(t, binary_params)
        times.append(t)
        sun_x.append(xs)
        sun_y.append(ys)
        planet_x.append(xp)
        planet_y.append(yp)
        merc_x.append(x_merc)
        merc_y.append(y_merc)
        merc_vx.append(vx_merc)
        merc_vy.append(vy_merc)
        jacobi.append(
            jacobi_constant(t, x_merc, y_merc, vx_merc, vy_merc, binary_params)
        )
        dt_used.append(step_dt)

    def collision_at_state() -> Optional[str]:
        r_sun, r_planet = distances_to_primaries(
            t, x_merc, y_merc, binary_params
        )
        if run_params.sun_collision_radius > 0.0 and r_sun <= run_params.sun_collision_radius:
            return "Sun"
        if (
            run_params.companion_collision_radius > 0.0
            and r_planet <= run_params.companion_collision_radius
        ):
            return "companion"
        return None

    # Initial sample. dt_used=0 identifies the starting point.
    record_state(0.0)
    initial_collision = collision_at_state()
    if initial_collision is not None:
        return MercPertOutput(
            times, sun_x, sun_y, planet_x, planet_y,
            merc_x, merc_y, merc_vx, merc_vy, jacobi, dt_used,
            accepted_steps=0,
            termination_reason=f"collision with {initial_collision}",
            collision_body=initial_collision,
        )

    accepted_steps = 0
    termination_reason = "max_steps reached"
    collision_body: Optional[str] = None

    while accepted_steps < run_params.max_steps:
        ax0, ay0 = mercury_acceleration(
            t, x_merc, y_merc, binary_params
        )

        accepted = False

        for _retry in range(max_retries_per_step):
            # Euler predictor, used as the inexpensive eps1 accuracy gate.
            x_pred = x_merc + vx_merc * dt_work
            y_pred = y_merc + vy_merc * dt_work
            vx_pred = vx_merc + ax0 * dt_work
            vy_pred = vy_merc + ay0 * dt_work

            ax_pred, ay_pred = mercury_acceleration(
                t + dt_work, x_pred, y_pred, binary_params
            )

            acc_change = _vector_relative_change(
                ax0, ay0, ax_pred, ay_pred
            )
            if acc_change > run_params.eps1:
                dt_work *= 0.5
                continue

            # Iterated trapezoidal corrector.
            x_new, y_new = x_pred, y_pred
            vx_new, vy_new = vx_pred, vy_pred
            ax1, ay1 = ax_pred, ay_pred
            converged = False

            for _ in range(max_corrector_iterations):
                vx_corr = vx_merc + 0.5 * (ax0 + ax1) * dt_work
                vy_corr = vy_merc + 0.5 * (ay0 + ay1) * dt_work
                x_corr = x_merc + 0.5 * (vx_merc + vx_corr) * dt_work
                y_corr = y_merc + 0.5 * (vy_merc + vy_corr) * dt_work

                velocity_change = _vector_relative_change(
                    vx_new, vy_new, vx_corr, vy_corr
                )

                x_new, y_new = x_corr, y_corr
                vx_new, vy_new = vx_corr, vy_corr

                if velocity_change < run_params.eps2:
                    converged = True
                    break

                ax1, ay1 = mercury_acceleration(
                    t + dt_work, x_new, y_new, binary_params
                )

            if not converged:
                dt_work *= 0.5
                continue

            accepted = True
            break

        if not accepted:
            raise RuntimeError(
                "MercPert could not find a converged timestep after "
                f"{max_retries_per_step} retries. Try a smaller dt or less "
                "extreme initial conditions."
            )

        # Before committing the accepted endpoint, test the entire accepted
        # segment against each moving finite-radius primary.  Endpoint-only
        # testing can miss a fast crossing that begins and ends outside a star.
        t_start = t
        particle_start = (x_merc, y_merc)
        particle_end = (x_new, y_new)
        sun_start, companion_start = binary_positions(t_start, binary_params)
        sun_end, companion_end = binary_positions(t_start + dt_work, binary_params)

        crossing_candidates = []
        s_sun = _moving_circle_crossing_fraction(
            particle_start,
            particle_end,
            sun_start,
            sun_end,
            run_params.sun_collision_radius,
        )
        if s_sun is not None:
            crossing_candidates.append((s_sun, "Sun"))

        s_companion = _moving_circle_crossing_fraction(
            particle_start,
            particle_end,
            companion_start,
            companion_end,
            run_params.companion_collision_radius,
        )
        if s_companion is not None:
            crossing_candidates.append((s_companion, "companion"))

        if crossing_candidates:
            s_hit, collision_body = min(crossing_candidates, key=lambda item: item[0])

            # Stop at the first boundary crossing. Linear interpolation here is
            # consistent with the segment used by the event detector.
            t += s_hit * dt_work
            x_merc += s_hit * (x_new - x_merc)
            y_merc += s_hit * (y_new - y_merc)
            vx_merc += s_hit * (vx_new - vx_merc)
            vy_merc += s_hit * (vy_new - vy_merc)
            accepted_steps += 1
            record_state(s_hit * dt_work)
            termination_reason = f"collision with {collision_body}"
            break

        # No collision: accept and record the full endpoint.
        t += dt_work
        x_merc, y_merc = x_new, y_new
        vx_merc, vy_merc = vx_new, vy_new
        accepted_steps += 1
        record_state(dt_work)

        # Recover gradually toward the user-specified maximum timestep.
        dt_work = min(dt_work * 1.1, run_params.dt)

    return MercPertOutput(
        times=times,
        sun_x=sun_x,
        sun_y=sun_y,
        planet_x=planet_x,
        planet_y=planet_y,
        merc_x=merc_x,
        merc_y=merc_y,
        merc_vx=merc_vx,
        merc_vy=merc_vy,
        jacobi=jacobi,
        dt_used=dt_used,
        accepted_steps=accepted_steps,
        termination_reason=termination_reason,
        collision_body=collision_body,
    )
