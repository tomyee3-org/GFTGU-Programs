"""
Adaptive predictor-corrector driver for Orbit.

The solver integrates Newtonian test-particle motion around a fixed point mass.
It uses:
  * a predictor followed by an iterated trapezoidal corrector,
  * adaptive timestep reduction and gradual recovery,
  * Euclidean vector convergence tests,
  * accumulated unwrapped azimuth for revolution counting,
  * explicit termination reasons and conservation diagnostics.

The requested maximum timestep is dt0.  The working timestep may shrink where
the orbit changes rapidly and then recover toward dt0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math

import numpy as np
from numpy.typing import NDArray

from physics_orbit import (
    compute_acceleration,
    specific_angular_momentum,
    specific_energy,
)


OutputType = Literal[
    "orbit",
    "velocity",
    "position_time",
    "velocity_time",
    "energy",
]


@dataclass
class OrbitResult:
    """Full trajectory history plus run diagnostics."""

    xs: NDArray[np.float64]
    ys: NDArray[np.float64]
    vxs: NDArray[np.float64]
    vys: NDArray[np.float64]
    ts: NDArray[np.float64]
    PEs: NDArray[np.float64]
    KEs: NDArray[np.float64]
    Hs: NDArray[np.float64]

    termination_reason: str
    accepted_steps: int
    final_time: float
    revolutions_completed: float

    max_fractional_energy_drift: float
    max_fractional_angular_momentum_drift: float | None

    closure_radius_residual: float | None
    closure_velocity_residual: float | None


def _validate_inputs(
    xInit: float,
    yInit: float,
    vxInit: float,
    vyInit: float,
    mu: float,
    dt0: float,
    maxSteps: int,
    eps1: float,
    eps2: float,
    maxOrbits: float,
) -> None:
    values = {
        "xInit": xInit,
        "yInit": yInit,
        "vxInit": vxInit,
        "vyInit": vyInit,
        "mu": mu,
        "dt0": dt0,
        "eps1": eps1,
        "eps2": eps2,
        "maxOrbits": maxOrbits,
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

    if math.hypot(xInit, yInit) <= 0.0:
        raise ValueError("The initial position must not be at r=0.")
    if mu <= 0.0:
        raise ValueError("mu=GM must be positive.")
    if dt0 <= 0.0:
        raise ValueError("dt0 must be positive.")
    if not isinstance(maxSteps, int) or isinstance(maxSteps, bool) or maxSteps <= 0:
        raise ValueError("maxSteps must be a positive integer.")
    if eps1 <= 0.0:
        raise ValueError("eps1 must be positive.")
    if eps2 <= 0.0:
        raise ValueError("eps2 must be positive.")
    if maxOrbits <= 0.0:
        raise ValueError("maxOrbits must be positive.")


def _relative_vector_change(
    old_x: float,
    old_y: float,
    new_x: float,
    new_y: float,
) -> float:
    difference = math.hypot(new_x - old_x, new_y - old_y)
    scale = max(math.hypot(old_x, old_y), math.hypot(new_x, new_y))
    if scale == 0.0:
        return 0.0 if difference == 0.0 else math.inf
    return difference / scale


def _relative_increment_change(
    old_dx: float,
    old_dy: float,
    new_dx: float,
    new_dy: float,
    reference_dx: float,
    reference_dy: float,
) -> float:
    """Change between correction increments, scaled by the initial increment."""
    difference = math.hypot(new_dx - old_dx, new_dy - old_dy)
    scale = math.hypot(reference_dx, reference_dy)
    if scale == 0.0:
        return 0.0 if difference == 0.0 else math.inf
    return difference / scale


def _unwrap_delta(new_angle: float, old_angle: float) -> float:
    """Signed angular change in (-pi, pi]."""
    delta = new_angle - old_angle
    while delta <= -math.pi:
        delta += 2.0 * math.pi
    while delta > math.pi:
        delta -= 2.0 * math.pi
    return delta


def _fractional_drift(value: float, reference: float) -> float:
    scale = abs(reference)
    if scale == 0.0:
        return abs(value - reference)
    return abs(value - reference) / scale


def _minimum_segment_radius(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    """Minimum distance from the origin to the straight endpoint segment."""
    dx = x1 - x0
    dy = y1 - y0
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return math.hypot(x0, y0)

    t = -(x0 * dx + y0 * dy) / denom
    t = min(1.0, max(0.0, t))
    return math.hypot(x0 + t * dx, y0 + t * dy)


def run_orbit(
    xInit: float,
    yInit: float,
    vxInit: float,
    vyInit: float,
    k: float,
    dt0: float,
    maxSteps: int,
    eps1: float,
    eps2: float,
    maxOrbits: float = 1.0,
) -> OrbitResult:
    """
    Integrate a Newtonian test-particle orbit.

    k is the central gravitational parameter GM.

    A bound orbit normally terminates at maxOrbits accumulated azimuthal
    revolutions.  An unbound or radial orbit generally terminates at maxSteps,
    unless the trajectory approaches the mathematical point-mass singularity.
    """
    _validate_inputs(
        xInit, yInit, vxInit, vyInit,
        k, dt0, maxSteps, eps1, eps2, maxOrbits,
    )

    x = float(xInit)
    y = float(yInit)
    vx = float(vxInit)
    vy = float(vyInit)
    t = 0.0

    initial_radius = math.hypot(x, y)
    initial_speed = math.hypot(vx, vy)
    energy0 = specific_energy(x, y, vx, vy, k)
    h0 = specific_angular_momentum(x, y, vx, vy)

    # Numerical guard only, not a physical stellar radius.  It prevents the
    # ideal point-mass singularity from generating infinities/NaNs.
    singularity_guard = max(1.0e-12 * initial_radius, 1.0e-6)

    xs = [x]
    ys = [y]
    vxs = [vx]
    vys = [vy]
    ts = [t]
    PEs = [-k / initial_radius]
    KEs = [0.5 * initial_speed * initial_speed]
    Hs = [h0]

    max_energy_drift = 0.0

    # A fractional angular-momentum drift is not meaningful when the initial
    # angular momentum is zero (or numerically indistinguishable from zero).
    h_scale = initial_radius * max(initial_speed, 1.0)
    report_fractional_h_drift = abs(h0) > 1.0e-12 * h_scale
    max_h_drift = 0.0 if report_fractional_h_drift else None

    dt_work = float(dt0)
    max_corrector_iterations = 10
    max_retries_per_step = 80

    angle_previous = math.atan2(y, x)
    accumulated_angle = 0.0
    target_angle = 2.0 * math.pi * maxOrbits

    termination_reason = "max_steps"
    accepted_steps = 0
    closure_radius_residual = None
    closure_velocity_residual = None

    while accepted_steps < maxSteps:
        accepted = False

        for _retry in range(max_retries_per_step):
            ax0, ay0 = compute_acceleration(x, y, k)

            # Constant-acceleration predictor.
            vx_pred = vx + ax0 * dt_work
            vy_pred = vy + ay0 * dt_work
            x_pred = x + 0.5 * (vx + vx_pred) * dt_work
            y_pred = y + 0.5 * (vy + vy_pred) * dt_work

            if _minimum_segment_radius(x, y, x_pred, y_pred) <= singularity_guard:
                termination_reason = "central_singularity"
                accepted = False
                break

            ax_pred, ay_pred = compute_acceleration(x_pred, y_pred, k)

            if _relative_vector_change(ax0, ay0, ax_pred, ay_pred) > eps1:
                dt_work *= 0.5
                continue

            # Iterated trapezoidal corrector.  Convergence is measured on
            # the velocity increment over the step, scaled by the initial
            # predicted increment.  This keeps eps2 tied to the correction
            # itself rather than to the much larger orbital velocity.
            vx_guess, vy_guess = vx_pred, vy_pred
            x_guess, y_guess = x_pred, y_pred
            ax_end, ay_end = ax_pred, ay_pred
            dvx_reference = vx_pred - vx
            dvy_reference = vy_pred - vy
            dvx_guess = dvx_reference
            dvy_guess = dvy_reference
            converged = False

            for _ in range(max_corrector_iterations):
                vx_corr = vx + 0.5 * (ax0 + ax_end) * dt_work
                vy_corr = vy + 0.5 * (ay0 + ay_end) * dt_work
                x_corr = x + 0.5 * (vx + vx_corr) * dt_work
                y_corr = y + 0.5 * (vy + vy_corr) * dt_work

                if _minimum_segment_radius(x, y, x_corr, y_corr) <= singularity_guard:
                    termination_reason = "central_singularity"
                    converged = False
                    break

                dvx_corr = vx_corr - vx
                dvy_corr = vy_corr - vy
                correction_change = _relative_increment_change(
                    dvx_guess,
                    dvy_guess,
                    dvx_corr,
                    dvy_corr,
                    dvx_reference,
                    dvy_reference,
                )

                x_guess, y_guess = x_corr, y_corr
                vx_guess, vy_guess = vx_corr, vy_corr
                dvx_guess, dvy_guess = dvx_corr, dvy_corr

                if correction_change <= eps2:
                    converged = True
                    break

                ax_end, ay_end = compute_acceleration(x_guess, y_guess, k)

            if termination_reason == "central_singularity":
                break

            if not converged:
                dt_work *= 0.5
                continue

            accepted = True
            break

        if termination_reason == "central_singularity":
            break

        if not accepted:
            raise RuntimeError(
                "Orbit could not find a converged timestep after "
                f"{max_retries_per_step} retries at t={t:.6g} s, "
                f"r={math.hypot(x, y):.6g} m, dt={dt_work:.6g} s."
            )

        x1, y1 = x_guess, y_guess
        vx1, vy1 = vx_guess, vy_guess
        t1 = t + dt_work

        angle_new = math.atan2(y1, x1)
        delta_angle = _unwrap_delta(angle_new, angle_previous)
        accumulated_before = accumulated_angle
        accumulated_after = accumulated_angle + delta_angle

        # If this accepted step crosses the requested accumulated revolution
        # count, interpolate to that angular crossing.  Radius, velocity, and
        # time are linearly interpolated over this final small step, while
        # position is reconstructed at the exact target azimuth.
        crossed_target = (
            abs(accumulated_before) < target_angle
            and abs(accumulated_after) >= target_angle
            and delta_angle != 0.0
        )

        if crossed_target:
            needed = target_angle - abs(accumulated_before)
            fraction = min(1.0, max(0.0, needed / abs(delta_angle)))
            direction = 1.0 if accumulated_after > 0.0 else -1.0

            r0 = math.hypot(x, y)
            r1 = math.hypot(x1, y1)
            r_final = r0 + fraction * (r1 - r0)

            final_angle = math.atan2(yInit, xInit) + direction * target_angle
            x1 = r_final * math.cos(final_angle)
            y1 = r_final * math.sin(final_angle)
            vx1 = vx + fraction * (vx1 - vx)
            vy1 = vy + fraction * (vy1 - vy)
            t1 = t + fraction * dt_work

            accumulated_after = direction * target_angle
            angle_new = math.atan2(y1, x1)
            termination_reason = "max_orbits"

        x, y, vx, vy, t = x1, y1, vx1, vy1, t1
        accumulated_angle = accumulated_after
        angle_previous = angle_new

        radius = math.hypot(x, y)
        speed2 = vx * vx + vy * vy
        pe = -k / radius
        ke = 0.5 * speed2
        h_now = specific_angular_momentum(x, y, vx, vy)
        energy_now = ke + pe

        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)
        ts.append(t)
        PEs.append(pe)
        KEs.append(ke)
        Hs.append(h_now)

        accepted_steps += 1

        max_energy_drift = max(
            max_energy_drift,
            _fractional_drift(energy_now, energy0),
        )
        if max_h_drift is not None:
            max_h_drift = max(
                max_h_drift,
                _fractional_drift(h_now, h0),
            )

        if crossed_target:
            # Closure residuals compare the final state with the initial state
            # and are meaningful only after an integral number of revolutions.
            nearest_integer_orbits = round(maxOrbits)
            if math.isclose(maxOrbits, nearest_integer_orbits, rel_tol=0.0, abs_tol=1.0e-12):
                closure_radius_residual = abs(radius - initial_radius) / initial_radius
                if initial_speed > 0.0:
                    closure_velocity_residual = (
                        math.hypot(vx - vxInit, vy - vyInit) / initial_speed
                    )
                else:
                    closure_velocity_residual = math.hypot(vx - vxInit, vy - vyInit)
            break

        # Gradual recovery after demanding portions of the orbit.
        dt_work = min(dt_work * 1.1, dt0)

    revolutions = abs(accumulated_angle) / (2.0 * math.pi)

    return OrbitResult(
        xs=np.asarray(xs, dtype=float),
        ys=np.asarray(ys, dtype=float),
        vxs=np.asarray(vxs, dtype=float),
        vys=np.asarray(vys, dtype=float),
        ts=np.asarray(ts, dtype=float),
        PEs=np.asarray(PEs, dtype=float),
        KEs=np.asarray(KEs, dtype=float),
        Hs=np.asarray(Hs, dtype=float),
        termination_reason=termination_reason,
        accepted_steps=accepted_steps,
        final_time=t,
        revolutions_completed=revolutions,
        max_fractional_energy_drift=max_energy_drift,
        max_fractional_angular_momentum_drift=max_h_drift,
        closure_radius_residual=closure_radius_residual,
        closure_velocity_residual=closure_velocity_residual,
    )
