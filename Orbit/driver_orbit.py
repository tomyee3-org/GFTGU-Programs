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
from numbers import Integral, Real
from typing import Literal
import math

import numpy as np
from numpy.typing import NDArray

import physics_orbit as phys
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

    model_version: str
    build_id: str

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

    max_fractional_energy_drift: float | None
    max_absolute_specific_energy_drift: float
    max_fractional_angular_momentum_drift: float | None
    max_absolute_specific_angular_momentum_drift: float

    closure_radius_residual: float | None
    closure_velocity_residual: float | None
    angular_step_rejections: int
    event_refinement_trials: int


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
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number.")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

    if math.hypot(xInit, yInit) <= 0.0:
        raise ValueError("The initial position must not be at r=0.")
    if mu <= 0.0:
        raise ValueError("mu=GM must be positive.")
    if dt0 <= 0.0:
        raise ValueError("dt0 must be positive.")
    if not isinstance(maxSteps, Integral) or isinstance(maxSteps, bool) or maxSteps <= 0:
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
    """Minimum segment radius using scaled coordinates to avoid overflow."""
    scale = max(abs(x0), abs(y0), abs(x1), abs(y1))
    if scale == 0.0:
        return 0.0

    sx0 = x0 / scale
    sy0 = y0 / scale
    sx1 = x1 / scale
    sy1 = y1 / scale
    dx = sx1 - sx0
    dy = sy1 - sy0
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return math.hypot(x0, y0)

    t = -(sx0 * dx + sy0 * dy) / denom
    t = min(1.0, max(0.0, t))
    return math.hypot(sx0 + t * dx, sy0 + t * dy) * scale


def _checked_time_advance(t: float, dt: float) -> float:
    """Return t + dt, rejecting overflow or loss of floating-point progress."""
    advanced = t + dt
    if not math.isfinite(advanced) or advanced <= t:
        raise RuntimeError(
            "The timestep can no longer advance simulated time at "
            f"t={t:.6g} s (working dt={dt:.6g} s)."
        )
    return advanced


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

    # Normalize accepted numeric scalar types (including NumPy real/integer
    # scalars) to the built-in types used throughout the integration.
    k = float(k)
    dt0 = float(dt0)
    maxSteps = int(maxSteps)
    eps1 = float(eps1)
    eps2 = float(eps2)
    maxOrbits = float(maxOrbits)

    x = float(xInit)
    y = float(yInit)
    vx = float(vxInit)
    vy = float(vyInit)
    t = 0.0

    initial_radius = math.hypot(x, y)
    initial_speed = math.hypot(vx, vy)
    if not math.isfinite(initial_radius):
        raise ValueError("The initial radius is outside floating-point range.")
    if not math.isfinite(initial_speed):
        raise ValueError("The initial speed is outside floating-point range.")
    energy0 = specific_energy(x, y, vx, vy, k)
    h0 = specific_angular_momentum(x, y, vx, vy)

    # Scale-relative numerical guard only, not a physical stellar radius.  The
    # ulp term keeps the threshold representable without imposing an SI length
    # floor that would exclude otherwise valid microscopic models.
    singularity_guard = max(
        1.0e-12 * initial_radius,
        32.0 * math.ulp(initial_radius),
    )
    # Stop accepted radial infall before roundoff-scale trial rejection makes
    # further progress unreliable.  This factor leaves room for timestep
    # refinement without assigning a physical radius to the central body.
    singularity_stop_factor = 1024.0
    singularity_stop_radius = singularity_stop_factor * singularity_guard

    xs = [x]
    ys = [y]
    vxs = [vx]
    vys = [vy]
    ts = [t]
    PEs = [-k / initial_radius]
    KEs = [0.5 * initial_speed * initial_speed]
    Hs = [h0]

    max_absolute_energy_drift = 0.0
    energy_scale = max(abs(KEs[0]), abs(PEs[0]))
    report_fractional_energy_drift = (
        energy_scale > 0.0 and abs(energy0) > 1.0e-12 * energy_scale
    )
    max_energy_drift = 0.0 if report_fractional_energy_drift else None

    # A fractional angular-momentum drift is not meaningful when the initial
    # angular momentum is zero (or numerically indistinguishable from zero).
    h_scale = initial_radius * initial_speed
    report_fractional_h_drift = (
        math.isfinite(h_scale)
        and h_scale > 0.0
        and abs(h0) > 1.0e-12 * h_scale
    )
    max_h_drift = 0.0 if report_fractional_h_drift else None
    max_absolute_h_drift = 0.0

    dt_work = float(dt0)
    max_corrector_iterations = 10
    max_retries_per_step = 80
    max_angular_step = 0.5 * math.pi
    max_event_refinement_trials = 80

    angle_previous = math.atan2(y, x)
    accumulated_angle = 0.0
    target_angle = 2.0 * math.pi * maxOrbits

    termination_reason = "max_steps"
    accepted_steps = 0
    closure_radius_residual = None
    closure_velocity_residual = None
    angular_step_rejections = 0
    event_refinement_trials = 0

    # Event refinement always re-integrates from the current accepted state.
    # These variables bracket the final timestep and its angular advance.
    event_needed = None
    event_lower_dt = 0.0
    event_lower_angle = 0.0
    event_upper_dt = 0.0
    event_upper_angle = 0.0
    event_trials_this_step = 0

    while accepted_steps < maxSteps:
        if math.hypot(x, y) <= singularity_stop_radius:
            termination_reason = "central_singularity"
            break

        accepted = False

        for _retry in range(max_retries_per_step):
            ax0, ay0 = compute_acceleration(x, y, k)

            # Constant-acceleration predictor.
            vx_pred = vx + ax0 * dt_work
            vy_pred = vy + ay0 * dt_work
            x_pred = x + 0.5 * (vx + vx_pred) * dt_work
            y_pred = y + 0.5 * (vy + vy_pred) * dt_work

            predicted_values = (x_pred, y_pred, vx_pred, vy_pred)
            predicted_radius = math.hypot(x_pred, y_pred)
            if (
                not all(math.isfinite(value) for value in predicted_values)
                or not math.isfinite(predicted_radius)
                or predicted_radius <= singularity_guard
            ):
                dt_work *= 0.5
                continue

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
            trial_hits_guard = False

            for _ in range(max_corrector_iterations):
                vx_corr = vx + 0.5 * (ax0 + ax_end) * dt_work
                vy_corr = vy + 0.5 * (ay0 + ay_end) * dt_work
                x_corr = x + 0.5 * (vx + vx_corr) * dt_work
                y_corr = y + 0.5 * (vy + vy_corr) * dt_work

                corrected_values = (x_corr, y_corr, vx_corr, vy_corr)
                corrected_radius = math.hypot(x_corr, y_corr)
                if (
                    not all(math.isfinite(value) for value in corrected_values)
                    or not math.isfinite(corrected_radius)
                    or corrected_radius <= singularity_guard
                ):
                    trial_hits_guard = True
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

            if trial_hits_guard:
                dt_work *= 0.5
                continue

            if not converged:
                dt_work *= 0.5
                continue

            segment_radius = _minimum_segment_radius(x, y, x_guess, y_guess)
            if segment_radius <= singularity_guard:
                # A guard intersection in a coarse trial is not itself a
                # physical event.  Retry with a smaller step; genuine infall
                # terminates only after accepted states approach the guard.
                dt_work *= 0.5
                continue

            h_start = abs(specific_angular_momentum(x, y, vx, vy))
            h_end = abs(specific_angular_momentum(x_guess, y_guess, vx_guess, vy_guess))
            try:
                angular_step_estimate = (
                    max(h_start, h_end) / segment_radius / segment_radius * dt_work
                )
            except OverflowError:
                angular_step_estimate = math.inf
            if not math.isfinite(angular_step_estimate) or angular_step_estimate > max_angular_step:
                # This conservative endpoint/chord estimate is a numerical
                # safeguard, not a formal bound on the curved numerical path.
                angular_step_rejections += 1
                dt_work *= 0.5
                continue

            accepted = True
            break

        if not accepted:
            raise RuntimeError(
                "Orbit could not find a converged timestep after "
                f"{max_retries_per_step} retries at t={t:.6g} s, "
                f"r={math.hypot(x, y):.6g} m, dt={dt_work:.6g} s."
            )

        x1, y1 = x_guess, y_guess
        vx1, vy1 = vx_guess, vy_guess
        t1 = _checked_time_advance(t, dt_work)

        angle_new = math.atan2(y1, x1)
        delta_angle = _unwrap_delta(angle_new, angle_previous)
        accumulated_before = accumulated_angle
        accumulated_after = accumulated_angle + delta_angle

        # If this converged step overshoots the requested accumulated angle,
        # retry it with a shorter timestep.  This leaves the final state on the
        # predictor-corrector solution instead of linearly interpolating state
        # components after integration.
        crossed_target = (
            abs(accumulated_before) < target_angle
            and abs(accumulated_after) >= target_angle
            and delta_angle != 0.0
        )

        if crossed_target and event_needed is None:
            event_needed = target_angle - abs(accumulated_before)
            event_upper_dt = dt_work
            event_upper_angle = abs(delta_angle)

        if event_needed is not None:
            event_trials_this_step += 1
            event_refinement_trials += 1
            if event_trials_this_step > max_event_refinement_trials:
                raise RuntimeError(
                    "Orbit could not refine the final revolution endpoint after "
                    f"{max_event_refinement_trials} trials at t={t:.6g} s."
                )

            trial_angle = abs(delta_angle)
            event_error = trial_angle - event_needed
            event_tolerance = 1.0e-12 * max(1.0, target_angle)
            if abs(event_error) <= event_tolerance:
                # Keep the actual integrated azimuth so the returned arrays
                # and revolutions_completed describe the same endpoint.
                termination_reason = "max_orbits"
            else:
                if event_error > 0.0:
                    event_upper_dt = dt_work
                    event_upper_angle = trial_angle
                else:
                    event_lower_dt = dt_work
                    event_lower_angle = trial_angle

                angle_span = event_upper_angle - event_lower_angle
                if angle_span > 0.0:
                    refined_dt = event_lower_dt + (
                        (event_needed - event_lower_angle)
                        * (event_upper_dt - event_lower_dt)
                        / angle_span
                    )
                else:
                    refined_dt = 0.5 * (event_lower_dt + event_upper_dt)

                # Keep the proposal strictly inside the bracket so rounding
                # cannot repeat an endpoint indefinitely.
                dt_margin = 0.1 * (event_upper_dt - event_lower_dt)
                refined_dt = min(
                    event_upper_dt - dt_margin,
                    max(event_lower_dt + dt_margin, refined_dt),
                )
                _checked_time_advance(t, refined_dt)
                dt_work = refined_dt
                continue

        x, y, vx, vy, t = x1, y1, vx1, vy1, t1
        accumulated_angle = accumulated_after
        angle_previous = angle_new

        radius = math.hypot(x, y)
        speed = math.hypot(vx, vy)
        speed2 = speed * speed
        pe = -k / radius
        ke = 0.5 * speed2
        h_now = specific_angular_momentum(x, y, vx, vy)
        energy_now = specific_energy(x, y, vx, vy, k)

        if not all(math.isfinite(value) for value in (radius, pe, ke)):
            raise RuntimeError(
                "The accepted state produced a non-finite derived quantity at "
                f"t={t:.6g} s, r={radius:.6g} m."
            )

        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)
        ts.append(t)
        PEs.append(pe)
        KEs.append(ke)
        Hs.append(h_now)

        accepted_steps += 1

        absolute_energy_drift = abs(energy_now - energy0)
        absolute_h_drift = abs(h_now - h0)
        if not all(math.isfinite(value) for value in (absolute_energy_drift, absolute_h_drift)):
            raise RuntimeError("A conservation diagnostic overflowed floating-point range.")

        max_absolute_energy_drift = max(
            max_absolute_energy_drift,
            absolute_energy_drift,
        )
        max_absolute_h_drift = max(max_absolute_h_drift, absolute_h_drift)
        if max_energy_drift is not None:
            max_energy_drift = max(
                max_energy_drift,
                _fractional_drift(energy_now, energy0),
            )
        if max_h_drift is not None:
            max_h_drift = max(
                max_h_drift,
                _fractional_drift(h_now, h0),
            )

        if termination_reason == "max_orbits":
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
        model_version=phys.MODEL_VERSION,
        build_id=phys.BUILD_ID,
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
        max_absolute_specific_energy_drift=max_absolute_energy_drift,
        max_fractional_angular_momentum_drift=max_h_drift,
        max_absolute_specific_angular_momentum_drift=max_absolute_h_drift,
        closure_radius_residual=closure_radius_residual,
        closure_velocity_residual=closure_velocity_residual,
        angular_step_rejections=angular_step_rejections,
        event_refinement_trials=event_refinement_trials,
    )
