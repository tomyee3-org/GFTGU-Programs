"""
Adaptive predictor-corrector driver for RelativisticOrbit.

The independent variable is the test particle's proper time tau.  The driver
also tracks unwrapped azimuth, periapsides, conserved-quantity drift, horizon
crossing, and the reason the integration terminated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import physics_relativistic_orbit as phys
from physics_relativistic_orbit import (
    HORIZON_RADIUS,
    central_acceleration,
    effective_specific_energy,
    orbital_constants,
    specific_angular_momentum,
)


@dataclass
class RelativisticOrbitParams:
    x_init: float
    u_init: float
    dt: float
    max_steps: int
    max_orbits: int
    eps1: float
    eps2: float
    model: str = "schwarzschild"  # "schwarzschild" or "newtonian"


@dataclass
class RelativisticOrbitResult:
    model_version: str
    build_id: str
    x: list[float]
    y: list[float]
    vx: list[float]
    vy: list[float]
    tau: list[float]
    azimuth_unwrapped: list[float]
    n_orbits: float
    final_step: int
    fell_into_hole: bool
    termination_reason: str
    model: str
    periapsis_indices: list[int] = field(default_factory=list)
    periapsis_tau: list[float] = field(default_factory=list)
    periapsis_radius: list[float] = field(default_factory=list)
    periapsis_azimuth: list[float] = field(default_factory=list)
    mean_periapsis_advance: float | None = None
    max_fractional_h_drift: float = 0.0
    max_fractional_energy_drift: float = 0.0


def _validate_params(params: RelativisticOrbitParams) -> None:
    if not isinstance(params.model, str):
        raise ValueError('model must be "schwarzschild" or "newtonian".')

    finite_values = {
        "x_init": params.x_init,
        "u_init": params.u_init,
        "dt": params.dt,
        "eps1": params.eps1,
        "eps2": params.eps2,
    }
    for name, value in finite_values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

    if params.x_init <= 0.0:
        raise ValueError("x_init must be positive.")
    if params.model.lower() == "schwarzschild" and params.x_init <= HORIZON_RADIUS:
        raise ValueError("x_init must lie outside the Schwarzschild horizon.")
    if params.model.lower() not in ("schwarzschild", "newtonian"):
        raise ValueError('model must be "schwarzschild" or "newtonian".')
    if params.dt <= 0.0:
        raise ValueError("dt must be positive.")
    if not isinstance(params.max_steps, int) or isinstance(params.max_steps, bool) or params.max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")
    if not isinstance(params.max_orbits, int) or isinstance(params.max_orbits, bool) or params.max_orbits <= 0:
        raise ValueError("max_orbits must be a positive integer.")
    if not (0.0 < params.eps2 < params.eps1 < 1.0):
        raise ValueError("Require 0 < eps2 < eps1 < 1.")


def _relative_vector_change(
    old_x: float,
    old_y: float,
    new_x: float,
    new_y: float,
) -> float:
    diff = math.hypot(new_x - old_x, new_y - old_y)
    scale = max(math.hypot(old_x, old_y), math.hypot(new_x, new_y))
    if scale == 0.0:
        return 0.0 if diff == 0.0 else math.inf
    return diff / scale


def _unwrap_delta(new_angle: float, old_angle: float) -> float:
    """Return the signed angular change in (-pi, pi]."""
    delta = new_angle - old_angle
    while delta <= -math.pi:
        delta += 2.0 * math.pi
    while delta > math.pi:
        delta -= 2.0 * math.pi
    return delta


def _segment_circle_first_fraction(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    radius: float,
) -> float | None:
    """Return first t in [0,1] where the line segment intersects the circle."""
    dx = x1 - x0
    dy = y1 - y0
    a = dx * dx + dy * dy
    if a == 0.0:
        return None

    b = 2.0 * (x0 * dx + y0 * dy)
    c = x0 * x0 + y0 * y0 - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None

    root = math.sqrt(max(0.0, disc))
    candidates = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
    valid = [t for t in candidates if 0.0 <= t <= 1.0]
    return min(valid) if valid else None


def _fractional_drift(value: float, reference: float) -> float:
    scale = abs(reference)
    if scale == 0.0:
        return abs(value - reference)
    return abs(value - reference) / scale


def _require_finite_state(stage: str, *values: float) -> None:
    """Raise a student-facing error if an attempted integration state is non-finite."""
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(
            f"RelativisticOrbit produced a non-finite value during {stage}. "
            "Use less extreme initial data and/or a smaller dt."
        )


def integrate_relativistic_orbit(
    params: RelativisticOrbitParams,
) -> RelativisticOrbitResult:
    """Integrate until max_orbits, max_steps, or horizon crossing."""
    _validate_params(params)

    model = params.model.lower()
    dt_work = float(params.dt)
    max_corrector_iterations = 10
    max_retries_per_step = 80

    # State: vx=dx/dtau, vy=dy/dtau.
    x0 = float(params.x_init)
    y0 = 0.0
    vx0 = 0.0
    vy0 = float(params.u_init)
    tau0 = 0.0

    h_constant, _q = orbital_constants(x0, vy0)
    ax0, ay0 = central_acceleration(x0, y0, h_constant, model)

    h_initial = specific_angular_momentum(x0, y0, vx0, vy0)
    energy_initial = effective_specific_energy(
        x0, y0, vx0, vy0, h_constant, model
    )

    x = [x0]
    y = [y0]
    vx = [vx0]
    vy = [vy0]
    tau = [tau0]

    angle0 = math.atan2(y0, x0)
    azimuth_unwrapped = [angle0]
    accumulated_angle = 0.0

    max_h_drift = 0.0
    max_energy_drift = 0.0

    periapsis_indices: list[int] = []
    periapsis_tau: list[float] = []
    periapsis_radius: list[float] = []
    periapsis_azimuth: list[float] = []

    termination_reason = "max_steps"
    fell_into_hole = False

    accepted_steps = 0

    while accepted_steps < params.max_steps:
        if abs(accumulated_angle) >= 2.0 * math.pi * params.max_orbits:
            termination_reason = "max_orbits"
            break

        accepted = False

        for _retry in range(max_retries_per_step):
            # Constant-acceleration predictor.
            x_pred = x0 + vx0 * dt_work + 0.5 * ax0 * dt_work * dt_work
            y_pred = y0 + vy0 * dt_work + 0.5 * ay0 * dt_work * dt_work
            vx_pred = vx0 + ax0 * dt_work
            vy_pred = vy0 + ay0 * dt_work
            _require_finite_state(
                "the predictor", x_pred, y_pred, vx_pred, vy_pred
            )

            ax_pred, ay_pred = central_acceleration(
                x_pred, y_pred, h_constant, model
            )
            _require_finite_state("the predicted acceleration", ax_pred, ay_pred)

            if _relative_vector_change(ax0, ay0, ax_pred, ay_pred) > params.eps1:
                dt_work *= 0.5
                continue

            # Iterated trapezoidal corrector.
            x_guess, y_guess = x_pred, y_pred
            vx_guess, vy_guess = vx_pred, vy_pred
            ax_end, ay_end = ax_pred, ay_pred
            converged = False

            for _ in range(max_corrector_iterations):
                vx_corr = vx0 + 0.5 * (ax0 + ax_end) * dt_work
                vy_corr = vy0 + 0.5 * (ay0 + ay_end) * dt_work
                x_corr = x0 + 0.5 * (vx0 + vx_corr) * dt_work
                y_corr = y0 + 0.5 * (vy0 + vy_corr) * dt_work
                _require_finite_state(
                    "the corrector", x_corr, y_corr, vx_corr, vy_corr
                )

                velocity_change = _relative_vector_change(
                    vx_guess, vy_guess, vx_corr, vy_corr
                )

                x_guess, y_guess = x_corr, y_corr
                vx_guess, vy_guess = vx_corr, vy_corr

                if velocity_change < params.eps2:
                    converged = True
                    break

                ax_end, ay_end = central_acceleration(
                    x_guess, y_guess, h_constant, model
                )
                _require_finite_state(
                    "the corrected acceleration", ax_end, ay_end
                )

            if not converged:
                dt_work *= 0.5
                continue

            accepted = True
            break

        if not accepted:
            raise RuntimeError(
                "RelativisticOrbit could not find a converged timestep after "
                f"{max_retries_per_step} retries."
            )

        x1, y1 = x_guess, y_guess
        vx1, vy1 = vx_guess, vy_guess
        tau1 = tau0 + dt_work

        # Detect the first crossing of the true Schwarzschild horizon.  This is
        # disabled for the Newtonian comparison model.
        horizon_fraction = None
        if model == "schwarzschild":
            horizon_fraction = _segment_circle_first_fraction(
                x0, y0, x1, y1, HORIZON_RADIUS
            )

        if horizon_fraction is not None:
            f = horizon_fraction
            x1 = x0 + f * (x1 - x0)
            y1 = y0 + f * (y1 - y0)
            vx1 = vx0 + f * (vx1 - vx0)
            vy1 = vy0 + f * (vy1 - vy0)
            tau1 = tau0 + f * dt_work
            fell_into_hole = True
            termination_reason = "horizon"

        angle_old = math.atan2(y0, x0)
        angle_new = math.atan2(y1, x1)
        delta_angle = _unwrap_delta(angle_new, angle_old)
        accumulated_angle += delta_angle

        x.append(x1)
        y.append(y1)
        vx.append(vx1)
        vy.append(vy1)
        tau.append(tau1)
        azimuth_unwrapped.append(azimuth_unwrapped[-1] + delta_angle)

        accepted_steps += 1

        h_now = specific_angular_momentum(x1, y1, vx1, vy1)
        e_now = effective_specific_energy(
            x1, y1, vx1, vy1, h_constant, model
        )
        max_h_drift = max(max_h_drift, _fractional_drift(h_now, h_initial))
        max_energy_drift = max(
            max_energy_drift,
            _fractional_drift(e_now, energy_initial),
        )

        # Local radius minimum at the previous accepted point.
        if len(x) >= 3:
            r_a = math.hypot(x[-3], y[-3])
            r_b = math.hypot(x[-2], y[-2])
            r_c = math.hypot(x[-1], y[-1])
            if r_b < r_a and r_b <= r_c:
                idx = len(x) - 2
                periapsis_indices.append(idx)
                periapsis_tau.append(tau[idx])
                periapsis_radius.append(r_b)
                periapsis_azimuth.append(azimuth_unwrapped[idx])

        x0, y0, vx0, vy0, tau0 = x1, y1, vx1, vy1, tau1
        ax0, ay0 = central_acceleration(x0, y0, h_constant, model)

        if fell_into_hole:
            break

        if abs(accumulated_angle) >= 2.0 * math.pi * params.max_orbits:
            termination_reason = "max_orbits"
            break

        # Recover gradually after a close passage, never above the user's dt.
        dt_work = min(dt_work * 1.1, params.dt)

    else:
        termination_reason = "max_steps"

    n_orbits = abs(accumulated_angle) / (2.0 * math.pi)

    mean_advance = None
    if len(periapsis_azimuth) >= 2:
        direction = 1.0 if periapsis_azimuth[-1] >= periapsis_azimuth[0] else -1.0
        advances = []
        for a0, a1 in zip(periapsis_azimuth[:-1], periapsis_azimuth[1:]):
            radial_period_angle = direction * (a1 - a0)
            advances.append(radial_period_angle - 2.0 * math.pi)
        mean_advance = sum(advances) / len(advances)

    return RelativisticOrbitResult(
        model_version=phys.MODEL_VERSION,
        build_id=phys.BUILD_ID,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        tau=tau,
        azimuth_unwrapped=azimuth_unwrapped,
        n_orbits=n_orbits,
        final_step=accepted_steps,
        fell_into_hole=fell_into_hole,
        termination_reason=termination_reason,
        model=model,
        periapsis_indices=periapsis_indices,
        periapsis_tau=periapsis_tau,
        periapsis_radius=periapsis_radius,
        periapsis_azimuth=periapsis_azimuth,
        mean_periapsis_advance=mean_advance,
        max_fractional_h_drift=max_h_drift,
        max_fractional_energy_drift=max_energy_drift,
    )
