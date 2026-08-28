"""
Integration driver for Multiple.

The numerical integration uses an adaptive predictor plus iterated trapezoidal
corrector. Animation frames are sampled at uniform PHYSICAL simulation times,
so adaptive timestep changes do not distort movie playback speed.
"""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import List, Dict, Any, Optional

import numpy as np

import physics_multiple as phys
from physics_multiple import compute_accelerations, conservation_state


MAX_ANIMATION_FRAMES = 1_000_000
ENERGY_CANCELLATION_TOLERANCE = 128.0 * np.finfo(float).eps


@dataclass
class SimulationParams:
    n_bodies: int
    masses_solar: List[float]
    positions_init: List[List[float]]
    velocities_init: List[List[float]]

    dt: float
    max_steps: int
    output_type: str                   # "trajectories" or "animation"
    eps1: float
    eps2: float

    animation_mode: str = "trails"             # or "current positions"
    frame_time: float = 2.0e5          # simulated seconds between frames
    frame_interval_ms: int = 50        # real milliseconds between frames
    trail_time: float = 6.0e5          # simulated seconds shown behind each body
    projection: str = "xy"             # "xy", "xz", or "yz"
    axis_mode: str = "fixed"           # "fixed" or "auto"


def _validate_params(params: SimulationParams) -> None:
    if not isinstance(params.n_bodies, Integral) or isinstance(params.n_bodies, bool):
        raise ValueError("n_bodies must be an integer.")
    if params.n_bodies < 2:
        raise ValueError("n_bodies must be at least 2.")

    try:
        masses = np.asarray(params.masses_solar, dtype=float)
        positions = np.asarray(params.positions_init, dtype=float)
        velocities = np.asarray(params.velocities_init, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Masses, positions, and velocities must contain numeric values."
        ) from exc

    if masses.shape != (params.n_bodies,):
        raise ValueError("masses_solar must contain exactly n_bodies values.")
    if positions.shape != (params.n_bodies, 3):
        raise ValueError(
            "positions_init must contain exactly n_bodies three-component vectors."
        )
    if velocities.shape != (params.n_bodies, 3):
        raise ValueError(
            "velocities_init must contain exactly n_bodies three-component vectors."
        )

    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("All masses must be finite and positive.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("All initial positions must be finite.")
    if not np.all(np.isfinite(velocities)):
        raise ValueError("All initial velocities must be finite.")

    if (
        not isinstance(params.dt, Real)
        or isinstance(params.dt, bool)
        or not np.isfinite(params.dt)
        or params.dt <= 0.0
    ):
        raise ValueError("dt must be a positive finite number.")
    if (
        not isinstance(params.max_steps, Integral)
        or isinstance(params.max_steps, bool)
        or params.max_steps <= 0
    ):
        raise ValueError("max_steps must be a positive integer.")

    if (
        not isinstance(params.eps1, Real)
        or isinstance(params.eps1, bool)
        or not np.isfinite(params.eps1)
        or not (0.0 < params.eps1 < 1.0)
    ):
        raise ValueError("eps1 must satisfy 0 < eps1 < 1.")
    if (
        not isinstance(params.eps2, Real)
        or isinstance(params.eps2, bool)
        or not np.isfinite(params.eps2)
        or not (0.0 < params.eps2 < params.eps1)
    ):
        raise ValueError("eps2 must satisfy 0 < eps2 < eps1.")

    if not isinstance(params.output_type, str):
        raise ValueError('output_type must be "trajectories" or "animation".')
    if params.output_type.lower() not in ("trajectories", "animation"):
        raise ValueError('output_type must be "trajectories" or "animation".')

    # Projection applies to both output modes.
    if not isinstance(params.projection, str):
        raise ValueError('projection must be "xy", "xz", or "yz".')
    if params.projection.lower() not in ("xy", "xz", "yz"):
        raise ValueError('projection must be "xy", "xz", or "yz".')

    # The remaining display controls matter only for animation.
    if params.output_type.lower() == "animation":
        if not isinstance(params.animation_mode, str):
            raise ValueError(
                'animation_mode must be "current positions" or "trails".'
            )
        if params.animation_mode.lower() not in ("current positions", "trails"):
            raise ValueError(
                'animation_mode must be "current positions" or "trails".'
            )
        if (
            not isinstance(params.frame_time, Real)
            or isinstance(params.frame_time, bool)
            or not np.isfinite(params.frame_time)
            or params.frame_time <= 0.0
        ):
            raise ValueError("frame_time must be a positive finite number.")
        if (
            not isinstance(params.frame_interval_ms, Integral)
            or isinstance(params.frame_interval_ms, bool)
            or params.frame_interval_ms <= 0
        ):
            raise ValueError("frame_interval_ms must be a positive integer.")
        if (
            not isinstance(params.trail_time, Real)
            or isinstance(params.trail_time, bool)
            or not np.isfinite(params.trail_time)
            or params.trail_time < 0.0
        ):
            raise ValueError("trail_time must be finite and non-negative.")
        if not isinstance(params.axis_mode, str):
            raise ValueError('axis_mode must be "fixed" or "auto".')
        if params.axis_mode.lower() not in ("fixed", "auto"):
            raise ValueError('axis_mode must be "fixed" or "auto".')

        maximum_duration = float(params.dt) * int(params.max_steps)
        maximum_frames = maximum_duration / float(params.frame_time) + 1.0
        if not np.isfinite(maximum_frames) or maximum_frames > MAX_ANIMATION_FRAMES:
            raise ValueError(
                "The requested animation could require more than "
                f"{MAX_ANIMATION_FRAMES:,} stored frames. Increase frame_time "
                "or reduce dt or max_steps."
            )


def _max_relative_vector_change(old: np.ndarray, new: np.ndarray) -> float:
    """
    Maximum relative change among body vectors.

    old and new have shape (n_bodies, 3). Each body is tested independently so
    a close encounter involving one body cannot be diluted by quiet bodies.
    """
    if not (np.all(np.isfinite(old)) and np.all(np.isfinite(new))):
        return float("inf")

    changes = np.hypot.reduce(new - old, axis=1)
    scales = np.maximum(
        np.hypot.reduce(old, axis=1),
        np.hypot.reduce(new, axis=1),
    )

    ratios = np.zeros_like(changes)
    nonzero = scales > 0.0
    ratios[nonzero] = changes[nonzero] / scales[nonzero]
    ratios[(~nonzero) & (changes > 0.0)] = np.inf
    return float(np.max(ratios))


def _hermite_state(
    t0: float,
    p0: np.ndarray,
    v0: np.ndarray,
    t1: float,
    p1: np.ndarray,
    v1: np.ndarray,
    target_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cubic-Hermite dense interpolation between two accepted states.

    Endpoint positions and velocities are matched exactly. This is used only for
    display sampling, not to advance the physical integration.
    """
    h = t1 - t0
    if h <= 0.0:
        return p1.copy(), v1.copy()

    s = (target_time - t0) / h
    s = min(1.0, max(0.0, s))

    h00 = 2*s**3 - 3*s**2 + 1
    h10 = s**3 - 2*s**2 + s
    h01 = -2*s**3 + 3*s**2
    h11 = s**3 - s**2

    pos = h00*p0 + h10*h*v0 + h01*p1 + h11*h*v1

    dh00 = (6*s**2 - 6*s) / h
    dh10 = 3*s**2 - 4*s + 1
    dh01 = (-6*s**2 + 6*s) / h
    dh11 = 3*s**2 - 2*s

    vel = dh00*p0 + dh10*v0 + dh01*p1 + dh11*v1
    return pos, vel


def _fractional_scalar_drift(
    value: float,
    reference: float,
    scale: Optional[float] = None,
) -> float:
    if scale is None:
        scale = abs(reference)
    if scale == 0.0:
        return abs(value - reference)
    return abs(value - reference) / scale


def _energy_drift_scale(
    kinetic: float,
    potential: float,
) -> tuple[float, str]:
    """Choose a stable, dimensionally consistent energy-error denominator."""
    characteristic = kinetic + abs(potential)
    if not np.isfinite(characteristic) or characteristic <= 0.0:
        raise ValueError(
            "The characteristic energy scale is outside the floating-point "
            "range."
        )

    total = kinetic + potential
    if abs(total) <= ENERGY_CANCELLATION_TOLERANCE * characteristic:
        return characteristic, "characteristic_energy"
    return abs(total), "initial_energy"


def _vector_drift(value: np.ndarray, reference: np.ndarray) -> float:
    if not (
        np.all(np.isfinite(value))
        and np.all(np.isfinite(reference))
    ):
        raise ValueError("Conservation vectors must contain finite values.")

    with np.errstate(over="ignore", invalid="ignore"):
        difference = value - reference
    if not np.all(np.isfinite(difference)):
        raise ValueError(
            "Conservation-vector drift is outside the floating-point range."
        )

    scale = float(np.hypot.reduce(reference))
    diff = float(np.hypot.reduce(difference))
    if not (np.isfinite(scale) and np.isfinite(diff)):
        raise ValueError(
            "Conservation-vector norm is outside the floating-point range."
        )
    if scale == 0.0:
        return diff
    drift = diff / scale
    if not np.isfinite(drift):
        raise ValueError(
            "Conservation-vector drift is outside the floating-point range."
        )
    return drift


def _vector_drift_metadata(reference: np.ndarray) -> tuple[Optional[float], str]:
    """Return the vector-drift scale and its public normalization label."""
    if not np.all(np.isfinite(reference)):
        raise ValueError("Conservation vectors must contain finite values.")
    scale = float(np.hypot.reduce(reference))
    if not np.isfinite(scale):
        raise ValueError(
            "Conservation-vector norm is outside the floating-point range."
        )
    if scale == 0.0:
        return None, "absolute_scaled"
    return scale, "initial_norm"


def _checked_maximum_drift(
    current_maximum: float,
    candidate: float,
    quantity: str,
) -> float:
    """Update a maximum without silently swallowing a NaN diagnostic."""
    if not np.isfinite(candidate):
        raise RuntimeError(f"Non-finite {quantity} drift was produced.")
    return max(current_maximum, candidate)


def run_simulation(params: SimulationParams) -> Dict[str, Any]:
    """
    Run the Multiple simulation.

    "trajectories" stores every accepted position and velocity state.

    "animation" stores cubic-Hermite-interpolated frames at uniformly spaced
    physical times t = 0, frame_time, 2*frame_time, ... independent of the
    adaptive integration timestep.
    """
    _validate_params(params)

    masses = np.asarray(params.masses_solar, dtype=float)
    positions = np.asarray(params.positions_init, dtype=float)
    velocities = np.asarray(params.velocities_init, dtype=float)

    dt_work = float(params.dt)
    time = 0.0
    output_type = params.output_type.lower()

    max_corrector_iterations = 10
    max_retries_per_step = 80

    initial_cons = conservation_state(positions, velocities, masses)
    energy_drift_scale, energy_drift_normalization = _energy_drift_scale(
        initial_cons["kinetic_energy"],
        initial_cons["potential_energy"],
    )
    momentum_drift_scale, momentum_drift_normalization = (
        _vector_drift_metadata(initial_cons["momentum"])
    )
    angular_drift_scale, angular_drift_normalization = (
        _vector_drift_metadata(initial_cons["angular_momentum"])
    )
    max_energy_drift = 0.0
    max_momentum_drift = 0.0
    max_angular_momentum_drift = 0.0

    if output_type == "trajectories":
        output_times = [0.0]
        output_positions = [positions.copy()]
        output_velocities = [velocities.copy()]
        energies = [initial_cons["energy"]]
        momenta = [initial_cons["momentum"].copy()]
        angular_momenta = [initial_cons["angular_momentum"].copy()]
        dt_used = [0.0]
    else:
        frame_times = [0.0]
        frame_positions = [positions.copy()]
        frame_velocities = [velocities.copy()]
        next_frame_index = 1
        next_frame_time = next_frame_index * params.frame_time

    accepted_steps = 0

    while accepted_steps < params.max_steps:
        acc0 = compute_accelerations(positions, masses)
        accepted = False

        for _retry in range(max_retries_per_step):
            # Predictor.
            pos_pred = (
                positions
                + velocities * dt_work
                + 0.5 * acc0 * dt_work * dt_work
            )
            vel_pred = velocities + acc0 * dt_work

            if not (
                np.all(np.isfinite(pos_pred))
                and np.all(np.isfinite(vel_pred))
            ):
                dt_work *= 0.5
                continue

            acc_pred = compute_accelerations(pos_pred, masses)
            if not np.all(np.isfinite(acc_pred)):
                dt_work *= 0.5
                continue

            # eps1: test each body's acceleration vector independently.
            if _max_relative_vector_change(acc0, acc_pred) > params.eps1:
                dt_work *= 0.5
                continue

            # Iterated trapezoidal corrector.
            pos_guess = pos_pred
            vel_guess = vel_pred
            acc_end = acc_pred
            converged = False

            for _ in range(max_corrector_iterations):
                vel_corr = velocities + 0.5 * (acc0 + acc_end) * dt_work
                pos_corr = positions + 0.5 * (velocities + vel_corr) * dt_work

                if not (
                    np.all(np.isfinite(pos_corr))
                    and np.all(np.isfinite(vel_corr))
                ):
                    break

                # Compare velocity *increments*, not absolute coordinate
                # velocities. This makes the convergence decision invariant
                # under a uniform Galilean boost of the entire system.
                velocity_change = _max_relative_vector_change(
                    vel_guess - velocities,
                    vel_corr - velocities,
                )

                pos_guess = pos_corr
                vel_guess = vel_corr

                if velocity_change < params.eps2:
                    # acc_end belongs to the position estimate used to form
                    # pos_corr, not to the newly accepted pos_corr itself. It
                    # therefore cannot safely be cached as the next step's
                    # starting acceleration without another force evaluation.
                    converged = True
                    break

                acc_end = compute_accelerations(pos_guess, masses)
                if not np.all(np.isfinite(acc_end)):
                    break

            if not converged:
                dt_work *= 0.5
                continue

            accepted = True
            break

        if not accepted:
            raise RuntimeError(
                "Multiple could not find a converged timestep after "
                f"{max_retries_per_step} retries. A near-collision or extreme "
                "initial condition may require different parameters."
            )

        previous_time = time
        previous_positions = positions.copy()
        previous_velocities = velocities.copy()

        new_time = time + dt_work
        if new_time <= time:
            raise RuntimeError(
                "The adaptive timestep became too small to advance simulation "
                "time in floating-point arithmetic."
            )
        time = new_time
        positions = pos_guess
        velocities = vel_guess
        accepted_steps += 1

        if not (
            np.all(np.isfinite(positions))
            and np.all(np.isfinite(velocities))
        ):
            raise RuntimeError(
                "Multiple produced a non-finite position or velocity. "
                "Try a smaller dt or less extreme initial conditions."
            )

        current_cons = conservation_state(positions, velocities, masses)
        if not (
            np.isfinite(current_cons["energy"])
            and np.all(np.isfinite(current_cons["momentum"]))
            and np.all(np.isfinite(current_cons["angular_momentum"]))
        ):
            raise RuntimeError(
                "Multiple produced a non-finite conservation diagnostic. "
                "Try a smaller dt or less extreme initial conditions."
            )

        max_energy_drift = _checked_maximum_drift(
            max_energy_drift,
            _fractional_scalar_drift(
                current_cons["energy"],
                initial_cons["energy"],
                energy_drift_scale,
            ),
            "energy",
        )
        max_momentum_drift = _checked_maximum_drift(
            max_momentum_drift,
            _vector_drift(
                current_cons["momentum"], initial_cons["momentum"]
            ),
            "momentum",
        )
        max_angular_momentum_drift = _checked_maximum_drift(
            max_angular_momentum_drift,
            _vector_drift(
                current_cons["angular_momentum"],
                initial_cons["angular_momentum"],
            ),
            "angular-momentum",
        )

        if output_type == "trajectories":
            output_times.append(time)
            output_positions.append(positions.copy())
            output_velocities.append(velocities.copy())
            energies.append(current_cons["energy"])
            momenta.append(current_cons["momentum"].copy())
            angular_momenta.append(
                current_cons["angular_momentum"].copy()
            )
            dt_used.append(dt_work)
        else:
            # One accepted step can cross multiple requested frame times.
            while next_frame_time <= time:
                if not previous_time <= next_frame_time <= time:
                    raise RuntimeError(
                        "An animation frame time fell outside its accepted-step "
                        "interpolation bracket."
                    )
                p_frame, v_frame = _hermite_state(
                    previous_time,
                    previous_positions,
                    previous_velocities,
                    time,
                    positions,
                    velocities,
                    next_frame_time,
                )
                frame_positions.append(p_frame)
                frame_velocities.append(v_frame)
                frame_times.append(next_frame_time)
                if len(frame_times) > MAX_ANIMATION_FRAMES:
                    raise RuntimeError(
                        "The animation exceeded the stored-frame safety limit. "
                        "Increase frame_time or reduce the simulated duration."
                    )
                next_frame_index += 1
                next_frame_time = next_frame_index * params.frame_time

        # Gradually recover after close encounters, never exceeding user dt.
        dt_work = min(dt_work * 1.1, params.dt)

    common = {
        "model_version": phys.MODEL_VERSION,
        "build_id": phys.BUILD_ID,
        "accepted_steps": accepted_steps,
        "final_time": time,
        "initial_conservation": initial_cons,
        "final_conservation": current_cons,
        "energy_drift_scale": energy_drift_scale,
        "energy_drift_normalization": energy_drift_normalization,
        "momentum_drift_scale": momentum_drift_scale,
        "momentum_drift_normalization": momentum_drift_normalization,
        "angular_momentum_drift_scale": angular_drift_scale,
        "angular_momentum_drift_normalization": angular_drift_normalization,
        "max_fractional_energy_drift": max_energy_drift,
        "max_momentum_drift": max_momentum_drift,
        "max_angular_momentum_drift": max_angular_momentum_drift,
        # Backward-compatible aliases retained for existing callers. Consult
        # the normalization metadata above before interpreting these values.
        "max_fractional_momentum_drift": max_momentum_drift,
        "max_fractional_angular_momentum_drift": max_angular_momentum_drift,
    }

    if output_type == "trajectories":
        return {
            "type": "trajectories",
            "times": np.asarray(output_times),
            "positions": np.stack(output_positions, axis=0),
            "velocities": np.stack(output_velocities, axis=0),
            "energies": np.asarray(energies),
            "momenta": np.stack(momenta, axis=0),
            "angular_momenta": np.stack(angular_momenta, axis=0),
            "dt_used": np.asarray(dt_used),
            **common,
        }

    return {
        "type": "animation",
        "frame_times": np.asarray(frame_times),
        "frame_positions": np.stack(frame_positions, axis=0),
        "frame_velocities": np.stack(frame_velocities, axis=0),
        "animation_mode": params.animation_mode.lower(),
        "frame_time": params.frame_time,
        "frame_interval_ms": params.frame_interval_ms,
        "trail_time": params.trail_time,
        "projection": params.projection.lower(),
        "axis_mode": params.axis_mode.lower(),
        **common,
    }
