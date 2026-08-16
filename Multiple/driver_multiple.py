"""
Integration driver for Multiple.

The numerical integration uses an adaptive predictor plus iterated trapezoidal
corrector. Animation frames are sampled at uniform PHYSICAL simulation times,
so adaptive timestep changes do not distort movie playback speed.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np

from physics_multiple import compute_accelerations, conservation_state


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

    animation_mode: str = "current positions"  # or "trails"
    frame_time: float = 2.0e5          # simulated seconds between frames
    frame_interval_ms: int = 50        # real milliseconds between frames
    trail_time: float = 6.0e5          # simulated seconds shown behind each body
    projection: str = "xy"             # "xy", "xz", or "yz"
    axis_mode: str = "fixed"           # "fixed" or "auto"


def _validate_params(params: SimulationParams) -> None:
    if not isinstance(params.n_bodies, int) or isinstance(params.n_bodies, bool):
        raise ValueError("n_bodies must be an integer.")
    if params.n_bodies < 2:
        raise ValueError("n_bodies must be at least 2.")

    if len(params.masses_solar) != params.n_bodies:
        raise ValueError("masses_solar must contain exactly n_bodies values.")
    if len(params.positions_init) != params.n_bodies:
        raise ValueError("positions_init must contain exactly n_bodies vectors.")
    if len(params.velocities_init) != params.n_bodies:
        raise ValueError("velocities_init must contain exactly n_bodies vectors.")
    if any(len(v) != 3 for v in params.positions_init):
        raise ValueError("Each initial position must contain exactly 3 values.")
    if any(len(v) != 3 for v in params.velocities_init):
        raise ValueError("Each initial velocity must contain exactly 3 values.")

    masses = np.asarray(params.masses_solar, dtype=float)
    positions = np.asarray(params.positions_init, dtype=float)
    velocities = np.asarray(params.velocities_init, dtype=float)

    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("All masses must be finite and positive.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("All initial positions must be finite.")
    if not np.all(np.isfinite(velocities)):
        raise ValueError("All initial velocities must be finite.")

    if not np.isfinite(params.dt) or params.dt <= 0.0:
        raise ValueError("dt must be a positive finite number.")
    if (
        not isinstance(params.max_steps, int)
        or isinstance(params.max_steps, bool)
        or params.max_steps <= 0
    ):
        raise ValueError("max_steps must be a positive integer.")

    if not np.isfinite(params.eps1) or not (0.0 < params.eps1 < 1.0):
        raise ValueError("eps1 must satisfy 0 < eps1 < 1.")
    if not np.isfinite(params.eps2) or not (0.0 < params.eps2 < params.eps1):
        raise ValueError("eps2 must satisfy 0 < eps2 < eps1.")

    if params.output_type.lower() not in ("trajectories", "animation"):
        raise ValueError('output_type must be "trajectories" or "animation".')

    # Projection applies to both output modes.
    if params.projection.lower() not in ("xy", "xz", "yz"):
        raise ValueError('projection must be "xy", "xz", or "yz".')

    # The remaining display controls matter only for animation.
    if params.output_type.lower() == "animation":
        if params.animation_mode.lower() not in ("current positions", "trails"):
            raise ValueError(
                'animation_mode must be "current positions" or "trails".'
            )
        if not np.isfinite(params.frame_time) or params.frame_time <= 0.0:
            raise ValueError("frame_time must be a positive finite number.")
        if (
            not isinstance(params.frame_interval_ms, int)
            or isinstance(params.frame_interval_ms, bool)
            or params.frame_interval_ms <= 0
        ):
            raise ValueError("frame_interval_ms must be a positive integer.")
        if not np.isfinite(params.trail_time) or params.trail_time < 0.0:
            raise ValueError("trail_time must be finite and non-negative.")
        if params.axis_mode.lower() not in ("fixed", "auto"):
            raise ValueError('axis_mode must be "fixed" or "auto".')


def _max_relative_vector_change(old: np.ndarray, new: np.ndarray) -> float:
    """
    Maximum relative change among body vectors.

    old and new have shape (n_bodies, 3). Each body is tested independently so
    a close encounter involving one body cannot be diluted by quiet bodies.
    """
    changes = np.linalg.norm(new - old, axis=1)
    scales = np.maximum(
        np.linalg.norm(old, axis=1),
        np.linalg.norm(new, axis=1),
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


def _fractional_scalar_drift(value: float, reference: float) -> float:
    scale = abs(reference)
    if scale == 0.0:
        return abs(value - reference)
    return abs(value - reference) / scale


def _vector_drift(value: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.linalg.norm(reference))
    diff = float(np.linalg.norm(value - reference))
    if scale == 0.0:
        return diff
    return diff / scale


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
        next_frame_time = params.frame_time

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
            acc_pred = compute_accelerations(pos_pred, masses)

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

                velocity_change = _max_relative_vector_change(
                    vel_guess, vel_corr
                )

                pos_guess = pos_corr
                vel_guess = vel_corr

                if velocity_change < params.eps2:
                    converged = True
                    break

                acc_end = compute_accelerations(pos_guess, masses)

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

        time += dt_work
        positions = pos_guess
        velocities = vel_guess
        accepted_steps += 1

        current_cons = conservation_state(positions, velocities, masses)
        max_energy_drift = max(
            max_energy_drift,
            _fractional_scalar_drift(
                current_cons["energy"], initial_cons["energy"]
            ),
        )
        max_momentum_drift = max(
            max_momentum_drift,
            _vector_drift(
                current_cons["momentum"], initial_cons["momentum"]
            ),
        )
        max_angular_momentum_drift = max(
            max_angular_momentum_drift,
            _vector_drift(
                current_cons["angular_momentum"],
                initial_cons["angular_momentum"],
            ),
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
                next_frame_time += params.frame_time

        # Gradually recover after close encounters, never exceeding user dt.
        dt_work = min(dt_work * 1.1, params.dt)

    common = {
        "accepted_steps": accepted_steps,
        "final_time": time,
        "initial_conservation": initial_cons,
        "final_conservation": current_cons,
        "max_fractional_energy_drift": max_energy_drift,
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
