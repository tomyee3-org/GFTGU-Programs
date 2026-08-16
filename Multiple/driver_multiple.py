"""
Driver module for Multiple.

The ordinary "trajectories" output stores every accepted state. The
"animation" output instead samples the adaptive integration at uniformly spaced
SIMULATION times. This decouples animation cadence from the numerical timestep:
a close encounter may require many small integration steps without making the
movie run in artificial slow motion.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np

from physics_multiple import compute_accelerations


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
    if not isinstance(params.max_steps, int) or isinstance(params.max_steps, bool) or params.max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")
    if not np.isfinite(params.eps1) or not (0.0 < params.eps1 < 1.0):
        raise ValueError("eps1 must satisfy 0 < eps1 < 1.")
    if not np.isfinite(params.eps2) or not (0.0 < params.eps2 < 1.0):
        raise ValueError("eps2 must satisfy 0 < eps2 < 1.")

    if params.output_type.lower() not in ("trajectories", "animation"):
        raise ValueError('output_type must be "trajectories" or "animation".')
    if params.animation_mode.lower() not in ("current positions", "trails"):
        raise ValueError('animation_mode must be "current positions" or "trails".')
    if not np.isfinite(params.frame_time) or params.frame_time <= 0.0:
        raise ValueError("frame_time must be a positive finite number.")
    if not isinstance(params.frame_interval_ms, int) or isinstance(params.frame_interval_ms, bool) or params.frame_interval_ms <= 0:
        raise ValueError("frame_interval_ms must be a positive integer.")
    if not np.isfinite(params.trail_time) or params.trail_time < 0.0:
        raise ValueError("trail_time must be finite and non-negative.")
    if params.projection.lower() not in ("xy", "xz", "yz"):
        raise ValueError('projection must be "xy", "xz", or "yz".')
    if params.axis_mode.lower() not in ("fixed", "auto"):
        raise ValueError('axis_mode must be "fixed" or "auto".')


def _interpolate_positions(t0, p0, t1, p1, target_time):
    """Linearly interpolate all body positions to target_time."""
    if t1 <= t0:
        return p1.copy()
    alpha = (target_time - t0) / (t1 - t0)
    alpha = min(1.0, max(0.0, alpha))
    return p0 + alpha * (p1 - p0)


def run_simulation(params: SimulationParams) -> Dict[str, Any]:
    """
    Run the Multiple simulation.

    "trajectories" stores every accepted state.
    "animation" stores interpolated frames at t = 0, frame_time,
    2*frame_time, ... regardless of the adaptive integration timestep.
    """
    _validate_params(params)

    masses = np.array(params.masses_solar, dtype=float)
    positions = np.array(params.positions_init, dtype=float)
    velocities = np.array(params.velocities_init, dtype=float)
    dt = float(params.dt)
    time = 0.0
    output_type = params.output_type.lower()

    if output_type == "trajectories":
        output_times = [0.0]
        output_positions = [positions.copy()]
    else:
        frame_times = [0.0]
        frame_positions = [positions.copy()]
        next_frame_time = params.frame_time

    step = 0
    while step < params.max_steps:
        previous_time = time
        previous_positions = positions.copy()

        acc = compute_accelerations(positions, masses)
        pos_pred = positions + velocities * dt + 0.5 * acc * dt * dt
        vel_pred = velocities + acc * dt
        acc_pred = compute_accelerations(pos_pred, masses)

        delta_acc = np.linalg.norm(acc_pred - acc)
        norm_acc = max(np.linalg.norm(acc), np.linalg.norm(acc_pred))
        rel_change_acc = 0.0 if norm_acc == 0.0 else delta_acc / norm_acc
        if rel_change_acc > params.eps1:
            dt *= 0.5
            continue

        vel_corr = velocities + 0.5 * (acc + acc_pred) * dt
        pos_corr = positions + 0.5 * (velocities + vel_corr) * dt

        delta_pos = np.linalg.norm(pos_corr - pos_pred)
        norm_pos = max(np.linalg.norm(pos_corr), np.linalg.norm(pos_pred))
        rel_change_pos = 0.0 if norm_pos == 0.0 else delta_pos / norm_pos
        if rel_change_pos > params.eps2:
            dt *= 0.5
            continue

        positions = pos_corr
        velocities = vel_corr
        time += dt
        step += 1

        if output_type == "trajectories":
            output_times.append(time)
            output_positions.append(positions.copy())
        else:
            while next_frame_time <= time:
                frame_positions.append(_interpolate_positions(
                    previous_time, previous_positions,
                    time, positions,
                    next_frame_time,
                ))
                frame_times.append(next_frame_time)
                next_frame_time += params.frame_time

    if output_type == "trajectories":
        return {
            "type": "trajectories",
            "times": np.asarray(output_times),
            "positions": np.stack(output_positions, axis=0),
            "accepted_steps": step,
            "final_time": time,
        }

    return {
        "type": "animation",
        "frame_times": np.asarray(frame_times),
        "frame_positions": np.stack(frame_positions, axis=0),
        "accepted_steps": step,
        "final_time": time,
        "animation_mode": params.animation_mode.lower(),
        "frame_time": params.frame_time,
        "frame_interval_ms": params.frame_interval_ms,
        "trail_time": params.trail_time,
        "projection": params.projection.lower(),
        "axis_mode": params.axis_mode.lower(),
    }
