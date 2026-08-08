"""
Driver module for Multiple

Implements time-step halving and a simple predictor-corrector,
"""

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np

from physics_multiple import compute_accelerations


@dataclass
class SimulationParams:
    n_bodies: int
    masses_solar: List[float]          # masses in solar masses
    positions_init: List[List[float]]  # list of [x, y, z] in meters
    velocities_init: List[List[float]] # list of [vx, vy, vz] in m/s
    dt: float                          # initial time-step (s)
    max_steps: int
    out_steps: int                     # steps between output events
    output_type: str                   # "trajectories" or "current positions"
    eps1: float                        # accuracy for time-step halving
    eps2: float                        # accuracy for predictor-corrector


def run_simulation(params: SimulationParams) -> Dict[str, Any]:
    """
    Run the Multiple simulation with Schutz-style adaptive dt and predictor-corrector.

    Returns a structured result suitable for plotting.
    """
    n = params.n_bodies
    masses = np.array(params.masses_solar, dtype=float)
    positions = np.array(params.positions_init, dtype=float)  # (n, 3)
    velocities = np.array(params.velocities_init, dtype=float)

    dt = params.dt
    max_steps = params.max_steps
    out_steps = params.out_steps
    output_type = params.output_type.lower()
    eps1 = params.eps1
    eps2 = params.eps2

    time = 0.0
    steps_since_output = 0

    # Storage
    if output_type == "trajectories":
        traj_times: List[float] = [time]
        traj_positions: List[np.ndarray] = [positions.copy()]
    else:
        snapshots: List[Dict[str, Any]] = []

    # Main time loop
    step = 0
    while step < max_steps:
        # Compute accelerations at current positions
        acc = compute_accelerations(positions, masses)

        # Predictor step
        pos_pred = positions + velocities * dt + 0.5 * acc * dt * dt
        vel_pred = velocities + acc * dt

        # New accelerations at predicted positions
        acc_pred = compute_accelerations(pos_pred, masses)

        # Time-step halving criterion (relative change in acceleration)
        delta_acc = np.linalg.norm(acc_pred - acc)
        norm_acc = np.linalg.norm(acc) if np.linalg.norm(acc) != 0.0 else 1.0
        rel_change_acc = delta_acc / norm_acc

        if rel_change_acc > eps1:
            # Cut time-step in half and retry this step
            dt *= 0.5
            continue

        # Predictor-corrector: average accelerations
        vel_corr = velocities + 0.5 * (acc + acc_pred) * dt
        pos_corr = positions + 0.5 * (velocities + vel_corr) * dt

        # Simple convergence check (one iteration, as in Schutz's spirit)
        delta_pos = np.linalg.norm(pos_corr - pos_pred)
        norm_pos = np.linalg.norm(pos_pred) if np.linalg.norm(pos_pred) != 0.0 else 1.0
        rel_change_pos = delta_pos / norm_pos

        if rel_change_pos > eps2:
            # If not converged, reduce dt and retry
            dt *= 0.5
            continue

        # Accept corrected step
        positions = pos_corr
        velocities = vel_corr
        time += dt
        step += 1
        steps_since_output += 1

        # Output handling
        if output_type == "trajectories":
            traj_times.append(time)
            traj_positions.append(positions.copy())
        else:
            if steps_since_output >= out_steps:
                snapshots.append(
                    {
                        "time": time,
                        "positions": positions.copy(),
                    }
                )
                steps_since_output = 0

        # Reset output counter if we never output in trajectories mode
        if output_type == "trajectories" and steps_since_output >= out_steps:
            steps_since_output = 0

    if output_type == "trajectories":
        return {
            "type": "trajectories",
            "times": np.array(traj_times),
            "positions": np.stack(traj_positions, axis=0),  # (n_steps+1, n_bodies, 3)
        }
    else:
        return {
            "type": "current positions",
            "snapshots": snapshots,
        }
