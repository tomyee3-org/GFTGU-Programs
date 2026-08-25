"""
Driver for CannonTrajectory.

The trajectory is sampled at fixed time intervals until the first stored point
below ground. The below-ground endpoint is intentionally retained so students
can investigate timestep error and interpolate the ground crossing themselves.
"""

import math
import numpy as np

import physics_cannon as phys
from physics_cannon import euler_step, improved_euler_step


def version_info():
    """Return machine-readable model and source-build identifiers."""
    return {
        "model_version": phys.MODEL_VERSION,
        "build_id": phys.BUILD_ID,
    }


def _validate_inputs(speed, angle_deg, dt, max_steps, method):
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("speed must be a finite positive number")

    if not math.isfinite(angle_deg):
        raise ValueError("angle_deg must be finite")
    if not 0.0 <= angle_deg <= 90.0:
        raise ValueError(
            "angle_deg must be between 0 and 90 degrees for a projectile "
            "launched from ground level"
        )

    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive number")

    if (
        not isinstance(max_steps, int)
        or isinstance(max_steps, bool)
        or max_steps < 2
    ):
        raise ValueError("max_steps must be an integer of at least 2")

    if method not in {"euler", "improved"}:
        raise ValueError('method must be "euler" or "improved"')


def run_cannon_trajectory(
    speed=100.0,
    angle_deg=45.0,
    dt=0.1,
    max_steps=100_000,
    method="improved",
):
    """
    Run the projectile simulation.

    max_steps counts stored points, including the initial point. Normal output
    includes the initial point and the first sampled point below ground.

    Raises ValueError for invalid input and RuntimeError if max_steps is
    exhausted before a below-ground sample is reached.
    """
    _validate_inputs(speed, angle_deg, dt, max_steps, method)

    theta = np.radians(angle_deg)
    u = speed * np.cos(theta)
    v = speed * np.sin(theta)
    state = np.array([0.0, 0.0, u, v], dtype=float)

    xs = np.zeros(max_steps, dtype=float)
    hs = np.zeros(max_steps, dtype=float)
    xs[0] = state[0]
    hs[0] = state[1]

    stepper = {
        "euler": euler_step,
        "improved": improved_euler_step,
    }[method]

    j = 1
    while j < max_steps and state[1] >= 0.0:
        state = stepper(state, dt)
        xs[j] = state[0]
        hs[j] = state[1]
        j += 1

    if state[1] >= 0.0:
        raise RuntimeError(
            "max_steps was reached before the projectile landed; increase "
            "max_steps or use a larger timestep"
        )

    return xs[:j], hs[:j]
