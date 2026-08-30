"""
Driver for CannonTrajectory.

The trajectory is sampled at fixed time intervals until the first stored point
below ground. The below-ground endpoint is intentionally retained so students
can investigate timestep error and interpolate the ground crossing themselves.
"""

import math
import numbers
import operator
import numpy as np

import physics_cannon as phys
from physics_cannon import euler_step, improved_euler_step


def version_info():
    """Return machine-readable model and source-build identifiers."""
    return {
        "model_version": phys.MODEL_VERSION,
        "build_id": phys.BUILD_ID,
    }


def _require_real(name, value):
    """Reject non-real settings before numerical operations begin."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_inputs(speed, angle_deg, dt, max_steps, method):
    _require_real("speed", speed)
    if speed <= 0.0:
        raise ValueError("speed must be a finite positive number")

    _require_real("angle_deg", angle_deg)
    if not 0.0 <= angle_deg <= 90.0:
        raise ValueError(
            "angle_deg must be between 0 and 90 degrees for a projectile "
            "launched from ground level"
        )

    _require_real("dt", dt)
    if dt <= 0.0:
        raise ValueError("dt must be a finite positive number")

    if isinstance(max_steps, bool):
        raise TypeError("max_steps must be an integer")
    try:
        max_steps = operator.index(max_steps)
    except TypeError as exc:
        raise TypeError("max_steps must be an integer") from exc
    if max_steps < 2:
        raise ValueError("max_steps must be an integer of at least 2")

    if not isinstance(method, str) or method not in {"euler", "improved"}:
        raise ValueError('method must be "euler" or "improved"')

    return max_steps


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
    max_steps = _validate_inputs(speed, angle_deg, dt, max_steps, method)

    theta = math.radians(angle_deg)
    u = speed * math.cos(theta)
    v = speed * math.sin(theta)
    state = np.array([0.0, 0.0, u, v], dtype=float)

    # Store only points actually calculated.  In Python, append-only lists
    # avoid allocating memory in proportion to an untrusted safety ceiling.
    xs = [state[0]]
    hs = [state[1]]

    stepper = {
        "euler": euler_step,
        "improved": improved_euler_step,
    }[method]

    try:
        with np.errstate(over="raise", invalid="raise"):
            while len(xs) < max_steps and state[1] >= 0.0:
                state = stepper(state, dt)
                if not np.all(np.isfinite(state)):
                    raise FloatingPointError
                xs.append(state[0])
                hs.append(state[1])
    except FloatingPointError as exc:
        raise FloatingPointError(
            "trajectory became non-finite; reduce speed or timestep"
        ) from exc

    if state[1] >= 0.0:
        raise RuntimeError(
            "max_steps was reached before the projectile landed; increase "
            "max_steps or use a larger timestep"
        )

    return np.asarray(xs, dtype=float), np.asarray(hs, dtype=float)
