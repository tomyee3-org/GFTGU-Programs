"""
driver_cannon.py

Driver for the CannonTrajectory simulation.

The loop stops when:
    - The projectile returns to the ground (h < 0)
    - The step counter reaches 1000 (array size limit)
"""

import numpy as np
from physics_cannon import euler_step, improved_euler_step


def run_cannon_trajectory(speed=100.0, angle_deg=45.0, dt=0.1,
                          max_steps=100_000, method="improved"):
    """
    Run the cannonball trajectory simulation.

    Parameters:
        speed      — launch speed (m/s)
        angle_deg  — launch angle in degrees
        dt         — timestep (s)
        max_steps  — maximum number of steps (default 1000)
        method     — "euler" or "improved"

    Returns:
        xs, hs — arrays of horizontal and vertical positions
    """

    # Convert angle to radians (Java used Math.PI)
    theta = np.radians(angle_deg)

    # Compute velocity components (Java used Math.cos, Math.sin)
    u = speed * np.cos(theta)
    v = speed * np.sin(theta)

    # Initial positions
    x = 0.0
    h = 0.0

    # Arrays equivalent to horizontalDistance[] and verticalDistance[]
    xs = np.zeros(max_steps)
    hs = np.zeros(max_steps)

    xs[0] = x
    hs[0] = h

    # Choose numerical method
    stepper = euler_step if method == "euler" else improved_euler_step

    # State vector matches Java variables: x, h, u, v
    state = np.array([x, h, u, v])

    # Java loop: for (j = 1; (h >= 0.0) && (j < 1000); j++)
    j = 1
    while j < max_steps and state[1] >= 0.0:
        state = stepper(state, dt)
        xs[j] = state[0]
        hs[j] = state[1]
        j += 1

    # Trim unused entries
    return xs[:j], hs[:j]
