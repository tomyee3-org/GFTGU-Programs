"""
Physics and numerical integration routines for CannonTrajectory.

State vector:
    x  — horizontal position (m)
    h  — vertical position / height (m)
    u  — horizontal velocity (m/s)
    v  — vertical velocity (m/s)

Model:
    Horizontal acceleration is zero.
    Vertical acceleration is the constant -g.
"""

import numpy as np

g = 9.8  # Near-surface gravitational acceleration (m/s^2)


def derivs_cannon(state):
    """Return [dx/dt, dh/dt, du/dt, dv/dt] for the projectile state."""
    x, h, u, v = state
    return np.array([u, v, 0.0, -g], dtype=float)


def euler_step(state, dt):
    """Advance one timestep with the forward Euler method."""
    return state + dt * derivs_cannon(state)


def improved_euler_step(state, dt):
    """Advance one timestep with improved Euler (Heun's method)."""
    ds1 = derivs_cannon(state)
    predictor = state + dt * ds1
    ds2 = derivs_cannon(predictor)
    return state + 0.5 * dt * (ds1 + ds2)
