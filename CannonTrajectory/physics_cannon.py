"""
physics_cannon.py

Physics and numerical integration routines for CannonTrajectory.

State vector:
    x  — horizontal position
    h  — vertical position (height)
    u  — horizontal velocity
    v  — vertical velocity

The physics:
    Horizontal motion has no acceleration.
    Vertical motion has constant downward acceleration g.
"""

import numpy as np

# Acceleration of gravity (m/s^2)
g = 9.8


def derivs_cannon(state):
    """
    Compute time derivatives for projectile motion.

    Equivalent to the Java logic:
        dx/dt = u
        dh/dt = v
        du/dt = 0
        dv/dt = -g
    """
    x, h, u, v = state
    return np.array([u, v, 0.0, -g])


def euler_step(state, dt):
    """
    Basic Euler step.
    This matches the first numerical method introduced in
    Investigation 1.3 of Gravity From the Ground Up.
    """
    return state + dt * derivs_cannon(state)


def improved_euler_step(state, dt):
    """
    Improved Euler (Heun’s method).

    This mirrors Schutz’s Java implementation:
        - Compute w = v - g*dt
        - Use average vertical speed (v + w)/2 for height update
        - Use predictor–corrector logic for full state update

    In Python, we implement this using the standard Heun method,
    which is mathematically identical to Schutz’s step.
    """
    ds1 = derivs_cannon(state)
    predictor = state + dt * ds1
    ds2 = derivs_cannon(predictor)
    return state + 0.5 * dt * (ds1 + ds2)
