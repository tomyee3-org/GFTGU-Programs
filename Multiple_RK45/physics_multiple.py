"""
Physics module for Multiple_RK45
Encodes the gravitational physics

All literal numbers here represent physical constants only.
"""

import numpy as np

# Newton's gravitational constant (SI units)
G = 6.6726e-11

# Mass of the Sun (kg), for converting masses given in solar masses
M_SUN = 1.989e30


def accelerations(t, state, masses):
    """
    Compute accelerations for all bodies under mutual Newtonian gravity.

    Parameters
    ----------
    t : float
        Time (not used explicitly, but required by solve_ivp).
    state : ndarray
        State vector of length 6*N:
            [x1, y1, z1, v1x, v1y, v1z, x2, y2, z2, v2x, ...]
    masses : ndarray
        Array of masses for all bodies (in kg).

    Returns
    -------
    derivs : ndarray
        Time derivatives of the state vector:
            dx/dt = vx
            dy/dt = vy
            dz/dt = vz
            dv/dt = ax, ay, az
    """

    n = len(masses)
    derivs = np.zeros_like(state)

    # Extract positions and velocities
    positions = state.reshape(n, 6)[:, 0:3]
    velocities = state.reshape(n, 6)[:, 3:6]

    # First fill in dx/dt = vx, etc.
    derivs.reshape(n, 6)[:, 0:3] = velocities

    # Compute accelerations
    acc = np.zeros((n, 3))

    for A in range(n):
        for B in range(A + 1, n):
            dx = positions[A] - positions[B]
            r = np.linalg.norm(dx)
            r3 = r**3

            # Newtonian pairwise acceleration
            aA = -G * masses[B] * dx / r3
            aB = +G * masses[A] * dx / r3

            acc[A] += aA
            acc[B] += aB

    # Fill in dv/dt = ax, ay, az
    derivs.reshape(n, 6)[:, 3:6] = acc

    return derivs
