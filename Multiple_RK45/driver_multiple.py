# driver_multiple.py
"""
Driver module for Multiple (Gravity From the Ground Up).
Uses RK45 (solve_ivp) to integrate the N-body system.
"""

import numpy as np
from scipy.integrate import solve_ivp
from physics_multiple import accelerations, M_SUN


def run_simulation(masses_solar, positions_init, velocities_init,
                   t_max, dt_output):
    """
    Integrate the N-body system using RK45.

    Parameters
    ----------
    masses_solar : sequence of length N
        Mass of each body, in solar masses.
    positions_init : sequence of N [x, y, z] triples
        Initial position of each body (m).
    velocities_init : sequence of N [vx, vy, vz] triples
        Initial velocity of each body (m/s).
    t_max : total simulation time (seconds)
    dt_output : time spacing between output samples

    Returns
    -------
    times : array of output times
    trajectories : list of arrays, one per body:
        each array has shape (len(times), 3)
    """

    masses_solar = np.asarray(masses_solar, dtype=float)
    positions_init = np.asarray(positions_init, dtype=float)   # (n, 3)
    velocities_init = np.asarray(velocities_init, dtype=float)  # (n, 3)

    n = len(masses_solar)
    masses_kg = masses_solar * M_SUN

    # Build initial state vector: each body contributes
    # [x, y, z, vx, vy, vz], read straight off its own row.
    state0 = np.zeros(6 * n)
    for i in range(n):
        state0[6*i:6*i + 3] = positions_init[i]
        state0[6*i + 3:6*i + 6] = velocities_init[i]

    # Output times
    times = np.arange(0, t_max, dt_output)

    # Integrate
    sol = solve_ivp(
        fun=lambda t, y: accelerations(t, y, masses_kg),
        t_span=(0, t_max),
        y0=state0,
        t_eval=times,
        method="RK45",
        rtol=1e-9,
        atol=1e-9
    )

    # Extract trajectories
    trajectories = []
    for i in range(n):
        xyz = sol.y[6*i:6*i+3, :].T
        trajectories.append(xyz)

    return times, trajectories
