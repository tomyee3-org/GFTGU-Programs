"""
Driver module for RK45 integration of binary orbits.
"""

import numpy as np
from scipy.integrate import solve_ivp
from physics_binary import acceleration_binary

def run_binary_orbit(
    MA, MB,
    xInitA, yInitA, xInitB, yInitB,
    vInitA, uInitA, vInitB, uInitB,
    t_max=5e7,
    rtol=1e-9,
    atol=1e-9,
    max_step=np.inf
):
    """
    Integrates the binary system using RK45.
    Returns a dictionary containing the full trajectory.
    """

    # Initial state vector
    y0 = np.array([
        xInitA, yInitA,
        xInitB, yInitB,
        vInitA, uInitA,
        vInitB, uInitB
    ])

    # Wrapper for solve_ivp
    def rhs(t, state):
        return acceleration_binary(t, state, MA, MB)

    sol = solve_ivp(
        rhs,
        t_span=(0, t_max),
        y0=y0,
        method="RK45",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=True
    )

    return {
        "t": sol.t,
        "xA": sol.y[0],
        "yA": sol.y[1],
        "xB": sol.y[2],
        "yB": sol.y[3],
        "vA": sol.y[4],
        "uA": sol.y[5],
        "vB": sol.y[6],
        "uB": sol.y[7],
        "sol": sol
    }
