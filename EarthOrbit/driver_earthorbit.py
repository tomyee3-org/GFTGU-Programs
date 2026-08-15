"""
driver_earthorbit.py

Driver for the EarthOrbit simulation.

In the integration loop employed here, acceleration is evaluated at the beginning of each
step, velocity is advanced with that acceleration, and position is advanced
using the average of the old and new velocities.

This is not velocity Verlet/leapfrog and is not generally symplectic.
"""

import numpy as np
from physics_earthorbit import compute_acceleration, R_EARTH, ForceLaw


def run_earth_orbit(
    h0=300.0,
    uInit=7900.0,
    vInit=0.0,
    dt=0.4,
    maxSteps=15000,
    force_law: ForceLaw = "simplified",
    return_diagnostics=False,
):
    """
    Integrate an EarthOrbit trajectory.

    Parameters
    ----------
    h0 : float
        Initial altitude above the reference Earth radius, in metres.
    uInit : float
        Initial x-velocity in m/s. Because the launch point is on the
        positive y-axis, this is the tangential/horizontal component.
    vInit : float
        Initial y-velocity in m/s. At launch this is the radial/vertical
        component.
    dt : float
        Fixed timestep in seconds.
    maxSteps : int
        Maximum number of stored trajectory points.
    force_law : {"simplified", "inverse_square"}
        Select the constant-magnitude-g model or the inverse-square
        extension.
    return_diagnostics : bool
        If False (default), preserve the original four-array return value.
        If True, also return time and velocity arrays for quantitative
        experiments.

    Returns
    -------
    xs, ys, xEarth, yEarth
        Returned when return_diagnostics is False.
    xs, ys, xEarth, yEarth, ts, us, vs
        Returned when return_diagnostics is True.
    """
    x = 0.0
    y = R_EARTH + h0

    u0 = uInit
    v0 = vInit

    xs = np.zeros(maxSteps)
    ys = np.zeros(maxSteps)
    xs[0] = x
    ys[0] = y

    if return_diagnostics:
        us = np.zeros(maxSteps)
        vs = np.zeros(maxSteps)
        us[0] = u0
        vs[0] = v0

    r = np.sqrt(x * x + y * y)

    j = 1
    while r >= R_EARTH and j < maxSteps:
        # The integration algorithm evaluates acceleration only at the
        # position at the beginning of the step.
        ax, ay = compute_acceleration(x, y, force_law)

        u1 = u0 + ax * dt
        v1 = v0 + ay * dt

        # Advance position using the average of old and new velocities.
        x = x + (u0 + u1) * 0.5 * dt
        y = y + (v0 + v1) * 0.5 * dt

        r = np.sqrt(x * x + y * y)

        xs[j] = x
        ys[j] = y

        if return_diagnostics:
            us[j] = u1
            vs[j] = v1

        u0 = u1
        v0 = v1
        j += 1

    xs = xs[:j]
    ys = ys[:j]

    # Build a closed Earth-surface reference curve. 
    angleStep = np.pi / 200
    xEarth = np.zeros(401)
    yEarth = np.zeros(401)

    for k in range(400):
        angle = angleStep * k
        xEarth[k] = R_EARTH * np.cos(angle)
        yEarth[k] = R_EARTH * np.sin(angle)

    xEarth[400] = xEarth[0]
    yEarth[400] = yEarth[0]

    if return_diagnostics:
        ts = np.arange(j, dtype=float) * dt
        return xs, ys, xEarth, yEarth, ts, us[:j], vs[:j]

    return xs, ys, xEarth, yEarth
