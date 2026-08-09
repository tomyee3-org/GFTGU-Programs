"""
driver_earthorbit.py

Driver for the EarthOrbit simulation.
"""

import numpy as np
from physics_earthorbit import compute_acceleration, R_EARTH, ForceLaw   # fix 1


def run_earth_orbit(h0=300.0, uInit=0.0, vInit=7900.0,
                    dt=0.4, maxSteps=15000,
                    force_law: ForceLaw = "simplified"):
    # Initial position: on the y-axis at R_EARTH + h0
    x = 0.0
    y = R_EARTH + h0          # fix 2

    u0 = uInit
    v0 = vInit

    xs = np.zeros(maxSteps)
    ys = np.zeros(maxSteps)
    xs[0] = x
    ys[0] = y

    r = np.sqrt(x*x + y*y)

    j = 1
    while r >= R_EARTH and j < maxSteps:    # fix 3
        ax, ay = compute_acceleration(x, y, force_law)

        u1 = u0 + ax * dt
        v1 = v0 + ay * dt

        x = x + (u0 + u1) * 0.5 * dt
        y = y + (v0 + v1) * 0.5 * dt

        r = np.sqrt(x*x + y*y)

        xs[j] = x
        ys[j] = y

        u0 = u1
        v0 = v1
        j += 1

    xs = xs[:j]
    ys = ys[:j]

    # Build Earth surface curve (Java used 400 points)
    angleStep = np.pi / 200
    xEarth = np.zeros(401)
    yEarth = np.zeros(401)

    for k in range(400):
        angle = angleStep * k
        xEarth[k] = R_EARTH * np.cos(angle)     # fix 4
        yEarth[k] = R_EARTH * np.sin(angle)     # fix 5

    xEarth[400] = xEarth[0]
    yEarth[400] = yEarth[0]

    return xs, ys, xEarth, yEarth
