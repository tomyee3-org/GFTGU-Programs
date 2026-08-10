"""
Driver for the Orbit simulation.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from physics_orbit import compute_acceleration

OutputType = Literal["orbit", "velocity", "position_time", "velocity_time", "energy"]


@dataclass
class OrbitResult:
    """
    Full trajectory history. run_orbit() always computes and returns
    every one of these arrays -- the computation does not depend on
    which output mode the user eventually wants to look at, only the
    plotting does (see plot_orbit.py).

    xs, ys       -- position (m)
    vxs, vys     -- velocity (m/s)
    ts           -- time since the start of the orbit (s)
    PEs, KEs     -- specific (per unit mass) potential and kinetic
                    energy (J/kg), since the orbiting body's own mass
                    never enters these calculations
    """
    xs: NDArray[np.float64]
    ys: NDArray[np.float64]
    vxs: NDArray[np.float64]
    vys: NDArray[np.float64]
    ts: NDArray[np.float64]
    PEs: NDArray[np.float64]
    KEs: NDArray[np.float64]


def run_orbit(
    xInit, yInit,            # initial position
    vxInit, vyInit,          # initial velocity
    k,             # GM of central mass
    dt0,                     # initial time-step
    maxSteps,                # maximum number of steps
    eps1,          # threshold for time-step reduction
    eps2,          # threshold for predictor-corrector convergence
) -> OrbitResult:
    """
    Returns
    -------
    OrbitResult with the full position/velocity/time/energy history.
    Which of these arrays actually gets plotted, and how, is decided
    separately by plot_orbit()'s `output` argument -- see that
    function's docstring for what each of the five output modes
    ("orbit", "velocity", "position_time", "velocity_time", "energy")
    actually shows.
    """

    # Initial state
    x = xInit
    y = yInit
    vx = vxInit
    vy = vyInit

    # Arrays for storing the full trajectory: position, velocity, time,
    # and energy history at every accepted step.
    xs = np.zeros(maxSteps)
    ys = np.zeros(maxSteps)
    vxs = np.zeros(maxSteps)
    vys = np.zeros(maxSteps)
    ts = np.zeros(maxSteps)
    PEs = np.zeros(maxSteps)
    KEs = np.zeros(maxSteps)

    xs[0] = x
    ys[0] = y
    vxs[0] = vx
    vys[0] = vy
    ts[0] = 0.0
    r0 = np.sqrt(x*x + y*y)
    PEs[0] = -k / r0
    KEs[0] = 0.5 * (vx*vx + vy*vy)

    # Initial time-step
    dt1 = dt0
    t = 0.0

    # Orbit closure detection.
    # angleInitPos/angleInitVel are computed once from the initial data,
    # and the direction of travel (counterclockwise) is determined from
    # the angle BETWEEN them -- not assumed. Every step's angular
    # position is then compared against angleInitPos,
    # so closure is detected correctly regardless of where the orbit
    # starts or which way it travels.
    angleInitPos = np.arctan2(yInit, xInit)
    angleInitVel = np.arctan2(vyInit, vxInit)
    anglediff0 = angleInitVel - angleInitPos
    if anglediff0 > np.pi:
        anglediff0 -= 2 * np.pi
    elif anglediff0 < -np.pi:
        anglediff0 += 2 * np.pi
    counterclockwise = anglediff0 > 0

    halfOrbit = False
    fullOrbit = False

    j = 1
    while j < maxSteps and not fullOrbit:
        # Compute acceleration at beginning of step
        ax0, ay0 = compute_acceleration(x, y, k)

        # Predictor step: constant acceleration assumption
        dv = ax0 * dt1
        du = ay0 * dt1
        dx = vx * dt1
        dy = vy * dt1

        ddx0 = dv * dt1 / 2
        ddy0 = du * dt1 / 2

        x1 = x + dx + ddx0
        y1 = y + dy + ddy0

        # Acceleration at predicted position
        ax1, ay1 = compute_acceleration(x1, y1, k)

        # Time-step adjustment test
        if abs(ax1 - ax0) + abs(ay1 - ay0) > eps1 * (abs(ax0) + abs(ay0)):
            dt1 /= 2
            continue

        # Predictor-corrector iteration
        testPrediction = abs(ddx0) + abs(ddy0)
        ddx1 = ddx0
        ddy1 = ddy0

        for _ in range(10):
            dv = (ax0 + ax1) * dt1 / 2
            du = (ay0 + ay1) * dt1 / 2

            ddx1 = dv * dt1 / 2
            ddy1 = du * dt1 / 2

            if abs(ddx1 - ddx0) + abs(ddy1 - ddy0) > eps2 * testPrediction:
                ddx0 = ddx1
                ddy0 = ddy1

                x1 = x + dx + ddx0
                y1 = y + dy + ddy0

                ax1, ay1 = compute_acceleration(x1, y1, k)
            else:
                break

        # Commit step
        t += dt1
        x += dx + ddx1
        y += dy + ddy1
        vx += dv
        vy += du

        xs[j] = x
        ys[j] = y
        vxs[j] = vx
        vys[j] = vy
        ts[j] = t
        r = np.sqrt(x*x + y*y)
        PEs[j] = -k / r
        KEs[j] = 0.5 * (vx*vx + vy*vy)

        # Orbit closure detection using the angle relative to the
        # starting position, exactly as in Orbit.java.
        angleNow = np.arctan2(y, x)
        anglediff = angleNow - angleInitPos
        if anglediff > np.pi:
            anglediff -= 2 * np.pi
        elif anglediff < -np.pi:
            anglediff += 2 * np.pi

        if not halfOrbit:
            halfOrbit = (anglediff < 0) if counterclockwise else (anglediff > 0)
        else:
            fullOrbit = (anglediff > 0) if counterclockwise else (anglediff < 0)

        j += 1

    # Trim arrays
    return OrbitResult(
        xs=xs[:j],
        ys=ys[:j],
        vxs=vxs[:j],
        vys=vys[:j],
        ts=ts[:j],
        PEs=PEs[:j],
        KEs=KEs[:j],
    )
