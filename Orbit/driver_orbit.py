"""
Driver for the Orbit simulation.
"""

import numpy as np
from physics_orbit import compute_acceleration

def run_orbit(
    xInit, yInit,
    vxInit, vyInit,
    k,             # GM of central mass
    dt0,           # initial time-step
    maxSteps,
    eps1,          # time-step accuracy threshold
    eps2,          # predictor-corrector accuracy threshold
    output="orbit" # "orbit", "velocity", "position_time", "velocity_time", "energy"
):
    """
    Run the Orbit simulation.

    Parameters match Schutz's Java program:
        xInit, yInit   — initial position
        vxInit, vyInit — initial velocity
        k              — GM of central mass
        dt0            — initial time-step
        maxSteps       — maximum number of steps
        eps1           — threshold for time-step reduction
        eps2           — threshold for predictor-corrector convergence
        output         — type of output data

    Returns:
        Data arrays depending on output mode.
    """

    # Initial state
    x = xInit
    y = yInit
    vx = vxInit
    vy = vyInit

    # Arrays for storing the full trajectory: position, velocity, time,
    # and energy history. Schutz's Java stores all of these
    # (xCoordinate/yCoordinate, xVelocity/yVelocity, time,
    # potentialEnergy/kineticEnergy) at every accepted step -- earlier
    # versions of this driver only tracked vx/vy as scalars that got
    # overwritten each step, which silently broke every output mode
    # except "orbit".
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

    # Orbit closure detection. This mirrors Schutz's Java exactly:
    # angleInitPos/angleInitVel are computed once from the initial data,
    # and the direction of travel (counterclockwise) is determined from
    # the angle BETWEEN them -- not assumed. Every step's angular
    # position is then compared against angleInitPos (not against zero),
    # so closure is detected correctly regardless of where the orbit
    # starts or which way it travels. The previous version of this
    # driver skipped angleInitPos/angleInitVel/counterclockwise
    # entirely and just re-wrapped the raw position angle against zero,
    # which only happened to work for an orbit starting on the positive
    # x-axis and moving counterclockwise.
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
    xs = xs[:j]
    ys = ys[:j]
    vxs = vxs[:j]
    vys = vys[:j]
    ts = ts[:j]
    PEs = PEs[:j]
    KEs = KEs[:j]

    # Output modes
    if output == "orbit":
        return xs, ys

    elif output == "velocity":
        return vxs, vys

    elif output == "position_time":
        return ts, xs, ys

    elif output == "velocity_time":
        return ts, vxs, vys

    elif output == "energy":
        TE = KEs + PEs
        return ts, KEs, PEs, TE

    else:
        return xs, ys
