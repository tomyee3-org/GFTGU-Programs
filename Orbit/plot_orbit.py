"""
plot_orbit.py

Plotting routine for Orbit.

Five output modes are available (selected via the `output` argument,
not by run_orbit() -- the driver always computes everything; this
module decides what to show):

    "orbit"          -- the trajectory in the x-y plane (closed ellipse
                        for a bound orbit). Equal-aspect axes, since
                        the true shape of the orbit is the point.
    "velocity"       -- the velocity-space "hodograph": the curve
                        traced by (vx, vy) as the body moves around
                        its orbit. For Newtonian 1/r^2 gravity this is
                        always a circle, regardless of the orbit's own
                        eccentricity -- one of the elegant, easy-to-
                        miss consequences of the inverse-square law.
                        Equal-aspect axes, so that circle actually
                        looks like a circle.
    "position_time"  -- x(t) and y(t) together on a time axis. Lets
                        you read off the orbital period directly, and
                        see the unequal speed through perihelion vs.
                        aphelion (Kepler's second law).
    "velocity_time"  -- vx(t) and vy(t) together on a time axis. Shows
                        that the velocity is periodic too, with the
                        same period as the position.
    "energy"         -- specific kinetic, potential, and total energy
                        vs. time. The total (dashed) should stay flat;
                        how much it drifts is a direct visual measure
                        of the integrator's numerical accuracy.

None of these axis pairs share the same units, so none except "orbit"
and "velocity" use equal-aspect scaling.
"""

import matplotlib.pyplot as plt

from driver_orbit import OrbitResult, OutputType


def plot_orbit(result: OrbitResult, output: OutputType = "orbit") -> None:
    fig, ax = plt.subplots(figsize=(8, 8) if output in ("orbit", "velocity") else (9, 6))

    if output == "orbit":
        ax.plot(result.xs, result.ys, linewidth=2, color="red", label="Orbit")
        ax.scatter([0], [0], color="orange", label="Central mass")
        ax.set_xlabel("x (meters)")
        ax.set_ylabel("y (meters)")
        ax.set_title("Orbit — Newtonian Planetary Motion")
        ax.set_aspect("equal", "box")
        ax.legend()

    elif output == "velocity":
        ax.plot(result.vxs, result.vys, linewidth=2, color="purple", label="Velocity")
        ax.set_xlabel("vx (m/s)")
        ax.set_ylabel("vy (m/s)")
        ax.set_title("Orbit — Velocity Space (Hodograph)")
        ax.set_aspect("equal", "box")
        ax.legend()

    elif output == "position_time":
        ax.plot(result.ts, result.xs, label="x(t)")
        ax.plot(result.ts, result.ys, label="y(t)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (m)")
        ax.set_title("Orbit — Position vs Time")
        ax.legend()

    elif output == "velocity_time":
        ax.plot(result.ts, result.vxs, label="vx(t)")
        ax.plot(result.ts, result.vys, label="vy(t)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Orbit — Velocity vs Time")
        ax.legend()

    elif output == "energy":
        TEs = [ke + pe for ke, pe in zip(result.KEs, result.PEs)]
        ax.plot(result.ts, result.KEs, label="Kinetic (specific)")
        ax.plot(result.ts, result.PEs, label="Potential (specific)")
        ax.plot(result.ts, TEs, label="Total (specific)", linestyle="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Specific energy (J/kg)")
        ax.set_title("Orbit — Energy vs Time")
        ax.legend()

    else:
        raise ValueError(f"Unknown output mode: {output!r}")

    ax.grid(True)
    plt.tight_layout()
    plt.show()
