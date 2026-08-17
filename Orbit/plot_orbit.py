"""
Plotting routine for Orbit.

Five display modes are available:
  "orbit"          -- x-y trajectory
  "velocity"       -- velocity-space hodograph
  "position_time"  -- x(t), y(t)
  "velocity_time"  -- vx(t), vy(t)
  "energy"         -- specific kinetic, potential, and total energy
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from driver_orbit import OrbitResult, OutputType


def plot_orbit(result: OrbitResult, output: OutputType = "orbit") -> None:
    """Display the selected view of an already-computed orbit."""
    fig, ax = plt.subplots(
        figsize=(8, 8) if output in ("orbit", "velocity") else (9, 6)
    )

    if output == "orbit":
        ax.plot(result.xs, result.ys, linewidth=2, label="Orbit")
        ax.scatter([0], [0], s=35, label="Central mass")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Orbit — Newtonian Planetary Motion")
        ax.set_aspect("equal", "box")
        ax.legend()

    elif output == "velocity":
        ax.plot(result.vxs, result.vys, linewidth=2, label="Velocity")
        ax.set_xlabel(r"$v_x$ (m/s)")
        ax.set_ylabel(r"$v_y$ (m/s)")
        ax.set_title("Orbit — Velocity Space (Hodograph)")
        ax.set_aspect("equal", "box")
        ax.legend()

    elif output == "position_time":
        ax.plot(result.ts, result.xs, label="x(t)")
        ax.plot(result.ts, result.ys, label="y(t)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (m)")
        ax.set_title("Orbit — Position vs. Time")
        ax.legend()

    elif output == "velocity_time":
        ax.plot(result.ts, result.vxs, label=r"$v_x(t)$")
        ax.plot(result.ts, result.vys, label=r"$v_y(t)$")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Orbit — Velocity vs. Time")
        ax.legend()

    elif output == "energy":
        total = result.KEs + result.PEs
        ax.plot(result.ts, result.KEs, label="Kinetic (specific)")
        ax.plot(result.ts, result.PEs, label="Potential (specific)")
        ax.plot(result.ts, total, label="Total (specific)", linestyle="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Specific energy (J/kg)")
        ax.set_title("Orbit — Energy vs. Time")
        ax.legend()

    else:
        raise ValueError(f"Unknown output mode: {output!r}")

    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()
