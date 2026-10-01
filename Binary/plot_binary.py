"""
Binary orbit plotting module.

This module takes the structured output from driver_binary
and produces graphs.
"""

from dataclasses import replace
from math import isfinite
from numbers import Real
from typing import Literal

import matplotlib.pyplot as plt

from driver_binary import BinaryResult
import physics_binary as phys


OutputType = Literal[
    "orbit",
    "orbits",
    "velocity space",
    "position vs. time, body A",
    "position vs. time, body B",
    "velocity vs. time, body A",
    "velocity vs. time, body B",
    "energy vs time",
]


Frame = Literal["user", "com"]


def in_plot_frame(result: BinaryResult, frame: Frame) -> BinaryResult:
    """Return display data without changing the input-frame integration history.

    COM coordinates use relative differences and mass fractions, equivalent to
    subtracting the mass-weighted centre and velocity at each saved sample.
    This avoids forming mass times position or subtracting large common offsets.
    """
    if frame == "user":
        return result
    if frame != "com":
        raise ValueError(f"Unknown frame: {frame}")
    for mass in (result.MA, result.MB):
        if (not isinstance(mass, Real) or isinstance(mass, bool)
                or not isfinite(mass) or mass <= 0):
            raise ValueError("COM plots require finite positive MA and MB in BinaryResult.")
    scale = max(result.MA, result.MB)
    a, b = result.MA / scale, result.MB / scale
    fraction_A, fraction_B = b / (a + b), a / (a + b)
    coordinates = {}
    for component in ("x", "y", "v", "u"):
        relative = [va - vb for va, vb in zip(getattr(result, component + "A"),
                                             getattr(result, component + "B"))]
        if not all(isfinite(value) for value in relative):
            raise ValueError("COM plot coordinates are outside the numerical range.")
        coordinates[component + "A"] = [fraction_A * value for value in relative]
        coordinates[component + "B"] = [-fraction_B * value for value in relative]
    # Recompute kinetic energy from relative velocities instead of subtracting
    # two large energies. Keep the input-frame positions for the same potential.
    energy = [phys.energies(result.MA, result.MB, xa, ya, va, ua, xb, yb, vb, ub)
              for xa, ya, va, ua, xb, yb, vb, ub in zip(
                  result.xA, result.yA, coordinates["vA"], coordinates["uA"],
                  result.xB, result.yB, coordinates["vB"], coordinates["uB"])]
    return replace(result, **coordinates,
                   K=[row[1] for row in energy], E=[row[2] for row in energy])


def plot_binary(result: BinaryResult, output_type: OutputType,
                *, frame: Frame = "user") -> None:
    """
    Plot the requested quantity in the input or centre-of-mass frame.
    """
    result = in_plot_frame(result, frame)

    if output_type in ("orbit", "orbits"):
        fig, ax = plt.subplots()
        ax.plot(result.xA, result.yA, label="Body A")
        ax.plot(result.xB, result.yB, label="Body B")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Binary orbits")
        ax.set_aspect("equal", "box")
        # A fixed legend location keeps it clear of the equal-aspect resizing.
        ax.legend(loc="upper right")

    elif output_type == "velocity space":
        fig, ax = plt.subplots()
        ax.plot(result.vA, result.uA, label="Body A")
        ax.plot(result.vB, result.uB, label="Body B")
        ax.set_xlabel("v_x (m/s)")
        ax.set_ylabel("v_y (m/s)")
        ax.set_title("Velocity space")
        # Both velocity axes use the same units, so equal aspect is appropriate.
        ax.set_aspect("equal", "box")
        ax.legend(loc="upper right")

    elif output_type == "position vs. time, body A":
        fig, ax = plt.subplots()
        ax.plot(result.times, result.xA, label="x_A(t)")
        ax.plot(result.times, result.yA, label="y_A(t)")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("position (m)")
        ax.set_title("Position vs time, body A")
        ax.legend()

    elif output_type == "position vs. time, body B":
        fig, ax = plt.subplots()
        ax.plot(result.times, result.xB, label="x_B(t)")
        ax.plot(result.times, result.yB, label="y_B(t)")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("position (m)")
        ax.set_title("Position vs time, body B")
        ax.legend()

    elif output_type == "velocity vs. time, body A":
        fig, ax = plt.subplots()
        ax.plot(result.times, result.vA, label="v_A(t)")
        ax.plot(result.times, result.uA, label="u_A(t)")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("velocity (m/s)")
        ax.set_title("Velocity vs time, body A")
        ax.legend()

    elif output_type == "velocity vs. time, body B":
        fig, ax = plt.subplots()
        ax.plot(result.times, result.vB, label="v_B(t)")
        ax.plot(result.times, result.uB, label="u_B(t)")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("velocity (m/s)")
        ax.set_title("Velocity vs time, body B")
        ax.legend()

    elif output_type == "energy vs time":
        fig, ax = plt.subplots()
        ax.plot(result.times, result.U, label="Potential U")
        ax.plot(result.times, result.K, label="Kinetic K")
        ax.plot(result.times, result.E, label="Total E")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("Energy (J)")
        ax.set_title("Energy vs time")
        ax.legend()

    else:
        raise ValueError(f"Unknown output_type: {output_type}")

    if frame == "com":
        ax.set_title(ax.get_title() + " (COM frame)")
    plt.tight_layout()
    plt.show()
