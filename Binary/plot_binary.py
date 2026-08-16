"""
Binary orbit plotting module.

This module takes the structured output from driver_binary
and produces graphs.
"""

from typing import Literal

import matplotlib.pyplot as plt

from driver_binary import BinaryResult


OutputType = Literal[
    "orbits",
    "velocity space",
    "position vs. time, body A",
    "position vs. time, body B",
    "velocity vs. time, body A",
    "velocity vs. time, body B",
    "energy vs time",
]


def plot_binary(result: BinaryResult, output_type: OutputType) -> None:
    """
    Plot the requested quantity
    """

    if output_type == "orbits":
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

    plt.tight_layout()
    plt.show()