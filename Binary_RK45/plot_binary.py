"""
Binary_RK45 orbit plotting module.

This module takes the structured output from driver_binary
and produces graphs.
"""

import matplotlib.pyplot as plt

def plot_orbits(data):
    xA, yA = data["xA"], data["yA"]
    xB, yB = data["xB"], data["yB"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xA, yA, label="Body A")
    ax.plot(xB, yB, label="Body B")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Binary Orbits (RK45)")
    ax.legend()
    ax.set_aspect("equal", "box")
    ax.grid(True)
    plt.show()
