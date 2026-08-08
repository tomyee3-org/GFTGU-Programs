"""
Plot module for Multiple_RK45.
Plots trajectories or current positions.
"""

import matplotlib.pyplot as plt


def plot_trajectories(trajectories, labels):
    plt.figure(figsize=(8, 8))
    for xyz, label in zip(trajectories, labels):
        plt.plot(xyz[:, 0], xyz[:, 1], label=label)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.title("Multiple: Trajectories (RK45)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def plot_positions(trajectories, labels):
    plt.figure(figsize=(8, 8))
    for xyz, label in zip(trajectories, labels):
        plt.scatter(xyz[-1, 0], xyz[-1, 1], s=40, label=label)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.title("Multiple: Current Positions (RK45)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
