"""
Plotting routines for the Random Walk program.
Uses matplotlib to produce the log-log plot
"""

import matplotlib.pyplot as plt

def plot_random_walk_results(lengths, avgDist):
    """
    Produces a log-log plot of average scaled distance vs number of steps.
    """

    fig, ax = plt.subplots(figsize=(8,6))

    ax.loglog(lengths, avgDist, marker='o', linestyle='-', color='blue')

    ax.set_xlabel("Number of Steps (log scale)")
    ax.set_ylabel("Scaled Distance (log scale)")
    ax.set_title("Random Walk: Scaled Distance vs Number of Steps")

    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.show()
