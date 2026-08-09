"""
Study random walks: compute the mean distance traveled.
This simulates the way a photon moves outwards from the
energy-generating center of a star to its surface.

Entry point for the Random Walk program.
Users may modify maxSteps and nTrials here.
"""

from random_driver import run_random_walk_experiments
from random_plot import plot_random_walk_results

def main():
    # Example parameters (user may overwrite)
    maxSteps = 4096     # analogous to Schutz's default large walk
    nTrials  = 100       # matches the Triana GUI default

    lengths, avgDist = run_random_walk_experiments(maxSteps, nTrials)

    plot_random_walk_results(lengths, avgDist)

if __name__ == "__main__":
    main()