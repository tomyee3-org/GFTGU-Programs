"""
Driver logic for the Random Walk program

This module performs the sequence of experiments:
starting with maxSteps, halving the number of steps each time,
and computing the average scaled walk distance for each.
"""

import math
from random_physics import perform_trials

def run_random_walk_experiments(max_steps, n_trials):
    """
    Runs the full suite of random-walk experiments.

    The logic:
        nWalks = floor(log2(maxSteps))
        nSteps starts at maxSteps
        For j = nWalks-1 down to 0:
            lengths[j] = nSteps
            avgDist[j] = average scaled distance over nTrials walks
            nSteps /= 2
            stop if nSteps <= 1

    Returns:
        lengths: list of step counts
        avgDist: list of average scaled distances
    """

    # Number of different walk lengths to test
    n_walks = int(math.floor(math.log(max_steps) / math.log(2.0)))

    lengths = [0.0] * n_walks
    avgDist = [0.0] * n_walks

    n_steps = max_steps

    for j in range(n_walks - 1, -1, -1):
        lengths[j] = n_steps

        # Compute average scaled distance for this number of steps
        avgDist[j] = perform_trials(n_steps, n_trials)

        # Halve the number of steps for the next experiment
        n_steps //= 2
        if n_steps <= 1:
            break

    return lengths, avgDist
