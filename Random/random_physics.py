"""
This module contains only the physics of the random walk:
generating random steps, accumulating displacement, and
computing scaled walk length. No plotting or I/O occurs here.
"""

import random
import math

def perform_single_walk(n_steps):
    """
    Perform one random walk consisting of n_steps steps.
    Each step has components dx, dy, dz uniformly distributed in [-1, 1].
    Returns:
        net_distance: length of the displacement vector
        avg_step_length: average length of the individual steps
    """

    # Initialize displacement and step-size accumulator
    x = 0.0
    y = 0.0
    z = 0.0
    step_size_total = 0.0

    for _ in range(n_steps):
        # Generate random step components uniformly in [-1, 1]
        dx = 2 * random.random() - 1
        dy = 2 * random.random() - 1
        dz = 2 * random.random() - 1

        # Accumulate total step length
        step_size_total += math.sqrt(dx*dx + dy*dy + dz*dz)

        # Update displacement
        x += dx
        y += dy
        z += dz

    net_distance = math.sqrt(x*x + y*y + z*z)
    avg_step_length = step_size_total / n_steps

    return net_distance, avg_step_length


def perform_trials(n_steps, n_trials):
    """
    Perform n_trials random walks of n_steps steps each.
    Returns the average scaled distance:
        scaled_distance = net_distance / avg_step_length
    """

    total_scaled_distance = 0.0

    for _ in range(n_trials):
        net_dist, avg_step = perform_single_walk(n_steps)
        scaled = net_dist / avg_step
        total_scaled_distance += scaled

    return total_scaled_distance / n_trials
