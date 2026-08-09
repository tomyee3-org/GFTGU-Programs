"""
random2_driver.py

Driver for Random2. Two independent experiments are supported:

    "scaled_distance" -- the same experiment as the original Random
        program (Investigation 8.7): for a sequence of step counts
        (halving from max_steps down), run many independent walks and
        average the net displacement, scaled by the mean step length.
        Reimplemented here in 2D (see random2_physics.generate_2d_step)
        rather than 3D, since Random2's other mode needs a 2D walk
        anyway and the sqrt(N) diffusion law this experiment
        demonstrates holds regardless of dimension.

    "walk2d" -- a small number of individual walks, each run for up to
        max_steps steps, starting at the origin ("center of the star")
        and stopping early if the walk crosses a circular boundary
        ("the surface", of a given or default radius) -- in which case
        a straight ray is recorded continuing in the walk's last
        direction of travel, representing the photon escaping into
        space. Walks that do not reach the surface within max_steps
        simply stop where they are.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math

from random2_physics import (
    Point,
    generate_2d_step,
    default_radius,
    circle_crossing_fraction,
    point_at,
)


# ---------------------------------------------------------------------
# "scaled_distance" mode
# ---------------------------------------------------------------------

def _perform_single_walk_2d(n_steps: int) -> Tuple[float, float]:
    """One 2D walk of n_steps steps. Returns (net_distance, avg_step_length)."""
    x = 0.0
    y = 0.0
    step_size_total = 0.0

    for _ in range(n_steps):
        dx, dy = generate_2d_step()
        step_size_total += math.sqrt(dx * dx + dy * dy)
        x += dx
        y += dy

    net_distance = math.sqrt(x * x + y * y)
    avg_step_length = step_size_total / n_steps
    return net_distance, avg_step_length


def _perform_trials_2d(n_steps: int, n_trials: int) -> float:
    """Average scaled distance (net_distance / avg_step_length) over n_trials walks."""
    total = 0.0
    for _ in range(n_trials):
        net_dist, avg_step = _perform_single_walk_2d(n_steps)
        total += net_dist / avg_step
    return total / n_trials


def run_scaled_distance_experiment(max_steps: int, n_trials: int
                                    ) -> Tuple[List[float], List[float]]:
    """
    Runs the same experiment as the original Random program, in 2D:
        nWalks = floor(log2(maxSteps))
        nSteps starts at maxSteps and halves each iteration
        avgDist[j] = average scaled distance over nTrials walks

    Returns (lengths, avg_dist).
    """
    n_walks = int(math.floor(math.log(max_steps) / math.log(2.0)))

    lengths = [0.0] * n_walks
    avg_dist = [0.0] * n_walks

    n_steps = max_steps
    for j in range(n_walks - 1, -1, -1):
        lengths[j] = n_steps
        avg_dist[j] = _perform_trials_2d(n_steps, n_trials)
        n_steps //= 2
        if n_steps <= 1:
            break

    return lengths, avg_dist


# ---------------------------------------------------------------------
# "walk2d" mode
# ---------------------------------------------------------------------

@dataclass
class WalkPath:
    points: List[Point]              # the walk itself, from the origin
    escaped: bool                    # whether it crossed the boundary
    ray: Optional[Tuple[Point, Point]] = None  # (exit point, ray end point), if escaped


@dataclass
class Walk2DResult:
    radius: float
    mean_free_path: float
    max_steps: int
    walks: List[WalkPath] = field(default_factory=list)


def run_walk2d(
    reference_steps: int = 2000,
    n_walks: int = 6,
    radius: Optional[float] = None,
    mean_free_path: float = 1.0,
    radius_factor: float = 2.0,
    ray_length_factor: float = 0.6,
    step_cap: int = 200_000,
) -> Walk2DResult:
    """
    Run n_walks independent 2D random walks, each starting at the
    origin and continuing until it crosses the circular boundary
    ("the star's surface") of the given (or default) radius -- so that,
    as in the photon-diffusion picture this models, a walk almost
    always eventually escapes; visual variety comes from how long and
    wandering each escaping path is, not from a coin-flip on whether it
    escapes at all. A walk stops early if it crosses the boundary, in
    which case a straight ray is recorded continuing in the direction
    of the walk's final step, extending an additional
    ray_length_factor*radius beyond the crossing point. step_cap is a
    generous safety limit only -- reaching it (a walk that still hasn't
    escaped) should be rare with the default radius_factor, but is
    handled the same way an incomplete walk always is: plotted as-is,
    with no ray.

    Parameters
    ----------
    reference_steps : int
        The "N" used only for the default radius formula (see
        random2_physics.default_radius) -- NOT a hard step budget for
        the walk itself. Historically this was found empirically to
        need to be much smaller than the number of steps an individual
        walk actually takes to reach that radius (by roughly 2-20x, for
        radius_factor=2) -- see the Random2 documentation.
    n_walks : int
        Number of independent walks to draw.
    radius : float, optional
        Star radius. Defaults to random2_physics.default_radius(...).
    mean_free_path : float
        Used only for the default radius formula, unless radius is given.
    radius_factor : float
        Used only for the default radius formula, unless radius is given.
    ray_length_factor : float
        Length of the escaping ray beyond the boundary, as a fraction
        of the radius.
    step_cap : int
        Safety limit on steps per walk, to guarantee termination even
        in the (rare, long-tail) case a walk hasn't escaped yet.
    """
    if radius is None:
        radius = default_radius(reference_steps, mean_free_path, radius_factor)

    walks: List[WalkPath] = []

    for _ in range(n_walks):
        x, y = 0.0, 0.0
        points: List[Point] = [(x, y)]
        escaped = False
        ray = None

        for _ in range(step_cap):
            dx, dy = generate_2d_step()
            x_new, y_new = x + dx, y + dy

            if math.hypot(x_new, y_new) >= radius:
                t = circle_crossing_fraction((x, y), (x_new, y_new), radius)
                if t is None:
                    # Shouldn't happen since we already know p1 is
                    # outside, but fall back to the endpoint itself
                    # rather than crash if it ever does.
                    exit_point = (x_new, y_new)
                else:
                    exit_point = point_at((x, y), (x_new, y_new), t)

                points.append(exit_point)

                # Continue the ray in the same direction as the final step.
                step_len = math.hypot(dx, dy)
                if step_len > 0.0:
                    ux, uy = dx / step_len, dy / step_len
                else:
                    ux, uy = exit_point[0] / radius, exit_point[1] / radius
                ray_end = (
                    exit_point[0] + ux * ray_length_factor * radius,
                    exit_point[1] + uy * ray_length_factor * radius,
                )
                ray = (exit_point, ray_end)
                escaped = True
                break

            x, y = x_new, y_new
            points.append((x, y))

        walks.append(WalkPath(points=points, escaped=escaped, ray=ray))

    return Walk2DResult(radius=radius, mean_free_path=mean_free_path,
                         max_steps=reference_steps, walks=walks)
