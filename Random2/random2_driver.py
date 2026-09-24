"""
random2_driver.py

Driver routines for Random2.

"scaled_distance"
    For a sequence of step counts, perform many independent 3D random
    walks and average the net displacement after dividing by that
    walk's mean step length. The experiment demonstrates the sqrt(N)
    scaling of random-walk displacement.

"walk2d"
    Draw a small number of fixed-step, isotropic 2D walks from the
    center of a circular schematic star. Each walk continues until it
    crosses the boundary or reaches a generous safety cap.

The result types are frozen dataclasses holding tuples, so a result
cannot be changed after it is returned to main.py or the plotter.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import statistics

import random2_physics as phys
from random2_physics import (
    Point,
    StepDistribution,
    circle_crossing_fraction,
    default_radius,
    generate_component_step,
    generate_isotropic_step,
    point_at,
    require_nonnegative_finite_number,
    require_positive_finite_number,
    require_positive_int,
)


# ---------------------------------------------------------------------
# "scaled_distance" mode
# ---------------------------------------------------------------------

def _perform_single_walk_3d(
    n_steps: int,
    step_distribution: StepDistribution = "uniform",
) -> Tuple[float, float]:
    """Return (net_distance, mean_step_length) for one 3D walk."""
    require_positive_int("n_steps", n_steps)

    x = 0.0
    y = 0.0
    z = 0.0
    step_size_total = 0.0

    for _ in range(n_steps):
        dx, dy, dz = generate_component_step(step_distribution)
        step_size_total += math.sqrt(dx * dx + dy * dy + dz * dz)
        x += dx
        y += dy
        z += dz

    net_distance = math.sqrt(x * x + y * y + z * z)
    mean_step_length = step_size_total / n_steps
    return net_distance, mean_step_length


def _perform_trials_3d_with_error(
    n_steps: int,
    n_trials: int,
    step_distribution: StepDistribution = "uniform",
) -> Tuple[float, Optional[float]]:
    """Return (mean scaled distance, standard error of that mean).

    The standard error is the sample standard deviation of the trial values
    divided by sqrt(n_trials); it is None when n_trials is 1.
    """
    require_positive_int("n_trials", n_trials)

    total = 0.0
    total_squares = 0.0
    for _ in range(n_trials):
        net_distance, mean_step = _perform_single_walk_3d(
            n_steps, step_distribution
        )
        if not math.isfinite(mean_step) or mean_step <= 0.0:
            raise RuntimeError(
                "generated walk has a nonpositive or non-finite mean step length."
            )
        scaled = net_distance / mean_step
        total += scaled
        total_squares += scaled * scaled
    mean = total / n_trials
    if n_trials == 1:
        return mean, None
    variance = max(0.0, (total_squares - n_trials * mean * mean) / (n_trials - 1))
    return mean, math.sqrt(variance / n_trials)


def _perform_trials_3d(
    n_steps: int,
    n_trials: int,
    step_distribution: StepDistribution = "uniform",
) -> float:
    """Average the scaled distance over n_trials independent walks."""
    return _perform_trials_3d_with_error(n_steps, n_trials, step_distribution)[0]


@dataclass(frozen=True)
class ScaledDistanceResult:
    """Scaled-distance statistics, ordered from the shortest to the longest walk."""
    model_version: str
    build_id: str
    step_distribution: str
    n_trials: int
    lengths: Tuple[float, ...]
    averages: Tuple[float, ...]
    standard_errors: Tuple[Optional[float], ...]


def run_scaled_distance_statistics(
    max_steps: int,
    n_trials: int,
    step_distribution: StepDistribution = "uniform",
) -> ScaledDistanceResult:
    """
    Run scaled-distance experiments at max_steps, max_steps//2, ...,
    stopping before a one-step walk, and return the averages together
    with their standard errors, ordered from smallest to largest step count.
    """
    require_positive_int("max_steps", max_steps)
    require_positive_int("n_trials", n_trials)
    if max_steps < 2:
        raise ValueError("max_steps must be at least 2.")
    if step_distribution not in ("uniform", "gaussian"):
        raise ValueError(
            'step_distribution must be "uniform" or "gaussian".'
        )

    rows = []
    n_steps = max_steps
    while n_steps > 1:
        mean, error = _perform_trials_3d_with_error(
            n_steps,
            n_trials,
            step_distribution,
        )
        rows.append((float(n_steps), mean, error))
        n_steps //= 2

    rows.reverse()
    return ScaledDistanceResult(
        model_version=phys.MODEL_VERSION,
        build_id=phys.BUILD_ID,
        step_distribution=step_distribution,
        n_trials=n_trials,
        lengths=tuple(row[0] for row in rows),
        averages=tuple(row[1] for row in rows),
        standard_errors=tuple(row[2] for row in rows),
    )


def run_scaled_distance_experiment(
    max_steps: int,
    n_trials: int,
    step_distribution: StepDistribution = "uniform",
) -> Tuple[List[float], List[float]]:
    """
    Run scaled-distance experiments at max_steps, max_steps//2, ...,
    stopping before a one-step walk.

    Returns
    -------
    lengths, avg_dist
        Step counts and the corresponding average scaled distances,
        ordered from smallest to largest step count.
    """
    result = run_scaled_distance_statistics(max_steps, n_trials, step_distribution)
    return list(result.lengths), list(result.averages)


# ---------------------------------------------------------------------
# "walk2d" mode
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class WalkPath:
    points: Tuple[Point, ...]
    escaped: bool
    steps_taken: int
    ray: Optional[Tuple[Point, Point]] = None


@dataclass(frozen=True)
class Walk2DResult:
    model_version: str
    build_id: str
    radius: float
    mean_free_path: float
    reference_steps: int
    step_cap: int
    walks: Tuple[WalkPath, ...] = ()


@dataclass(frozen=True)
class EscapeStatistics:
    """Summary of the escape step counts of the walks that escaped."""
    n_walks: int
    n_escaped: int
    mean: Optional[float]
    standard_error: Optional[float]
    median: Optional[float]
    minimum: Optional[int]
    maximum: Optional[int]


def escape_statistics(result: Walk2DResult) -> EscapeStatistics:
    """Return escape-step statistics; walks stopped by step_cap are excluded."""
    steps = [walk.steps_taken for walk in result.walks if walk.escaped]
    if not steps:
        return EscapeStatistics(len(result.walks), 0, None, None, None, None, None)
    mean = statistics.fmean(steps)
    error = (
        statistics.stdev(steps) / math.sqrt(len(steps)) if len(steps) > 1 else None
    )
    return EscapeStatistics(
        n_walks=len(result.walks),
        n_escaped=len(steps),
        mean=mean,
        standard_error=error,
        median=float(statistics.median(steps)),
        minimum=min(steps),
        maximum=max(steps),
    )


def run_walk2d(
    reference_steps: int = 2000,
    n_walks: int = 4,
    radius: Optional[float] = None,
    mean_free_path: float = 1.0,
    radius_factor: float = 2.0,
    ray_length_factor: float = 0.6,
    step_cap: int = 200_000,
) -> Walk2DResult:
    """
    Run fixed-length isotropic 2D random walks from the origin.

    If radius is None, the schematic star radius is

        radius_factor * mean_free_path * sqrt(reference_steps).

    A walk ends when it crosses the circular boundary or reaches
    step_cap. When a crossing occurs, the plotted path terminates at the
    exact line-circle intersection and a short straight ray is recorded
    to indicate that the toy model has stopped scattering the photon.
    """
    require_positive_int("reference_steps", reference_steps)
    require_positive_int("n_walks", n_walks)
    require_positive_int("step_cap", step_cap)

    mean_free_path = require_positive_finite_number(
        "mean_free_path", mean_free_path
    )
    radius_factor = require_positive_finite_number(
        "radius_factor", radius_factor
    )
    ray_length_factor = require_nonnegative_finite_number(
        "ray_length_factor", ray_length_factor
    )

    if radius is None:
        radius = default_radius(
            reference_steps,
            mean_free_path,
            radius_factor,
        )
    else:
        radius = require_positive_finite_number("radius", radius)

    walks: List[WalkPath] = []

    for _ in range(n_walks):
        x, y = 0.0, 0.0
        points: List[Point] = [(x, y)]
        escaped = False
        ray = None
        steps_taken = 0

        for step_number in range(1, step_cap + 1):
            dx, dy = generate_isotropic_step(mean_free_path)
            x_new, y_new = x + dx, y + dy
            steps_taken = step_number

            if math.hypot(x_new, y_new) >= radius:
                t = circle_crossing_fraction(
                    (x, y),
                    (x_new, y_new),
                    radius,
                )
                if t is None:
                    raise RuntimeError(
                        "boundary crossing was detected but no circle "
                        "intersection could be computed."
                    )
                exit_point = point_at((x, y), (x_new, y_new), t)
                points.append(exit_point)

                if ray_length_factor > 0.0:
                    # An isotropic step has length mean_free_path > 0.
                    ux = dx / mean_free_path
                    uy = dy / mean_free_path
                    ray_end = (
                        exit_point[0] + ux * ray_length_factor * radius,
                        exit_point[1] + uy * ray_length_factor * radius,
                    )
                    ray = (exit_point, ray_end)

                escaped = True
                break

            x, y = x_new, y_new
            points.append((x, y))

        walks.append(
            WalkPath(
                points=tuple(points),
                escaped=escaped,
                steps_taken=steps_taken,
                ray=ray,
            )
        )

    return Walk2DResult(
        model_version=phys.MODEL_VERSION,
        build_id=phys.BUILD_ID,
        radius=radius,
        mean_free_path=mean_free_path,
        reference_steps=reference_steps,
        step_cap=step_cap,
        walks=tuple(walks),
    )
