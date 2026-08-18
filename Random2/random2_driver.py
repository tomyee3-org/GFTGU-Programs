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
"""

from dataclasses import dataclass, field
from numbers import Real
from typing import List, Optional, Tuple
import math

from random2_physics import (
    Point,
    StepDistribution,
    circle_crossing_fraction,
    default_radius,
    generate_component_step,
    generate_isotropic_step,
    point_at,
)


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _require_positive_finite_number(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return numeric


def _require_nonnegative_finite_number(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a nonnegative finite number.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite number.")
    return numeric


# ---------------------------------------------------------------------
# "scaled_distance" mode
# ---------------------------------------------------------------------

def _perform_single_walk_3d(
    n_steps: int,
    step_distribution: StepDistribution = "uniform",
) -> Tuple[float, float]:
    """Return (net_distance, mean_step_length) for one 3D walk."""
    _require_positive_int("n_steps", n_steps)

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


def _perform_trials_3d(
    n_steps: int,
    n_trials: int,
    step_distribution: StepDistribution = "uniform",
) -> float:
    """Average the scaled distance over n_trials independent walks."""
    _require_positive_int("n_trials", n_trials)

    total = 0.0
    for _ in range(n_trials):
        net_distance, mean_step = _perform_single_walk_3d(
            n_steps, step_distribution
        )
        if not math.isfinite(mean_step) or mean_step <= 0.0:
            raise RuntimeError(
                "generated walk has a nonpositive or non-finite mean step length."
            )
        total += net_distance / mean_step
    return total / n_trials


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
    _require_positive_int("max_steps", max_steps)
    _require_positive_int("n_trials", n_trials)
    if max_steps < 2:
        raise ValueError("max_steps must be at least 2.")
    if step_distribution not in ("uniform", "gaussian"):
        raise ValueError(
            'step_distribution must be "uniform" or "gaussian".'
        )

    pairs = []
    n_steps = max_steps
    while n_steps > 1:
        pairs.append(
            (
                float(n_steps),
                _perform_trials_3d(
                    n_steps,
                    n_trials,
                    step_distribution,
                ),
            )
        )
        n_steps //= 2

    pairs.reverse()
    lengths = [p[0] for p in pairs]
    avg_dist = [p[1] for p in pairs]
    return lengths, avg_dist


# ---------------------------------------------------------------------
# "walk2d" mode
# ---------------------------------------------------------------------

@dataclass
class WalkPath:
    points: List[Point]
    escaped: bool
    steps_taken: int
    ray: Optional[Tuple[Point, Point]] = None


@dataclass
class Walk2DResult:
    radius: float
    mean_free_path: float
    reference_steps: int
    step_cap: int
    walks: List[WalkPath] = field(default_factory=list)


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
    _require_positive_int("reference_steps", reference_steps)
    _require_positive_int("n_walks", n_walks)
    _require_positive_int("step_cap", step_cap)

    mean_free_path = _require_positive_finite_number(
        "mean_free_path", mean_free_path
    )
    radius_factor = _require_positive_finite_number(
        "radius_factor", radius_factor
    )
    ray_length_factor = _require_nonnegative_finite_number(
        "ray_length_factor", ray_length_factor
    )

    if radius is None:
        radius = default_radius(
            reference_steps,
            mean_free_path,
            radius_factor,
        )
    else:
        radius = _require_positive_finite_number("radius", radius)

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
                points=points,
                escaped=escaped,
                steps_taken=steps_taken,
                ray=ray,
            )
        )

    return Walk2DResult(
        radius=radius,
        mean_free_path=mean_free_path,
        reference_steps=reference_steps,
        step_cap=step_cap,
        walks=walks,
    )
