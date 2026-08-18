"""
random2_physics.py

Physics and geometry for Random2.

Two step generators are provided for two different purposes:

    generate_component_step()
        Used by the scaled-distance experiment. The x, y, and z components
        are independent random variables. The default distribution is
        uniform on [-1, 1], matching the component-based construction
        used in Schutz's original Random program. A Gaussian option is
        also provided for the suggested distribution-comparison
        experiment.

    generate_isotropic_step()
        Used by the 2D star visualization. It chooses a direction
        uniformly on [0, 2*pi) and gives every step the same length,
        mean_free_path. This makes the displayed walk isotropic in the
        plane.

The module also contains the default star-radius relation and the
geometry for locating the exact point at which a walk segment crosses
a circular boundary.
"""

import math
import random
from numbers import Real
from typing import Literal, Optional, Tuple

Point = Tuple[float, float]
ComponentStep3D = Tuple[float, float, float]
StepDistribution = Literal["uniform", "gaussian"]


def _require_positive_finite_number(name: str, value: Real) -> float:
    """Validate and return a positive finite real number."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return numeric


def generate_component_step(
    distribution: StepDistribution = "uniform",
) -> ComponentStep3D:
    """
    Generate one 3D step for the scaled-distance experiment.

    "uniform":
        dx, dy, and dz are independent and uniform on [-1, 1].

    "gaussian":
        dx, dy, and dz are independent standard normal random variables.

    These component-based steps are not isotropic in direction for the
    uniform case; that does not affect the sqrt(N) scaling experiment.
    """
    if distribution == "uniform":
        dx = 2.0 * random.random() - 1.0
        dy = 2.0 * random.random() - 1.0
        dz = 2.0 * random.random() - 1.0
    elif distribution == "gaussian":
        dx = random.gauss(0.0, 1.0)
        dy = random.gauss(0.0, 1.0)
        dz = random.gauss(0.0, 1.0)
    else:
        raise ValueError(
            'distribution must be "uniform" or "gaussian".'
        )
    return dx, dy, dz


def generate_isotropic_step(mean_free_path: float = 1.0) -> Point:
    """
    Generate one fixed-length isotropic step in the plane.

    theta is uniform on [0, 2*pi), so every direction is equally
    probable, and the step length is exactly mean_free_path.
    """
    mean_free_path = _require_positive_finite_number(
        "mean_free_path", mean_free_path
    )

    theta = 2.0 * math.pi * random.random()
    return (
        mean_free_path * math.cos(theta),
        mean_free_path * math.sin(theta),
    )


def default_radius(
    reference_steps: int,
    mean_free_path: float = 1.0,
    radius_factor: float = 2.0,
) -> float:
    """
    Return the default radius for the 2D star display.

        radius = radius_factor * mean_free_path * sqrt(reference_steps)

    For a fixed-length isotropic random walk, the RMS displacement after
    N steps is mean_free_path * sqrt(N). The radius_factor therefore
    chooses a multiple of that characteristic diffusion distance.
    """
    if not isinstance(reference_steps, int) or isinstance(reference_steps, bool):
        raise ValueError("reference_steps must be a positive integer.")
    if reference_steps <= 0:
        raise ValueError("reference_steps must be a positive integer.")

    mean_free_path = _require_positive_finite_number(
        "mean_free_path", mean_free_path
    )
    radius_factor = _require_positive_finite_number(
        "radius_factor", radius_factor
    )

    return radius_factor * mean_free_path * math.sqrt(reference_steps)


def circle_crossing_fraction(
    p0: Point,
    p1: Point,
    radius: float,
) -> Optional[float]:
    """
    Given a straight segment from p0 (inside a circle centered at the
    origin) to p1, return the first fraction t in (0, 1] for which

        |p0 + t*(p1-p0)| = radius.

    Return None if the segment has no crossing in (0, 1].
    """
    radius = _require_positive_finite_number("radius", radius)

    x0, y0 = p0
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]

    # |p0 + t*d|^2 = radius^2  =>  a*t^2 + b*t + c = 0
    a = dx * dx + dy * dy
    b = 2.0 * (x0 * dx + y0 * dy)
    c = x0 * x0 + y0 * y0 - radius * radius

    if a == 0.0:
        return None

    disc = b * b - 4.0 * a * c
    # Allow tiny negative values from floating-point roundoff.
    if disc < 0.0:
        if disc > -1.0e-12 * max(1.0, b * b, abs(4.0 * a * c)):
            disc = 0.0
        else:
            return None

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    for t in (t1, t2):
        if 0.0 < t <= 1.0:
            return t
    return None


def point_at(p0: Point, p1: Point, t: float) -> Point:
    """Return the linearly interpolated point p0 + t*(p1-p0)."""
    return (
        p0[0] + t * (p1[0] - p0[0]),
        p0[1] + t * (p1[1] - p0[1]),
    )
