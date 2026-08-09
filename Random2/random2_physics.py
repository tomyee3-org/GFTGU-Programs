"""
random2_physics.py

Physics for Random2: a 2D random walk, following the same step-generation
convention as Schutz's Random program (Investigation 8.7) -- each step's
components are drawn uniformly in [-1, 1] -- but restricted to two
dimensions so the walk can be drawn directly on a plane, representing
(schematically) a photon random-walking outward from a star's center.

No plotting or driver-loop logic lives here -- only the physics of a
single step and the geometry of where a straight segment crosses a
circular boundary (the "surface" of the star).
"""

import math
import random
from typing import Optional, Tuple

Point = Tuple[float, float]


def generate_2d_step() -> Point:
    """
    One random step (dx, dy), each component uniform in [-1, 1],
    matching the same 2 * random.random() - 1 convention used in the
    original Random program's 3D walk.
    """
    dx = 2.0 * random.random() - 1.0
    dy = 2.0 * random.random() - 1.0
    return dx, dy


def default_radius(n_steps: int, mean_free_path: float = 1.0,
                    radius_factor: float = 2.0) -> float:
    """
    Default star radius, following the scheme described for the
    original 2D visualization this program is modeled on:
        radius = radius_factor * sqrt(n_steps * mean_free_path)
    i.e. a small multiple of the walk's characteristic (RMS) diffusion
    distance for a walk of n_steps steps with the given mean free path.
    Because a walk's actual net distance is itself a random quantity,
    using radius_factor > 1 (default 2) means only a minority of
    n_steps-step walks will typically reach the surface -- by design,
    so that a batch of walks shows a mix of escaped (ray-producing) and
    still-wandering paths.
    """
    return radius_factor * math.sqrt(n_steps * mean_free_path)


def circle_crossing_fraction(p0: Point, p1: Point, radius: float) -> Optional[float]:
    """
    Given a straight segment from p0 (assumed strictly inside the
    circle of the given radius, centered at the origin) to p1, find the
    fraction t in (0, 1] along the segment where the path first crosses
    |p(t)| = radius, with p(t) = p0 + t*(p1 - p0).

    Returns None if the segment does not cross the circle at all
    (i.e. p1 is also inside), which the caller should check for before
    calling this (see driver: only called when p1 is outside).
    """
    x0, y0 = p0
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]

    # |p0 + t*d|^2 = radius^2  =>  a t^2 + b t + c = 0
    a = dx * dx + dy * dy
    b = 2.0 * (x0 * dx + y0 * dy)
    c = x0 * x0 + y0 * y0 - radius * radius

    if a == 0.0:
        return None

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # We want the first crossing at t in (0, 1]; t1 <= t2 always.
    for t in (t1, t2):
        if 0.0 < t <= 1.0:
            return t
    return None


def point_at(p0: Point, p1: Point, t: float) -> Point:
    """Linear interpolation p0 + t*(p1 - p0)."""
    return (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))
