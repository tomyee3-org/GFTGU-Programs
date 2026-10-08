"""
Integration driver for Multiple.

The numerical integration uses an adaptive predictor plus iterated trapezoidal
corrector. Animation frames are sampled at uniform PHYSICAL simulation times,
so adaptive timestep changes do not distort movie playback speed.

Optional massless test particles are advanced with the same predictor and
corrector after each accepted step of the massive bodies. They feel the
massive bodies, pull on nothing, and never influence the massive bodies'
timestep, so the massive bodies' motion is identical with or without them.
The trapezoidal corrector needs the force only at the two ends of a step,
so a test particle's corrector uses the massive bodies' accepted end
positions. A particle whose corrector does not converge, whose acceleration
changes too much, or whose straight path crosses a removal radius has its
own step halved, with the massive bodies' positions inside the step taken
from cubic-Hermite interpolation, and removal is checked at the end of every
such substep.
"""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import List, Dict, Any, Optional, Union

import numpy as np

import physics_multiple as phys
from physics_multiple import compute_accelerations, conservation_state


MAX_ANIMATION_FRAMES = 1_000_000
# Ceiling on stored test-particle states (states times particles) in the
# trajectories and animation modes. Survival mode stores no path history.
MAX_TEST_PARTICLE_STATES = 5_000_000
# Test-particle step refinement: a particle's step is halved while its
# acceleration changes by more than eps1 across the step (the same test the
# massive bodies pass), its corrector does not reach eps2, or its path
# crosses a removal radius, down to 1/2**MAX_TEST_SUBSTEP_DEPTH of the
# massive bodies' step.
MAX_TEST_SUBSTEP_DEPTH = 16
ENERGY_CANCELLATION_TOLERANCE = 128.0 * np.finfo(float).eps


@dataclass
class SimulationParams:
    n_bodies: int
    masses_solar: List[float]
    positions_init: List[List[float]]
    velocities_init: List[List[float]]

    dt: float
    max_steps: int
    output_type: str                   # "trajectories", "animation" or "survival"
    eps1: float
    eps2: float

    animation_mode: str = "trails"             # or "current positions"
    frame_time: float = 2.0e5          # simulated seconds between frames
    frame_interval_ms: int = 50        # real milliseconds between frames
    trail_time: float = 6.0e5          # simulated seconds shown behind each body
    projection: str = "xy"             # "xy", "xz", or "yz"
    axis_mode: str = "fixed"           # "fixed" or "auto"
    display_frame: str = "com"         # "com" or "user"

    # Optional massless test particles. They feel the massive bodies but
    # exert no force, so they never change the massive bodies' motion.
    test_positions_init: Optional[List[List[float]]] = None
    test_velocities_init: Optional[List[List[float]]] = None
    # Removal criteria checked after every accepted step. A particle within
    # removal_radii[k] of body k is removed as an "encounter" (use a star's
    # radius for a collision, or a planet's Hill radius for a close approach).
    removal_radii: Optional[List[float]] = None   # [m], one per massive body
    escape_radius: Optional[float] = None         # [m], from the massive COM
    # Reference for each particle's starting distance, a and e: "com" or a
    # 1-based massive-body number.
    test_center: Union[str, int] = "com"


def _validate_params(params: SimulationParams) -> None:
    if not isinstance(params.n_bodies, Integral) or isinstance(params.n_bodies, bool):
        raise ValueError("n_bodies must be an integer.")
    if params.n_bodies < 2:
        raise ValueError("n_bodies must be at least 2.")

    try:
        masses = np.asarray(params.masses_solar, dtype=float)
        positions = np.asarray(params.positions_init, dtype=float)
        velocities = np.asarray(params.velocities_init, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Masses, positions, and velocities must contain numeric values."
        ) from exc

    if masses.shape != (params.n_bodies,):
        raise ValueError("masses_solar must contain exactly n_bodies values.")
    if positions.shape != (params.n_bodies, 3):
        raise ValueError(
            "positions_init must contain exactly n_bodies three-component vectors."
        )
    if velocities.shape != (params.n_bodies, 3):
        raise ValueError(
            "velocities_init must contain exactly n_bodies three-component vectors."
        )

    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("All masses must be finite and positive.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("All initial positions must be finite.")
    if not np.all(np.isfinite(velocities)):
        raise ValueError("All initial velocities must be finite.")

    if (
        not isinstance(params.dt, Real)
        or isinstance(params.dt, bool)
        or not np.isfinite(params.dt)
        or params.dt <= 0.0
    ):
        raise ValueError("dt must be a positive finite number.")
    if (
        not isinstance(params.max_steps, Integral)
        or isinstance(params.max_steps, bool)
        or params.max_steps <= 0
    ):
        raise ValueError("max_steps must be a positive integer.")

    if (
        not isinstance(params.eps1, Real)
        or isinstance(params.eps1, bool)
        or not np.isfinite(params.eps1)
        or not (0.0 < params.eps1 < 1.0)
    ):
        raise ValueError("eps1 must satisfy 0 < eps1 < 1.")
    if (
        not isinstance(params.eps2, Real)
        or isinstance(params.eps2, bool)
        or not np.isfinite(params.eps2)
        or not (0.0 < params.eps2 < params.eps1)
    ):
        raise ValueError("eps2 must satisfy 0 < eps2 < eps1.")

    if not isinstance(params.output_type, str):
        raise ValueError(
            'output_type must be "trajectories", "animation" or "survival".'
        )
    if params.output_type.lower() not in ("trajectories", "animation", "survival"):
        raise ValueError(
            'output_type must be "trajectories", "animation" or "survival".'
        )

    # Projection applies to both output modes.
    if not isinstance(params.projection, str):
        raise ValueError('projection must be "xy", "xz", or "yz".')
    if params.projection.lower() not in ("xy", "xz", "yz"):
        raise ValueError('projection must be "xy", "xz", or "yz".')

    # Display frame applies to both output modes. Integration always uses the
    # user-supplied inertial coordinates; "com" only recenters the plot.
    if not isinstance(params.display_frame, str):
        raise ValueError('display_frame must be "com" or "user".')
    if params.display_frame.lower() not in ("com", "user"):
        raise ValueError('display_frame must be "com" or "user".')

    # The remaining display controls matter only for animation.
    if params.output_type.lower() == "animation":
        if not isinstance(params.animation_mode, str):
            raise ValueError(
                'animation_mode must be "current positions" or "trails".'
            )
        if params.animation_mode.lower() not in ("current positions", "trails"):
            raise ValueError(
                'animation_mode must be "current positions" or "trails".'
            )
        if (
            not isinstance(params.frame_time, Real)
            or isinstance(params.frame_time, bool)
            or not np.isfinite(params.frame_time)
            or params.frame_time <= 0.0
        ):
            raise ValueError("frame_time must be a positive finite number.")
        if (
            not isinstance(params.frame_interval_ms, Integral)
            or isinstance(params.frame_interval_ms, bool)
            or params.frame_interval_ms <= 0
        ):
            raise ValueError("frame_interval_ms must be a positive integer.")
        if (
            not isinstance(params.trail_time, Real)
            or isinstance(params.trail_time, bool)
            or not np.isfinite(params.trail_time)
            or params.trail_time < 0.0
        ):
            raise ValueError("trail_time must be finite and non-negative.")
        if not isinstance(params.axis_mode, str):
            raise ValueError('axis_mode must be "fixed" or "auto".')
        if params.axis_mode.lower() not in ("fixed", "auto"):
            raise ValueError('axis_mode must be "fixed" or "auto".')

        maximum_duration = float(params.dt) * int(params.max_steps)
        maximum_frames = maximum_duration / float(params.frame_time) + 1.0
        if not np.isfinite(maximum_frames) or maximum_frames > MAX_ANIMATION_FRAMES:
            raise ValueError(
                "The requested animation could require more than "
                f"{MAX_ANIMATION_FRAMES:,} stored frames. Increase frame_time "
                "or reduce dt or max_steps."
            )

    _validate_test_particle_params(params, positions, velocities, masses)


def _test_particle_arrays(params: SimulationParams):
    """Return (positions, velocities) of the test particles, possibly empty."""
    if params.test_positions_init is None and params.test_velocities_init is None:
        return np.zeros((0, 3)), np.zeros((0, 3))
    if params.test_positions_init is None or params.test_velocities_init is None:
        raise ValueError(
            "test_positions_init and test_velocities_init must be given together."
        )
    try:
        test_pos = np.asarray(params.test_positions_init, dtype=float)
        test_vel = np.asarray(params.test_velocities_init, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Test-particle positions and velocities must contain numeric values."
        ) from exc
    if test_pos.size == 0 and test_vel.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    if test_pos.ndim != 2 or test_pos.shape[1] != 3:
        raise ValueError(
            "test_positions_init must be a list of three-component vectors."
        )
    if test_vel.shape != test_pos.shape:
        raise ValueError(
            "test_velocities_init must contain one three-component vector per "
            "test particle."
        )
    if not (np.all(np.isfinite(test_pos)) and np.all(np.isfinite(test_vel))):
        raise ValueError("All test-particle positions and velocities must be finite.")
    return test_pos, test_vel


def _normalized_test_center(value, n_bodies: int):
    """Return "com" or a 0-based massive-body index."""
    if isinstance(value, str):
        if value.lower() == "com":
            return "com"
        try:
            value = int(value)
        except ValueError:
            raise ValueError(
                'test_center must be "com" or a body number from 1 to n_bodies.'
            ) from None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            'test_center must be "com" or a body number from 1 to n_bodies.'
        )
    if not 1 <= int(value) <= n_bodies:
        raise ValueError(
            'test_center must be "com" or a body number from 1 to n_bodies.'
        )
    return int(value) - 1


def _validate_test_particle_params(params, positions, velocities, masses) -> None:
    test_pos, _ = _test_particle_arrays(params)
    n_test = test_pos.shape[0]
    output_type = params.output_type.lower()
    if output_type == "survival" and n_test == 0:
        raise ValueError("output_type 'survival' needs at least one test particle.")

    _normalized_test_center(params.test_center, params.n_bodies)

    radii = _removal_radii(params)
    if params.escape_radius is not None:
        value = params.escape_radius
        if (
            not isinstance(value, Real) or isinstance(value, bool)
            or not np.isfinite(value) or value <= 0.0
        ):
            raise ValueError("escape_radius must be a positive finite number.")

    if n_test:
        # A particle that starts inside a removal radius or beyond the escape
        # radius is removed at t = 0 by run_simulation. Only a particle exactly
        # on a body with no removal radius is singular and therefore an error.
        separation = positions[None, :, :] - test_pos[:, None, :]
        distance = np.hypot.reduce(separation, axis=2)
        if np.any((distance == 0.0) & (radii[None, :] == 0.0)):
            raise ValueError("A test particle starts exactly on a massive body.")

        if output_type == "trajectories":
            stored = (int(params.max_steps) + 1) * n_test
            if stored > MAX_TEST_PARTICLE_STATES:
                raise ValueError(
                    "Storing every test-particle state could exceed "
                    f"{MAX_TEST_PARTICLE_STATES:,} particle-states. Use "
                    "output_type survival, fewer particles, or fewer steps."
                )
        elif output_type == "animation":
            frames = float(params.dt) * int(params.max_steps) / float(params.frame_time) + 1.0
            if frames * n_test > MAX_TEST_PARTICLE_STATES:
                raise ValueError(
                    "The animation could store more than "
                    f"{MAX_TEST_PARTICLE_STATES:,} test-particle frame states. "
                    "Increase frame_time, use fewer particles, or use "
                    "output_type survival."
                )


def _removal_radii(params: SimulationParams) -> np.ndarray:
    """Return one non-negative removal radius per massive body."""
    if params.removal_radii is None:
        return np.zeros(params.n_bodies)
    try:
        radii = np.asarray(params.removal_radii, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("removal_radii must contain numeric values.") from exc
    if radii.shape != (params.n_bodies,):
        raise ValueError("removal_radii must contain exactly n_bodies values.")
    if not np.all(np.isfinite(radii)) or np.any(radii < 0.0):
        raise ValueError("Removal radii must be finite and non-negative.")
    return radii


def _max_relative_vector_change(old: np.ndarray, new: np.ndarray) -> float:
    """
    Maximum relative change among body vectors.

    old and new have shape (n_bodies, 3). Each body is tested independently so
    a close encounter involving one body cannot be diluted by quiet bodies.
    """
    if not (np.all(np.isfinite(old)) and np.all(np.isfinite(new))):
        return float("inf")

    changes = np.hypot.reduce(new - old, axis=1)
    scales = np.maximum(
        np.hypot.reduce(old, axis=1),
        np.hypot.reduce(new, axis=1),
    )

    ratios = np.zeros_like(changes)
    nonzero = scales > 0.0
    ratios[nonzero] = changes[nonzero] / scales[nonzero]
    ratios[(~nonzero) & (changes > 0.0)] = np.inf
    return float(np.max(ratios))


def _hermite_state(
    t0: float,
    p0: np.ndarray,
    v0: np.ndarray,
    t1: float,
    p1: np.ndarray,
    v1: np.ndarray,
    target_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cubic-Hermite dense interpolation between two accepted states.

    Endpoint positions and velocities are matched exactly. This is used only for
    display sampling, not to advance the physical integration.
    """
    h = t1 - t0
    if h <= 0.0:
        return p1.copy(), v1.copy()

    s = (target_time - t0) / h
    s = min(1.0, max(0.0, s))

    h00 = 2*s**3 - 3*s**2 + 1
    h10 = s**3 - 2*s**2 + s
    h01 = -2*s**3 + 3*s**2
    h11 = s**3 - s**2

    pos = h00*p0 + h10*h*v0 + h01*p1 + h11*h*v1

    dh00 = (6*s**2 - 6*s) / h
    dh10 = 3*s**2 - 4*s + 1
    dh01 = (-6*s**2 + 6*s) / h
    dh11 = 3*s**2 - 2*s

    vel = dh00*p0 + dh10*v0 + dh01*p1 + dh11*v1
    return pos, vel


def _fractional_scalar_drift(
    value: float,
    reference: float,
    scale: Optional[float] = None,
) -> float:
    if scale is None:
        scale = abs(reference)
    if scale == 0.0:
        return abs(value - reference)
    return abs(value - reference) / scale


def _energy_drift_scale(
    kinetic: float,
    potential: float,
) -> tuple[float, str]:
    """Choose a stable, dimensionally consistent energy-error denominator."""
    characteristic = kinetic + abs(potential)
    if not np.isfinite(characteristic) or characteristic <= 0.0:
        raise ValueError(
            "The characteristic energy scale is outside the floating-point "
            "range."
        )

    total = kinetic + potential
    if abs(total) <= ENERGY_CANCELLATION_TOLERANCE * characteristic:
        return characteristic, "characteristic_energy"
    return abs(total), "initial_energy"


def _vector_drift(
    value: np.ndarray,
    reference: np.ndarray,
    scale: Optional[float] = None,
) -> float:
    """
    Return the norm of (value - reference), optionally divided by `scale`.

    `scale` should be the same scale chosen once, at the initial state, by
    `_vector_drift_metadata` -- either the initial vector's own norm, a
    near-cancellation-safe characteristic scale, or None for an absolute
    (unscaled) drift. Recomputing a fresh denominator from `reference` at
    every step would defeat the point of choosing a stable scale up front.
    """
    if not (
        np.all(np.isfinite(value))
        and np.all(np.isfinite(reference))
    ):
        raise ValueError("Conservation vectors must contain finite values.")

    with np.errstate(over="ignore", invalid="ignore"):
        difference = value - reference
    if not np.all(np.isfinite(difference)):
        raise ValueError(
            "Conservation-vector drift is outside the floating-point range."
        )

    diff = float(np.hypot.reduce(difference))
    if not np.isfinite(diff):
        raise ValueError(
            "Conservation-vector norm is outside the floating-point range."
        )
    if scale is None:
        return diff
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError(
            "Conservation-vector drift scale is outside the floating-point "
            "range."
        )
    if scale == 0.0:
        return diff
    drift = diff / scale
    if not np.isfinite(drift):
        raise ValueError(
            "Conservation-vector drift is outside the floating-point range."
        )
    return drift


def _vector_drift_metadata(
    reference: np.ndarray,
    characteristic: Optional[float] = None,
) -> tuple[Optional[float], str]:
    """
    Return the vector-drift scale and its public normalization label.

    `characteristic` is the sum of individual-body magnitudes (from
    `scaled_characteristic_momentum` or
    `scaled_characteristic_angular_momentum`), which is always at least as
    large as the norm of `reference` itself. When the total is small only
    because individually nonzero contributions nearly or exactly cancel --
    the same kind of near-cancellation `_energy_drift_scale` already
    detects for energy -- using `characteristic` as the scale gives a
    stable fractional drift instead of an unscaled, hard-to-interpret
    absolute one. The absolute fallback is reserved for the case where
    there is genuinely no such quantity to divide by (`characteristic` is
    zero, absent, or every body is exactly still).
    """
    if not np.all(np.isfinite(reference)):
        raise ValueError("Conservation vectors must contain finite values.")
    scale = float(np.hypot.reduce(reference))
    if not np.isfinite(scale):
        raise ValueError(
            "Conservation-vector norm is outside the floating-point range."
        )
    if characteristic is not None:
        if not np.isfinite(characteristic) or characteristic < 0.0:
            raise ValueError(
                "Conservation-vector characteristic scale is outside the "
                "floating-point range."
            )
        if characteristic > 0.0 and scale <= ENERGY_CANCELLATION_TOLERANCE * characteristic:
            return characteristic, "characteristic_scale"
    if scale == 0.0:
        return None, "absolute_scaled"
    return scale, "initial_norm"


def _checked_maximum_drift(
    current_maximum: float,
    candidate: float,
    quantity: str,
) -> float:
    """Update a maximum without silently swallowing a NaN diagnostic."""
    if not np.isfinite(candidate):
        raise RuntimeError(f"Non-finite {quantity} drift was produced.")
    return max(current_maximum, candidate)


def _row_relative_change(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Relative change of each row vector; inf where it cannot be judged."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        changes = np.hypot.reduce(new - old, axis=1)
        scales = np.maximum(np.hypot.reduce(old, axis=1),
                            np.hypot.reduce(new, axis=1))
        ratio = np.where(scales > 0.0, changes / scales,
                         np.where(changes > 0.0, np.inf, 0.0))
    ratio[~np.isfinite(ratio)] = np.inf
    return ratio


def _test_particle_trial(p0, v0, acc_a, massive_end, masses, h, eps2,
                         max_iterations):
    """One predictor-corrector trial for each test particle, row by row.

    Each row iterates the trapezoidal corrector until its own velocity
    increment changes by less than eps2, and is then frozen, so one hard
    particle never changes the result for the others. Returns the end
    position, end velocity, the end acceleration used in the last
    correction, and a per-row convergence flag.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        p_guess = p0 + v0 * h + 0.5 * acc_a * h * h
        v_guess = v0 + acc_a * h
    acc_end = phys._test_accelerations_unchecked(p_guess, massive_end, masses)
    converged = np.zeros(p0.shape[0], dtype=bool)
    open_rows = np.ones(p0.shape[0], dtype=bool)
    for _ in range(max_iterations):
        rows = np.flatnonzero(open_rows)
        if rows.size == 0:
            break
        with np.errstate(over="ignore", invalid="ignore"):
            v_corr = v0[rows] + 0.5 * (acc_a[rows] + acc_end[rows]) * h
            p_corr = p0[rows] + 0.5 * (v0[rows] + v_corr) * h
        change = _row_relative_change(v_guess[rows] - v0[rows], v_corr - v0[rows])
        p_guess[rows] = p_corr
        v_guess[rows] = v_corr
        done = change < eps2
        converged[rows[done]] = True
        failed = ~np.isfinite(change)
        open_rows[rows[done | failed]] = False
        still = rows[~(done | failed)]
        if still.size:
            acc_end[still] = phys._test_accelerations_unchecked(
                p_guess[still], massive_end, masses)
    return p_guess, v_guess, acc_end, converged


def _segment_closest_approach(ra, rb):
    """Closest approach of the straight segment ra -> rb to the origin.

    ra, rb: shape (n, m, 3), particle positions relative to each massive
    body at the start and end of a (sub)step. Returns (distance, fraction)
    with shape (n, m); fraction is where along the step it occurs.
    """
    d = rb - ra
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        dd = np.sum(d * d, axis=2)
        frac = np.where(dd > 0.0, -np.sum(ra * d, axis=2) / dd, 0.0)
    frac = np.clip(np.nan_to_num(frac, nan=0.0), 0.0, 1.0)
    closest = ra + frac[:, :, None] * d
    return np.hypot.reduce(closest, axis=2), frac


def _advance_test_particles(
    test_pos, test_vel, act, test_acc0, t0, massive0, massive_v0, t1,
    massive1, massive_v1, masses, eps1, eps2, max_iterations, removal_radii,
    escape_radius, active, removal_time, removal_reason, removal_body,
    on_commit=None,
) -> None:
    """Advance the active test particles in place across one accepted step.

    A particle is first tried in one step, with the massive bodies' accepted
    start and end positions supplying the forces. The step is accepted for
    that particle only if its corrector converges (eps2), its acceleration
    changes by no more than eps1 (the same tests the massive bodies pass),
    and the straight path
    between its start and end points relative to each massive body does not
    pass through that body's removal radius. Otherwise the particle's step is
    halved, with the massive bodies' positions inside the step taken from
    the same cubic-Hermite interpolation used for animation frames, down to
    1/2**MAX_TEST_SUBSTEP_DEPTH of the step. Removal is checked at the end of
    every particle (sub)step. At the smallest substep (or when floating-point
    time cannot be halved further) all three tests must still pass; a
    particle that fails any of them there is removed with fate "numerical"
    at the start of that substep, keeping its last trustworthy state. The one
    exception is a straight-line crossing of a removal radius by an
    otherwise accepted substep, which is recorded as an encounter at the
    crossing point. on_commit(indices, positions, velocities, time,
    massive_positions) is called for every accepted (sub)step state, in time
    order for each particle. The massive bodies are not affected.
    """
    def massive_at(t):
        if t == t0:
            return massive0
        if t == t1:
            return massive1
        position, _ = _hermite_state(t0, massive0, massive_v0, t1, massive1,
                                     massive_v1, t)
        return position

    def remove(index, time, reason, body):
        active[index] = False
        removal_time[index] = time
        removal_reason[index] = reason
        removal_body[index] = body

    def advance(idx, ta, tb, acc_a, depth):
        h = tb - ta
        pa, pb = massive_at(ta), massive_at(tb)
        p0, v0 = test_pos[idx], test_vel[idx]
        p, v, acc_b, converged = _test_particle_trial(
            p0, v0, acc_a, pb, masses, h, eps2, max_iterations)
        finite = np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(v), axis=1)
        steady = _row_relative_change(acc_a, acc_b) <= eps1
        with np.errstate(over="ignore", invalid="ignore"):
            ra = p0[:, None, :] - pa[None, :, :]
            rb = p[:, None, :] - pb[None, :, :]
            end_distance = np.hypot.reduce(rb, axis=2)
            inside = end_distance <= removal_radii[None, :]
            closest, frac = _segment_closest_approach(ra, rb)
            crossed = (closest <= removal_radii[None, :]) & ~inside
        crossing = np.any(crossed, axis=1) & ~np.any(inside, axis=1)
        last_level = depth >= MAX_TEST_SUBSTEP_DEPTH or not (ta < ta + 0.5 * h < tb)
        good = finite & converged & steady & ~crossing

        commit = idx[good]
        test_pos[commit] = p[good]
        test_vel[commit] = v[good]
        if on_commit is not None and commit.size:
            on_commit(commit, p[good], v[good], tb, pb)
        any_inside = np.any(inside, axis=1)
        escaped = np.zeros_like(good)
        if escape_radius is not None:
            com_b = np.sum(masses[:, None] * pb, axis=0) / float(np.sum(masses))
            with np.errstate(over="ignore", invalid="ignore"):
                escaped = np.hypot.reduce(p - com_b, axis=1) > escape_radius
        for local in np.flatnonzero(good & (any_inside | escaped)):
            index = idx[local]
            if any_inside[local]:
                hits = np.flatnonzero(inside[local])
                remove(index, tb, "encounter",
                       int(hits[np.argmin(end_distance[local, hits])]) + 1)
            else:
                remove(index, tb, "escaped", 0)

        bad = np.flatnonzero(~good)
        if bad.size == 0:
            return
        if last_level:
            for local in bad:
                index = idx[local]
                if (finite[local] and converged[local] and steady[local]
                        and crossing[local]):
                    hits = np.flatnonzero(crossed[local])
                    body = int(hits[np.argmin(closest[local, hits])])
                    f = frac[local, body]
                    time = ta + f * h
                    test_pos[index] = p0[local] + f * (p[local] - p0[local])
                    test_vel[index] = v0[local] + f * (v[local] - v0[local])
                    if on_commit is not None:
                        on_commit(np.array([index]), test_pos[index][None, :],
                                  test_vel[index][None, :], time, massive_at(time))
                    remove(index, time, "encounter", body + 1)
                else:
                    remove(index, ta, "numerical", 0)
            return
        mid = ta + 0.5 * h
        retry = idx[bad]
        advance(retry, ta, mid, acc_a[bad], depth + 1)
        alive = retry[active[retry]]
        if alive.size:
            acc_mid = phys._test_accelerations_unchecked(
                test_pos[alive], massive_at(mid), masses)
            advance(alive, mid, tb, acc_mid, depth + 1)

    advance(np.asarray(act), t0, t1, test_acc0, 0)


def _remove_test_particles(
    test_pos, test_vel, previous_pos, previous_vel, act, active, positions,
    masses, removal_radii, escape_radius, time, removal_time,
    removal_reason, removal_body,
) -> None:
    """Apply the removal criteria to the particles advanced this step."""
    p = test_pos[act]
    v = test_vel[act]
    finite = np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(v), axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        distance = np.hypot.reduce(positions[None, :, :] - p[:, None, :], axis=2)
        inside = distance <= removal_radii[None, :]
        flagged = ~finite | np.any(inside, axis=1)
        if escape_radius is not None:
            com = (np.sum(masses[:, None] * positions, axis=0)
                   / float(np.sum(masses)))
            from_com = np.hypot.reduce(p - com, axis=1)
            flagged |= from_com > escape_radius
    for local in np.flatnonzero(flagged):
        index = act[local]
        if not finite[local]:
            reason, body = "numerical", 0
            test_pos[index] = previous_pos[index]
            test_vel[index] = previous_vel[index]
        elif np.any(inside[local]):
            hits = np.flatnonzero(inside[local])
            reason = "encounter"
            body = int(hits[np.argmin(distance[local, hits])]) + 1
        else:
            reason, body = "escaped", 0
        active[index] = False
        removal_time[index] = time
        removal_reason[index] = reason
        removal_body[index] = body


def _test_frame_positions(frame_time, n_test, history, removal_time):
    """Test-particle positions at one animation frame time.

    Each particle is interpolated (cubic Hermite) inside the bracket of its
    own accepted substep states that contains frame_time, so a refined path
    is drawn as computed. A particle is NaN at and after its removal time,
    and when it has no bracket (removed earlier, or never active).
    """
    frame = np.full((n_test, 3), np.nan)
    for index, states in history.items():
        removed_at = removal_time[index]
        if np.isfinite(removed_at) and frame_time >= removed_at:
            continue
        times = [state[0] for state in states]
        k = int(np.searchsorted(times, frame_time, side="left"))
        if k == 0:
            if times and times[0] == frame_time:
                frame[index] = states[0][1]
            continue
        if k >= len(states):
            continue
        t_a, p_a, v_a = states[k - 1]
        t_b, p_b, v_b = states[k]
        frame[index], _ = _hermite_state(t_a, p_a, v_a, t_b, p_b, v_b, frame_time)
    return frame


def _phase_reference(positions, masses, test_center) -> int:
    """Return the 0-based massive body that angles are measured from.

    It is the massive body farthest from the ring centre (Jupiter for a
    Sun-Jupiter ring about the centre of mass; the companion star for a ring
    about one star), excluding the centre body itself.
    """
    center = (np.sum(masses[:, None] * positions, axis=0) / float(np.sum(masses))
              if test_center == "com" else positions[test_center])
    offsets = np.hypot.reduce(positions - center, axis=1)
    if test_center != "com":
        offsets[test_center] = -1.0
    return int(np.argmax(offsets))


def _phase_angles(test_pos, positions, masses, test_center, reference):
    """Angle [rad] of each test particle about the centre, from the reference body."""
    if test_center == "com":
        center = np.sum(masses[:, None] * positions, axis=0) / float(np.sum(masses))
    else:
        center = positions[test_center]
    ref_vec = positions[reference] - center
    rel = test_pos - center
    with np.errstate(invalid="ignore"):
        return np.arctan2(rel[:, 1], rel[:, 0]) - np.arctan2(ref_vec[1], ref_vec[0])


def _wrapped(angle):
    """Wrap radians to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


ROUTH_MASS_FRACTION = phys.ROUTH_MASS_FRACTION


# Phase-path evidence: a turning point counts only when the angle has come
# back by at least this much from its most recent extreme, and the swing is
# judged against the whole range finally reached, within the same margin.
TURN_THRESHOLD_DEG = 1.0
# A particle is treated as co-orbital with the reference body only if its
# starting semi-major axis is within this many of the body's Hill radii of
# the body's own semi-major axis.
CO_ORBITAL_HILL_RADII = 2.0


def _track_turns(u, start, heading, extreme, turn_max, turn_min, threshold):
    """Advance the turning-point detector by one new angle per particle.

    heading is 0 until the angle has moved threshold from its start, then
    +1 (rising) or -1 (falling); extreme is the running extreme in that
    direction. A turning maximum (minimum) is recorded only when the angle
    has come back threshold from an extreme reached while rising (falling),
    so the start of the run is never counted as a turning point. Returns the
    updated (heading, extreme, turn_max, turn_min).
    """
    d = np.asarray(heading)
    ext = np.asarray(extreme, dtype=float)
    start_up = (d == 0) & (u - start >= threshold)
    start_down = (d == 0) & (start - u >= threshold)
    rising, falling = d == 1, d == -1
    top = rising & (ext - u >= threshold)
    bottom = falling & (u - ext >= threshold)
    turn_max = np.where(top, np.maximum(turn_max, ext), turn_max)
    turn_min = np.where(bottom, np.minimum(turn_min, ext), turn_min)
    new_d = d.copy()
    new_d[start_up | bottom] = 1
    new_d[start_down | top] = -1
    new_ext = np.where(start_up | start_down | top | bottom, u,
                       np.where(rising, np.maximum(ext, u),
                                np.where(falling, np.minimum(ext, u), ext)))
    return new_d, new_ext, turn_max, turn_min


def swing_complete_for(angles_deg, threshold_deg: float = TURN_THRESHOLD_DEG) -> bool:
    """Replay one particle's sequence of unwrapped angles [deg] through the
    turning-point detector and report whether both ends of the final range
    were reached at observed turning points (within threshold_deg)."""
    angles = np.radians(np.asarray(angles_deg, dtype=float))
    turn = np.radians(threshold_deg)
    start = angles[:1]
    heading, extreme = np.zeros(1, dtype=int), start.copy()
    turn_max, turn_min = np.full(1, -np.inf), np.full(1, np.inf)
    for value in angles[1:]:
        heading, extreme, turn_max, turn_min = _track_turns(
            np.array([value]), start, heading, extreme, turn_max, turn_min, turn)
    low, high = float(np.min(angles)), float(np.max(angles))
    return bool(np.isfinite(turn_max[0]) and np.isfinite(turn_min[0])
                and turn_max[0] >= high - turn and turn_min[0] <= low + turn)


def classify_phase_motion(
    phase_min_deg: float,
    phase_max_deg: float,
    swing_complete: bool = True,
    co_orbital: bool = True,
    sense: float = 1.0,
) -> str:
    """Describe the path traced by a survivor's angle from the reference body.

    phase_min_deg and phase_max_deg are the lowest and highest values of the
    angle, followed continuously. swing_complete says whether both ends of
    that final range were reached at observed turning points (the angle came
    back from each by at least TURN_THRESHOLD_DEG), not at the start or end
    of the run. co_orbital says whether co-orbital names may be used at all
    (planar, nearly circular pair; the reference body light enough for
    stable L4/L5; the particle starting near the body's semi-major axis).
    sense is the pair's orbital direction (+1 counterclockwise from +z).

    "circulating": the angle went all the way round (360 degrees or more).
    "passed body": it crossed the reference body's direction without going
    all the way round.
    "tadpole L4" / "tadpole L5": co-orbital, a full swing that stayed ahead of
    / behind the body in its orbital direction.
    "horseshoe": co-orbital, a full swing that passed the far side of the
    orbit but never the body.
    "unfinished": anything else; this run cannot name the path.
    These describe this run only: a longer run can change them.
    """
    span = phase_max_deg - phase_min_deg
    if not np.isfinite(span):
        return "-"
    if span >= 360.0:
        return "circulating"
    shift = 360.0 * np.floor((phase_min_deg + 180.0) / 360.0)
    low, high = phase_min_deg - shift, phase_max_deg - shift   # low in [-180, 180)
    if low <= 0.0 <= high or high >= 360.0:
        return "passed body"
    if not (swing_complete and co_orbital):
        return "unfinished"
    ahead, behind = ("tadpole L4", "tadpole L5") if sense >= 0.0 else ("tadpole L5", "tadpole L4")
    if 0.0 < low and high < 180.0:
        return ahead
    if high < 0.0 and low > -180.0:
        return behind
    return "horseshoe"


def _test_particle_summary(
    positions0, velocities0, masses, test_center, initial_pos, initial_vel,
    final_pos, final_vel, removal_time, removal_reason, removal_body,
    removal_radii, escape_radius, reference, phase_min, phase_max,
    final_massive, swing_complete,
) -> Dict[str, Any]:
    """Collect each particle's starting orbit and fate."""
    if test_center == "com":
        center_p = phys.center_of_mass(positions0, masses)
        center_v = phys.center_of_mass_velocity(velocities0, masses)
        center_mass = float(np.sum(masses))
        center_label = "com"
    else:
        center_p = positions0[test_center]
        center_v = velocities0[test_center]
        center_mass = float(masses[test_center])
        center_label = f"body {test_center + 1}"
    rel_p = initial_pos - center_p
    rel_v = initial_vel - center_v
    # Two-body elements are undefined for a particle exactly at the centre
    # (possible for the centre of mass, or a particle removed at t = 0 on a
    # body); those rows get NaN instead of losing the whole result.
    semi_major = np.full(rel_p.shape[0], np.nan)
    ecc = np.full(rel_p.shape[0], np.nan)
    defined = np.hypot.reduce(rel_p, axis=1) > 0.0
    if np.any(defined):
        semi_major[defined], ecc[defined] = phys.osculating_elements(
            rel_p[defined], rel_v[defined], center_mass)

    # Starting angle in the x-y plane, measured from the reference body.
    phase = np.degrees(_wrapped(
        _phase_angles(initial_pos, positions0, masses, test_center, reference)))
    # Ending angle and distance of each survivor, in the frame that turns with
    # the reference body (NaN for removed particles).
    survived = np.array([fate == "survived" for fate in removal_reason])
    final_phase = np.degrees(_wrapped(
        _phase_angles(final_pos, final_massive, masses, test_center, reference)))
    if test_center == "com":
        final_center = (np.sum(masses[:, None] * final_massive, axis=0)
                        / float(np.sum(masses)))
    else:
        final_center = final_massive[test_center]
    # Distance in the x-y plane, matching the angle (a view from above).
    final_distance = np.hypot(final_pos[:, 0] - final_center[0],
                              final_pos[:, 1] - final_center[1])
    final_phase[~survived] = np.nan
    final_distance[~survived] = np.nan
    ref_final = (final_massive[reference] - final_center)[:2]
    # Co-orbital names need the planar circular two-primary model, a ring
    # about the pair's centre of mass, the reference body as the light
    # primary below Routh's limit, and each particle starting with a
    # semi-major axis within CO_ORBITAL_HILL_RADII Hill radii of the body's.
    pair = phys.two_body_pair_orbit(positions0, velocities0, masses)
    sense = 1.0 if pair is None else pair["sense"]
    co_orbital_names = bool(
        pair is not None and pair["circular_planar"] and test_center == "com"
        and float(masses[reference]) == float(np.min(masses))
        and pair["minor_mass_fraction"] < ROUTH_MASS_FRACTION
    )
    co_orbital = np.zeros(semi_major.shape[0], dtype=bool)
    if co_orbital_names:
        a_ref = pair["semi_major_axis_m"]
        hill = a_ref * (pair["minor_mass_fraction"] / 3.0) ** (1.0 / 3.0)
        with np.errstate(invalid="ignore"):
            co_orbital = np.abs(semi_major - a_ref) <= CO_ORBITAL_HILL_RADII * hill
    return {
        "center": center_label,
        "center_mass_solar": center_mass,
        "initial_positions": initial_pos,
        "initial_velocities": initial_vel,
        "initial_distance_m": np.hypot.reduce(rel_p, axis=1),
        "initial_semi_major_axis_m": semi_major,
        "initial_eccentricity": ecc,
        "initial_phase_deg": phase,
        "phase_reference_body": reference + 1,
        # Lowest and highest angle from the reference body reached during
        # the run, followed continuously (not wrapped), in degrees.
        "phase_min_deg": np.degrees(phase_min),
        "phase_max_deg": np.degrees(phase_max),
        "final_phase_deg": final_phase,
        "final_distance_m": final_distance,
        "reference_final_distance_m": float(np.hypot.reduce(ref_final)),
        # Only a particle followed for the whole run has a meaningful path.
        "phase_swing_complete": swing_complete.copy(),
        "co_orbital_names": bool(co_orbital_names),
        "co_orbital_start": co_orbital,
        "orbital_sense": sense,
        "phase_motion": [
            classify_phase_motion(lo, hi, bool(swing), bool(coorb), sense)
            if fate == "survived" else "-"
            for lo, hi, swing, coorb, fate in zip(np.degrees(phase_min),
                                                  np.degrees(phase_max),
                                                  swing_complete, co_orbital,
                                                  removal_reason)],
        "final_positions": final_pos.copy(),
        "final_velocities": final_vel.copy(),
        "removal_time_s": removal_time.copy(),
        "fate": list(removal_reason),
        "removal_body": removal_body.copy(),
        "removal_radii_m": removal_radii.copy(),
        "escape_radius_m": escape_radius,
    }


def run_simulation(params: SimulationParams) -> Dict[str, Any]:
    """
    Run the Multiple simulation.

    "trajectories" stores every accepted position and velocity state.

    "animation" stores cubic-Hermite-interpolated frames at uniformly spaced
    physical times t = 0, frame_time, 2*frame_time, ... independent of the
    adaptive integration timestep.

    "survival" stores no path history. It records when and why each test
    particle is removed, and stops early once every particle is gone.

    Test-particle states are stored as NaN after a particle is removed. The
    conservation diagnostics describe the massive bodies only.
    """
    _validate_params(params)

    masses = np.asarray(params.masses_solar, dtype=float)
    positions = np.asarray(params.positions_init, dtype=float)
    velocities = np.asarray(params.velocities_init, dtype=float)

    dt_work = float(params.dt)
    time = 0.0
    output_type = params.output_type.lower()

    max_corrector_iterations = 10
    max_retries_per_step = 80

    initial_cons = conservation_state(positions, velocities, masses)
    energy_drift_scale, energy_drift_normalization = _energy_drift_scale(
        initial_cons["kinetic_energy"],
        initial_cons["potential_energy"],
    )
    momentum_characteristic = phys.scaled_characteristic_momentum(
        velocities, masses
    )
    angular_characteristic = phys.scaled_characteristic_angular_momentum(
        positions, velocities, masses
    )
    momentum_drift_scale, momentum_drift_normalization = (
        _vector_drift_metadata(initial_cons["momentum"], momentum_characteristic)
    )
    angular_drift_scale, angular_drift_normalization = (
        _vector_drift_metadata(
            initial_cons["angular_momentum"], angular_characteristic
        )
    )
    max_energy_drift = 0.0
    max_momentum_drift = 0.0
    max_angular_momentum_drift = 0.0

    # Keep only conservation quantities at accepted states. Animation frames
    # are interpolated separately and need not coincide with these times.
    total_mass = float(np.sum(masses))

    def diagnostic_values(state):
        p = np.asarray(state["momentum"], dtype=float)
        angular = np.asarray(state["angular_momentum"], dtype=float)
        internal = float(state["energy"]) - float(np.dot(p, p)) / (2 * total_mass)
        values = np.array((state["energy"], internal, state["kinetic_energy"],
                           *p, *angular), dtype=float)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Multiple produced a non-finite conservation diagnostic.")
        return values

    conservation_times = [0.0]
    conservation_values = [diagnostic_values(initial_cons)]

    if output_type == "trajectories":
        output_times = [0.0]
        output_positions = [positions.copy()]
        output_velocities = [velocities.copy()]
        energies = [initial_cons["energy"]]
        momenta = [initial_cons["momentum"].copy()]
        angular_momenta = [initial_cons["angular_momentum"].copy()]
        dt_used = [0.0]
    else:
        frame_times = [0.0]
        frame_positions = [positions.copy()]
        frame_velocities = [velocities.copy()]
        next_frame_index = 1
        next_frame_time = next_frame_index * params.frame_time

    # Massless test particles. They are advanced after each accepted step of
    # the massive bodies, using the massive bodies' start and end positions,
    # so they can never alter the massive bodies' motion or the timestep.
    test_pos, test_vel = _test_particle_arrays(params)
    n_test = test_pos.shape[0]
    has_tests = n_test > 0
    removal_radii = _removal_radii(params)
    escape_radius = params.escape_radius
    test_center = _normalized_test_center(params.test_center, params.n_bodies)
    active = np.ones(n_test, dtype=bool)
    removal_time = np.full(n_test, np.nan)
    removal_reason = ["survived"] * n_test
    removal_body = np.zeros(n_test, dtype=int)
    test_initial_pos = test_pos.copy()
    test_initial_vel = test_vel.copy()
    if has_tests:
        phase_reference = _phase_reference(positions, masses, test_center)
        phase_last = _wrapped(_phase_angles(
            test_pos, positions, masses, test_center, phase_reference))
        phase_unwrapped = phase_last.copy()
        phase_min = phase_last.copy()
        phase_max = phase_last.copy()
        # Turning-point detector for each particle's angle: direction of
        # travel (0 until the angle has moved TURN_THRESHOLD_DEG from its
        # start), the running extreme in that direction, and the highest
        # turning maximum and lowest turning minimum observed.
        turn = np.radians(TURN_THRESHOLD_DEG)
        phase_start = phase_last.copy()
        heading = np.zeros(n_test, dtype=int)
        running_extreme = phase_last.copy()
        turn_max = np.full(n_test, -np.inf)
        turn_min = np.full(n_test, np.inf)
        frame_history = {} if output_type == "animation" else None

        def on_commit(indices, positions_now, velocities_now, time_now, massive_now):
            """Follow angles (and animation history) at every accepted substep."""
            now = _phase_angles(positions_now, massive_now, masses, test_center,
                                phase_reference)
            phase_unwrapped[indices] += _wrapped(now - phase_last[indices])
            phase_last[indices] = now
            u = phase_unwrapped[indices]
            phase_max[indices] = np.maximum(phase_max[indices], u)
            phase_min[indices] = np.minimum(phase_min[indices], u)
            (heading[indices], running_extreme[indices], turn_max[indices],
             turn_min[indices]) = _track_turns(
                u, phase_start[indices], heading[indices], running_extreme[indices],
                turn_max[indices], turn_min[indices], turn)
            if frame_history is not None:
                for k, index in enumerate(indices):
                    frame_history.setdefault(int(index), []).append(
                        (time_now, positions_now[k].copy(), velocities_now[k].copy()))
        # Particles that start inside a removal radius or beyond the escape
        # radius are removed at t = 0, so a dense ring never fails to start.
        _remove_test_particles(
            test_pos, test_vel, test_pos.copy(), test_vel.copy(),
            np.arange(n_test), active, positions, masses, removal_radii,
            escape_radius, 0.0, removal_time, removal_reason, removal_body,
        )

    def _masked(array):
        shown = array.copy()
        shown[~active] = np.nan
        return shown

    if has_tests:
        if output_type == "trajectories":
            test_out_positions = [_masked(test_pos)]
            test_out_velocities = [_masked(test_vel)]
        elif output_type == "animation":
            test_frame_positions = [_masked(test_pos)]

    accepted_steps = 0
    current_cons = initial_cons

    # A survival scan whose particles were all removed at t = 0 has nothing
    # to follow: return the initial state with zero steps and zero time.
    while accepted_steps < params.max_steps and not (
        output_type == "survival" and not np.any(active)
    ):
        acc0 = compute_accelerations(positions, masses)
        if has_tests:
            act = np.flatnonzero(active)
            test_acc0 = phys._test_accelerations_unchecked(
                test_pos[act], positions, masses
            )
        accepted = False

        for _retry in range(max_retries_per_step):
            # Predictor.
            pos_pred = (
                positions
                + velocities * dt_work
                + 0.5 * acc0 * dt_work * dt_work
            )
            vel_pred = velocities + acc0 * dt_work

            if not (
                np.all(np.isfinite(pos_pred))
                and np.all(np.isfinite(vel_pred))
            ):
                dt_work *= 0.5
                continue

            # pos_pred is a *trial* configuration, not the accepted state:
            # a predictor overshoot can drive two bodies onto exactly the
            # same point even when the accepted state (used for acc0) was
            # never singular. compute_accelerations() raises ValueError for
            # that, same as it would for the genuinely singular initial or
            # accepted state -- so treat only a trial-step ValueError as a
            # rejected step (halve dt and retry), the same response already
            # given to a non-finite trial acceleration below.
            try:
                acc_pred = compute_accelerations(pos_pred, masses)
            except ValueError:
                dt_work *= 0.5
                continue
            if not np.all(np.isfinite(acc_pred)):
                dt_work *= 0.5
                continue

            # eps1: test each body's acceleration vector independently.
            if _max_relative_vector_change(acc0, acc_pred) > params.eps1:
                dt_work *= 0.5
                continue

            # Iterated trapezoidal corrector.
            pos_guess = pos_pred
            vel_guess = vel_pred
            acc_end = acc_pred
            converged = False

            for _ in range(max_corrector_iterations):
                vel_corr = velocities + 0.5 * (acc0 + acc_end) * dt_work
                pos_corr = positions + 0.5 * (velocities + vel_corr) * dt_work

                if not (
                    np.all(np.isfinite(pos_corr))
                    and np.all(np.isfinite(vel_corr))
                ):
                    break

                # Compare velocity *increments*, not absolute coordinate
                # velocities. This makes the convergence decision invariant
                # under a uniform Galilean boost of the entire system.
                velocity_change = _max_relative_vector_change(
                    vel_guess - velocities,
                    vel_corr - velocities,
                )

                pos_guess = pos_corr
                vel_guess = vel_corr

                if velocity_change < params.eps2:
                    # acc_end belongs to the position estimate used to form
                    # pos_corr, not to the newly accepted pos_corr itself. It
                    # therefore cannot safely be cached as the next step's
                    # starting acceleration without another force evaluation.
                    converged = True
                    break

                # pos_guess is also a trial (uncommitted) corrector
                # position; a singular ValueError here is the same kind of
                # rejected-iterate outcome as the non-finite check below,
                # not a genuine singularity of the accepted state.
                try:
                    acc_end = compute_accelerations(pos_guess, masses)
                except ValueError:
                    break
                if not np.all(np.isfinite(acc_end)):
                    break

            if not converged:
                dt_work *= 0.5
                continue

            accepted = True
            break

        if not accepted:
            raise RuntimeError(
                "Multiple could not find a converged timestep after "
                f"{max_retries_per_step} retries. A near-collision or extreme "
                "initial condition may require different parameters."
            )

        previous_time = time
        previous_positions = positions.copy()
        previous_velocities = velocities.copy()

        new_time = time + dt_work
        if new_time <= time:
            raise RuntimeError(
                "The adaptive timestep became too small to advance simulation "
                "time in floating-point arithmetic."
            )
        time = new_time
        positions = pos_guess
        velocities = vel_guess
        accepted_steps += 1

        if has_tests:
            previous_test_pos = test_pos.copy()
            previous_test_vel = test_vel.copy()
            active_before = active.copy()
            if frame_history is not None:
                frame_history.clear()
                for index in act:
                    frame_history[int(index)] = [
                        (previous_time, previous_test_pos[index].copy(),
                         previous_test_vel[index].copy())]
            if act.size:
                _advance_test_particles(
                    test_pos, test_vel, act, test_acc0,
                    previous_time, previous_positions, previous_velocities,
                    time, positions, velocities, masses, params.eps1, params.eps2,
                    max_corrector_iterations, removal_radii, escape_radius,
                    active, removal_time, removal_reason, removal_body,
                    on_commit,
                )

        if not (
            np.all(np.isfinite(positions))
            and np.all(np.isfinite(velocities))
        ):
            raise RuntimeError(
                "Multiple produced a non-finite position or velocity. "
                "Try a smaller dt or less extreme initial conditions."
            )

        current_cons = conservation_state(positions, velocities, masses)
        if not (
            np.isfinite(current_cons["energy"])
            and np.all(np.isfinite(current_cons["momentum"]))
            and np.all(np.isfinite(current_cons["angular_momentum"]))
        ):
            raise RuntimeError(
                "Multiple produced a non-finite conservation diagnostic. "
                "Try a smaller dt or less extreme initial conditions."
            )
        conservation_times.append(time)
        conservation_values.append(diagnostic_values(current_cons))

        max_energy_drift = _checked_maximum_drift(
            max_energy_drift,
            _fractional_scalar_drift(
                current_cons["energy"],
                initial_cons["energy"],
                energy_drift_scale,
            ),
            "energy",
        )
        max_momentum_drift = _checked_maximum_drift(
            max_momentum_drift,
            _vector_drift(
                current_cons["momentum"], initial_cons["momentum"],
                momentum_drift_scale,
            ),
            "momentum",
        )
        max_angular_momentum_drift = _checked_maximum_drift(
            max_angular_momentum_drift,
            _vector_drift(
                current_cons["angular_momentum"],
                initial_cons["angular_momentum"],
                angular_drift_scale,
            ),
            "angular-momentum",
        )

        if output_type == "trajectories":
            output_times.append(time)
            output_positions.append(positions.copy())
            output_velocities.append(velocities.copy())
            energies.append(current_cons["energy"])
            momenta.append(current_cons["momentum"].copy())
            angular_momenta.append(
                current_cons["angular_momentum"].copy()
            )
            dt_used.append(dt_work)
            if has_tests:
                test_out_positions.append(_masked(test_pos))
                test_out_velocities.append(_masked(test_vel))
        elif output_type == "animation":
            # One accepted step can cross multiple requested frame times.
            while next_frame_time <= time:
                if not previous_time <= next_frame_time <= time:
                    raise RuntimeError(
                        "An animation frame time fell outside its accepted-step "
                        "interpolation bracket."
                    )
                p_frame, v_frame = _hermite_state(
                    previous_time,
                    previous_positions,
                    previous_velocities,
                    time,
                    positions,
                    velocities,
                    next_frame_time,
                )
                frame_positions.append(p_frame)
                frame_velocities.append(v_frame)
                frame_times.append(next_frame_time)
                if has_tests:
                    test_frame_positions.append(_test_frame_positions(
                        next_frame_time, n_test, frame_history,
                        removal_time))
                if len(frame_times) > MAX_ANIMATION_FRAMES:
                    raise RuntimeError(
                        "The animation exceeded the stored-frame safety limit. "
                        "Increase frame_time or reduce the simulated duration."
                    )
                next_frame_index += 1
                next_frame_time = next_frame_index * params.frame_time

        # Gradually recover after close encounters, never exceeding user dt.
        dt_work = min(dt_work * 1.1, params.dt)


    history_times = np.asarray(conservation_times, dtype=float)
    history_values = np.stack(conservation_values)
    conservation_samples = []
    for index in range(11):
        target = time * index / 10.0
        right = int(np.searchsorted(history_times, target, side="left"))
        if right == 0:
            values = history_values[0]
        elif right == len(history_times):
            values = history_values[-1]
        elif history_times[right] == target:
            values = history_values[right]
        else:
            left = right - 1
            neighbours = [i for i in (left - 1, right + 1)
                          if 0 <= i < len(history_times)]
            if neighbours:
                third = min(neighbours, key=lambda i: abs(history_times[i] - target))
                indices = (left, right, third)
                # Local offsets avoid products of large absolute times.
                offsets = history_times[list(indices)] - target
                values = np.zeros_like(history_values[0])
                for k, node in enumerate(indices):
                    others = [j for j in range(3) if j != k]
                    numerator = offsets[others[0]] * offsets[others[1]]
                    denominator = ((offsets[k] - offsets[others[0]]) *
                                   (offsets[k] - offsets[others[1]]))
                    values += (numerator / denominator) * history_values[node]
            else:
                # A one-step run supplies only two accepted states.
                weight = ((target - history_times[left]) /
                          (history_times[right] - history_times[left]))
                values = history_values[left] + weight * (
                    history_values[right] - history_values[left])
        conservation_samples.append({
            "fraction": index / 10.0, "time": target,
            "energy": float(values[0]), "internal_energy": float(values[1]),
            "kinetic_energy": float(values[2]),
            "momentum": values[3:6].copy(),
            "angular_momentum": values[6:9].copy(),
        })

    common = {
        "model_version": phys.MODEL_VERSION,
        "build_id": phys.BUILD_ID,
        "accepted_steps": accepted_steps,
        "final_time": time,
        "masses_solar": masses.copy(),
        "display_frame": params.display_frame.lower(),
        "initial_conservation": initial_cons,
        "final_conservation": current_cons,
        "conservation_samples": conservation_samples,
        "energy_drift_scale": energy_drift_scale,
        "energy_drift_normalization": energy_drift_normalization,
        "momentum_drift_scale": momentum_drift_scale,
        "momentum_drift_normalization": momentum_drift_normalization,
        "angular_momentum_drift_scale": angular_drift_scale,
        "angular_momentum_drift_normalization": angular_drift_normalization,
        "max_fractional_energy_drift": max_energy_drift,
        "max_momentum_drift": max_momentum_drift,
        "max_angular_momentum_drift": max_angular_momentum_drift,
        # Backward-compatible aliases retained for existing callers. Consult
        # the normalization metadata above before interpreting these values.
        "max_fractional_momentum_drift": max_momentum_drift,
        "max_fractional_angular_momentum_drift": max_angular_momentum_drift,
        "n_test_particles": n_test,
        # Massive-body state at the end of the run, in every output mode.
        "final_massive_positions": positions.copy(),
        "final_massive_velocities": velocities.copy(),
    }

    if has_tests:
        common["test_particles"] = _test_particle_summary(
            np.asarray(params.positions_init, dtype=float),
            np.asarray(params.velocities_init, dtype=float),
            masses, test_center, test_initial_pos, test_initial_vel,
            test_pos, test_vel, removal_time, removal_reason, removal_body,
            removal_radii, escape_radius, phase_reference, phase_min, phase_max,
            positions,
            np.isfinite(turn_max) & np.isfinite(turn_min)
            & (turn_max >= phase_max - turn) & (turn_min <= phase_min + turn),
        )

    if output_type == "survival":
        return {"type": "survival", **common}

    if output_type == "trajectories":
        return {
            "type": "trajectories",
            "times": np.asarray(output_times),
            "positions": np.stack(output_positions, axis=0),
            "velocities": np.stack(output_velocities, axis=0),
            "energies": np.asarray(energies),
            "momenta": np.stack(momenta, axis=0),
            "angular_momenta": np.stack(angular_momenta, axis=0),
            "dt_used": np.asarray(dt_used),
            **common,
            **({"test_positions": np.stack(test_out_positions, axis=0),
                "test_velocities": np.stack(test_out_velocities, axis=0)}
               if has_tests else {}),
        }

    return {
        "type": "animation",
        "frame_times": np.asarray(frame_times),
        "frame_positions": np.stack(frame_positions, axis=0),
        "frame_velocities": np.stack(frame_velocities, axis=0),
        "animation_mode": params.animation_mode.lower(),
        "frame_time": params.frame_time,
        "frame_interval_ms": params.frame_interval_ms,
        "trail_time": params.trail_time,
        "projection": params.projection.lower(),
        "axis_mode": params.axis_mode.lower(),
        **common,
        **({"test_frame_positions": np.stack(test_frame_positions, axis=0)}
           if has_tests else {}),
    }
