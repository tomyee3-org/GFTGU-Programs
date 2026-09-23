"""
driver_earthorbit.py

Driver for the EarthOrbit simulation.

In the integration loop employed here, acceleration is evaluated at the beginning of each
step, velocity is advanced with that acceleration, and position is advanced
using the average of the old and new velocities.

This is not velocity Verlet/leapfrog and is not generally symplectic.
"""

import math
import numbers

import numpy as np
import physics_earthorbit as phys
from physics_earthorbit import compute_acceleration, R_EARTH, ForceLaw


def version_info():
    """Return machine-readable model and source-build identifiers."""
    return {
        "model_version": phys.MODEL_VERSION,
        "build_id": phys.BUILD_ID,
    }


def run_earth_orbit(
    h0=300.0,
    uInit=7900.0,
    vInit=0.0,
    dt=0.4,
    maxSteps=15000,
    force_law: ForceLaw = "simplified",
    return_diagnostics=False,
):
    """
    Integrate an EarthOrbit trajectory.

    Parameters
    ----------
    h0 : float
        Initial altitude above the reference Earth radius, in metres.
    uInit : float
        Initial x-velocity in m/s. Because the launch point is on the
        positive y-axis, this is the tangential/horizontal component.
    vInit : float
        Initial y-velocity in m/s. At launch this is the radial/vertical
        component.
    dt : float
        Fixed timestep in seconds.
    maxSteps : int
        Maximum number of stored trajectory points.
    force_law : {"simplified", "inverse_square"}
        Select the constant-magnitude-g model or the inverse-square
        extension.
    return_diagnostics : bool
        If False (default), preserve the original four-array return value.
        If True, also return time and velocity arrays for quantitative
        experiments.

    Returns
    -------
    xs, ys, xEarth, yEarth
        Returned when return_diagnostics is False.
    xs, ys, xEarth, yEarth, ts, us, vs
        Returned when return_diagnostics is True.

    Raises
    ------
    FloatingPointError
        If a step produces a non-finite state, or if a step's straight-line
        displacement passes through the reference Earth sphere without
        either endpoint landing inside it (dt too coarse to resolve the
        surface crossing). In both cases, reduce dt.
    """
    # Validate user-editable inputs before allocating arrays or integrating.
    if force_law not in ("simplified", "inverse_square"):
        raise ValueError("force_law must be 'simplified' or 'inverse_square'")

    if not isinstance(return_diagnostics, bool):
        raise ValueError("return_diagnostics must be True or False")

    for name, value in (("h0", h0), ("uInit", uInit), ("vInit", vInit), ("dt", dt)):
        if not isinstance(value, numbers.Real) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite real number")

    if h0 < 0:
        raise ValueError("h0 must be non-negative for this Earth-surface launch model")
    if dt <= 0:
        raise ValueError("dt must be a positive finite timestep in seconds")
    if not isinstance(maxSteps, numbers.Integral) or isinstance(maxSteps, bool) or maxSteps < 2:
        raise ValueError("maxSteps must be an integer >= 2")

    # NumPy accepts integral scalar types, but normalising to int makes array
    # allocation behavior consistent across supported NumPy versions.
    maxSteps = int(maxSteps)

    x = 0.0
    y = R_EARTH + h0

    u0 = float(uInit)
    v0 = float(vInit)

    try:
        xs = np.zeros(maxSteps)
        ys = np.zeros(maxSteps)
        if return_diagnostics:
            us = np.zeros(maxSteps)
            vs = np.zeros(maxSteps)
    except (MemoryError, OverflowError, ValueError) as exc:
        raise ValueError(
            "maxSteps is too large for the available memory or platform"
        ) from exc

    xs[0] = x
    ys[0] = y

    if return_diagnostics:
        us[0] = u0
        vs[0] = v0

    r = math.hypot(x, y)

    j = 1
    while r >= R_EARTH and j < maxSteps:
        # The integration algorithm evaluates acceleration only at the
        # position at the beginning of the step.
        ax, ay = compute_acceleration(x, y, force_law)

        x_prev, y_prev = x, y

        u1 = u0 + ax * dt
        v1 = v0 + ay * dt

        # Advance position using the average of old and new velocities.
        x = x_prev + (u0 + u1) * 0.5 * dt
        y = y_prev + (v0 + v1) * 0.5 * dt

        if not all(math.isfinite(value) for value in (x, y, u1, v1)):
            raise FloatingPointError(
                "integration produced a non-finite state; reduce the timestep "
                "or use less extreme initial values"
            )

        r = math.hypot(x, y)

        if r >= R_EARTH and _segment_dips_inside_earth(x_prev, y_prev, x, y):
            # Both the start and end of this step lie outside the reference
            # sphere, but the straight-line displacement between them passed
            # through it. A large dt can carry the projectile clean through
            # Earth in a single step; left unchecked, the loop above would
            # keep running and analyze_earth_orbit would misreport this as
            # an ordinary escape or orbit instead of an impact.
            raise FloatingPointError(
                "this step's straight-line displacement passed through "
                "Earth's reference surface without either endpoint landing "
                "inside it; dt is too coarse to resolve the surface "
                "crossing -- reduce dt"
            )

        xs[j] = x
        ys[j] = y

        if return_diagnostics:
            us[j] = u1
            vs[j] = v1

        u0 = u1
        v0 = v1
        j += 1

    xs = xs[:j]
    ys = ys[:j]

    # Build a closed Earth-surface reference curve. 
    angleStep = np.pi / 200
    xEarth = np.zeros(401)
    yEarth = np.zeros(401)

    for k in range(400):
        angle = angleStep * k
        xEarth[k] = R_EARTH * np.cos(angle)
        yEarth[k] = R_EARTH * np.sin(angle)

    xEarth[400] = xEarth[0]
    yEarth[400] = yEarth[0]

    if return_diagnostics:
        ts = np.arange(j, dtype=float) * dt
        return xs, ys, xEarth, yEarth, ts, us[:j], vs[:j]

    return xs, ys, xEarth, yEarth


def _five_significant(value):
    """Format a finite number with exactly five significant digits."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("summary values must be finite")
    if value == 0.0:
        return "0.0000"

    def _format_at(exponent):
        if -4 <= exponent < 5:
            decimal_places = max(0, 4 - exponent)
            return f"{value:.{decimal_places}f}"
        return f"{value:.4e}"

    exponent = math.floor(math.log10(abs(value)))
    text = _format_at(exponent)

    # Rounding to five significant digits at this exponent can carry the
    # magnitude up by a power of ten -- 99999.9 rounds to "100000" and
    # 0.999999 rounds to "1.00000", each with six digits, unless the
    # exponent used for formatting is re-derived from the rounded value.
    # The %e branch already renormalizes its own mantissa, so this only
    # ever fires for the fixed-point branch, and never needs a second pass:
    # rounding to ~5 significant digits shifts the magnitude by at most one
    # decade.
    rounded_exponent = math.floor(math.log10(abs(float(text))))
    if rounded_exponent != exponent:
        text = _format_at(rounded_exponent)

    return text


def _segment_dips_inside_earth(x0, y0, x1, y1):
    """Return True if the segment from (x0, y0) to (x1, y1) comes within
    R_EARTH of the origin anywhere strictly between its endpoints.

    Both endpoints are assumed to already be at or outside the reference
    surface; this only detects a straight-line displacement large enough to
    cut through the disk of radius R_EARTH and back out again -- a single
    fixed-step update whose start and end points are both above the surface
    but whose connecting chord passes through it.

    Coordinates are rescaled by the largest magnitude present (endpoints or
    R_EARTH) before any products are formed. Classroom-scale displacements
    (order 1e6-1e7 m) are unaffected to floating-point precision, but the
    rescaling also keeps this finite for wildly unphysical inputs -- e.g. a
    single-step displacement of order 1e155 m -- where squaring the raw
    coordinates would overflow to inf/nan and silently return False for a
    segment that does cross the sphere.
    """
    scale = max(abs(x0), abs(y0), abs(x1), abs(y1), R_EARTH, 1.0)
    x0, y0, x1, y1 = x0 / scale, y0 / scale, x1 / scale, y1 / scale

    dx = x1 - x0
    dy = y1 - y0
    segment_length_squared = dx * dx + dy * dy
    if segment_length_squared == 0.0:
        return False

    # Fraction along the segment of the closest approach to the origin.
    closest_fraction = -(x0 * dx + y0 * dy) / segment_length_squared
    if closest_fraction <= 0.0 or closest_fraction >= 1.0:
        # The closest approach to Earth's centre is at or beyond an
        # endpoint, both of which are already confirmed to be outside the
        # reference surface, so the whole segment stays outside.
        return False

    closest_x = x0 + closest_fraction * dx
    closest_y = y0 + closest_fraction * dy
    return math.hypot(closest_x, closest_y) < (R_EARTH / scale)


def _surface_crossing_fraction(x0, y0, x1, y1):
    """Return the line-segment fraction at the reference Earth surface."""
    dx = x1 - x0
    dy = y1 - y0
    a = dx * dx + dy * dy
    if a == 0.0:
        return 0.0
    b = 2.0 * (x0 * dx + y0 * dy)
    c = x0 * x0 + y0 * y0 - R_EARTH * R_EARTH
    discriminant = max(0.0, b * b - 4.0 * a * c)
    root = math.sqrt(discriminant)
    candidates = (
        (-b - root) / (2.0 * a),
        (-b + root) / (2.0 * a),
    )
    in_segment = [value for value in candidates if -1.0e-12 <= value <= 1.0 + 1.0e-12]
    if not in_segment:
        raise ValueError("final trajectory segment does not cross Earth's surface")
    return min(1.0, max(0.0, min(in_segment)))


def _parabolic_candidate(times, values, index):
    """Return a three-point parabolic vertex when it lies between neighbours."""
    if index <= 0 or index >= len(values) - 1:
        return float(values[index])

    x0 = float(times[index - 1] - times[index])
    x2 = float(times[index + 1] - times[index])
    y0 = float(values[index - 1])
    y1 = float(values[index])
    y2 = float(values[index + 1])
    determinant = x0 * x2 * (x0 - x2)
    if determinant == 0.0:
        return y1

    a = ((y0 - y1) * x2 - (y2 - y1) * x0) / determinant
    b = (x0 * x0 * (y2 - y1) - x2 * x2 * (y0 - y1)) / determinant
    if a == 0.0:
        return y1

    vertex_x = -b / (2.0 * a)
    if not x0 <= vertex_x <= x2:
        return y1
    return float(a * vertex_x * vertex_x + b * vertex_x + y1)


def _interpolated_extreme(times, values, mode, eligible=None, boundaries=()):
    """Return a sampled/parabolic extreme, also considering boundary values."""
    values = np.asarray(values, dtype=float)
    if eligible is None:
        eligible = np.arange(values.size)
    else:
        eligible = np.asarray(eligible, dtype=int)
    if eligible.size == 0 and not boundaries:
        raise ValueError("at least one value is required for an extreme")

    candidates = [float(value) for value in boundaries]
    if eligible.size:
        selected = values[eligible]
        local = int(np.argmax(selected) if mode == "maximum" else np.argmin(selected))
        index = int(eligible[local])
        candidates.append(float(values[index]))
        if index - 1 in eligible and index + 1 in eligible:
            candidates.append(_parabolic_candidate(times, values, index))

    return max(candidates) if mode == "maximum" else min(candidates)


def _progress_crossing(progress, target, *arrays):
    """Linearly interpolate arrays where monotone angular progress meets target."""
    upper = int(np.searchsorted(progress, target, side="left"))
    if upper == 0:
        return tuple(float(values[0]) for values in arrays)
    if upper >= len(progress):
        return tuple(float(values[-1]) for values in arrays)
    lower = upper - 1
    span = progress[upper] - progress[lower]
    fraction = 0.0 if span == 0.0 else (target - progress[lower]) / span
    return tuple(
        float(values[lower] + fraction * (values[upper] - values[lower]))
        for values in arrays
    )


def _inverse_square_elements(x, y, u, v):
    """Return osculating two-body elements derived from the initial state."""
    radius = math.hypot(x, y)
    speed_squared = u * u + v * v
    angular_momentum = x * v - y * u
    energy = 0.5 * speed_squared - phys.MU_EARTH / radius
    eccentricity_squared = (
        1.0
        + 2.0 * energy * angular_momentum * angular_momentum
        / (phys.MU_EARTH * phys.MU_EARTH)
    )
    eccentricity = math.sqrt(max(0.0, eccentricity_squared))

    if energy < 0.0:
        semimajor_axis = -phys.MU_EARTH / (2.0 * energy)
        perigee_radius = semimajor_axis * (1.0 - eccentricity)
        apogee_radius = semimajor_axis * (1.0 + eccentricity)
    else:
        semilatus_rectum = (
            angular_momentum * angular_momentum / phys.MU_EARTH
        )
        perigee_radius = semilatus_rectum / (1.0 + eccentricity)
        apogee_radius = None

    return {
        "specific_energy": energy,
        "eccentricity": eccentricity,
        "perigee_altitude": perigee_radius - R_EARTH,
        "apogee_altitude": (
            None if apogee_radius is None else apogee_radius - R_EARTH
        ),
        "unbound": energy >= 0.0,
    }


def analyze_earth_orbit(
    xs, ys, ts, us, vs, *, force_law="simplified", max_steps=None
):
    """Analyze a driver-produced trajectory and build its console summary.

    Surface impact is interpolated with the line segment joining the final two
    stored positions.  Angular crossings are interpolated between samples, and
    local altitude extrema use three-point parabolic interpolation where the
    vertex lies inside the relevant interval.
    """
    arrays = []
    for name, values in (("xs", xs), ("ys", ys), ("ts", ts), ("us", us), ("vs", vs)):
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric values") from exc
        if array.ndim != 1 or array.size < 2:
            raise ValueError(f"{name} must be a one-dimensional array with at least two values")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        arrays.append(array)

    xs, ys, ts, us, vs = arrays
    if not all(array.size == xs.size for array in arrays[1:]):
        raise ValueError("trajectory and diagnostic arrays must have equal lengths")
    if force_law not in ("simplified", "inverse_square"):
        raise ValueError("force_law must be 'simplified' or 'inverse_square'")
    if np.any(np.diff(ts) <= 0.0):
        raise ValueError("diagnostic times must increase strictly")

    radii = np.hypot(xs, ys)
    impacted = bool(radii[-1] < R_EARTH)
    impact_fraction = None
    if impacted:
        if radii[-2] < R_EARTH:
            raise ValueError("impact trajectory must have an above-surface penultimate sample")
        impact_fraction = _surface_crossing_fraction(
            xs[-2], ys[-2], xs[-1], ys[-1]
        )
        x_end = xs[-2] + impact_fraction * (xs[-1] - xs[-2])
        y_end = ys[-2] + impact_fraction * (ys[-1] - ys[-2])
        t_end = ts[-2] + impact_fraction * (ts[-1] - ts[-2])
        x_used = np.concatenate((xs[:-1], [x_end]))
        y_used = np.concatenate((ys[:-1], [y_end]))
        t_used = np.concatenate((ts[:-1], [t_end]))
    else:
        x_used, y_used, t_used = xs, ys, ts

    radii_used = np.hypot(x_used, y_used)
    altitudes = radii_used - R_EARTH
    angles = np.unwrap(np.arctan2(y_used, x_used))
    initial_angular_momentum = x_used[0] * vs[0] - y_used[0] * us[0]
    if initial_angular_momentum > 0.0:
        direction = 1.0
    elif initial_angular_momentum < 0.0:
        direction = -1.0
    else:
        angular_change = angles[-1] - angles[0]
        direction = 1.0 if angular_change >= 0.0 else -1.0
    progress = direction * (angles - angles[0])
    progress = np.maximum.accumulate(np.maximum(progress, 0.0))

    total_angle = float(progress[-1])
    revolutions = total_angle / (2.0 * math.pi)
    completed_revolutions = int(math.floor(revolutions + 1.0e-12))
    total_time = float(t_used[-1] - t_used[0])
    segment_lengths = np.hypot(np.diff(x_used), np.diff(y_used))
    total_distance = float(np.sum(segment_lengths))
    maximum_altitude = _interpolated_extreme(
        t_used, altitudes, "maximum"
    )

    elements = None
    if force_law == "inverse_square":
        elements = _inverse_square_elements(xs[0], ys[0], us[0], vs[0])

    escaped = bool(elements is not None and elements["unbound"] and not impacted)
    reached_orbit = completed_revolutions >= 1 and not escaped

    revolution_data = []
    if reached_orbit:
        for number in range(1, completed_revolutions + 1):
            start_target = 2.0 * math.pi * (number - 1)
            end_target = 2.0 * math.pi * number
            start_time, start_x, start_y = _progress_crossing(
                progress, start_target, t_used, x_used, y_used
            )
            end_time, end_x, end_y = _progress_crossing(
                progress, end_target, t_used, x_used, y_used
            )
            start_altitude = math.hypot(start_x, start_y) - R_EARTH
            end_altitude = math.hypot(end_x, end_y) - R_EARTH
            eligible = np.nonzero(
                (progress > start_target) & (progress < end_target)
            )[0]
            minimum = _interpolated_extreme(
                t_used, altitudes, "minimum", eligible,
                (start_altitude, end_altitude),
            )
            maximum = _interpolated_extreme(
                t_used, altitudes, "maximum", eligible,
                (start_altitude, end_altitude),
            )
            revolution_data.append({
                "number": number,
                "time": end_time - start_time,
                "minimum_altitude": minimum,
                "maximum_altitude": maximum,
            })

    if reached_orbit:
        termination = "surface impact" if impacted else "maxSteps reached"
        status = f"Orbit completed; termination: {termination}"
    elif impacted:
        status = "Surface impact before one revolution"
    elif escaped:
        status = "Escape trajectory; maxSteps reached"
    else:
        status = "No complete revolution before maxSteps"

    lines = [f"Outcome: {status}"]
    if reached_orbit:
        lines.extend((
            f"Completed revolutions: {completed_revolutions}",
            f"Total flight time: {_five_significant(total_time)} s",
            f"Total angular travel: {_five_significant(math.degrees(total_angle))} deg",
        ))
        for item in revolution_data:
            lines.append(
                f"Revolution {item['number']}: "
                f"time {_five_significant(item['time'])} s; "
                f"minimum altitude {_five_significant(item['minimum_altitude'])} m; "
                f"maximum altitude {_five_significant(item['maximum_altitude'])} m"
            )
    else:
        distance_label = (
            "Total distance traveled (interpolated to impact)"
            if impacted
            else "Total distance traveled through final sample"
        )
        lines.extend((
            f"Fraction of a revolution: {_five_significant(revolutions)}",
            f"Total flight time: {_five_significant(total_time)} s",
            f"Total angular travel: {_five_significant(math.degrees(total_angle))} deg",
            f"Maximum altitude (interpolated): {_five_significant(maximum_altitude)} m",
            f"{distance_label}: {_five_significant(total_distance)} m",
        ))

    if elements is not None:
        lines.append(
            f"Osculating eccentricity: {_five_significant(elements['eccentricity'])}"
        )
        lines.append(
            "Osculating perigee altitude: "
            f"{_five_significant(elements['perigee_altitude'])} m"
        )
        if elements["apogee_altitude"] is not None:
            lines.append(
                "Osculating apogee altitude: "
                f"{_five_significant(elements['apogee_altitude'])} m"
            )

    if escaped:
        final_altitude = float(altitudes[-1])
        if _five_significant(final_altitude) == _five_significant(maximum_altitude):
            lines.append(
                "Final/maximum altitude at maxSteps: "
                f"{_five_significant(final_altitude)} m"
            )
        else:
            lines.append(
                f"Final altitude at maxSteps: {_five_significant(final_altitude)} m"
            )

    return {
        "outcome": status,
        "impact": impacted,
        "escape": escaped,
        "reached_orbit": reached_orbit,
        "completed_revolutions": completed_revolutions,
        "revolutions": revolutions,
        "total_time": total_time,
        "total_angle_radians": total_angle,
        "maximum_altitude": maximum_altitude,
        "total_distance": total_distance,
        "revolution_data": revolution_data,
        "elements": elements,
        "impact_fraction": impact_fraction,
        "max_steps": max_steps,
        "lines": lines,
    }
