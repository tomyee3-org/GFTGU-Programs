"""
Newtonian gravity and conservation diagnostics for Multiple.
"""

import numpy as np

# Public release metadata. MODEL_VERSION changes when the model's documented
# behaviour changes; BUILD_ID changes whenever one of the core source files
# changes.
MODEL_VERSION = "1.2.2"
BUILD_ID_COVERS = (
    "physics_multiple.py",
    "driver_multiple.py",
    "main.py",
    "plot_multiple.py",
)


def _compute_build_id() -> str:
    """Return a short, reproducible identifier for the core source files.

    Files are read as UTF-8 text with universal-newline conversion, so merely
    switching between LF and CRLF line endings does not create a new build.
    Filename and byte-length framing prevents ambiguous concatenations.
    """
    import hashlib
    import os

    try:
        here = os.path.dirname(os.path.abspath(__file__))
        digest = hashlib.sha256()
        for name in BUILD_ID_COVERS:
            path = os.path.join(here, name)
            with open(path, "r", encoding="utf-8", newline=None) as source:
                content = source.read().encode("utf-8")
            digest.update(name.encode("utf-8"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        return digest.hexdigest()[:12]
    except (OSError, UnicodeDecodeError):
        return "unknown"


BUILD_ID = _compute_build_id()

# IAU 2015 nominal solar mass parameter (exact nominal conversion constant).
GM_SUN = 1.3271244e20  # m^3 s^-2


def _as_finite_float_array(values, name: str) -> np.ndarray:
    """Convert array-like input to a finite floating-point NumPy array."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validated_positions_masses(
    positions,
    masses_solar,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated position and mass arrays for the public physics API."""
    masses = _as_finite_float_array(masses_solar, "masses_solar")
    pos = _as_finite_float_array(positions, "positions")

    if masses.ndim != 1 or masses.size < 2:
        raise ValueError(
            "masses_solar must be a one-dimensional array with at least "
            "two values."
        )
    if np.any(masses <= 0.0):
        raise ValueError("All masses must be positive.")
    if pos.shape != (masses.size, 3):
        raise ValueError(
            "positions must have shape (number of masses, 3)."
        )
    return pos, masses


def _validated_state(
    positions,
    velocities,
    masses_solar,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return validated position, velocity, and mass arrays."""
    pos, masses = _validated_positions_masses(positions, masses_solar)
    vel = _as_finite_float_array(velocities, "velocities")
    if vel.shape != pos.shape:
        raise ValueError("velocities must have the same shape as positions.")
    return pos, vel, masses


def compute_accelerations(
    positions: np.ndarray,
    masses_solar: np.ndarray,
) -> np.ndarray:
    """
    Compute mutual Newtonian accelerations.

    positions:    shape (n_bodies, 3), metres
    masses_solar: shape (n_bodies,), masses in solar-mass units

    Returns an array of shape (n_bodies, 3), m/s^2.

    Bodies are mathematical point masses. Exact coincidence makes the
    Newtonian force singular and therefore raises ValueError.
    """
    positions, masses_solar = _validated_positions_masses(
        positions, masses_solar
    )
    n_bodies = positions.shape[0]
    acc = np.zeros_like(positions, dtype=float)

    for a in range(n_bodies):
        for b in range(a + 1, n_bodies):
            with np.errstate(over="ignore", invalid="ignore"):
                r_vec = positions[b] - positions[a]
            if not np.all(np.isfinite(r_vec)):
                raise ValueError(
                    f"The separation of bodies {a + 1} and {b + 1} is "
                    "outside the floating-point range."
                )

            # Repeated hypot avoids the avoidable overflow that can occur in
            # sqrt(x*x + y*y + z*z) for large but finite coordinates.
            r = float(np.hypot.reduce(r_vec))
            if r == 0.0:
                raise ValueError(
                    f"Bodies {a + 1} and {b + 1} occupy exactly the same "
                    "position; the Newtonian point-mass force is singular."
                )

            if not np.isfinite(r):
                raise ValueError(
                    f"The separation of bodies {a + 1} and {b + 1} is "
                    "outside the floating-point range."
                )

            # r_vec points from a to b. Dividing before multiplying avoids an
            # unnecessary r**3 overflow for very large separations.
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                direction = r_vec / r
                scale_a = (GM_SUN / r) * (masses_solar[b] / r)
                scale_b = (GM_SUN / r) * (masses_solar[a] / r)
                contribution_a = scale_a * direction
                contribution_b = scale_b * direction

            if not (
                np.all(np.isfinite(contribution_a))
                and np.all(np.isfinite(contribution_b))
            ):
                raise ValueError(
                    f"The acceleration of bodies {a + 1} and {b + 1} is "
                    "outside the floating-point range."
                )

            acc[a] += contribution_a
            acc[b] -= contribution_b

    if not np.all(np.isfinite(acc)):
        raise ValueError(
            "The combined acceleration is outside the floating-point range."
        )

    return acc


def scaled_energy_components(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses_solar: np.ndarray,
) -> tuple[float, float]:
    """
    Return kinetic and potential energy per solar-mass unit.

    Both values have units m^2/s^2. The overall solar-mass factor is omitted
    because conservation diagnostics depend on relative changes, not joules.
    """
    positions, velocities, masses_solar = _validated_state(
        positions, velocities, masses_solar
    )
    with np.errstate(over="ignore", invalid="ignore"):
        kinetic = 0.5 * np.sum(
            masses_solar[:, None] * velocities * velocities
        )

    potential = 0.0
    n_bodies = len(masses_solar)
    for a in range(n_bodies):
        for b in range(a + 1, n_bodies):
            with np.errstate(over="ignore", invalid="ignore"):
                r_vec = positions[b] - positions[a]
            if not np.all(np.isfinite(r_vec)):
                raise ValueError(
                    f"The separation of bodies {a + 1} and {b + 1} is "
                    "outside the floating-point range."
                )
            r = float(np.hypot.reduce(r_vec))
            if r == 0.0:
                raise ValueError(
                    f"Bodies {a + 1} and {b + 1} occupy exactly the same "
                    "position; gravitational potential energy is singular."
                )
            if not np.isfinite(r):
                raise ValueError(
                    f"The separation of bodies {a + 1} and {b + 1} is "
                    "outside the floating-point range."
                )
            with np.errstate(over="ignore", invalid="ignore"):
                potential -= (
                    (GM_SUN / r) * masses_solar[a] * masses_solar[b]
                )

    kinetic = float(kinetic)
    potential = float(potential)
    if not (np.isfinite(kinetic) and np.isfinite(potential)):
        raise ValueError("Energy components are outside the floating-point range.")
    return kinetic, potential


def scaled_total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses_solar: np.ndarray,
) -> float:
    """Return total mechanical energy divided by one solar-mass unit."""
    kinetic, potential = scaled_energy_components(
        positions, velocities, masses_solar
    )
    energy = kinetic + potential
    if not np.isfinite(energy):
        raise ValueError("Total energy is outside the floating-point range.")
    return float(energy)


def scaled_total_momentum(
    velocities: np.ndarray,
    masses_solar: np.ndarray,
) -> np.ndarray:
    """
    Return total linear momentum divided by one solar-mass unit.

    Units are m/s. Fractional or absolute drift provides a numerical diagnostic.
    """
    velocities = _as_finite_float_array(velocities, "velocities")
    masses_solar = _as_finite_float_array(masses_solar, "masses_solar")
    if masses_solar.ndim != 1 or masses_solar.size < 2:
        raise ValueError(
            "masses_solar must be a one-dimensional array with at least "
            "two values."
        )
    if np.any(masses_solar <= 0.0):
        raise ValueError("All masses must be positive.")
    if velocities.shape != (masses_solar.size, 3):
        raise ValueError("velocities must have shape (number of masses, 3).")

    with np.errstate(over="ignore", invalid="ignore"):
        momentum = np.sum(masses_solar[:, None] * velocities, axis=0)
    if not np.all(np.isfinite(momentum)):
        raise ValueError("Total momentum is outside the floating-point range.")
    return momentum


def scaled_total_angular_momentum(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses_solar: np.ndarray,
) -> np.ndarray:
    """
    Return total angular momentum divided by one solar-mass unit.

    Units are m^2/s. Fractional or absolute drift provides a numerical diagnostic.
    """
    positions, velocities, masses_solar = _validated_state(
        positions, velocities, masses_solar
    )
    with np.errstate(over="ignore", invalid="ignore"):
        angular_momentum = np.sum(
            masses_solar[:, None] * np.cross(positions, velocities),
            axis=0,
        )
    if not np.all(np.isfinite(angular_momentum)):
        raise ValueError(
            "Total angular momentum is outside the floating-point range."
        )
    return angular_momentum


def conservation_state(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses_solar: np.ndarray,
) -> dict:
    """Return energy, linear momentum, and angular momentum diagnostics."""
    kinetic, potential = scaled_energy_components(
        positions, velocities, masses_solar
    )
    return {
        "energy": kinetic + potential,
        "kinetic_energy": kinetic,
        "potential_energy": potential,
        "momentum": scaled_total_momentum(velocities, masses_solar),
        "angular_momentum": scaled_total_angular_momentum(
            positions, velocities, masses_solar
        ),
    }


def _normalized_display_frame(frame) -> str:
    """Return a canonical display-frame name or raise ValueError."""
    if not isinstance(frame, str):
        raise ValueError('display_frame must be "com" or "user".')
    normalized = frame.lower()
    if normalized not in ("com", "user"):
        raise ValueError('display_frame must be "com" or "user".')
    return normalized


def center_of_mass(
    positions: np.ndarray,
    masses_solar: np.ndarray,
) -> np.ndarray:
    """
    Return the mass-weighted center of mass.

    positions may have shape (n_bodies, 3) or (n_states, n_bodies, 3).
    The result has shape (3,) or (n_states, 3) respectively.
    """
    positions = _as_finite_float_array(positions, "positions")
    masses_solar = _as_finite_float_array(masses_solar, "masses_solar")
    if masses_solar.ndim != 1 or masses_solar.size < 2:
        raise ValueError(
            "masses_solar must be a one-dimensional array with at least "
            "two values."
        )
    if np.any(masses_solar <= 0.0):
        raise ValueError("All masses must be positive.")

    if positions.ndim == 2:
        if positions.shape != (masses_solar.size, 3):
            raise ValueError(
                "positions must have shape (number of masses, 3)."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            weighted = np.sum(masses_solar[:, None] * positions, axis=0)
            com = weighted / float(np.sum(masses_solar))
    elif positions.ndim == 3:
        if positions.shape[1:] != (masses_solar.size, 3):
            raise ValueError(
                "stacked positions must have shape "
                "(n_states, number of masses, 3)."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            weighted = np.sum(masses_solar[None, :, None] * positions, axis=1)
            com = weighted / float(np.sum(masses_solar))
    else:
        raise ValueError(
            "positions must have shape (n_bodies, 3) or "
            "(n_states, n_bodies, 3)."
        )

    if not np.all(np.isfinite(com)):
        raise ValueError(
            "Center of mass is outside the floating-point range."
        )
    return com


def center_of_mass_velocity(
    velocities: np.ndarray,
    masses_solar: np.ndarray,
) -> np.ndarray:
    """
    Return the mass-weighted center-of-mass velocity V_CM.

    velocities may have shape (n_bodies, 3) or (n_states, n_bodies, 3).
    For an isolated Newtonian system the inertial frame that moves at
    V_CM is the center-of-momentum frame, meaning the frame in which the
    total linear momentum vanishes. V_CM itself is a velocity, not the
    velocity of a "center-of-momentum point."
    """
    return center_of_mass(velocities, masses_solar)


def positions_in_display_frame(
    positions: np.ndarray,
    masses_solar: np.ndarray,
    display_frame: str = "com",
) -> np.ndarray:
    """
    Return positions expressed in the requested display frame.

    "user" leaves the integration coordinates unchanged. "com" subtracts the
    instantaneous center of mass from every body so the plotted origin sits
    at R_CM. The integration itself is never modified.
    """
    frame = _normalized_display_frame(display_frame)
    positions = _as_finite_float_array(positions, "positions")
    if frame == "user":
        if positions.ndim == 2:
            _validated_positions_masses(positions, masses_solar)
        elif positions.ndim == 3:
            if positions.shape[0] == 0:
                raise ValueError("positions must contain at least one state.")
            _validated_positions_masses(positions[0], masses_solar)
        else:
            raise ValueError(
                "positions must have shape (n_bodies, 3) or "
                "(n_states, n_bodies, 3)."
            )
        return positions

    com = center_of_mass(positions, masses_solar)
    if positions.ndim == 2:
        shifted = positions - com
    else:
        shifted = positions - com[:, None, :]
    if not np.all(np.isfinite(shifted)):
        raise ValueError(
            "COM-frame positions are outside the floating-point range."
        )
    return shifted
