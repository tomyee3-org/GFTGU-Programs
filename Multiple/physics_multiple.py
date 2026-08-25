"""
Newtonian gravity and conservation diagnostics for Multiple.
"""

import numpy as np

# Public release metadata. MODEL_VERSION changes when the model's documented
# behaviour changes; BUILD_ID changes whenever one of the core source files
# changes.
MODEL_VERSION = "1.0.0"
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
    n_bodies = positions.shape[0]
    acc = np.zeros_like(positions, dtype=float)

    for a in range(n_bodies):
        for b in range(a + 1, n_bodies):
            r_vec = positions[b] - positions[a]
            r = float(np.linalg.norm(r_vec))
            if r == 0.0:
                raise ValueError(
                    f"Bodies {a + 1} and {b + 1} occupy exactly the same "
                    "position; the Newtonian point-mass force is singular."
                )

            inv_r3 = 1.0 / (r * r * r)

            # r_vec points from a to b.
            acc[a] += GM_SUN * masses_solar[b] * inv_r3 * r_vec
            acc[b] -= GM_SUN * masses_solar[a] * inv_r3 * r_vec

    return acc


def scaled_total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses_solar: np.ndarray,
) -> float:
    """
    Return total mechanical energy divided by one solar-mass unit.

    The result has units m^2/s^2. The overall solar-mass factor is omitted
    because conservation tests depend on fractional changes, not on joules.
    """
    kinetic = 0.5 * np.sum(
        masses_solar[:, None] * velocities * velocities
    )

    potential = 0.0
    n_bodies = len(masses_solar)
    for a in range(n_bodies):
        for b in range(a + 1, n_bodies):
            r = float(np.linalg.norm(positions[b] - positions[a]))
            if r == 0.0:
                raise ValueError(
                    f"Bodies {a + 1} and {b + 1} occupy exactly the same "
                    "position; gravitational potential energy is singular."
                )
            potential -= (
                GM_SUN * masses_solar[a] * masses_solar[b] / r
            )

    return float(kinetic + potential)


def scaled_total_momentum(
    velocities: np.ndarray,
    masses_solar: np.ndarray,
) -> np.ndarray:
    """
    Return total linear momentum divided by one solar-mass unit.

    Units are m/s. Fractional or absolute drift provides a numerical diagnostic.
    """
    return np.sum(masses_solar[:, None] * velocities, axis=0)


def scaled_total_angular_momentum(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses_solar: np.ndarray,
) -> np.ndarray:
    """
    Return total angular momentum divided by one solar-mass unit.

    Units are m^2/s. Fractional or absolute drift provides a numerical diagnostic.
    """
    return np.sum(
        masses_solar[:, None] * np.cross(positions, velocities),
        axis=0,
    )


def conservation_state(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses_solar: np.ndarray,
) -> dict:
    """Return energy, linear momentum, and angular momentum diagnostics."""
    return {
        "energy": scaled_total_energy(positions, velocities, masses_solar),
        "momentum": scaled_total_momentum(velocities, masses_solar),
        "angular_momentum": scaled_total_angular_momentum(
            positions, velocities, masses_solar
        ),
    }
