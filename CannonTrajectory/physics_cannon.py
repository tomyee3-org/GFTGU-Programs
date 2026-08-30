"""
Physics and numerical integration routines for CannonTrajectory.

State vector:
    x  — horizontal position (m)
    h  — vertical position / height (m)
    u  — horizontal velocity (m/s)
    v  — vertical velocity (m/s)

Model:
    Horizontal acceleration is zero.
    Vertical acceleration is the constant -g.

These low-level teaching functions deliberately expose the integration
formulas without duplicating the driver's validation machinery.  Direct
callers must supply a finite, one-dimensional four-value state and a finite
scalar timestep.  ``run_cannon_trajectory`` validates ordinary user settings.
"""

import numpy as np

MODEL_VERSION = "1.1.0"


#: The exact source files this build identifier covers: a documentation-only
#: change, a sample-output file, or an edit to the test suite does not change
#: this value -- only the four core program modules listed here do.  Exposed
#: so callers can determine precisely what BUILD_ID covers without duplicating
#: this list.
BUILD_ID_COVERS = (
    "physics_cannon.py",
    "driver_cannon.py",
    "main.py",
    "plot_cannon.py",
)


def _compute_build_id():
    """Return a short identifier derived from the core source files.

    MODEL_VERSION records the program's declared release version.  BUILD_ID
    additionally distinguishes source revisions that retain the same declared
    version.  The hash is independent of LF versus CRLF line endings and
    frames each file with its name and length so file-boundary changes cannot
    collide with an unchanged concatenated byte stream.

    Return ``"unknown"`` rather than preventing the program from running if
    the source files cannot be located or decoded, as can happen in some
    frozen or zipped distributions.
    """
    import hashlib
    import os

    try:
        here = os.path.dirname(os.path.abspath(__file__))
        digest = hashlib.sha256()
        for name in BUILD_ID_COVERS:
            with open(os.path.join(here, name), "r", encoding="utf-8",
                      newline=None) as source:
                content = source.read().encode("utf-8")
            digest.update(name.encode("utf-8"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        return digest.hexdigest()[:12]
    except (OSError, UnicodeDecodeError):
        return "unknown"


BUILD_ID = _compute_build_id()


# Standard acceleration due to gravity, defined exactly by international
# agreement.  It is a representative near-surface value, not a claim that
# gravity has precisely this value at every point on Earth.
g = 9.80665  # m/s^2


def derivs_cannon(state):
    """Return [dx/dt, dh/dt, du/dt, dv/dt] for the projectile state."""
    x, h, u, v = state
    return np.array([u, v, 0.0, -g], dtype=float)


def euler_step(state, dt):
    """Advance one timestep with the forward Euler method."""
    return state + dt * derivs_cannon(state)


def improved_euler_step(state, dt):
    """Advance one timestep with improved Euler (Heun's method)."""
    ds1 = derivs_cannon(state)
    predictor = state + dt * ds1
    ds2 = derivs_cannon(predictor)
    return state + 0.5 * dt * (ds1 + ds2)
