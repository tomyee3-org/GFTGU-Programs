"""
Encodes Newtonian gravity for an arbitrary number of bodies.
Only physical constants appear as literals.
"""

import numpy as np

# Physical constants
G_NEWTON = 6.67430e-11          # m^3 / (kg s^2)
M_SUN = 1.98847e30              # kg
K_GRAVITY = G_NEWTON * M_SUN    # G * M_sun, as in Schutz's code


def compute_accelerations(positions: np.ndarray, masses_solar: np.ndarray) -> np.ndarray:
    """
    Compute accelerations of all bodies due to mutual gravity.

    positions: shape (n_bodies, 3), in meters
    masses_solar: shape (n_bodies,), in solar masses

    Returns accelerations: shape (n_bodies, 3), in m/s^2
    """
    n_bodies = positions.shape[0]
    acc = np.zeros_like(positions)

    # Convert masses to kg via K_GRAVITY convention (masses in solar units)
    for a in range(n_bodies):
        for b in range(a + 1, n_bodies):
            r_vec = positions[b] - positions[a]
            r = np.linalg.norm(r_vec)
            if r == 0.0:
                continue
            r3 = r ** 3

            # r_vec points from a to b (opposite of Schutz's xAB = xA - xB
            # convention). Gravity pulls a TOWARD b, i.e. along +r_vec, and
            # pulls b TOWARD a, i.e. along -r_vec -- so both coefficients
            # here are positive; the "-=" below is what supplies b's
            # opposite direction.

            # Acceleration on a due to b
            factor_ab = K_GRAVITY * masses_solar[b] / r3
            # Acceleration on b due to a
            factor_ba = K_GRAVITY * masses_solar[a] / r3

            acc[a] += factor_ab * r_vec
            acc[b] -= factor_ba * r_vec  # opposite direction

    return acc