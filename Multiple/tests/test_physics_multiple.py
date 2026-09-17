"""Regression tests for the Multiple Newtonian N-body module.

The locator intentionally supports both repository layouts used for review:

* canonical: ``Multiple/tests/test_physics_multiple.py``;
* flattened: the test file copied beside the four program modules.
"""

import ast
import contextlib
import hashlib
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


CORE_MODULE_FILES = (
    "physics_multiple.py",
    "driver_multiple.py",
    "main.py",
    "plot_multiple.py",
)
HELP_FILE = "Multiple.html"


def find_module_dir(start) -> Path:
    """Find the nearest ancestor containing all four Multiple modules."""
    path = Path(start).resolve()
    directory = path if path.is_dir() else path.parent
    for candidate in (directory, *directory.parents):
        if all((candidate / name).is_file() for name in CORE_MODULE_FILES):
            return candidate
    names = ", ".join(CORE_MODULE_FILES)
    raise FileNotFoundError(
        f"Could not find a directory containing all core modules: {names}"
    )


def find_help_file(module_dir):
    """Find Help in a flattened upload or the GFTGU-Documentation tree.

    Documentation folders no longer use chapter-number prefixes, and the
    Help files live under the sibling ``GFTGU-Documentation`` repository
    rather than beside the program modules inside ``GFTGU-Programs``.
    """
    program_name = Path(HELP_FILE).stem
    candidates = [module_dir / HELP_FILE]
    for ancestor in (module_dir, *module_dir.parents):
        candidates.append(
            ancestor / "GFTGU-Documentation" / program_name / HELP_FILE
        )
        if ancestor.name != program_name:
            candidates.append(ancestor / program_name / HELP_FILE)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {HELP_FILE} beside the program or in "
        f"GFTGU-Documentation/{program_name}/."
    )


MODULE_DIR = find_module_dir(Path(__file__))
HELP_PATH = find_help_file(MODULE_DIR)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

os.environ.setdefault("MPLBACKEND", "Agg")

import driver_multiple as driver  # noqa: E402
import physics_multiple as phys  # noqa: E402
import plot_multiple as plotting  # noqa: E402


def make_params(**overrides):
    """Return a small, valid two-body configuration with optional changes."""
    values = {
        "n_bodies": 2,
        "masses_solar": [1.0, 2.0],
        "positions_init": [[-1.0e10, 0.0, 0.0], [1.0e10, 0.0, 0.0]],
        "velocities_init": [[0.0, -1000.0, 0.0], [0.0, 500.0, 0.0]],
        "dt": 100.0,
        "max_steps": 4,
        "output_type": "trajectories",
        "eps1": 0.05,
        "eps2": 1.0e-4,
        "animation_mode": "trails",
        "frame_time": 50.0,
        "frame_interval_ms": 20,
        "trail_time": 100.0,
        "projection": "xy",
        "axis_mode": "fixed",
    }
    values.update(overrides)
    return driver.SimulationParams(**values)


class TestPortableLocator(unittest.TestCase):
    def test_finds_canonical_parent_from_test_file(self):
        self.assertEqual(find_module_dir(Path(__file__)), MODULE_DIR)

    def test_finds_flattened_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in CORE_MODULE_FILES:
                (root / name).touch()
            flattened_test = root / "test_physics_multiple.py"
            flattened_test.touch()
            self.assertEqual(find_module_dir(flattened_test), root)

    def test_nearest_matching_ancestor_wins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "outer" / "inner"
            nested.mkdir(parents=True)
            for directory in (root, root / "outer"):
                for name in CORE_MODULE_FILES:
                    (directory / name).touch()
            self.assertEqual(find_module_dir(nested / "test.py"), root / "outer")

    def test_missing_modules_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "all core modules"):
                find_module_dir(Path(temporary) / "test.py")


class TestPhysicsAccelerations(unittest.TestCase):
    def test_nominal_solar_mass_parameter(self):
        self.assertEqual(phys.GM_SUN, 1.3271244e20)

    def test_model_version_format(self):
        self.assertRegex(phys.MODEL_VERSION, r"^\d+\.\d+\.\d+$")

    def test_equal_mass_two_body_acceleration(self):
        separation = 2.0e10
        positions = np.array([[-separation / 2, 0, 0], [separation / 2, 0, 0]])
        acceleration = phys.compute_accelerations(positions, np.array([1.0, 1.0]))
        expected = phys.GM_SUN / separation**2
        np.testing.assert_allclose(
            acceleration,
            [[expected, 0.0, 0.0], [-expected, 0.0, 0.0]],
            rtol=2.0e-15,
            atol=0.0,
        )

    def test_unequal_mass_action_reaction(self):
        positions = np.array([[0.0, 0.0, 0.0], [3.0e10, 4.0e10, 0.0]])
        masses = np.array([2.0, 5.0])
        acceleration = phys.compute_accelerations(positions, masses)
        np.testing.assert_allclose(
            masses[0] * acceleration[0] + masses[1] * acceleration[1],
            np.zeros(3),
            rtol=0.0,
            atol=1.0e-16,
        )
        self.assertAlmostEqual(
            np.linalg.norm(acceleration[0]) / np.linalg.norm(acceleration[1]),
            masses[1] / masses[0],
            places=14,
        )

    def test_three_body_superposition(self):
        distance = 1.0e10
        positions = np.array([[0, 0, 0], [distance, 0, 0], [0, distance, 0]], float)
        acceleration = phys.compute_accelerations(positions, np.ones(3))
        expected = phys.GM_SUN / distance**2
        np.testing.assert_allclose(
            acceleration[0], [expected, expected, 0.0], rtol=2.0e-15
        )

    def test_inverse_square_scaling(self):
        masses = [1.0, 1.0]
        near = phys.compute_accelerations([[0, 0, 0], [1.0e10, 0, 0]], masses)
        far = phys.compute_accelerations([[0, 0, 0], [2.0e10, 0, 0]], masses)
        self.assertAlmostEqual(near[0, 0] / far[0, 0], 4.0, places=14)

    def test_permutation_equivariance(self):
        rng = np.random.default_rng(20260828)
        positions = rng.normal(size=(5, 3)) * 1.0e11
        masses = rng.uniform(0.2, 5.0, size=5)
        permutation = np.array([3, 0, 4, 1, 2])
        original = phys.compute_accelerations(positions, masses)
        permuted = phys.compute_accelerations(
            positions[permutation], masses[permutation]
        )
        np.testing.assert_allclose(
            permuted, original[permutation], rtol=3.0e-15, atol=2.0e-15
        )

    def test_random_system_has_zero_mass_weighted_net_acceleration(self):
        rng = np.random.default_rng(314159)
        positions = rng.normal(size=(8, 3)) * 1.0e12
        masses = rng.uniform(0.1, 10.0, size=8)
        acceleration = phys.compute_accelerations(positions, masses)
        weighted = np.sum(masses[:, None] * acceleration, axis=0)
        scale = np.sum(masses[:, None] * np.abs(acceleration), axis=0)
        np.testing.assert_allclose(
            weighted,
            np.zeros(3),
            rtol=0.0,
            atol=3.0e-15 * float(np.max(scale)),
        )

    def test_translation_invariance(self):
        positions = np.array([[1.0e10, 2.0e10, 3.0e10], [-4.0e10, 1.0e10, 0]])
        masses = np.array([1.0, 3.0])
        offset = np.array([7.0e11, -2.0e11, 9.0e11])
        original = phys.compute_accelerations(positions, masses)
        translated = phys.compute_accelerations(positions + offset, masses)
        np.testing.assert_allclose(translated, original, rtol=2.0e-15, atol=1.0e-15)

    def test_rotation_covariance(self):
        positions = np.array([[1.0e10, 2.0e10, 0.0], [-3.0e10, 5.0e10, 0.0]])
        masses = np.array([1.0, 2.0])
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        original = phys.compute_accelerations(positions, masses)
        rotated = phys.compute_accelerations(positions @ rotation.T, masses)
        np.testing.assert_allclose(rotated, original @ rotation.T, rtol=2.0e-15)

    def test_single_body_is_rejected_consistently_with_program_contract(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            phys.compute_accelerations([[1, 2, 3]], [4])

    def test_inputs_are_not_modified(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.0e10, 0.0, 0.0]])
        masses = np.array([1.0, 2.0])
        positions_before = positions.copy()
        masses_before = masses.copy()
        phys.compute_accelerations(positions, masses)
        np.testing.assert_array_equal(positions, positions_before)
        np.testing.assert_array_equal(masses, masses_before)

    def test_coincident_positions_raise(self):
        with self.assertRaisesRegex(ValueError, "point-mass force is singular"):
            phys.compute_accelerations([[0, 0, 0], [0, 0, 0]], [1, 1])

    def test_bad_shapes_raise(self):
        cases = [
            ([[0, 0], [1, 0]], [1, 1]),
            ([[0, 0, 0]], [1, 1]),
            ([[0, 0, 0], [1, 0, 0]], [[1, 1]]),
            ([], []),
        ]
        for positions, masses in cases:
            with self.subTest(positions=positions, masses=masses):
                with self.assertRaises(ValueError):
                    phys.compute_accelerations(positions, masses)

    def test_nonpositive_or_nonfinite_masses_raise(self):
        for masses in ([1, 0], [1, -1], [1, np.inf], [1, np.nan]):
            with self.subTest(masses=masses):
                with self.assertRaises(ValueError):
                    phys.compute_accelerations([[0, 0, 0], [1, 0, 0]], masses)

    def test_nonnumeric_and_nonfinite_positions_raise(self):
        cases = (
            [[0, 0, 0], ["one", 0, 0]],
            [[0, 0, 0], [np.inf, 0, 0]],
            [[0, 0, 0], [np.nan, 0, 0]],
        )
        for positions in cases:
            with self.subTest(positions=positions):
                with self.assertRaises(ValueError):
                    phys.compute_accelerations(positions, [1, 1])

    def test_out_of_range_separation_raises(self):
        with self.assertRaisesRegex(ValueError, "separation.*floating-point range"):
            phys.compute_accelerations(
                [[-1.0e308, 0, 0], [1.0e308, 0, 0]], [1, 1]
            )

    def test_very_close_noncoincident_pair_remains_finite(self):
        acceleration = phys.compute_accelerations(
            [[-0.5, 0, 0], [0.5, 0, 0]], [1.0e-20, 2.0e-20]
        )
        self.assertTrue(np.all(np.isfinite(acceleration)))
        np.testing.assert_allclose(
            acceleration,
            [
                [2.0e-20 * phys.GM_SUN, 0.0, 0.0],
                [-1.0e-20 * phys.GM_SUN, 0.0, 0.0],
            ],
            rtol=0.0,
            atol=5.0e-8,
        )

    def test_large_n_force_sanity(self):
        rng = np.random.default_rng(8675309)
        n_bodies = 200
        positions = rng.normal(size=(n_bodies, 3)) * 1.0e13
        masses = rng.uniform(0.1, 10.0, size=n_bodies)
        acceleration = phys.compute_accelerations(positions, masses)
        self.assertEqual(acceleration.shape, (n_bodies, 3))
        self.assertTrue(np.all(np.isfinite(acceleration)))


class TestConservationFunctions(unittest.TestCase):
    def test_known_energy(self):
        positions = np.array([[-5.0e10, 0, 0], [5.0e10, 0, 0]])
        velocities = np.array([[0, -2.0e4, 0], [0, 2.0e4, 0]])
        masses = np.array([1.0, 1.0])
        expected = 4.0e8 - phys.GM_SUN / 1.0e11
        self.assertAlmostEqual(
            phys.scaled_total_energy(positions, velocities, masses), expected
        )

        kinetic, potential = phys.scaled_energy_components(
            positions, velocities, masses
        )
        self.assertAlmostEqual(kinetic, 4.0e8)
        self.assertAlmostEqual(potential, -phys.GM_SUN / 1.0e11)

    def test_known_momentum(self):
        momentum = phys.scaled_total_momentum(
            [[1, 2, 3], [-2, 4, 1]], [2, 3]
        )
        np.testing.assert_array_equal(momentum, [-4, 16, 9])

    def test_known_angular_momentum(self):
        angular = phys.scaled_total_angular_momentum(
            [[1, 0, 0], [0, 2, 0]],
            [[0, 3, 0], [-4, 0, 0]],
            [2, 5],
        )
        np.testing.assert_array_equal(angular, [0, 0, 46])

    def test_conservation_state_keys_and_values(self):
        positions = [[-1, 0, 0], [1, 0, 0]]
        velocities = [[0, -2, 0], [0, 2, 0]]
        masses = [1, 1]
        state = phys.conservation_state(positions, velocities, masses)
        self.assertEqual(
            set(state),
            {
                "energy",
                "kinetic_energy",
                "potential_energy",
                "momentum",
                "angular_momentum",
            },
        )
        self.assertEqual(
            state["energy"],
            state["kinetic_energy"] + state["potential_energy"],
        )
        self.assertTrue(np.isfinite(state["energy"]))
        np.testing.assert_array_equal(state["momentum"], np.zeros(3))

    def test_energy_coincidence_raises(self):
        with self.assertRaisesRegex(ValueError, "potential energy is singular"):
            phys.scaled_total_energy(
                [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [1, 1]
            )

    def test_velocity_shape_and_values_are_validated(self):
        for velocities in ([[0, 0, 0]], [[0, 0, 0], [np.inf, 0, 0]], "bad"):
            with self.subTest(velocities=velocities):
                with self.assertRaises(ValueError):
                    phys.scaled_total_energy(
                        [[0, 0, 0], [1, 0, 0]], velocities, [1, 1]
                    )

    def test_momentum_validates_shape_and_finiteness(self):
        cases = (([[1, 2]], [1]), ([[1, 2, 3]], [0]), ([[np.inf, 0, 0]], [1]))
        for velocities, masses in cases:
            with self.subTest(velocities=velocities, masses=masses):
                with self.assertRaises(ValueError):
                    phys.scaled_total_momentum(velocities, masses)

    def test_energy_overflow_raises_cleanly(self):
        with self.assertRaisesRegex(ValueError, "[Ee]nergy.*floating-point range"):
            phys.scaled_total_energy(
                [[-1, 0, 0], [1, 0, 0]],
                [[0, 0, 0], [0, 0, 0]],
                [1.0e300, 1.0e300],
            )

    def test_center_of_mass_of_known_pair(self):
        positions = np.array([[-2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        masses = np.array([2.0, 1.0])
        np.testing.assert_allclose(
            phys.center_of_mass(positions, masses),
            [0.0, 0.0, 0.0],
            atol=1.0e-15,
        )
        velocities = np.array([[0.0, -1.0, 0.0], [0.0, 2.0, 3.0]])
        np.testing.assert_allclose(
            phys.center_of_mass_velocity(velocities, masses),
            [0.0, 0.0, 1.0],
            atol=1.0e-15,
        )

    def test_com_frame_positions_have_zero_mass_weighted_mean(self):
        rng = np.random.default_rng(42)
        positions = rng.normal(size=(5, 4, 3)) * 1.0e11
        masses = rng.uniform(0.3, 4.0, size=4)
        shifted = phys.positions_in_display_frame(positions, masses, "com")
        com = phys.center_of_mass(shifted, masses)
        np.testing.assert_allclose(com, np.zeros((5, 3)), atol=1.0e-3)
        original = phys.positions_in_display_frame(positions, masses, "USER")
        np.testing.assert_array_equal(original, positions)

    def test_uniform_boost_adds_total_mass_times_boost_to_momentum(self):
        masses = np.array([1.0, 1.0, 1.0])
        velocities = np.array(
            [[0.0, 0.0, 0.0], [0.0, -30000.0, 0.0], [-30000.0, 0.0, 0.0]]
        )
        original = phys.scaled_total_momentum(velocities, masses)
        np.testing.assert_allclose(original, [-3.0e4, -3.0e4, 0.0])
        boost = np.array([1.0e5, 0.0, 0.0])
        boosted = phys.scaled_total_momentum(velocities + boost, masses)
        np.testing.assert_allclose(
            boosted, original + float(np.sum(masses)) * boost
        )
        np.testing.assert_allclose(boosted, [2.7e5, -3.0e4, 0.0])

    def test_display_frame_rejects_unknown_names(self):
        with self.assertRaisesRegex(ValueError, "com"):
            phys.positions_in_display_frame(
                [[0, 0, 0], [1, 0, 0]], [1, 1], "lab"
            )


class TestParameterValidation(unittest.TestCase):
    def assert_invalid(self, field, values):
        for value in values:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    driver._validate_params(make_params(**{field: value}))

    def test_valid_parameters(self):
        driver._validate_params(make_params())
        driver._validate_params(
            make_params(n_bodies=np.int64(2), max_steps=np.int64(3))
        )

    def test_n_bodies_validation(self):
        self.assert_invalid("n_bodies", [True, 1, 2.0, "2", None])

    def test_mass_validation(self):
        self.assert_invalid(
            "masses_solar",
            [[1], [1, 0], [1, -1], [1, np.nan], [1, np.inf], [1, "one"]],
        )

    def test_position_validation(self):
        self.assert_invalid(
            "positions_init",
            [
                [[0, 0, 0]],
                [[0, 0], [1, 0]],
                [[0, 0, 0], [np.inf, 0, 0]],
                [[0, 0, 0], ["one", 0, 0]],
            ],
        )

    def test_velocity_validation(self):
        self.assert_invalid(
            "velocities_init",
            [
                [[0, 0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0, 0], [np.nan, 0, 0]],
                [[0, 0, 0], ["fast", 0, 0]],
            ],
        )

    def test_dt_validation(self):
        self.assert_invalid("dt", [True, 0, -1, np.inf, np.nan, "100"])

    def test_max_steps_validation(self):
        self.assert_invalid("max_steps", [True, 0, -1, 2.5, "3"])

    def test_eps1_validation(self):
        self.assert_invalid("eps1", [True, 0, -0.1, 1, np.inf, np.nan, "0.1"])

    def test_eps2_validation(self):
        self.assert_invalid("eps2", [True, 0, -0.1, 0.05, 1, np.nan, "0.001"])

    def test_output_type_validation_and_case_acceptance(self):
        self.assert_invalid("output_type", [None, 3, "movie", " trajectories "])
        driver._validate_params(make_params(output_type="TRAJECTORIES"))

    def test_projection_validation_and_case_acceptance(self):
        self.assert_invalid("projection", [None, 2, "xyz", " xy "])
        driver._validate_params(make_params(projection="YZ"))

    def test_display_frame_validation_and_case_acceptance(self):
        self.assert_invalid("display_frame", [None, 2, "lab", " com "])
        driver._validate_params(make_params(display_frame="USER"))
        driver._validate_params(make_params(display_frame="COM"))

    def test_animation_control_validation(self):
        base = {"output_type": "animation"}
        cases = {
            "animation_mode": [None, 3, "points"],
            "frame_time": [True, 0, -1, np.inf, "50"],
            "frame_interval_ms": [True, 0, -1, 2.5, "20"],
            "trail_time": [True, -1, np.inf, "10"],
            "axis_mode": [None, 4, "dynamic"],
        }
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    params = make_params(**base, **{field: value})
                    with self.assertRaises(ValueError):
                        driver._validate_params(params)

    def test_trajectory_mode_ignores_unused_animation_controls(self):
        driver._validate_params(
            make_params(
                animation_mode=None,
                frame_time=None,
                frame_interval_ms=None,
                trail_time=None,
                axis_mode=None,
            )
        )

    def test_animation_frame_safety_limit_boundary(self):
        driver._validate_params(
            make_params(
                output_type="animation",
                dt=driver.MAX_ANIMATION_FRAMES - 1,
                max_steps=1,
                frame_time=1.0,
            )
        )
        with self.assertRaisesRegex(ValueError, "1,000,000 stored frames"):
            driver._validate_params(
                make_params(
                    output_type="animation",
                    dt=driver.MAX_ANIMATION_FRAMES,
                    max_steps=1,
                    frame_time=1.0,
                )
            )


class TestDriverHelpers(unittest.TestCase):
    def test_relative_vector_change(self):
        old = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
        new = np.array([[6.0, 8.0, 0.0], [0.0, 0.0, 2.0]])
        self.assertAlmostEqual(driver._max_relative_vector_change(old, new), 0.5)

    def test_relative_vector_change_zero_and_nonfinite(self):
        zeros = np.zeros((2, 3))
        self.assertEqual(driver._max_relative_vector_change(zeros, zeros), 0.0)
        changed = zeros.copy()
        changed[0, 0] = 1.0
        self.assertEqual(driver._max_relative_vector_change(zeros, changed), 1.0)
        changed[1, 1] = np.inf
        self.assertEqual(driver._max_relative_vector_change(zeros, changed), np.inf)

    def test_relative_vector_change_handles_large_finite_components(self):
        old = np.array([[1.0e308, 0.0, 0.0]])
        new = np.array([[9.0e307, 0.0, 0.0]])
        self.assertAlmostEqual(driver._max_relative_vector_change(old, new), 0.1)

    def test_hermite_linear_motion_and_endpoints(self):
        p0 = np.array([[1.0, 2.0, 3.0]])
        velocity = np.array([[4.0, -2.0, 1.0]])
        p1 = p0 + velocity * 10.0
        p_mid, v_mid = driver._hermite_state(0, p0, velocity, 10, p1, velocity, 4)
        np.testing.assert_allclose(p_mid, p0 + velocity * 4)
        np.testing.assert_allclose(v_mid, velocity)
        for target, expected in ((0, p0), (10, p1)):
            position, endpoint_velocity = driver._hermite_state(
                0, p0, velocity, 10, p1, velocity, target
            )
            np.testing.assert_allclose(position, expected)
            np.testing.assert_allclose(endpoint_velocity, velocity)

    def test_hermite_clamps_and_handles_nonpositive_interval(self):
        p0 = np.zeros((1, 3))
        p1 = np.ones((1, 3))
        v0 = np.zeros((1, 3))
        v1 = np.ones((1, 3))
        before, _ = driver._hermite_state(0, p0, v0, 1, p1, v1, -1)
        after, _ = driver._hermite_state(0, p0, v0, 1, p1, v1, 2)
        np.testing.assert_allclose(before, p0)
        np.testing.assert_allclose(after, p1)
        same, same_velocity = driver._hermite_state(1, p0, v0, 1, p1, v1, 1)
        np.testing.assert_array_equal(same, p1)
        np.testing.assert_array_equal(same_velocity, v1)

    def test_drift_helpers_zero_reference(self):
        self.assertEqual(driver._fractional_scalar_drift(3, 0), 3)
        self.assertEqual(driver._fractional_scalar_drift(9, 10), 0.1)
        self.assertEqual(driver._vector_drift(np.array([3, 4, 0]), np.zeros(3)), 5)

    def test_vector_drift_avoids_large_finite_norm_overflow(self):
        reference = np.array([1.0e308, 1.0e308, 0.0])
        value = np.array([9.0e307, 1.0e308, 0.0])
        drift = driver._vector_drift(value, reference)
        self.assertTrue(np.isfinite(drift))
        self.assertAlmostEqual(drift, 1.0 / np.sqrt(200.0), places=15)

    def test_vector_drift_rejects_out_of_range_difference(self):
        reference = np.array([-1.0e308, 0.0, 0.0])
        value = np.array([1.0e308, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "drift.*floating-point range"):
            driver._vector_drift(value, reference)

    def test_nonfinite_candidate_cannot_be_swallowed_by_maximum(self):
        for quantity in ("momentum", "angular-momentum"):
            with self.subTest(quantity=quantity):
                with self.assertRaisesRegex(RuntimeError, quantity):
                    driver._checked_maximum_drift(
                        0.0, float("nan"), quantity
                    )

    def test_energy_drift_scale_detects_near_cancellation(self):
        scale, mode = driver._energy_drift_scale(1.0, -1.0 + 1.0e-16)
        self.assertEqual(mode, "characteristic_energy")
        self.assertAlmostEqual(scale, 2.0)

        scale, mode = driver._energy_drift_scale(1.0, -0.9)
        self.assertEqual(mode, "initial_energy")
        self.assertAlmostEqual(scale, 0.1)


class TestSimulation(unittest.TestCase):
    def test_trajectory_result_shapes_metadata_and_initial_state(self):
        params = make_params(max_steps=5)
        result = driver.run_simulation(params)
        self.assertEqual(result["type"], "trajectories")
        self.assertEqual(result["accepted_steps"], 5)
        self.assertEqual(result["positions"].shape, (6, 2, 3))
        self.assertEqual(result["velocities"].shape, (6, 2, 3))
        self.assertEqual(result["times"].shape, (6,))
        self.assertEqual(result["dt_used"].shape, (6,))
        self.assertEqual(result["dt_used"][0], 0.0)
        np.testing.assert_array_equal(result["positions"][0], params.positions_init)
        np.testing.assert_array_equal(result["velocities"][0], params.velocities_init)
        self.assertEqual(result["model_version"], phys.MODEL_VERSION)
        self.assertEqual(result["build_id"], phys.BUILD_ID)
        self.assertEqual(
            result["max_momentum_drift"],
            result["max_fractional_momentum_drift"],
        )
        self.assertEqual(
            result["max_angular_momentum_drift"],
            result["max_fractional_angular_momentum_drift"],
        )
        self.assertEqual(
            result["momentum_drift_normalization"], "absolute_scaled"
        )
        self.assertIsNone(result["momentum_drift_scale"])
        self.assertEqual(
            result["angular_momentum_drift_normalization"], "initial_norm"
        )
        self.assertGreater(result["angular_momentum_drift_scale"], 0.0)

    def test_no_input_mutation(self):
        params = make_params()
        positions_before = [row[:] for row in params.positions_init]
        velocities_before = [row[:] for row in params.velocities_init]
        masses_before = params.masses_solar[:]
        driver.run_simulation(params)
        self.assertEqual(params.positions_init, positions_before)
        self.assertEqual(params.velocities_init, velocities_before)
        self.assertEqual(params.masses_solar, masses_before)

    def test_equal_mass_circular_binary(self):
        separation = 1.0e11
        speed = np.sqrt(phys.GM_SUN / (2.0 * separation))
        period = 2.0 * np.pi * (separation / 2.0) / speed
        steps = 3000
        params = make_params(
            masses_solar=[1.0, 1.0],
            positions_init=[[-separation / 2, 0, 0], [separation / 2, 0, 0]],
            velocities_init=[[0, -speed, 0], [0, speed, 0]],
            dt=period / steps,
            max_steps=steps,
        )
        result = driver.run_simulation(params)
        relative = result["positions"][:, 1] - result["positions"][:, 0]
        separations = np.linalg.norm(relative, axis=1)
        self.assertLess(np.max(np.abs(separations / separation - 1.0)), 2.0e-6)
        np.testing.assert_allclose(relative[-1], relative[0], rtol=0.0, atol=2.0e6)
        self.assertLess(result["max_fractional_energy_drift"], 2.0e-8)

    def test_center_of_mass_moves_uniformly(self):
        params = make_params(max_steps=100, dt=500)
        result = driver.run_simulation(params)
        masses = np.asarray(params.masses_solar)
        center = np.sum(
            result["positions"] * masses[None, :, None], axis=1
        ) / masses.sum()
        initial_velocity = np.sum(
            np.asarray(params.velocities_init) * masses[:, None], axis=0
        ) / masses.sum()
        expected = center[0] + result["times"][:, None] * initial_velocity
        np.testing.assert_allclose(center, expected, rtol=0.0, atol=3.0e-5)

    def test_galilean_boost_preserves_relative_integration(self):
        separation = 1.0e11
        speed = np.sqrt(phys.GM_SUN / (2.0 * separation))
        base_velocities = np.array([[0, -speed, 0], [0, speed, 0]])
        common = {
            "masses_solar": [1.0, 1.0],
            "positions_init": [[-separation / 2, 0, 0], [separation / 2, 0, 0]],
            "dt": 1.0e5,
            "max_steps": 200,
            "eps2": 1.0e-7,
        }
        unboosted = driver.run_simulation(
            make_params(**common, velocities_init=base_velocities.tolist())
        )
        boosted = driver.run_simulation(
            make_params(
                **common,
                velocities_init=(base_velocities + [1.0e6, -2.0e6, 3.0e6]).tolist(),
            )
        )
        np.testing.assert_array_equal(unboosted["dt_used"], boosted["dt_used"])
        relative_0 = unboosted["positions"][:, 1] - unboosted["positions"][:, 0]
        relative_1 = boosted["positions"][:, 1] - boosted["positions"][:, 0]
        np.testing.assert_allclose(relative_1, relative_0, rtol=0.0, atol=0.5)

    def test_large_initial_step_is_reduced(self):
        result = driver.run_simulation(
            make_params(
                positions_init=[[-5.0e8, 0, 0], [5.0e8, 0, 0]],
                velocities_init=[[0, 0, 0], [0, 0, 0]],
                masses_solar=[1, 1],
                dt=1.0e5,
                max_steps=5,
            )
        )
        self.assertLess(np.min(result["dt_used"][1:]), 1.0e5)
        self.assertTrue(np.all(result["dt_used"][1:] > 0.0))

    def test_working_timestep_never_exceeds_requested_maximum(self):
        result = driver.run_simulation(make_params(dt=1000, max_steps=100))
        self.assertLessEqual(np.max(result["dt_used"]), 1000)

    def test_animation_frames_are_uniform_physical_times(self):
        result = driver.run_simulation(
            make_params(
                masses_solar=[1.0e-100, 1.0e-100],
                positions_init=[[-1.0e12, 0, 0], [1.0e12, 0, 0]],
                velocities_init=[[1, 2, 3], [-1, -2, -3]],
                dt=1000,
                max_steps=2,
                output_type="animation",
                frame_time=100,
            )
        )
        np.testing.assert_array_equal(result["frame_times"], np.arange(0, 2001, 100))
        self.assertEqual(result["frame_positions"].shape, (21, 2, 3))
        self.assertEqual(result["frame_velocities"].shape, (21, 2, 3))

    def test_frame_schedule_uses_index_multiplication_without_accumulation(self):
        frame_time = 0.1
        result = driver.run_simulation(
            make_params(
                masses_solar=[1.0e-100, 1.0e-100],
                positions_init=[[-1.0e12, 0, 0], [1.0e12, 0, 0]],
                velocities_init=[[0, 0, 0], [0, 0, 0]],
                dt=100.0,
                max_steps=1,
                output_type="animation",
                frame_time=frame_time,
            )
        )
        expected = np.arange(1001, dtype=float) * frame_time
        np.testing.assert_array_equal(result["frame_times"], expected)

    def test_animation_omits_unreached_partial_final_frame(self):
        result = driver.run_simulation(
            make_params(
                dt=100,
                max_steps=2,
                output_type="animation",
                frame_time=150,
            )
        )
        np.testing.assert_array_equal(result["frame_times"], [0, 150])

    def test_animation_metadata_is_normalized(self):
        result = driver.run_simulation(
            make_params(
                output_type="ANIMATION",
                animation_mode="CURRENT POSITIONS",
                projection="XZ",
                axis_mode="AUTO",
            )
        )
        self.assertEqual(result["type"], "animation")
        self.assertEqual(result["animation_mode"], "current positions")
        self.assertEqual(result["projection"], "xz")
        self.assertEqual(result["axis_mode"], "auto")
        self.assertEqual(result["display_frame"], "com")
        np.testing.assert_array_equal(result["masses_solar"], [1.0, 2.0])

    def test_result_records_requested_display_frame(self):
        result = driver.run_simulation(make_params(display_frame="user", max_steps=1))
        self.assertEqual(result["display_frame"], "user")
        defaulted = driver.run_simulation(make_params(max_steps=1))
        self.assertEqual(defaulted["display_frame"], "com")

    def test_com_display_removes_uniform_drift(self):
        params = make_params(
            velocities_init=[[1.0e6, -2.0e6, 3.0e5], [1.0e6, -2.0e6, 3.0e5]],
            max_steps=20,
            dt=200.0,
        )
        result = driver.run_simulation(params)
        displayed = phys.positions_in_display_frame(
            result["positions"], result["masses_solar"], "com"
        )
        com = phys.center_of_mass(displayed, result["masses_solar"])
        np.testing.assert_allclose(com, np.zeros_like(com), atol=1.0e-6)
        unboosted = driver.run_simulation(
            make_params(
                velocities_init=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                max_steps=20,
                dt=200.0,
            )
        )
        unboosted_display = phys.positions_in_display_frame(
            unboosted["positions"], unboosted["masses_solar"], "com"
        )
        np.testing.assert_allclose(
            displayed[:, 1] - displayed[:, 0],
            unboosted_display[:, 1] - unboosted_display[:, 0],
            rtol=0.0,
            atol=1.0,
        )

    def test_near_zero_initial_energy_uses_characteristic_scale(self):
        separation = 1.0e11
        speed = np.sqrt(phys.GM_SUN / separation)
        result = driver.run_simulation(
            make_params(
                masses_solar=[1.0, 1.0],
                positions_init=[[-separation / 2, 0, 0], [separation / 2, 0, 0]],
                velocities_init=[[0, -speed, 0], [0, speed, 0]],
                max_steps=1,
            )
        )
        self.assertEqual(
            result["energy_drift_normalization"], "characteristic_energy"
        )
        self.assertGreater(result["energy_drift_scale"], 0.0)
        self.assertTrue(np.isfinite(result["max_fractional_energy_drift"]))


class TestPlotting(unittest.TestCase):
    def tearDown(self):
        plotting.plt.close("all")

    def test_projection_indices(self):
        self.assertEqual(plotting._projection_indices("XY"), (0, 1, "x", "y"))
        self.assertEqual(plotting._projection_indices("xz"), (0, 2, "x", "z"))
        self.assertEqual(plotting._projection_indices("yz"), (1, 2, "y", "z"))
        with self.assertRaises(ValueError):
            plotting._projection_indices("xyz")
        for invalid in (None, 3, ["x", "y"]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    plotting._projection_indices(invalid)

    def test_fixed_limits_are_square_and_nonzero(self):
        projected = np.array([[[1.0, 2.0]], [[1.0, 2.0]]])
        xlim, ylim = plotting._fixed_limits(projected)
        self.assertAlmostEqual(xlim[1] - xlim[0], ylim[1] - ylim[0])
        self.assertGreater(xlim[1] - xlim[0], 0.0)

    def test_plot_trajectories_rejects_wrong_result_type(self):
        with self.assertRaisesRegex(ValueError, "trajectories result"):
            plotting.plot_trajectories({"type": "animation"})

    def test_plot_energy_rejects_wrong_result_type(self):
        with self.assertRaisesRegex(ValueError, "trajectories mode"):
            plotting.plot_energy_drift({"type": "animation"})

    def test_plot_energy_validates_arrays_and_normalization_metadata(self):
        malformed = (
            {"type": "trajectories"},
            {"type": "trajectories", "energies": [], "times": []},
            {
                "type": "trajectories",
                "energies": [1.0, 2.0],
                "times": [0.0],
            },
            {
                "type": "trajectories",
                "energies": [1.0, np.inf],
                "times": [0.0, 1.0],
            },
        )
        for result in malformed:
            with self.subTest(result=result):
                with self.assertRaises(ValueError):
                    plotting.plot_energy_drift(result)

        for normalization in ("initial_energy", "characteristic_energy"):
            with self.subTest(normalization=normalization):
                result = {
                    "type": "trajectories",
                    "energies": [1.0],
                    "times": [0.0],
                    "energy_drift_normalization": normalization,
                    "energy_drift_scale": None,
                }
                with self.assertRaisesRegex(ValueError, "energy_drift_scale"):
                    plotting.plot_energy_drift(result)

        result = {
            "type": "trajectories",
            "energies": [1.0],
            "times": [0.0],
            "energy_drift_normalization": "mystery",
        }
        with self.assertRaisesRegex(ValueError, "energy_drift_normalization"):
            plotting.plot_energy_drift(result)

    def test_animate_rejects_wrong_or_empty_result(self):
        with self.assertRaisesRegex(ValueError, "animation result"):
            plotting.animate_multiple({"type": "trajectories"})
        with self.assertRaisesRegex(ValueError, "No animation frames"):
            plotting.animate_multiple(
                {"type": "animation", "frame_times": [], "frame_positions": []}
            )

    @mock.patch.object(plotting.plt, "show")
    def test_static_plot_functions_run(self, show):
        result = driver.run_simulation(make_params(max_steps=2))
        plotting.plot_trajectories(result, projection="yz")
        plotting.plot_energy_drift(result)
        self.assertEqual(show.call_count, 2)

    @mock.patch.object(plotting.plt, "show")
    def test_near_zero_initial_energy_plot_uses_characteristic_label(self, show):
        result = {
            "type": "trajectories",
            "energies": np.array([1.0e-16, 2.5e-16]),
            "times": np.array([0.0, 86400.0]),
            "energy_drift_normalization": "characteristic_energy",
            "energy_drift_scale": 2.0,
        }
        plotting.plot_energy_drift(result)
        self.assertIn("K_0", plotting.plt.gca().get_ylabel())
        show.assert_called_once()

    @mock.patch.object(plotting.plt, "show")
    def test_animation_constructs_timer_and_state(self, show):
        result = driver.run_simulation(
            make_params(output_type="animation", max_steps=2, frame_time=50)
        )
        controls = plotting.animate_multiple(result)
        self.assertIn("timer", controls)
        self.assertEqual(controls["state"]["index"], 0)
        show.assert_called_once()

    @mock.patch.object(plotting.plt, "show")
    def test_animation_f_key_toggles_display_frame_and_note(self, show):
        result = driver.run_simulation(
            make_params(output_type="animation", max_steps=2, frame_time=50)
        )
        controls = plotting.animate_multiple(result)
        self.assertEqual(controls["state"]["display_frame"], "com")
        event = mock.Mock()
        event.key = "f"
        controls["on_key"](event)
        self.assertEqual(controls["state"]["display_frame"], "user")
        note = controls["frame_note"].get_text()
        self.assertTrue(note.startswith("Switched — Display frame: user"))
        self.assertNotIn("E=", note)
        self.assertIsNone(controls["frame_note"].get_bbox_patch())
        totals = controls["totals_note"].get_text()
        self.assertIn("interpolated-frame totals", totals)
        self.assertIn("E=", totals)
        self.assertIn("\nP=", totals)
        self.assertIn("\nL=", totals)
        event.key = "F"
        controls["on_key"](event)
        self.assertEqual(controls["state"]["display_frame"], "com")
        self.assertIn("COM", controls["frame_note"].get_text())
        show.assert_called_once()

    @mock.patch.object(plotting.plt, "show")
    def test_legacy_result_without_masses_is_labeled_user_frame(self, show):
        positions = np.array(
            [[[0.0, 0.0, 0.0], [1.0e10, 0.0, 0.0]],
             [[1.0e9, 0.0, 0.0], [1.1e10, 0.0, 0.0]]]
        )
        result = {
            "type": "trajectories",
            "positions": positions,
        }
        self.assertEqual(plotting._resolve_display_frame(result), "user")
        plotting.plot_trajectories(result)
        title = plotting.plt.gca().get_title()
        self.assertIn("user frame", title)
        show.assert_called_once()

    def test_explicit_com_without_masses_raises(self):
        result = {
            "type": "trajectories",
            "positions": np.zeros((2, 2, 3)),
            "display_frame": "com",
        }
        with self.assertRaisesRegex(ValueError, "masses_solar"):
            plotting._resolve_display_frame(result)


class TestCommandLineAndSamples(unittest.TestCase):
    def test_all_driver_fields_have_a_cli_default(self):
        import main as multiple_main

        args = multiple_main.parse_args([])
        for name in driver.SimulationParams.__dataclass_fields__:
            self.assertTrue(hasattr(args, name), name)
        self.assertEqual(args.animation_mode, "trails")

    def test_cli_parses_matrix_and_normalized_selector(self):
        import main as multiple_main

        args = multiple_main.parse_args([
            "--n_bodies", "2", "--masses_solar", "1,2",
            "--positions_init", "1e10,0,0;-1e10,0,0",
            "--velocities_init", "0,1000,0;0,-1000,0",
            "--animation_mode", "current_positions", "--dt", "500",
        ])
        self.assertEqual(args.positions_init[1], [-1e10, 0, 0])
        self.assertEqual(args.animation_mode, "current_positions")
        self.assertEqual(args.dt, 500)
        negative_first = multiple_main.parse_args([
            "--n_bodies", "2", "--masses_solar", "1,1",
            "--positions_init", "-1e10,0,0;1e10,0,0",
            "--velocities_init", "0,0,0;0,0,0",
        ])
        self.assertEqual(negative_first.positions_init[0][0], -1e10)

    def test_cli_rejects_mismatched_body_count_and_negative_exponent(self):
        import main as multiple_main

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            multiple_main.parse_args(["--n_bodies", "2"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            multiple_main.parse_args(["--dt", "-1e2"])

    def test_samples_cover_exact_endpoints_and_interpolate_accepted_steps(self):
        for output_type in ("trajectories", "animation"):
            with self.subTest(output_type=output_type):
                result = driver.run_simulation(make_params(
                    output_type=output_type, max_steps=2, frame_time=50,
                ))
                samples = result["conservation_samples"]
                self.assertEqual(len(samples), 11)
                self.assertEqual([s["fraction"] for s in samples],
                                 [j / 10 for j in range(11)])
                self.assertEqual(samples[0]["energy"],
                                 result["initial_conservation"]["energy"])
                self.assertEqual(samples[-1]["energy"],
                                 result["final_conservation"]["energy"])
                self.assertAlmostEqual(samples[5]["time"], result["final_time"] / 2)
                if output_type == "trajectories":
                    t = np.asarray(result["times"])
                    fit = np.polynomial.polynomial.polyfit(
                        t / t[-1], result["energies"], deg=2,
                    )
                    expected = np.polynomial.polynomial.polyval(.5, fit)
                    self.assertAlmostEqual(samples[5]["energy"], expected,
                                           delta=1e-9 * max(abs(expected), 1))

    def test_one_step_samples_use_two_point_fallback(self):
        result = driver.run_simulation(make_params(
            output_type="animation", max_steps=1, frame_time=50,
        ))
        midpoint = result["conservation_samples"][5]
        first = result["initial_conservation"]
        last = result["final_conservation"]
        for key, name in (("energy", "energy"),
                          ("kinetic_energy", "kinetic_energy")):
            self.assertAlmostEqual(midpoint[key],
                                   (first[name] + last[name]) / 2,
                                   delta=1e-10 * max(abs(midpoint[key]), 1))

    @mock.patch.object(plotting.plt, "show")
    def test_frame_labels_are_small_and_unboxed(self, show):
        result = driver.run_simulation(make_params(
            output_type="trajectories", max_steps=2,
        ))
        plotting.plot_trajectories(result)
        notes = [text for text in plotting.plt.gca().texts
                 if "Display frame:" in text.get_text()]
        self.assertEqual(len(notes), 1)
        self.assertLessEqual(notes[0].get_fontsize(), 8)
        self.assertIsNone(notes[0].get_bbox_patch())
        show.assert_called_once()


class TestBuildDocumentationAndCompatibility(unittest.TestCase):
    @staticmethod
    def independent_build_id():
        digest = hashlib.sha256()
        for name in CORE_MODULE_FILES:
            content = (MODULE_DIR / name).read_text(encoding="utf-8").encode("utf-8")
            digest.update(name.encode("utf-8"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        return digest.hexdigest()[:12]

    def test_build_coverage_and_hash(self):
        self.assertEqual(phys.BUILD_ID_COVERS, CORE_MODULE_FILES)
        self.assertEqual(phys.BUILD_ID, self.independent_build_id())
        self.assertNotEqual(phys.BUILD_ID, "unknown")

    def test_all_core_sources_parse_as_python_310(self):
        for name in CORE_MODULE_FILES:
            with self.subTest(name=name):
                source = (MODULE_DIR / name).read_text(encoding="utf-8")
                ast.parse(source, filename=name, feature_version=(3, 10))

    def test_version_command_from_module_directory(self):
        completed = subprocess.run(
            [sys.executable, "main.py", "--version"],
            cwd=MODULE_DIR,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(
            completed.stdout.strip(),
            f"Multiple {phys.MODEL_VERSION} (build {phys.BUILD_ID})",
        )

    def test_help_version_and_build_match_program(self):
        help_text = HELP_PATH.read_text(encoding="utf-8")
        match = re.search(
            r'<p id="version_build"[^>]*>\s*Version\s+([0-9.]+)'
            r'(?:&nbsp;)+Build\s+([0-9a-f]+)\s*</p>',
            help_text,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), phys.MODEL_VERSION)
        self.assertEqual(match.group(2), phys.BUILD_ID)

    def test_help_describes_current_defaults_and_modes(self):
        help_text = HELP_PATH.read_text(encoding="utf-8")
        required_fragments = (
            "60000",
            "0.005",
            "1.0e-7",
            "2.0e5",
            "6.0e5",
            "'animation'",
            "'current positions'",
            "'trails'",
            "'xy'",
            "'xz'",
            "'yz'",
            "'com'",
            "display_frame",
            "cubic-Hermite",
            "velocity increment",
            "Introductory",
            "Intermediate",
            "Advanced",
            "E_internal",
            "interpolated-frame totals",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, help_text)

    def test_main_executable_configuration_has_documented_defaults(self):
        import main as multiple_main

        args = multiple_main.parse_args([])
        for name in ("max_steps", "eps1", "eps2", "display_frame"):
            self.assertEqual(getattr(args, name), multiple_main.DEFAULTS[name])
        self.assertEqual(args.max_steps, 60000)
        self.assertEqual(args.eps1, .005)
        self.assertEqual(args.eps2, 1e-7)
        self.assertEqual(args.display_frame, "com")

    def test_help_defines_current_output_terminology(self):
        help_text = HELP_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "not an <code>output_type</code> value",
            help_text,
        )
        self.assertIn("two <code>animation_mode</code> choices", help_text)
        self.assertNotIn("output_type='current positions'", help_text)

    def test_help_documents_build_id_coverage(self):
        help_text = HELP_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "Build identifier covers the four Python program modules",
            help_text,
        )
        self.assertIn("Help-only or test-only edits do not change it", help_text)

    def test_help_has_correct_energy_equation_without_malformed_residue(self):
        help_text = HELP_PATH.read_text(encoding="utf-8")
        self.assertIn(r"\frac{Gm_A m_B}{r_{AB}}", help_text)
        for residue in (r'\]=""', 'div="">', "gm_am_b", "</b}"):
            with self.subTest(residue=residue):
                self.assertNotIn(residue, help_text)

    def test_help_preserves_and_orders_key_exercises(self):
        help_text = HELP_PATH.read_text(encoding="utf-8")
        titles = (
            "Two-body sanity check",
            "Default three-body encounter",
            "Out-of-plane encounter",
            "Numerical convergence",
            "Center-of-mass frame",
            "Binary hardening",
            "Small star cluster",
            "Compare with MercPert",
            "Galaxy collision — optional toy model",
        )
        locations = [help_text.index(title) for title in titles]
        self.assertEqual(locations, sorted(locations))

    def test_development_history_is_confined_to_license_provenance(self):
        help_text = HELP_PATH.read_text(encoding="utf-8")
        pre_license, license_and_after = help_text.split('<section id="license">', 1)
        for suspicious in ("Copilot", "Gemini", "Claude", "Audit", "legacy fix"):
            with self.subTest(suspicious=suspicious):
                self.assertNotIn(suspicious, pre_license)
        self.assertIn("port and extension", license_and_after)
        self.assertIn("Triana/Java", license_and_after)

    def test_help_scenario_cards_each_have_one_difficulty_badge(self):
        help_text = HELP_PATH.read_text(encoding="utf-8")
        heads = re.findall(
            r'<div class="sc-head"><div class="sc-title">(.*?)</div>'
            r'<span class="diff diff-\w+">(.*?)</span></div>',
            help_text,
        )
        self.assertEqual(len(heads), 10)
        allowed = {"Introductory", "Intermediate", "Advanced"}
        expected = {
            "Two-body sanity check": "Introductory",
            "Default three-body encounter": "Introductory",
            "Change the masses": "Introductory",
            "Out-of-plane encounter": "Intermediate",
            "Numerical convergence": "Intermediate",
            "Center-of-mass frame": "Intermediate",
            "Binary hardening": "Intermediate",
            "Small star cluster": "Advanced",
            "Compare with MercPert": "Advanced",
            "Galaxy collision — optional toy model": "Advanced",
        }
        found = {}
        for title, badge in heads:
            self.assertIn(badge, allowed)
            found[title] = badge
        self.assertEqual(found, expected)
        self.assertEqual(help_text.count('class="diff '), 10)
        self.assertNotIn("equal the total mass times the boost", help_text)
        self.assertIn(r"P_{\rm new}=\mathbf P_{\rm old}+M\mathbf u", help_text)

    def test_console_totals_report_energy_not_kinetic_only(self):
        import main as multiple_main

        state = phys.conservation_state(
            [[-1.0e10, 0.0, 0.0], [1.0e10, 0.0, 0.0]],
            [[0.0, -1000.0, 0.0], [0.0, 500.0, 0.0]],
            [1.0, 2.0],
        )
        text = multiple_main._format_conservation_totals(
            "Initial", state, [1.0, 2.0]
        )
        self.assertIn("user-frame totals", text)
        self.assertIn("E=", text)
        self.assertIn("E_internal=", text)
        self.assertIn("P=", text)
        self.assertIn("L=", text)
        self.assertNotIn("KE=", text)
        energy = float(state["energy"])
        self.assertIn(f"{energy:.4e}", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
