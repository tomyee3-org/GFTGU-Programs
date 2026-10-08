"""Regression tests for the Multiple Newtonian N-body module.

The locator intentionally supports both repository layouts used for review:

* canonical: ``Multiple/tests/test_physics_multiple.py``;
* flattened: the test file copied beside the four program modules.
"""

import ast
import base64
import contextlib
import hashlib
from html import unescape
from html.parser import HTMLParser
import io
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Optional
import unittest
from unittest import mock

import numpy as np


CORE_MODULE_FILES = (
    "physics_multiple.py",
    "driver_multiple.py",
    "main.py",
    "plot_multiple.py",
)
# Both active tutorials are one documentation set. The archived Reference
# Guide remains optional for its historical text checks.
HELP_FILENAMES = ("Multiple.html",)
PROGRAM_NAME = "Multiple"


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


def find_help_files(module_dir):
    """Find the merged Multiple.html in the program or sibling docs tree."""
    module_dir = Path(module_dir)
    directories = [module_dir]
    for ancestor in (module_dir, *module_dir.parents):
        directories.extend((ancestor / "GFTGU-Documentation" / "Multiple",
                            ancestor / "Multiple-Documentation",
                            ancestor / "Multiple-docs"))
        if ancestor.name != "Multiple":
            directories.append(ancestor / "Multiple")
    for directory in directories:
        candidate = directory / "Multiple.html"
        if candidate.is_file():
            return (candidate,)
    raise FileNotFoundError("Could not find Multiple.html beside the program or in GFTGU-Documentation/Multiple/.")


def original_help_text(test) -> str:
    """Text of the Reference Guide Help, for the tests written for it."""
    if not ORIGINAL_HELP_FILE.is_file():
        test.skipTest("Multiple-original.html is not present; nothing else depends on it")
    return ORIGINAL_HELP_FILE.read_text(encoding="utf-8")


MODULE_DIR = find_module_dir(Path(__file__))
HELP_PATH, = find_help_files(MODULE_DIR)
GROK_HELP_PATH = HELP_PATH  # Both existing check groups read the merged baseline.
BEATS_HELP_FILE: Optional[Path] = HELP_PATH
ORIGINAL_HELP_FILE = HELP_PATH.parent / "Multiple-original.html"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

os.environ.setdefault("MPLBACKEND", "Agg")

import driver_multiple as driver  # noqa: E402
import main as entry  # noqa: E402
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
            with self.assertRaisesRegex(FileNotFoundError, r"all core modules"):
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

    def test_characteristic_momentum_sums_individual_magnitudes(self):
        # Exactly cancelling total momentum: m=[1,2], v=[(0,-1000,0),(0,500,0)].
        characteristic = phys.scaled_characteristic_momentum(
            [[0, -1000, 0], [0, 500, 0]], [1, 2]
        )
        self.assertAlmostEqual(characteristic, 1 * 1000.0 + 2 * 500.0)
        total = phys.scaled_total_momentum(
            [[0, -1000, 0], [0, 500, 0]], [1, 2]
        )
        np.testing.assert_allclose(total, [0.0, 0.0, 0.0])
        # The characteristic scale is always at least the norm of the total
        # (triangle inequality); here the total is exactly zero.
        self.assertGreaterEqual(characteristic, float(np.hypot.reduce(total)))

    def test_characteristic_angular_momentum_sums_individual_magnitudes(self):
        characteristic = phys.scaled_characteristic_angular_momentum(
            [[1, 0, 0], [-1, 0, 0]],
            [[0, 3, 0], [0, 3, 0]],
            [2, 5],
        )
        # Body 1: |r x v| = |(1,0,0)x(0,3,0)| = 3; body 2: |(-1,0,0)x(0,3,0)| = 3.
        self.assertAlmostEqual(characteristic, 2 * 3.0 + 5 * 3.0)

    def test_characteristic_momentum_validates_like_total_momentum(self):
        for velocities, masses in (
            ([[1, 2]], [1]), ([[1, 2, 3]], [0]), ([[np.inf, 0, 0]], [1]),
        ):
            with self.subTest(velocities=velocities, masses=masses):
                with self.assertRaises(ValueError):
                    phys.scaled_characteristic_momentum(velocities, masses)

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
        # The caller is expected to pass the scale decided once at the
        # initial state (as run_simulation does via _vector_drift_metadata),
        # not have _vector_drift recompute norm(reference) on every call.
        scale = float(np.hypot.reduce(reference))
        self.assertTrue(np.isfinite(scale))
        drift = driver._vector_drift(value, reference, scale)
        self.assertTrue(np.isfinite(drift))
        self.assertAlmostEqual(drift, 1.0 / np.sqrt(200.0), places=15)

    def test_vector_drift_with_no_scale_is_absolute(self):
        reference = np.array([1.0e308, 1.0e308, 0.0])
        value = np.array([9.0e307, 1.0e308, 0.0])
        drift = driver._vector_drift(value, reference)
        self.assertTrue(np.isfinite(drift))
        np.testing.assert_allclose(drift, 1.0e307, rtol=1.0e-12)

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

    def test_vector_drift_metadata_detects_near_cancellation(self):
        # Reference exactly zero, but individual contributions are not:
        # the scale-aware criterion should use the characteristic scale,
        # not fall back to an unscaled absolute drift.
        scale, mode = driver._vector_drift_metadata(
            np.zeros(3), characteristic=2000.0
        )
        self.assertEqual(mode, "characteristic_scale")
        self.assertAlmostEqual(scale, 2000.0)

        # A reference norm just barely inside the cancellation tolerance of
        # a large characteristic is treated the same way.
        tiny = driver.ENERGY_CANCELLATION_TOLERANCE * 2000.0 * 0.5
        scale, mode = driver._vector_drift_metadata(
            np.array([tiny, 0.0, 0.0]), characteristic=2000.0
        )
        self.assertEqual(mode, "characteristic_scale")
        self.assertAlmostEqual(scale, 2000.0)

        # A reference norm well outside the tolerance uses its own norm,
        # exactly as before this criterion was added.
        scale, mode = driver._vector_drift_metadata(
            np.array([500.0, 0.0, 0.0]), characteristic=2000.0
        )
        self.assertEqual(mode, "initial_norm")
        self.assertAlmostEqual(scale, 500.0)

    def test_vector_drift_metadata_falls_back_to_absolute_when_characteristic_is_zero(self):
        # No characteristic at all (every body exactly still, or no
        # characteristic supplied): the original absolute-drift fallback
        # is preserved for a genuinely zero reference.
        scale, mode = driver._vector_drift_metadata(np.zeros(3), characteristic=0.0)
        self.assertEqual(mode, "absolute_scaled")
        self.assertIsNone(scale)

        scale, mode = driver._vector_drift_metadata(np.zeros(3))
        self.assertEqual(mode, "absolute_scaled")
        self.assertIsNone(scale)

        # A nonzero reference with no characteristic behaves exactly as
        # the pre-existing initial-norm path did.
        scale, mode = driver._vector_drift_metadata(np.array([3.0, 4.0, 0.0]))
        self.assertEqual(mode, "initial_norm")
        self.assertAlmostEqual(scale, 5.0)


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
        # make_params()'s two bodies have exactly cancelling momentum
        # (m=[1,2], v=[(0,-1000,0),(0,500,0)] -> P=(0,0,0) exactly), but the
        # individual body momenta are not zero, so the scale-aware
        # near-cancellation criterion uses the characteristic momentum
        # scale (sum_A m_A|v_A| = 1*1000 + 2*500 = 2000) rather than the
        # unscaled absolute drift.
        self.assertEqual(
            result["momentum_drift_normalization"], "characteristic_scale"
        )
        self.assertAlmostEqual(result["momentum_drift_scale"], 2000.0)
        self.assertEqual(
            result["angular_momentum_drift_normalization"], "initial_norm"
        )
        self.assertGreater(result["angular_momentum_drift_scale"], 0.0)

    def test_predictor_singularity_retries_instead_of_aborting(self):
        # Audit32 Codex #1: a predictor overshoot can drive two bodies onto
        # exactly the same *trial* point even though the accepted state
        # never is singular. compute_accelerations() raises ValueError for
        # that, and the retry loop previously let it propagate uncaught
        # instead of treating it like the existing non-finite-trial
        # rejection (halve dt_work and retry). Reproduction from the
        # audit: with dt=1 the predictor initially overshoots the two
        # bodies onto the same point; the fixed retry loop halves dt_work
        # until the trial configuration is no longer singular, landing on
        # an accepted dt of 2**-8 for these initial conditions.
        params = make_params(
            n_bodies=2,
            masses_solar=[4.0 / phys.GM_SUN, 4.0 / phys.GM_SUN],
            positions_init=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            velocities_init=[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]],
            dt=1.0,
            max_steps=1,
            eps1=0.005,
            eps2=1.0e-7,
        )
        result = driver.run_simulation(params)
        self.assertEqual(result["accepted_steps"], 1)
        self.assertAlmostEqual(result["dt_used"][-1], 2.0**-8)
        self.assertFalse(
            np.allclose(result["positions"][-1, 0], result["positions"][-1, 1])
        )

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

    def test_non_dict_result_is_rejected_with_a_clear_error(self):
        # All three public plotting entry points must reject a non-dict
        # result the same way plot_energy_drift already did, rather than
        # letting the first result.get(...) raise a bare AttributeError.
        for fn in (plotting.plot_trajectories, plotting.plot_energy_drift,
                   plotting.animate_multiple):
            for bad_result in ("not a dict", ["type", "trajectories"], None, 3.0):
                with self.subTest(fn=fn.__name__, bad_result=bad_result):
                    with self.assertRaisesRegex(ValueError, "must be a dict"):
                        fn(bad_result)

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

    def test_plot_trajectories_validates_positions(self):
        malformed = (
            {"type": "trajectories"},
            {"type": "trajectories", "positions": []},
            {"type": "trajectories", "positions": [[1.0, 2.0, 3.0]]},
            {
                "type": "trajectories",
                "positions": [[[1.0, np.inf, 0.0], [0.0, 0.0, 0.0]]],
                "display_frame": "user",
            },
        )
        for result in malformed:
            with self.subTest(result=result):
                with self.assertRaises(ValueError):
                    plotting.plot_trajectories(result)

    def test_zero_body_plot_results_raise_clear_shape_errors(self):
        with self.assertRaisesRegex(ValueError, "positions must have shape.*one body"):
            plotting.plot_trajectories({
                "type": "trajectories", "display_frame": "user",
                "positions": np.empty((1, 0, 3)),
            })
        with self.assertRaisesRegex(ValueError, "frame_positions must have shape.*one body"):
            plotting.animate_multiple({
                "type": "animation", "frame_times": [0.0],
                "frame_positions": np.empty((1, 0, 3)),
            })

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

    def test_animate_validates_frame_arrays_before_other_fields(self):
        # Missing/malformed frame_times or frame_positions is diagnosed
        # without requiring the other animation fields to be present.
        malformed_frames = (
            {"type": "animation"},
            {"type": "animation", "frame_times": [0.0], "frame_positions": []},
            {
                "type": "animation",
                "frame_times": [0.0, 1.0],
                "frame_positions": [[[0.0, 0.0, 0.0]]],
            },
            {
                "type": "animation",
                "frame_times": [0.0, np.inf],
                "frame_positions": [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
            },
            # A scalar or otherwise wrong-ndim frame_positions becomes a
            # 0-d/2-d array: indexing .shape[0] before checking .ndim would
            # raise a bare IndexError instead of ValueError here.
            {"type": "animation", "frame_times": [0.0], "frame_positions": 7},
            {"type": "animation", "frame_times": [0.0], "frame_positions": None},
            {
                "type": "animation",
                "frame_times": [0.0],
                "frame_positions": [[1.0, 2.0]],
            },
        )
        for result in malformed_frames:
            with self.subTest(result=result):
                with self.assertRaises(ValueError):
                    plotting.animate_multiple(result)

    def test_animate_validates_remaining_scalar_and_selector_fields(self):
        base = {
            "type": "animation",
            "frame_times": [0.0, 1.0],
            "frame_positions": [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
            "animation_mode": "trails",
            "frame_time": 1.0,
            "frame_interval_ms": 50,
            "trail_time": 1.0,
            "projection": "xy",
            "axis_mode": "fixed",
            "display_frame": "user",
        }
        for key, bad_value in (
            ("animation_mode", "orbits"),
            ("frame_time", 0.0),
            ("frame_interval_ms", 0),
            ("trail_time", -1.0),
            ("axis_mode", "zoom"),
        ):
            with self.subTest(key=key, bad_value=bad_value):
                broken = dict(base, **{key: bad_value})
                with self.assertRaises(ValueError):
                    plotting.animate_multiple(broken)

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
    def test_animation_f_key_updates_frame_title_without_redundant_notes(self, show):
        result = driver.run_simulation(
            make_params(output_type="animation", max_steps=2, frame_time=50)
        )
        controls = plotting.animate_multiple(result)
        self.assertEqual(controls["state"]["display_frame"], "com")
        event = mock.Mock()
        event.key = "f"
        controls["on_key"](event)
        self.assertEqual(controls["state"]["display_frame"], "user")
        axes = controls["figure"].axes[0]
        self.assertIn("user frame", axes.get_title())
        self.assertEqual(len(axes.texts), 1)
        self.assertIn("frame 1 /", axes.texts[0].get_text())
        self.assertEqual(len(axes.get_legend().texts), 2)
        event.key = "F"
        controls["on_key"](event)
        self.assertEqual(controls["state"]["display_frame"], "com")
        self.assertIn("COM frame", axes.get_title())
        self.assertEqual(len(axes.texts), 1)
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
    def test_static_plot_title_identifies_frame_without_duplicate_note(self, show):
        result = driver.run_simulation(make_params(
            output_type="trajectories", max_steps=2,
        ))
        plotting.plot_trajectories(result)
        axes = plotting.plt.gca()
        self.assertIn("COM frame", axes.get_title())
        self.assertEqual(len(axes.texts), 0)
        self.assertEqual(len(axes.get_legend().texts), len(result["masses_solar"]))
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
        self._check_defaults_and_modes(HELP_PATH.read_text(encoding="utf-8"), original=False)

    def test_original_help_describes_current_defaults_and_modes(self):
        self._check_defaults_and_modes(original_help_text(self), original=True)

    def _check_defaults_and_modes(self, help_text, original):
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
        )
        if original:
            # this sentence belongs to the Reference Guide's wording
            required_fragments += ("console samples are independent of",)
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
        help_text = original_help_text(self)
        self.assertIn(
            "not an <code>output_type</code> value",
            help_text,
        )
        self.assertIn("two <code>animation_mode</code> choices", help_text)
        self.assertNotIn("output_type='current positions'", help_text)

    def test_help_documents_build_id_coverage(self):
        help_text = original_help_text(self)
        self.assertIn(
            "Build identifier covers the four Python program modules",
            help_text,
        )
        self.assertIn("Help-only or test-only edits do not change it", help_text)

    def test_help_has_no_malformed_equation_residue(self):
        help_text = HELP_PATH.read_text(encoding="utf-8")
        for residue in (r'\]=""', 'div="">', "gm_am_b", "</b}"):
            with self.subTest(residue=residue):
                self.assertNotIn(residue, help_text)

    def test_help_has_correct_energy_equation_without_malformed_residue(self):
        help_text = original_help_text(self)
        self.assertIn(r"\frac{Gm_A m_B}{r_{AB}}", help_text)
        for residue in (r'\]=""', 'div="">', "gm_am_b", "</b}"):
            with self.subTest(residue=residue):
                self.assertNotIn(residue, help_text)

    def test_help_preserves_and_orders_key_exercises(self):
        help_text = original_help_text(self)
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
        license_start = re.search(r'<section\b[^>]*\bid="license"[^>]*>', help_text)
        self.assertIsNotNone(license_start)
        pre_license = help_text[:license_start.start()]
        license_and_after = help_text[license_start.start():]
        for suspicious in ("Copilot", "Gemini", "Claude", "Audit", "legacy fix"):
            with self.subTest(suspicious=suspicious):
                self.assertNotIn(suspicious, pre_license)
        self.assertIn("Bernard Schutz", license_and_after)
        self.assertIn("CC BY-NC-SA 4.0", license_and_after)
        self.assertIn("Original Java", license_and_after)
        self.assertIn("Python port", license_and_after)

    def test_help_scenario_cards_each_have_one_difficulty_badge(self):
        help_text = original_help_text(self)
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


class _HelpInspector(HTMLParser):
    """Collect the small amount of HTML structure used by documentation tests."""

    def __init__(self) -> None:
        super().__init__()
        self.ids = []
        self.fragment_links = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.append(attributes["id"])
        href = attributes.get("href", "")
        if tag == "a" and href.startswith("#"):
            self.fragment_links.append(href[1:])


def tutorial_commands(source):
    """Read whole commands, including continued lines in edited tutorials."""
    commands = []
    for block in re.findall(r'<pre class="code-block"><code>(.*?)</code></pre>',
                            source, flags=re.DOTALL):
        block = unescape(block).replace("\\\r\n", " ").replace("\\\n", " ")
        commands.extend(" ".join(line.split()) for line in block.splitlines()
                        if line.strip().startswith("python main.py"))
    return commands


class PairedTutorialTests(unittest.TestCase):
    def test_merged_help_is_required_and_archives_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp_name:
            folder = Path(temp_name)
            for suffix in ("short", "extended", "original", "java"):
                (folder / "Multiple-{}.html".format(suffix)).write_text("archive", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, r"Multiple\.html"):
                find_help_files(folder)
            merged = folder / "Multiple.html"
            merged.write_text("merged", encoding="utf-8")
            self.assertEqual(find_help_files(folder), (merged,))

    def test_both_tutorials_share_build_options_and_navigation(self):
        for path in (HELP_PATH, GROK_HELP_PATH):
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                inspector = _HelpInspector()
                inspector.feed(source)
                self.assertEqual(len(inspector.ids), len(set(inspector.ids)))
                self.assertTrue(set(inspector.fragment_links).issubset(inspector.ids))
                for beat in range(9):
                    self.assertIn(f"beat{beat}", inspector.ids)
                self.assertRegex(
                    source,
                    rf"Version\s+{re.escape(phys.MODEL_VERSION)}(?:&nbsp;|\s)+"
                    rf"Build\s+{re.escape(phys.BUILD_ID)}",
                )
                for name in entry.DEFAULTS:
                    self.assertIn("--" + name, source)
                self.assertIn("(x, y)", source)
                self.assertNotIn("far fewer position and velocity states", source)
                self.assertNotIn("Claude tutorial", source)
                self.assertNotIn("Grok tutorial", source)

    def test_grok_commands_parse_and_activities_are_complete(self):
        source = GROK_HELP_PATH.read_text(encoding="utf-8")
        commands = tutorial_commands(source)
        self.assertGreaterEqual(len(commands), 14)
        for command in commands:
            with self.subTest(command=command), contextlib.redirect_stderr(io.StringIO()):
                entry.parse_args(shlex.split(command)[2:])
        self.assertEqual([int(i) for i in re.findall(r'<b>Try (\d+):</b>', source)],
                         list(range(1, 11)))

    def test_instructor_guide_matches_documented_release_and_has_nine_images(self):
        guide = (HELP_PATH.parent / "SampleOutputs" /
                 "Multiple-SampleOutputs_Guide.html").read_text(encoding="utf-8")
        # The guide and release notes are intentionally held at the previous
        # release during copy editing. Active Help follows the working core;
        # figure provenance follows the release recorded with those figures.
        notes = HELP_PATH.with_name("Multiple-ReleaseNotes.html").read_text(encoding="utf-8")
        version = re.search(r'<b>Version:</b>\s*([0-9.]+)', notes)
        build = re.search(r'<b>Build:</b>\s*([0-9a-f]+)', notes)
        self.assertIsNotNone(version)
        self.assertIsNotNone(build)
        self.assertIn(f"Version {version.group(1)}", guide)
        self.assertIn(f"Build {build.group(1)}", guide)
        images = re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)', guide)
        self.assertEqual(len(images), 9)
        for encoded in images:
            self.assertTrue(base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n"))


@unittest.skipIf(BEATS_HELP_FILE is None, "Beats-format Help not present")
class BeatsHelpTests(unittest.TestCase):
    """Structural checks on the Beats-format tutorial Help."""

    @classmethod
    def setUpClass(cls) -> None:
        assert BEATS_HELP_FILE is not None
        cls.text = BEATS_HELP_FILE.read_text(encoding="utf-8")

    def _section(self, section_id: str) -> str:
        match = re.search(rf'<section id="{section_id}">(.*?)</section>',
                          self.text, flags=re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        return match.group(1)

    def _commands(self):
        return [command for command in tutorial_commands(self.text)
                if command not in ("python main.py --help", "python main.py --version")]

    def test_version_and_build_match_program(self) -> None:
        match = re.search(
            r'<p\s+id="version_build"[^>]*>\s*'
            r'Version\s+([0-9]+\.[0-9]+\.[0-9]+)'
            r'(?:&nbsp;|\s)+Build\s+([0-9a-f]{12})',
            self.text,
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), phys.MODEL_VERSION)
        self.assertEqual(match.group(2), phys.BUILD_ID)

    def test_internal_links_and_ids_are_consistent(self) -> None:
        inspector = _HelpInspector()
        inspector.feed(self.text)
        self.assertEqual(len(inspector.ids), len(set(inspector.ids)))
        self.assertTrue(set(inspector.fragment_links).issubset(inspector.ids))
        for number in range(9):
            self.assertIn(f"beat{number}", inspector.ids)
        for number in range(1, 11):
            self.assertIn(f"exp{number}", inspector.ids)

    def test_numbered_equations_appear_once_and_in_order(self) -> None:
        numbers = [int(n) for n in re.findall(r"Eq\. (\d+) —", self.text)]
        self.assertEqual(numbers, list(range(1, 17)))

    def test_student_help_has_no_development_history(self) -> None:
        lowered = self.text.lower()
        for phrase in ("copilot", "gemini", "codex", "grok", "audit",
                       "ai-generated", "ported from java",
                       "development history"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lowered)

    def test_documents_every_command_line_option(self) -> None:
        for name in entry.DEFAULTS:
            self.assertIn("--" + name, self.text)
        for flag in ("--output_type", "--animation_mode", "--projection",
                     "--axis_mode", "--display_frame",
                     "--show_energy_diagnostic"):
            self.assertIn(flag, self.text)

    def test_every_tutorial_command_is_accepted_by_the_parser(self) -> None:
        commands = self._commands()
        self.assertGreater(len(commands), 20)
        for command in commands:
            with self.subTest(command=command), \
                    contextlib.redirect_stderr(io.StringIO()):
                entry.parse_args(shlex.split(command)[2:])

    def test_beat1_acceleration_one_liner_matches_claimed_magnitudes(self) -> None:
        # Live-verifies the numbers Beat 1's inline python3 -c snippet
        # reports for the three-body initial-acceleration cross-check.
        pos = np.array([[4.6e10, 0., 0.], [-4.6e10, 0., 0.], [0., 4.6e10, 0.]])
        masses = np.array([2., 1., 0.5])
        magnitudes = np.linalg.norm(phys.compute_accelerations(pos, masses), axis=1)
        for actual, claimed in zip(magnitudes, (0.028972, 0.043871, 0.070121)):
            self.assertAlmostEqual(actual, claimed, places=5)
        beat1 = self._section("beat1")
        for claimed in ("0.028972", "0.043871", "0.070121"):
            self.assertIn(claimed, beat1)

    def test_beat4_momentum_and_angular_momentum_fallback_labels(self) -> None:
        # Live-verifies Beat 4's characteristic-scale near-cancellation
        # demonstration, the centerpiece maintenance-item fix for 1.4.0.
        params = driver.SimulationParams(
            n_bodies=2, masses_solar=[1, 1],
            positions_init=[[5e10, 0, 0], [-5e10, 0, 0]],
            velocities_init=[[0, 25760, 0], [0, -25760, 0]],
            dt=2000., max_steps=3000, eps1=.005, eps2=1e-7,
            output_type="trajectories", animation_mode="trails",
            frame_time=2e5, frame_interval_ms=50, trail_time=6e5,
            projection="xy", axis_mode="fixed", display_frame="com",
        )
        result = driver.run_simulation(params)
        self.assertEqual(result["momentum_drift_normalization"], "characteristic_scale")
        self.assertAlmostEqual(result["momentum_drift_scale"], 51520.0, places=1)
        self.assertEqual(result["angular_momentum_drift_normalization"], "initial_norm")
        self.assertAlmostEqual(
            result["angular_momentum_drift_scale"], 2576000000000000.0, delta=1.0
        )
        beat4 = self._section("beat4")
        self.assertIn("characteristic_scale", beat4)
        self.assertIn("51520.0", beat4)
        self.assertIn("initial_norm", beat4)

    def test_galaxy_collision_singularity_command_actually_crashes(self) -> None:
        # Beat 8 / Experiment 10 quote an exact error message from a
        # deliberately over-extended head-on point-mass encounter; confirm
        # the quoted command still raises it (parser-valid, runtime crash).
        beat8 = " ".join(self._section("beat8").split())
        self.assertIn(
            "Multiple error: The adaptive timestep became too small to "
            "advance simulation time in floating-point arithmetic.",
            beat8,
        )
        params = driver.SimulationParams(
            n_bodies=2, masses_solar=[1e11, 1e11],
            positions_init=[[-7.7e20, 0, 0], [7.7e20, 0, 0]],
            velocities_init=[[50000, 0, 0], [-50000, 0, 0]],
            dt=1e12, max_steps=17800,
            eps1=.005, eps2=1e-7, output_type="trajectories",
            animation_mode="trails", frame_time=2e5, frame_interval_ms=50,
            trail_time=6e5, projection="xy", axis_mode="fixed",
            display_frame="com",
        )
        with self.assertRaises(RuntimeError) as failure:
            driver.run_simulation(params)
        self.assertIn("too small to advance simulation time", str(failure.exception))


class TestConservationTableFormatting(unittest.TestCase):
    @staticmethod
    def capture(samples):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            entry.print_conservation_samples({"conservation_samples": samples})
        return output.getvalue().splitlines()

    @staticmethod
    def sample(fraction, energy, momentum, angular):
        return {"fraction": fraction, "time": fraction * 1234567.0,
                "energy": energy, "internal_energy": -1.23456789e9,
                "kinetic_energy": 9.87654321e10,
                "momentum": momentum, "angular_momentum": angular}

    def test_eleven_rows_preserve_units_values_and_five_significant_digits(self):
        samples = [self.sample(i / 10, (-1) ** i * 7.377412345e9,
                               [(-1) ** i * 2.7e5, -3e4, -0.0],
                               [0.0, 1.234567e7, (-1) ** i * 1.84e15])
                   for i in range(11)]
        before = repr(samples)
        lines = self.capture(samples)
        self.assertEqual(len(lines), 14)
        self.assertEqual(lines[1], "t [days]; E, E_internal, K [m^2/s^2]; P [m/s]; L [m^2/s]")
        self.assertEqual(lines[2].split(), ["Fraction", "t", "E", "E_internal", "K", "P", "L"])
        for sample, line in zip(samples, lines[3:]):
            self.assertEqual(line.split()[0], f"{sample['fraction']:.1f}")
            values = [sample["time"] / 86400, sample["energy"],
                      sample["internal_energy"], sample["kinetic_energy"],
                      *sample["momentum"], *sample["angular_momentum"]]
            printed = re.findall(r"-?\d\.\d{4}e[+-]\d+", line)
            self.assertEqual(printed, [f"{float(value):.4e}" for value in values])
        self.assertEqual(repr(samples), before)

    def test_changing_signs_keeps_scalar_and_vector_columns_aligned(self):
        samples = [self.sample(0.0, 7.3774e9, [2.7e5, -3e4, 0], [0, 0, -1.84e15]),
                   self.sample(1.0, -1.2735e9, [-2.7e5, 3e4, -0.0], [-1, 2, 1.84e15])]
        lines = self.capture(samples)
        header, rows = lines[2], lines[3:]
        starts = [header.index(name) for name in ("t", "E", "E_internal", "K", "P", "L")]
        self.assertEqual(len(rows[0]), len(rows[1]))
        for a, b in zip(starts[:4], starts[1:5]):
            self.assertEqual(rows[0][a:b].index('.'), rows[1][a:b].index('.'))
        for row in rows:
            self.assertEqual(row[starts[4]], '(')
            self.assertEqual(row[starts[5]], '(')
        for vector_start in starts[4:]:
            self.assertEqual([m.start() for m in re.finditer(r"[(),]", rows[0][vector_start:])],
                             [m.start() for m in re.finditer(r"[(),]", rows[1][vector_start:])])
        self.assertIn("( 2.7000e+05, -3.0000e+04,  0.0000e+00)", rows[0])

    def test_three_digit_exponents_expand_columns_without_truncating(self):
        samples = [self.sample(0.0, 1e9, [1, 0, -1], [1, -1, 0]),
                   self.sample(1.0, -1e300, [-1e300, 1e-300, 0], [1e300, -1e-300, 0])]
        lines = self.capture(samples)
        header, rows = lines[2], lines[3:]
        self.assertEqual(len(rows[0]), len(rows[1]))
        for label in ("P", "L"):
            start = header.index(label)
            self.assertTrue(all(row[start] == '(' for row in rows))
        self.assertIn("-1.0000e+300", rows[1])
        self.assertIn("1.0000e-300", rows[1])



AU = 1.495978707e11
YEAR = 365.25 * 86400.0


def sun_jupiter_params(**overrides):
    """A circular Sun-Jupiter pair with its centre of mass at rest."""
    values = dict(
        n_bodies=2, masses_solar=[1.0, 9.548e-4],
        positions_init=[[-7.4241e8, 0.0, 0.0], [7.77555e11, 0.0, 0.0]],
        velocities_init=[[0.0, -12.462, 0.0], [0.0, 13051.96, 0.0]],
        dt=1.87e6, max_steps=200, eps1=0.05, eps2=1e-7,
        output_type="trajectories", display_frame="com",
    )
    values.update(overrides)
    return driver.SimulationParams(**values)


def kepler_position(time, aphelion, perihelion):
    """Independent two-body position for a particle that starts at aphelion
    (-aphelion, 0, 0) moving in -y about one solar mass at the origin."""
    a = 0.5 * (aphelion + perihelion)
    e = (aphelion - perihelion) / (aphelion + perihelion)
    mean_anomaly = np.pi + np.sqrt(phys.GM_SUN / a ** 3) * time
    E = mean_anomaly
    for _ in range(100):
        E = E - (E - e * np.sin(E) - mean_anomaly) / (1.0 - e * np.cos(E))
    return np.array([a * (np.cos(E) - e), a * np.sqrt(1.0 - e * e) * np.sin(E), 0.0])


def kepler_turning(total_time, aphelion, perihelion, samples=200000):
    """Total angle [deg] swept about the origin by that Kepler orbit."""
    points = np.array([kepler_position(t, aphelion, perihelion)
                       for t in np.linspace(0.0, total_time, samples)])
    angles = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    return float(np.degrees(angles[-1] - angles[0]))


class TestTestParticles(unittest.TestCase):
    """Massless test particles: physics, integration, removal and output."""

    def test_acceleration_matches_a_negligible_mass_body(self):
        massive = np.array([[0.0, 0.0, 0.0], [3.0e11, 1.0e11, -2.0e10]])
        masses = np.array([1.0, 0.3])
        test = np.array([[1.0e11, 2.0e10, 5.0e9], [-2.0e11, 0.0, 0.0]])
        got = phys.compute_test_particle_accelerations(test, massive, masses)
        for k, point in enumerate(test):
            reference = phys.compute_accelerations(
                np.vstack((massive, point)), np.append(masses, 1e-30))[-1]
            np.testing.assert_allclose(got[k], reference, rtol=1e-12)
        self.assertEqual(phys.compute_test_particle_accelerations(
            np.zeros((0, 3)), massive, masses).shape, (0, 3))
        with self.assertRaises(ValueError):
            phys.compute_test_particle_accelerations([[0.0, 0.0, 0.0]], massive, masses)
        with self.assertRaises(ValueError):
            phys.compute_test_particle_accelerations([[1.0, 2.0]], massive, masses)

    def test_ring_states_are_circular_and_spread_by_the_golden_angle(self):
        radii = np.array([1.0, 2.0, 3.0]) * AU
        center = np.array([1.0e10, -2.0e10, 0.0])
        center_v = np.array([100.0, 200.0, 0.0])
        pos, vel = phys.ring_test_particle_states(radii, 2.0, center, center_v)
        rel_p, rel_v = pos - center, vel - center_v
        np.testing.assert_allclose(np.linalg.norm(rel_p, axis=1), radii, rtol=1e-14)
        np.testing.assert_allclose(np.sum(rel_p * rel_v, axis=1), 0.0, atol=1e-3 * AU)
        a, e = phys.osculating_elements(rel_p, rel_v, 2.0)
        np.testing.assert_allclose(a, radii, rtol=1e-12)
        np.testing.assert_allclose(e, 0.0, atol=1e-7)
        angles = np.arctan2(rel_p[:, 1], rel_p[:, 0])
        np.testing.assert_allclose(np.mod(angles[1] - angles[0], 2 * np.pi),
                                   phys.GOLDEN_ANGLE, rtol=1e-12)
        self.assertTrue(np.all(np.cross(rel_p, rel_v)[:, 2] > 0.0))
        with self.assertRaises(ValueError):
            phys.ring_test_particle_states([0.0], 1.0)

    def test_massive_bodies_are_bitwise_unchanged_by_test_particles(self):
        test_p, test_v = phys.ring_test_particle_states(
            np.array([2.0e10, 6.0e10, 1.5e11]), 3.0)
        for output_type in ("trajectories", "animation"):
            with self.subTest(output_type=output_type):
                base = dict(
                    n_bodies=3, masses_solar=[1.0, 1.0, 1.0],
                    positions_init=entry.DEFAULTS["positions_init"],
                    velocities_init=entry.DEFAULTS["velocities_init"],
                    dt=2000.0, max_steps=600, eps1=0.005, eps2=1e-7,
                    output_type=output_type, frame_time=2e4,
                )
                plain = driver.run_simulation(driver.SimulationParams(**base))
                loaded = driver.run_simulation(driver.SimulationParams(
                    **base, test_positions_init=test_p.tolist(),
                    test_velocities_init=test_v.tolist(),
                    removal_radii=[1e9, 1e9, 1e9], escape_radius=1e13))
                self.assertEqual(plain["n_test_particles"], 0)
                self.assertEqual(loaded["n_test_particles"], 3)
                keys = (("times", "positions", "velocities", "energies", "dt_used")
                        if output_type == "trajectories"
                        else ("frame_times", "frame_positions", "frame_velocities"))
                for key in keys:
                    self.assertTrue(np.array_equal(plain[key], loaded[key]), key)
                for key in ("accepted_steps", "final_time", "max_fractional_energy_drift",
                            "max_momentum_drift", "max_angular_momentum_drift"):
                    self.assertEqual(plain[key], loaded[key], key)

    def test_test_particle_follows_a_kepler_orbit(self):
        # A Sun-like body with a negligible, distant companion: a test
        # particle on a circular 1-AU orbit must return to its start after
        # one Kepler period.
        period = 2 * np.pi * np.sqrt(AU ** 3 / phys.GM_SUN)
        pos, vel = phys.ring_test_particle_states([AU], 1.0, phases_rad=[0.0])
        result = driver.run_simulation(driver.SimulationParams(
            n_bodies=2, masses_solar=[1.0, 1e-12],
            positions_init=[[0.0, 0.0, 0.0], [1e16, 0.0, 0.0]],
            velocities_init=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dt=period / 2000, max_steps=2000, eps1=0.5, eps2=1e-12,
            output_type="trajectories", test_positions_init=pos.tolist(),
            test_velocities_init=vel.tolist(), test_center=1))
        self.assertAlmostEqual(result["final_time"] / period, 1.0, places=9)
        path = result["test_positions"][:, 0, :]
        radius = np.linalg.norm(path, axis=1)
        self.assertLess(np.max(np.abs(radius / AU - 1.0)), 1e-6)
        self.assertLess(np.linalg.norm(path[-1] - path[0]) / AU, 1e-4)
        self.assertEqual(result["test_particles"]["fate"], ["survived"])
        self.assertEqual(result["test_particles"]["center"], "body 1")

    def test_test_particle_matches_a_tiny_mass_body(self):
        pos, vel = phys.ring_test_particle_states([4.5 * AU], 1.0 + 9.548e-4)
        massless = driver.run_simulation(sun_jupiter_params(
            eps2=1e-12, test_positions_init=pos.tolist(),
            test_velocities_init=vel.tolist()))
        tiny = driver.run_simulation(sun_jupiter_params(
            n_bodies=3, masses_solar=[1.0, 9.548e-4, 1e-15], eps2=1e-12,
            positions_init=[*sun_jupiter_params().positions_init, pos[0].tolist()],
            velocities_init=[*sun_jupiter_params().velocities_init, vel[0].tolist()]))
        np.testing.assert_array_equal(massless["times"], tiny["times"])
        difference = np.linalg.norm(
            massless["test_positions"][:, 0, :] - tiny["positions"][:, 2, :], axis=1)
        self.assertLess(np.max(difference) / AU, 1e-9)

    def test_encounter_and_escape_are_recorded_and_paths_end(self):
        # Particle 1 falls straight onto the Sun; particle 2 leaves fast.
        # The 2000-s step resolves the fall: near the Sun a step moves the
        # particle less than its 0.01-AU removal radius.
        result = driver.run_simulation(sun_jupiter_params(
            dt=2000.0, max_steps=300,
            test_positions_init=[[0.1 * AU, 0.0, 0.0], [0.0, -4.5 * AU, 0.0]],
            test_velocities_init=[[0.0, 0.0, 0.0], [0.0, -3.0e5, 0.0]],
            removal_radii=[0.01 * AU, 0.355 * AU], escape_radius=5.0 * AU))
        info = result["test_particles"]
        self.assertEqual(info["fate"], ["encounter", "escaped"])
        self.assertEqual(list(info["removal_body"]), [1, 0])
        self.assertTrue(np.all(np.isfinite(info["removal_time_s"])))
        for k in range(2):
            # Removal can happen inside a step (at a particle substep), so
            # the stored path ends at the first accepted state after it.
            index = int(np.searchsorted(result["times"], info["removal_time_s"][k]))
            self.assertGreater(info["removal_time_s"][k], result["times"][index - 1])
            self.assertLessEqual(info["removal_time_s"][k], result["times"][index])
            self.assertTrue(np.all(np.isfinite(result["test_positions"][:index, k])))
            self.assertTrue(np.all(np.isnan(result["test_positions"][index:, k])))
        # Removal is detected at an accepted step, just past the boundary.
        last = info["final_positions"]
        self.assertLessEqual(np.linalg.norm(last[0] - result["positions"][-1, 0]),
                             0.01 * AU)

    def test_survival_mode_stores_no_paths_and_stops_when_empty(self):
        result = driver.run_simulation(sun_jupiter_params(
            output_type="survival", max_steps=500,
            test_positions_init=[[0.0, -2.0 * AU, 0.0]],
            test_velocities_init=[[0.0, -2.0e5, 0.0]], escape_radius=5.0 * AU))
        self.assertEqual(result["type"], "survival")
        self.assertLess(result["accepted_steps"], 500)
        removed_at = result["test_particles"]["removal_time_s"][0]
        self.assertLessEqual(removed_at, result["final_time"])
        self.assertGreater(removed_at, result["final_time"] - 1.87e6)
        for key in ("positions", "test_positions", "frame_positions"):
            self.assertNotIn(key, result)
        self.assertEqual(len(result["conservation_samples"]), 11)

    def test_invalid_test_particle_inputs_are_rejected(self):
        ring_p, ring_v = phys.ring_test_particle_states([3.0 * AU], 1.0)
        good = dict(test_positions_init=ring_p.tolist(), test_velocities_init=ring_v.tolist())
        cases = {
            "survival without particles": dict(output_type="survival"),
            "positions without velocities": dict(test_positions_init=ring_p.tolist()),
            "mismatched velocities": dict(test_positions_init=ring_p.tolist(),
                                          test_velocities_init=[[0, 0, 0], [0, 0, 0]]),
            "non-finite": dict(test_positions_init=[[np.nan, 0, 0]],
                               test_velocities_init=[[0, 0, 0]]),
            "starts on a body with no removal radius": dict(
                test_positions_init=[[-7.4241e8, 0.0, 0.0]],
                test_velocities_init=[[0, 0, 0]]),
            "removal radii length": dict(good, removal_radii=[1.0]),
            "negative removal radius": dict(good, removal_radii=[-1.0, 0.0]),
            "zero escape radius": dict(good, escape_radius=0.0),
            "test_center too large": dict(good, test_center=3),
            "test_center word": dict(good, test_center="sun"),
            "too many stored states": dict(good, max_steps=driver.MAX_TEST_PARTICLE_STATES),
        }
        for label, overrides in cases.items():
            with self.subTest(label), self.assertRaises(ValueError):
                driver.run_simulation(sun_jupiter_params(**overrides))
        # The same storage request is fine when nothing is stored per step.
        driver._validate_params(sun_jupiter_params(
            output_type="survival", max_steps=driver.MAX_TEST_PARTICLE_STATES, **good))

    def test_particles_starting_inside_a_removal_zone_are_removed_at_once(self):
        # Inside the Sun's removal radius, inside Jupiter's Hill radius,
        # beyond the escape radius, exactly on the Sun (radius > 0), and one
        # ordinary particle that must be unaffected.
        positions = [[0.5 * AU, 0.0, 0.0], [5.1 * AU, 0.0, 0.0],
                     [0.0, 30.0 * AU, 0.0], [-7.4241e8, 0.0, 0.0],
                     [3.0 * AU, 0.0, 0.0]]
        velocities = [[0.0, 0.0, 0.0]] * 4 + [[0.0, 17200.0, 0.0]]
        for output_type in ("trajectories", "survival", "animation"):
            with self.subTest(output_type=output_type):
                result = driver.run_simulation(sun_jupiter_params(
                    output_type=output_type, max_steps=5, frame_time=1.5e6,
                    test_positions_init=positions, test_velocities_init=velocities,
                    removal_radii=[AU, 0.355 * AU], escape_radius=20.0 * AU))
                info = result["test_particles"]
                self.assertEqual(info["fate"],
                                 ["encounter", "encounter", "escaped", "encounter", "survived"])
                self.assertEqual(list(info["removal_body"]), [1, 2, 0, 1, 0])
                np.testing.assert_array_equal(info["removal_time_s"][:4], 0.0)
                self.assertEqual(info["phase_motion"][:4], ["-"] * 4)
                key = {"trajectories": "test_positions", "animation": "test_frame_positions"}
                if output_type in key:
                    stored = result[key[output_type]]
                    self.assertTrue(np.all(np.isnan(stored[:, :4])))
                    self.assertTrue(np.all(np.isfinite(stored[:, 4])))

    def test_phase_motion_classification(self):
        classify = driver.classify_phase_motion
        self.assertEqual(classify(25.0, 160.0), "tadpole L4")
        self.assertEqual(classify(-160.0, -25.0), "tadpole L5")
        self.assertEqual(classify(385.0, 520.0), "tadpole L4")      # one turn later
        self.assertEqual(classify(20.0, 340.0), "horseshoe")
        self.assertEqual(classify(-340.0, -20.0), "horseshoe")
        self.assertEqual(classify(-10.0, 30.0), "passed body")   # crossed, never round
        self.assertEqual(classify(330.0, 370.0), "passed body")  # crossed at 360
        self.assertEqual(classify(0.0, 400.0), "circulating")
        self.assertEqual(classify(float("nan"), 1.0), "-")
        # Without one full swing nothing is called a tadpole or horseshoe.
        for low, high in ((25.0, 160.0), (-160.0, -25.0), (20.0, 340.0), (60.0, 70.5)):
            self.assertEqual(classify(low, high, swing_complete=False), "unfinished")
        # Where co-orbital names do not apply, a full swing is only "unfinished".
        self.assertEqual(classify(25.0, 160.0, co_orbital=False), "unfinished")
        self.assertEqual(classify(20.0, 340.0, co_orbital=False), "unfinished")
        self.assertEqual(classify(0.0, 400.0, co_orbital=False), "circulating")
        self.assertEqual(classify(-10.0, 30.0, co_orbital=False), "passed body")
        # L4 leads in the orbital direction: a clockwise pair swaps the names.
        self.assertEqual(classify(25.0, 160.0, sense=-1.0), "tadpole L5")
        self.assertEqual(classify(-160.0, -25.0, sense=-1.0), "tadpole L4")

    def test_co_orbital_particles_are_classified_by_their_angle_path(self):
        # On Jupiter's own circle, starting 60 degrees ahead gives a tadpole
        # about L4, 60 degrees behind a tadpole about L5, and 20 degrees ahead
        # a horseshoe; a particle at 3 AU circulates.
        radius = 5.2026 * AU
        phases = np.radians([60.0, -60.0, 20.0, 0.0])
        pos, vel = phys.ring_test_particle_states(
            [radius, radius, radius, 3.0 * AU], 1.0 + 9.548e-4, phases_rad=phases)
        result = driver.run_simulation(sun_jupiter_params(
            output_type="survival", max_steps=8000, dt=1.5e6,
            test_positions_init=pos.tolist(), test_velocities_init=vel.tolist(),
            removal_radii=[6.96e8, 5.31e10], escape_radius=3e12))
        info = result["test_particles"]
        self.assertEqual(info["fate"], ["survived"] * 4)
        self.assertEqual(info["phase_motion"],
                         ["tadpole L4", "tadpole L5", "horseshoe", "circulating"])
        self.assertEqual(info["phase_reference_body"], 2)
        # Starting exactly at L4 or L5, the angle barely moves.
        self.assertLess(info["phase_max_deg"][0] - info["phase_min_deg"][0], 10.0)
        self.assertLess(info["phase_max_deg"][1] - info["phase_min_deg"][1], 10.0)
        np.testing.assert_allclose(info["initial_phase_deg"], [60.0, -60.0, 20.0, 0.0],
                                   atol=0.1)

    def test_cli_random_phases_per_radius_and_seed(self):
        base = ["--n_bodies", "2", "--masses_solar", "1,9.548e-4",
                "--positions_init", "-7.4241e8,0,0;7.77555e11,0,0",
                "--velocities_init", "0,-12.462,0;0,13051.96,0",
                "--test_ring", "7.0e11,8.0e11,3", "--test_ring_per_radius", "4"]
        golden = entry.test_particle_initial_states(entry.parse_args(base))[0]
        first = entry.test_particle_initial_states(
            entry.parse_args(base + ["--test_phases", "random", "--test_seed", "7"]))[0]
        again = entry.test_particle_initial_states(
            entry.parse_args(base + ["--test_phases", "random", "--test_seed", "7"]))[0]
        other = entry.test_particle_initial_states(
            entry.parse_args(base + ["--test_phases", "random", "--test_seed", "8"]))[0]
        self.assertEqual(len(golden), 12)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        com = phys.center_of_mass([[-7.4241e8, 0, 0], [7.77555e11, 0, 0]], [1, 9.548e-4])
        radii = np.linalg.norm(np.array(first) - com, axis=1)
        np.testing.assert_allclose(radii, np.repeat([7.0e11, 7.5e11, 8.0e11], 4), rtol=1e-12)
        golden_angles = np.arctan2(np.array(golden)[:, 1] - com[1], np.array(golden)[:, 0] - com[0])
        np.testing.assert_allclose(np.mod(np.diff(golden_angles), 2 * np.pi),
                                   phys.GOLDEN_ANGLE, rtol=1e-9)
        for extra in (["--test_ring_per_radius", "0"], ["--test_phases", "uniform"],
                      ["--test_seed", "-1"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                entry.parse_args(base + extra)
        for extra in (["--test_ring_per_radius", "2"], ["--test_phases", "random"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                entry.parse_args(extra)

    def test_end_positions_of_survivors_are_recorded_and_plotted(self):
        radius = 5.2026 * AU
        pos, vel = phys.ring_test_particle_states(
            [radius, radius, 3.0 * AU, 5.1 * AU], 1.0 + 9.548e-4,
            phases_rad=np.radians([60.0, 20.0, 0.0, 0.0]))
        params = dict(max_steps=8000, dt=1.5e6, test_positions_init=pos.tolist(),
                      test_velocities_init=vel.tolist(), removal_radii=[6.96e8, 5.31e10],
                      escape_radius=3e12)
        survival = driver.run_simulation(sun_jupiter_params(output_type="survival", **params))
        info = survival["test_particles"]
        self.assertEqual(info["phase_motion"], ["tadpole L4", "horseshoe", "circulating", "-"])
        self.assertTrue(0.0 < info["final_phase_deg"][0] < 180.0)
        self.assertTrue(np.isnan(info["final_phase_deg"][3]))
        self.assertTrue(np.isnan(info["final_distance_m"][3]))
        self.assertAlmostEqual(info["final_distance_m"][2] / AU, 3.0, delta=0.1)
        # Jupiter's distance from the centre of mass is a / (1 + m_J / M_sun).
        self.assertAlmostEqual(info["reference_final_distance_m"] / AU,
                               5.2026 / (1.0 + 9.548e-4), delta=1e-3)
        trajectories = driver.run_simulation(sun_jupiter_params(
            output_type="trajectories", **dict(params, max_steps=50)))
        np.testing.assert_array_equal(trajectories["final_massive_positions"],
                                      trajectories["positions"][-1])
        np.testing.assert_array_equal(trajectories["final_massive_velocities"],
                                      trajectories["velocities"][-1])
        with mock.patch.object(plotting.plt, "show"):
            plotting.plot_survivor_positions(survival)
            plotting.plt.close("all")
        none_secure = dict(survival, test_particles=dict(
            info, phase_motion=["horseshoe", "horseshoe", "horseshoe", "-"]))
        with self.assertRaises(ValueError):
            plotting.plot_survivor_positions(none_secure)
        with self.assertRaises(ValueError):
            plotting.plot_survivor_positions({"type": "survival"})

    def test_survival_command_opens_the_end_position_window_second(self):
        command = ["--n_bodies", "2", "--masses_solar", "1,9.548e-4",
                   "--positions_init", "-7.4241e8,0,0;7.77555e11,0,0",
                   "--velocities_init", "0,-12.462,0;0,13051.96,0",
                   "--dt", "1.5e6", "--eps1", "0.05", "--output_type", "survival",
                   "--removal_radii", "6.96e8,5.31e10", "--escape_radius", "3e12"]
        calls = []
        with mock.patch.object(entry, "plot_survival", lambda r: calls.append("survival")), \
                mock.patch.object(entry, "plot_survivor_positions",
                                  lambda r: calls.append("positions")), \
                contextlib.redirect_stdout(io.StringIO()):
            # Two particles near 0.5 AU circulate past Jupiter within a year.
            entry.main([arg if arg != "1.5e6" else "1e5" for arg in command]
                       + ["--max_steps", "400", "--test_ring", "7.4e10,7.6e10,2"])
        self.assertEqual(calls, ["survival", "positions"])
        calls.clear()
        output = io.StringIO()
        with mock.patch.object(entry, "plot_survival", lambda r: calls.append("survival")), \
                mock.patch.object(entry, "plot_survivor_positions",
                                  lambda r: calls.append("positions")), \
                contextlib.redirect_stdout(output):
            # A single particle starting inside the Sun's (enlarged) removal radius.
            entry.main([arg if arg != "6.96e8,5.31e10" else "2e9,5.31e10" for arg in command]
                       + ["--max_steps", "2", "--test_ring", "5e8,5e8,1"])
        self.assertEqual(calls, ["survival"])
        self.assertIn("removed at the start: 1", output.getvalue())
        self.assertIn("end-position window is skipped", output.getvalue())

    def test_plots_accept_test_particles_and_reject_bad_survival_results(self):
        result = driver.run_simulation(sun_jupiter_params(
            max_steps=30,
            test_positions_init=[[3.0 * AU, 0.0, 0.0], [0.0, -2.0 * AU, 0.0]],
            test_velocities_init=[[0.0, 17000.0, 0.0], [0.0, -2.0e5, 0.0]],
            escape_radius=3.5 * AU))
        survival = driver.run_simulation(sun_jupiter_params(
            output_type="survival", max_steps=30,
            test_positions_init=[[3.0 * AU, 0.0, 0.0]],
            test_velocities_init=[[0.0, 17000.0, 0.0]]))
        animation = driver.run_simulation(sun_jupiter_params(
            output_type="animation", max_steps=30, frame_time=4e6,
            test_positions_init=[[3.0 * AU, 0.0, 0.0], [0.0, -2.0 * AU, 0.0]],
            test_velocities_init=[[0.0, 17000.0, 0.0], [0.0, -2.0e5, 0.0]],
            escape_radius=3.5 * AU))
        self.assertTrue(np.any(np.isnan(animation["test_frame_positions"][-1])))
        with mock.patch.object(plotting.plt, "show"):
            plotting.plot_trajectories(result)
            plotting.plot_survival(survival)
            handles = plotting.animate_multiple(animation)
            handles["timer"].stop()
            handles["on_key"](mock.Mock(key="f"))
            plotting.plt.close("all")
        with self.assertRaises(ValueError):
            plotting.plot_survival({"type": "survival", "final_time": 1.0})
        with self.assertRaises(ValueError):
            plotting.plot_survival([])

    def test_cli_ring_options_and_csv(self):
        args = entry.parse_args([
            "--n_bodies", "2", "--masses_solar", "1,1",
            "--positions_init", "7.48e10,0,0;-7.48e10,0,0",
            "--velocities_init", "0,21061,0;0,-21061,0",
            "--test_ring", "1.5e10,3e10,4", "--test_center", "1",
            "--removal_radii", "7e8,7e8", "--escape_radius", "3e12",
            "--output_type", "survival"])
        positions, velocities = entry.test_particle_initial_states(args)
        radii = np.linalg.norm(np.array(positions) - [7.48e10, 0, 0], axis=1)
        np.testing.assert_allclose(radii, [1.5e10, 2.0e10, 2.5e10, 3.0e10], rtol=1e-12)
        self.assertEqual(args.test_center, 1)
        bad = (
            ["--output_type", "survival"],
            ["--test_ring", "2e11,1e11,3"],
            ["--test_ring", "1e11,2e11"],
            ["--test_ring", "1e11,2e11,3", "--test_positions_init", "1e11,0,0",
             "--test_velocities_init", "0,1,0"],
            ["--test_positions_init", "1e11,0,0"],
            ["--test_center", "4"],
            ["--removal_radii", "1,2"],
            ["--escape_radius", "-5"],
        )
        for extra in bad:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                entry.parse_args(extra)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fates.csv"
            output = io.StringIO()
            with mock.patch.object(entry, "plot_survival"), \
                    contextlib.redirect_stdout(output):
                entry.main([
                    "--n_bodies", "2", "--masses_solar", "1,9.548e-4",
                    "--positions_init", "-7.4241e8,0,0;7.77555e11,0,0",
                    "--velocities_init", "0,-12.462,0;0,13051.96,0",
                    "--dt", "1.87e6", "--eps1", "0.05", "--max_steps", "40",
                    "--output_type", "survival", "--test_ring", "4.5e11,6e11,3",
                    "--escape_radius", "3e12", "--test_csv", str(path)])
            rows = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 4)
        self.assertTrue(rows[0].startswith("particle,x0_m,"))
        self.assertIn("Test particles: 3; survived 3", output.getvalue())
        self.assertIn("Fewest dt steps per starting test-particle orbit", output.getvalue())


class TestBeat9Help(unittest.TestCase):
    """Beat 9 and the two binary-star experiments use the live program."""

    @classmethod
    def setUpClass(cls):
        cls.text = HELP_PATH.read_text(encoding="utf-8")

    def section(self, section_id):
        match = re.search(rf'<section id="{section_id}">(.*?)</section>', self.text, re.S)
        self.assertIsNotNone(match, section_id)
        return match.group(1)

    def command(self, section_id, index=0):
        commands = tutorial_commands(self.section(section_id))
        return entry.parse_args(shlex.split(commands[index])[2:])

    def test_new_beat_and_experiments_are_linked(self):
        for anchor in ("beat9", "exp11", "exp12"):
            self.assertIn(f'id="{anchor}"', self.text)
            self.assertIn(f'href="#{anchor}"', self.text)

    def run_command(self, args):
        test_positions, test_velocities = entry.test_particle_initial_states(args)
        return driver.run_simulation(driver.SimulationParams(
            n_bodies=args.n_bodies, masses_solar=args.masses_solar,
            positions_init=args.positions_init, velocities_init=args.velocities_init,
            dt=args.dt, max_steps=args.max_steps, eps1=args.eps1, eps2=args.eps2,
            output_type=args.output_type, test_positions_init=test_positions,
            test_velocities_init=test_velocities, removal_radii=args.removal_radii,
            escape_radius=args.escape_radius, test_center=args.test_center))

    def test_beat9_trojan_numbers(self):
        args = self.command("beat9", 1)
        self.assertEqual((args.test_ring_per_radius, args.test_phases, args.test_seed),
                         (24, "random", 1))
        result = self.run_command(args)
        info = result["test_particles"]
        survivors = [p for p, f in zip(info["phase_motion"], info["fate"]) if f == "survived"]
        self.assertEqual(result["n_test_particles"], 216)
        self.assertEqual(int(np.sum(info["removal_time_s"] == 0.0)), 4)
        self.assertEqual(len(survivors), 189)
        self.assertEqual([survivors.count(name) for name in
                          ("tadpole L4", "tadpole L5", "horseshoe", "circulating")],
                         [72, 72, 44, 1])
        tadpoles = np.array([p.startswith("tadpole") for p in info["phase_motion"]])
        self.assertGreaterEqual(np.min(np.abs(info["initial_phase_deg"][tadpoles])), 26.0)
        self.assertLess(np.min(np.abs(info["initial_phase_deg"][tadpoles])), 27.0)
        self.assertAlmostEqual(result["final_time"] / YEAR, 950.6, places=1)
        motion = np.array(info["phase_motion"])
        ends = info["final_phase_deg"]
        l4, l5 = motion == "tadpole L4", motion == "tadpole L5"
        self.assertEqual((round(np.min(ends[l4])), round(np.max(ends[l4]))), (27, 162))
        self.assertEqual((round(-np.max(ends[l5])), round(-np.min(ends[l5]))), (29, 156))
        tadpole_distance = info["final_distance_m"][l4 | l5] / AU
        self.assertEqual((round(np.min(tadpole_distance), 2), round(np.max(tadpole_distance), 2)),
                         (4.96, 5.46))
        text = " ".join(self.section("beat9").split())
        for claim in ("between 27° and 162° ahead", "between 29° and 156° behind",
                      "between 4.96 and 5.46 AU", "Of the 216 particles, 4 start", "After 951 years, 189 survive",
                      "72 are tadpoles at L4 and 72 at L5, 44 are horseshoes and 1 circulates",
                      "at least 26°"):
            self.assertIn(claim, text)

    def test_beat9_jupiter_clearing_numbers(self):
        args = self.command("beat9")
        self.assertEqual(args.output_type, "survival")
        result = self.run_command(args)
        info = result["test_particles"]
        r0 = info["initial_distance_m"] / AU
        removed = np.array([f != "survived" for f in info["fate"]])
        self.assertEqual(int(np.sum(removed)), 12)
        self.assertEqual(set(info["fate"]), {"survived", "encounter"})
        self.assertEqual(set(info["removal_body"][removed]), {2})
        kept = ~removed
        # The ring radii are quoted to 0.01 AU in the Help.
        self.assertAlmostEqual(min(r0[removed]), 4.35, places=4)
        self.assertAlmostEqual(max(r0[removed]), 4.90, places=4)
        self.assertAlmostEqual(max(r0[kept][r0[kept] < 4.5]), 4.30, places=4)
        self.assertEqual([round(x, 2) for x in r0[kept][r0[kept] > 4.5]], [4.95, 5.0])
        motion = dict(zip(np.round(r0, 2), info["phase_motion"]))
        self.assertEqual((motion[4.95], motion[5.0]), ("horseshoe", "tadpole L4"))
        times = info["removal_time_s"][removed] / YEAR
        self.assertAlmostEqual(float(np.min(times)), 1.6, delta=0.05)
        last = int(np.argmax(np.where(removed, info["removal_time_s"], -1.0)))
        self.assertAlmostEqual(r0[last], 4.85, places=4)
        self.assertAlmostEqual(info["removal_time_s"][last] / YEAR, 552.0, delta=0.5)
        self.assertAlmostEqual(result["final_time"] / YEAR, 1426.0, places=1)
        beat9 = " ".join(self.section("beat9").split())
        for claim in ("4.35 AU", "4.30 AU", "1426 years", "removes 12 of the 41",
                      "after 1.6 years", "lasts 552 years", "about 131 steps per orbit",
                      "calls 4.95 AU a <code>horseshoe</code> and 5.00 AU a <code>tadpole L4</code>"):
            self.assertIn(claim, beat9)


class TestAudit51Regressions(unittest.TestCase):
    """Counterexamples from the Audit51 reviews."""

    two_stars = dict(
        n_bodies=2, masses_solar=[1.0, 1.0],
        positions_init=[[7.48e10, 0.0, 0.0], [-7.48e10, 0.0, 0.0]],
        velocities_init=[[0.0, 21061.0, 0.0], [0.0, -21061.0, 0.0]],
        eps1=0.05, eps2=1e-7)

    def test_a51_09_force_keeps_finite_values_at_huge_distances(self):
        acc = phys.compute_test_particle_accelerations(
            [[1e130, 0.0, 0.0]], [[0.0, 0.0, 0.0], [1e131, 0.0, 0.0]], [1.0, 1.0])
        self.assertTrue(np.all(np.isfinite(acc)))
        self.assertAlmostEqual(acc[0, 0] / -1.310740148e-240, 1.0, places=8)

    def test_a51_02_crossing_a_removal_sphere_inside_one_step_is_caught(self):
        common = dict(
            n_bodies=2, masses_solar=[1, 1e-12],
            positions_init=[[0, 0, 0], [1e16, 0, 0]],
            velocities_init=[[0, 0, 0], [0, 0, 0]], eps1=0.05, eps2=1e-7,
            test_positions_init=[[-2e9, 0, 0]], test_velocities_init=[[4e5, 0, 0]],
            removal_radii=[7e8, 0])
        for output_type in ("trajectories", "survival", "animation"):
            for dt, steps in ((10000, 1), (1000, 10)):
                with self.subTest(output_type=output_type, dt=dt):
                    result = driver.run_simulation(driver.SimulationParams(
                        **common, dt=dt, max_steps=steps, output_type=output_type,
                        frame_time=1000.0))
                    info = result["test_particles"]
                    self.assertEqual(info["fate"], ["encounter"])
                    self.assertEqual(list(info["removal_body"]), [1])
                    # The particle reaches 7e8 m from the star after
                    # (2e9 - 7e8) / 4e5 = 3250 s at constant speed; the
                    # star's pull makes it a little sooner.
                    self.assertTrue(2500.0 < info["removal_time_s"][0] <= 3250.0)

    def test_a51_01_unresolved_particles_never_get_a_physical_fate(self):
        common = dict(
            n_bodies=2, masses_solar=[1, 1e-12],
            positions_init=[[0, 0, 0], [1e16, 0, 0]],
            velocities_init=[[0, 0, 0], [0, 0, 0]], eps1=0.05, eps2=1e-7,
            dt=10000, max_steps=1, output_type="survival",
            test_positions_init=[[-2e9, 1e9, 0]], test_velocities_init=[[4e5, 0, 0]])
        with mock.patch.object(driver, "MAX_TEST_SUBSTEP_DEPTH", 0):
            result = driver.run_simulation(driver.SimulationParams(**common))
        info = result["test_particles"]
        self.assertEqual(info["fate"], ["numerical"])
        self.assertEqual(info["removal_time_s"][0], 0.0)
        np.testing.assert_array_equal(info["final_positions"][0], [-2e9, 1e9, 0])
        # With refinement allowed, the same particle is followed through its
        # close pass instead. (Aimed at 1e8 m it would pass about 6000 km
        # from the point mass, too close even for the smallest substep; Audit52
        # showed the old code accepted that pass by skipping the eps1 test.)
        result = driver.run_simulation(driver.SimulationParams(**common))
        self.assertEqual(result["test_particles"]["fate"], ["survived"])

    def test_a51_01_trial_reports_nonconvergence_row_by_row(self):
        star = np.array([[0.0, 0.0, 0.0], [1e16, 0.0, 0.0]])
        masses = np.array([1.0, 1e-12])
        p0 = np.array([[-2e9, 1e8, 0.0], [1.5e11, 0.0, 0.0]])
        v0 = np.array([[4e5, 0.0, 0.0], [0.0, 29780.0, 0.0]])
        acc0 = phys.compute_test_particle_accelerations(p0, star, masses)
        _, _, _, converged = driver._test_particle_trial(
            p0, v0, acc0, star, masses, 10000.0, 1e-7, 10)
        self.assertEqual(list(converged), [False, True])

    def test_a51_01_a_hard_particle_does_not_change_the_others(self):
        period = 2 * np.pi * np.sqrt(AU ** 3 / phys.GM_SUN)
        quiet_p, quiet_v = phys.ring_test_particle_states([AU, 2 * AU], 1.0)
        # An eccentric orbit whose perihelion pass (0.02 AU) is far shorter
        # than one step, although its starting orbit has over 100 steps.
        ra, rp = AU, 0.02 * AU
        a = 0.5 * (ra + rp)
        va = np.sqrt(phys.GM_SUN * (2 / ra - 1 / a))
        hard_p, hard_v = [[-ra, 0.0, 0.0]], [[0.0, -va, 0.0]]
        base = dict(n_bodies=2, masses_solar=[1, 1e-12],
                    positions_init=[[0, 0, 0], [1e16, 0, 0]],
                    velocities_init=[[0, 0, 0], [0, 0, 0]],
                    dt=period / 400, max_steps=400, eps1=0.05, eps2=1e-10,
                    output_type="trajectories")
        quiet = driver.run_simulation(driver.SimulationParams(
            **base, test_positions_init=quiet_p.tolist(),
            test_velocities_init=quiet_v.tolist()))
        mixed = driver.run_simulation(driver.SimulationParams(
            **base, test_positions_init=quiet_p.tolist() + hard_p,
            test_velocities_init=quiet_v.tolist() + hard_v))
        np.testing.assert_array_equal(quiet["test_positions"], mixed["test_positions"][:, :2])
        np.testing.assert_array_equal(quiet["positions"], mixed["positions"])
        # The hard particle is followed through two perihelion passes and
        # comes back close to where it started.
        hard_period = 2 * np.pi * np.sqrt(a ** 3 / phys.GM_SUN)
        self.assertEqual(mixed["test_particles"]["fate"][2], "survived")
        self.assertGreater(400 * period / 400 / hard_period, 2.0)
        # Compare with the independent Kepler solution at every stored time;
        # an earlier draft with a fixed 25% acceleration limit was 0.039 AU off
        # after one orbit.
        gaps = [np.linalg.norm(mixed["test_positions"][k, 2] - kepler_position(
                    mixed["times"][k], ra, rp)) / AU
                for k in range(0, len(mixed["times"]), 20)]
        self.assertLess(max(gaps), 0.006)

    def test_a51_03_undefined_elements_do_not_lose_the_result(self):
        for output_type in ("survival", "trajectories", "animation"):
            with self.subTest(output_type=output_type):
                at_com = driver.run_simulation(driver.SimulationParams(
                    **self.two_stars, dt=1000, max_steps=2, output_type=output_type,
                    frame_time=1000.0,
                    test_positions_init=[[0, 0, 0], [3e11, 0, 0]],
                    test_velocities_init=[[0, 0, 0], [0, 20000, 0]]))
                info = at_com["test_particles"]
                self.assertTrue(np.isnan(info["initial_semi_major_axis_m"][0]))
                self.assertTrue(np.isnan(info["initial_eccentricity"][0]))
                self.assertTrue(np.isfinite(info["initial_semi_major_axis_m"][1]))
        on_body = driver.run_simulation(driver.SimulationParams(
            **self.two_stars, dt=1000, max_steps=2, output_type="survival",
            test_positions_init=[[7.48e10, 0, 0]], test_velocities_init=[[0, 0, 0]],
            test_center=1, removal_radii=[7.5e9, 7.5e9]))
        self.assertEqual(on_body["test_particles"]["fate"], ["encounter"])
        # Console table and CSV cope with the undefined row.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.csv"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                entry.print_test_particle_table(at_com, 1000.0)
                entry.write_test_particle_csv(at_com, path)
            self.assertIn("       -", output.getvalue())
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 3)

    def test_a51_08_scan_with_every_particle_removed_at_start_takes_no_steps(self):
        cases = {
            "encounter": dict(test_positions_init=[[7.5e10, 0, 0]],
                              test_velocities_init=[[0, 21061, 0]],
                              removal_radii=[7.5e9, 7.5e9]),
            "escaped": dict(test_positions_init=[[4e12, 0, 0]],
                            test_velocities_init=[[0, 0, 0]], escape_radius=3e12),
        }
        for fate, extra in cases.items():
            with self.subTest(fate=fate):
                result = driver.run_simulation(driver.SimulationParams(
                    **self.two_stars, dt=1000, max_steps=5, output_type="survival", **extra))
                self.assertEqual(result["accepted_steps"], 0)
                self.assertEqual(result["final_time"], 0.0)
                self.assertEqual(result["test_particles"]["fate"], [fate])
                np.testing.assert_array_equal(result["final_massive_positions"],
                                              self.two_stars["positions_init"])
                with mock.patch.object(plotting.plt, "show"):
                    plotting.plot_survival(result)
                    plotting.plt.close("all")

    def test_a51_04_angle_paths_need_a_full_swing_and_the_right_configuration(self):
        # Codex's counterexample: an ordinary 1-AU orbit starting 60 degrees
        # ahead of Jupiter drifts for 0.03 years; that is not a Trojan.
        pos, vel = phys.ring_test_particle_states(
            [AU], 1.0 + 9.548e-4, phases_rad=[np.radians(60.0)])
        result = driver.run_simulation(sun_jupiter_params(
            dt=10000, max_steps=100, output_type="survival",
            test_positions_init=pos.tolist(), test_velocities_init=vel.tolist()))
        info = result["test_particles"]
        self.assertTrue(60.0 < info["phase_max_deg"][0] < 75.0)
        self.assertEqual(info["phase_motion"], ["unfinished"])
        # The same particle, followed for long enough, circulates.
        result = driver.run_simulation(sun_jupiter_params(
            dt=1e5, max_steps=400, output_type="survival",
            test_positions_init=pos.tolist(), test_velocities_init=vel.tolist()))
        self.assertEqual(result["test_particles"]["phase_motion"], ["circulating"])
        # An equal-mass pair has no co-orbital names, and circumbinary
        # orbits circulate.
        pos, vel = phys.ring_test_particle_states([3.0 * AU], 2.0)
        result = driver.run_simulation(driver.SimulationParams(
            **self.two_stars, dt=1.1e5, max_steps=2000, output_type="survival",
            test_positions_init=pos.tolist(), test_velocities_init=vel.tolist()))
        self.assertFalse(result["test_particles"]["co_orbital_names"])
        self.assertEqual(result["test_particles"]["phase_motion"], ["circulating"])

    def test_a51_05_equilateral_points_use_both_bodies(self):
        separation = 1.496e11
        positions = [[0.5 * separation, 0, 0], [-0.5 * separation, 0, 0]]
        velocities = [[0, 21061, 0], [0, -21061, 0]]
        points = dict((name, (angle, distance)) for name, angle, distance in
                      plotting.equilateral_points(positions, velocities, [1, 1], "com", 1))
        self.assertAlmostEqual(points["L4"][0], 90.0, places=6)
        self.assertAlmostEqual(points["L5"][0], -90.0, places=6)
        for name in ("L4", "L5"):
            self.assertAlmostEqual(points[name][1] / separation, np.sqrt(3) / 2, places=9)
        # Seen from body 2 (the centre), measured from body 1: 60 degrees,
        # at the separation.
        points = dict((name, (angle, distance)) for name, angle, distance in
                      plotting.equilateral_points(positions, velocities, [1, 1], 1, 1))
        self.assertAlmostEqual(abs(points["L4"][0]), 60.0, places=6)
        self.assertAlmostEqual(points["L4"][1] / separation, 1.0, places=9)
        # Sun-Jupiter about the centre of mass: close to +/-60 degrees.
        params = sun_jupiter_params()
        points = dict((name, (angle, distance)) for name, angle, distance in
                      plotting.equilateral_points(params.positions_init,
                                                  params.velocities_init,
                                                  params.masses_solar, "com", 2))
        self.assertAlmostEqual(points["L4"][0], 60.0, delta=0.1)
        self.assertAlmostEqual(points["L5"][0], -60.0, delta=0.1)
        # Not defined for three bodies.
        self.assertIsNone(plotting.equilateral_points(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 0]] * 3, [1, 1, 1], "com", 2))


class TestAudit52Regressions(unittest.TestCase):
    """Counterexamples from the Audit52 reviews."""

    lone_sun = dict(n_bodies=2, masses_solar=[1, 1e-12],
                    positions_init=[[0, 0, 0], [1e16, 0, 0]],
                    velocities_init=[[0, 0, 0], [0, 0, 0]])
    ra, rp = AU, 0.02 * AU

    def eccentric(self, steps, output_type, frame_time=2e5):
        a = 0.5 * (self.ra + self.rp)
        speed = np.sqrt(phys.GM_SUN * (2 / self.ra - 1 / a))
        period = 2 * np.pi * np.sqrt(AU ** 3 / phys.GM_SUN)
        return driver.run_simulation(driver.SimulationParams(
            **self.lone_sun, dt=period / steps, max_steps=steps, eps1=0.05, eps2=1e-10,
            output_type=output_type, frame_time=frame_time,
            test_positions_init=[[-self.ra, 0, 0]], test_velocities_init=[[0, -speed, 0]]))

    def test_a52_01_codex_tiny_orbit_ends_as_numerical(self):
        result = driver.run_simulation(driver.SimulationParams(
            **self.lone_sun, dt=10000, max_steps=1, eps1=0.05, eps2=1e-7,
            output_type="survival", test_positions_init=[[1e5, 0, 0]],
            test_velocities_init=[[0, np.sqrt(phys.GM_SUN / 1e5), 0]]))
        self.assertEqual(result["test_particles"]["fate"], ["numerical"])
        self.assertEqual(result["test_particles"]["removal_time_s"][0], 0.0)

    def test_a52_01_converged_but_unsteady_at_the_floor_is_numerical(self):
        # A 1-AU circle in 20 steps converges each step but turns its
        # acceleration by 18 degrees, far beyond eps1; with no refinement
        # allowed it must not be committed.
        period = 2 * np.pi * np.sqrt(AU ** 3 / phys.GM_SUN)
        pos, vel = phys.ring_test_particle_states([AU], 1.0, phases_rad=[0.0])
        params = driver.SimulationParams(
            **self.lone_sun, dt=period / 20, max_steps=20, eps1=0.05, eps2=1e-7,
            output_type="survival", test_positions_init=pos.tolist(),
            test_velocities_init=vel.tolist())
        star = np.array(self.lone_sun["positions_init"], dtype=float)
        acc0 = phys.compute_test_particle_accelerations(pos, star, [1, 1e-12])
        _, _, acc_end, converged = driver._test_particle_trial(
            pos, vel, acc0, star, np.array([1, 1e-12]), period / 20, 1e-7, 10)
        self.assertTrue(converged[0])
        self.assertGreater(driver._row_relative_change(acc0, acc_end)[0], 0.05)
        with mock.patch.object(driver, "MAX_TEST_SUBSTEP_DEPTH", 0):
            result = driver.run_simulation(params)
        self.assertEqual(result["test_particles"]["fate"], ["numerical"])
        np.testing.assert_array_equal(result["test_particles"]["final_positions"], pos)
        # With refinement allowed it is followed for the whole orbit.
        result = driver.run_simulation(params)
        self.assertEqual(result["test_particles"]["fate"], ["survived"])

    def test_a52_01_time_that_cannot_be_halved_is_the_floor_too(self):
        t0 = 1.0e20
        t1 = float(np.nextafter(t0, np.inf))
        star = np.array([[0.0, 0.0, 0.0], [1e16, 0.0, 0.0]])
        velocity = np.zeros((2, 3))
        masses = np.array([1.0, 1e-12])
        pos = np.array([[AU, 0.0, 0.0]])
        vel = np.array([[0.0, 29780.0, 0.0]])
        acc0 = phys.compute_test_particle_accelerations(pos, star, masses)
        active = np.ones(1, dtype=bool)
        removal_time = np.full(1, np.nan)
        reason, body = ["survived"], np.zeros(1, dtype=int)
        # eps1 so small that no step can pass the acceleration test.
        driver._advance_test_particles(
            pos, vel, np.array([0]), acc0, t0, star, velocity, t1, star, velocity,
            masses, 1e-300, 1e-7, 10, np.zeros(2), None, active, removal_time,
            reason, body)
        self.assertEqual(reason, ["numerical"])
        self.assertEqual(removal_time[0], t0)

    def test_a52_02_angle_history_keeps_every_turn(self):
        period = 2 * np.pi * np.sqrt(AU ** 3 / phys.GM_SUN)
        expected = kepler_turning(period, self.ra, self.rp)
        self.assertAlmostEqual(expected, 1072.5, delta=0.1)
        for output_type in ("trajectories", "survival", "animation"):
            with self.subTest(output_type=output_type):
                info = self.eccentric(400, output_type)["test_particles"]
                swept = info["phase_max_deg"][0] - info["phase_min_deg"][0]
                self.assertAlmostEqual(swept, expected, delta=1.0)
                self.assertAlmostEqual(info["phase_min_deg"][0], -180.0, places=6)
                self.assertEqual(info["phase_motion"], ["circulating"])

    def test_a52_03_turning_points_need_the_final_range(self):
        replay = driver.swing_complete_for
        self.assertFalse(replay([60.0, 61.0, 60.5, 80.0]))      # Grok: wiggle, then drift
        self.assertFalse(replay([60.0, 62.0, 60.5, 80.0]))      # turn, then drift past it
        self.assertFalse(replay([60.0, 60.2, 60.0, 90.0]))      # 0.2-degree wiggle
        self.assertFalse(replay([60.0, 84.0, 57.3]))            # still falling at the end
        self.assertFalse(replay([60.0, 70.0, 80.0, 90.0]))      # the start is not a turn
        self.assertTrue(replay([60.0, 62.0, 58.0, 62.0, 58.5]))
        self.assertTrue(replay([60.0, 84.0, 57.0, 70.0, 57.2, 84.0, 60.0]))

    def test_a52_03_eccentric_inner_orbit_is_not_a_trojan(self):
        params = sun_jupiter_params()
        com = phys.center_of_mass(params.positions_init, params.masses_solar)
        r0, angle = 3.6 * AU, np.radians(60.0)
        speed = np.sqrt(phys.GM_SUN * (1 + 9.548e-4) * (2 / r0 - 1 / (2 * AU)))
        position = com + [r0 * np.cos(angle), r0 * np.sin(angle), 0.0]
        velocity = [-speed * np.sin(angle), speed * np.cos(angle), 0.0]
        for steps in (400, 4000):
            with self.subTest(steps=steps):
                result = driver.run_simulation(sun_jupiter_params(
                    dt=1e5, max_steps=steps, output_type="survival",
                    test_positions_init=[position.tolist()], test_velocities_init=[velocity]))
                info = result["test_particles"]
                self.assertFalse(info["co_orbital_start"][0])
                self.assertNotIn(info["phase_motion"][0],
                                 ("tadpole L4", "tadpole L5", "horseshoe"))

    def test_a52_03_clockwise_names_agree_with_the_plot(self):
        params = sun_jupiter_params(velocities_init=[[0.0, 12.462, 0.0],
                                                     [0.0, -13051.96, 0.0]])
        pos, vel = phys.ring_test_particle_states(
            [5.2026 * AU] * 2, 1 + 9.548e-4, phases_rad=np.radians([-60.0, 60.0]))
        result = driver.run_simulation(sun_jupiter_params(
            velocities_init=params.velocities_init, dt=1.5e6, max_steps=6000,
            output_type="survival", test_positions_init=pos.tolist(),
            test_velocities_init=(-vel).tolist(), removal_radii=[6.96e8, 5.31e10],
            escape_radius=3e12))
        info = result["test_particles"]
        self.assertEqual(info["orbital_sense"], -1.0)
        self.assertEqual(info["phase_motion"], ["tadpole L4", "tadpole L5"])
        points = dict((name, angle) for name, angle, _ in plotting.equilateral_points(
            params.positions_init, params.velocities_init, params.masses_solar, "com", 2))
        self.assertAlmostEqual(points["L4"], -60.0, delta=0.1)
        self.assertAlmostEqual(points["L5"], 60.0, delta=0.1)

    def test_a52_04_animation_follows_substeps_and_hides_removed_particles(self):
        crossing = dict(**self.lone_sun, eps1=0.05, eps2=1e-7,
                        test_positions_init=[[-2e9, 0, 0]], test_velocities_init=[[4e5, 0, 0]],
                        removal_radii=[7e8, 0])
        coarse = driver.run_simulation(driver.SimulationParams(
            **crossing, dt=10000, max_steps=1, output_type="animation", frame_time=1000))
        fine = driver.run_simulation(driver.SimulationParams(
            **crossing, dt=100, max_steps=100, output_type="trajectories"))
        removed_at = coarse["test_particles"]["removal_time_s"][0]
        self.assertTrue(2000.0 < removed_at < 3000.0)
        frames = coarse["test_frame_positions"][:, 0, :]
        for k, frame_time in enumerate(coarse["frame_times"]):
            if frame_time >= removed_at:
                self.assertTrue(np.all(np.isnan(frames[k])), frame_time)
            else:
                index = int(np.argmin(np.abs(fine["times"] - frame_time)))
                self.assertEqual(fine["times"][index], frame_time)
                self.assertLess(abs(frames[k, 0] - fine["test_positions"][index, 0, 0]), 2e7)
        # A fast live orbit with several particle substeps between frames:
        # frames lie on the Kepler orbit.
        period = 2 * np.pi * np.sqrt(AU ** 3 / phys.GM_SUN)
        result = self.eccentric(400, "animation", frame_time=period / 997)
        positions = result["test_frame_positions"][:, 0, :]
        gaps = [np.linalg.norm(positions[k] - kepler_position(t, self.ra, self.rp)) / AU
                for k, t in enumerate(result["frame_times"])]
        # Interpolating the 400 macrosteps instead misses the perihelion
        # passes by far more than this.
        self.assertLess(max(gaps), 0.008)

    def test_a52_05_stability_label_does_not_depend_on_the_centre(self):
        params = sun_jupiter_params()
        self.assertTrue(plotting.triangular_points_stable(params.masses_solar))
        self.assertFalse(plotting.triangular_points_stable([1, 1]))
        jupiter = np.array(params.positions_init[1])
        local = np.sqrt(phys.GM_SUN * 9.548e-4 / 1e8)
        result = driver.run_simulation(sun_jupiter_params(
            dt=100, max_steps=1000, output_type="survival", test_center=2,
            test_positions_init=[(jupiter + [1e8, 0, 0]).tolist()],
            test_velocities_init=[[0, 13051.96 + local, 0]]))
        self.assertEqual(result["test_particles"]["phase_motion"], ["circulating"])
        labels = []
        def capture():
            for number in plotting.plt.get_fignums():
                for ax in plotting.plt.figure(number).axes:
                    legend = ax.get_legend()
                    if legend is not None:
                        labels.extend(text.get_text() for text in legend.get_texts())
            plotting.plt.close("all")
        with mock.patch.object(plotting.plt, "show", side_effect=capture):
            plotting.plot_survivor_positions(result)
        self.assertIn("L4 and L5", labels)
        self.assertFalse(any("not stable" in label for label in labels))

    def test_a52_05_points_are_omitted_outside_the_planar_circular_model(self):
        d = 1.496e11
        positions = [[d / 2, 0, 0], [-d / 2, 0, 0]]
        tilted = [[0, 0, 21061], [0, 0, -21061]]          # orbit in the x-z plane
        self.assertIsNone(plotting.equilateral_points(positions, tilted, [1, 1], "com", 1))
        radial = [[-5000, 0, 0], [5000, 0, 0]]            # falling straight in
        self.assertIsNone(plotting.equilateral_points(positions, radial, [1, 1], "com", 1))
        eccentric = [[0, 30000, 0], [0, -30000, 0]]       # e well above 0.1
        self.assertIsNone(plotting.equilateral_points(positions, eccentric, [1, 1], "com", 1))
        circular = [[0, 21061, 0], [0, -21061, 0]]
        self.assertIsNotNone(plotting.equilateral_points(positions, circular, [1, 1], "com", 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
