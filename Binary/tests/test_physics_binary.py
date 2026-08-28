"""Regression tests for the complete Binary teaching module.

The discovery logic supports both the canonical tests subdirectory and a
review upload in which this file is flattened beside the four core modules.
"""

from __future__ import annotations

import ast
import hashlib
from html.parser import HTMLParser
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


CORE_MODULE_FILENAMES = (
    "physics_binary.py",
    "driver_binary.py",
    "main.py",
    "plot_binary.py",
)


def find_module_dir(start: Path) -> Path:
    """Find the nearest ancestor containing every Binary core module."""
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if all((directory / name).is_file() for name in CORE_MODULE_FILENAMES):
            return directory
    names = ", ".join(CORE_MODULE_FILENAMES)
    raise FileNotFoundError(
        f"Could not find a directory containing all Binary core modules: {names}"
    )


MODULE_DIR = find_module_dir(Path(__file__))
HELP_FILE = MODULE_DIR / "Binary.html"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import driver_binary as driver  # noqa: E402
import physics_binary as physics  # noqa: E402
import plot_binary as plotting  # noqa: E402


DEFAULTS = {
    "MA": 2.0e30,
    "MB": 2.0e30,
    "xInitA": 4.6e10,
    "yInitA": 0.0,
    "vInitA": 0.0,
    "uInitA": 13000.0,
    "xInitB": -4.6e10,
    "yInitB": 0.0,
    "vInitB": 0.0,
    "uInitB": -13000.0,
    "dt": 2000.0,
    "max_steps": 10000,
    "eps1": 0.05,
    "eps2": 1.0e-4,
    "stop_after_one_orbit": True,
}


def integrate(**changes):
    parameters = dict(DEFAULTS)
    parameters.update(changes)
    return driver.integrate_binary(**parameters)


def energy_drift(result):
    initial = result.E[0]
    return max(abs(value - initial) for value in result.E) / abs(initial)


def unequal_moving_case():
    """Return a nonsymmetric circular case with a moving centre of mass."""
    mass_a, mass_b = 3e30, 1e30
    total_mass = mass_a + mass_b
    relative_x, relative_y = 9.2e10, 4.0e10
    separation = math.hypot(relative_x, relative_y)
    com_x, com_y = 2e10, -3e10
    com_vx, com_vy = 500.0, -800.0
    relative_speed = math.sqrt(physics.G * total_mass / separation)
    relative_vx = -relative_speed * relative_y / separation
    relative_vy = relative_speed * relative_x / separation
    fraction_a = mass_b / total_mass
    fraction_b = mass_a / total_mass
    return {
        "MA": mass_a,
        "MB": mass_b,
        "xInitA": com_x + fraction_a * relative_x,
        "yInitA": com_y + fraction_a * relative_y,
        "vInitA": com_vx + fraction_a * relative_vx,
        "uInitA": com_vy + fraction_a * relative_vy,
        "xInitB": com_x - fraction_b * relative_x,
        "yInitB": com_y - fraction_b * relative_y,
        "vInitB": com_vx - fraction_b * relative_vx,
        "uInitB": com_vy - fraction_b * relative_vy,
        "dt": 1000.0,
        "max_steps": 1000,
        "eps1": 0.05,
        "eps2": 1e-4,
        "stop_after_one_orbit": False,
    }


def fitted_circle_radial_variation(x_values, y_values):
    """Fit a circle through three separated samples and return radial spread."""
    indices = (0, len(x_values) // 3, 2 * len(x_values) // 3)
    x1, x2, x3 = (x_values[index] for index in indices)
    y1, y2, y3 = (y_values[index] for index in indices)
    denominator = 2.0 * (
        x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    )
    center_x = (
        (x1 * x1 + y1 * y1) * (y2 - y3)
        + (x2 * x2 + y2 * y2) * (y3 - y1)
        + (x3 * x3 + y3 * y3) * (y1 - y2)
    ) / denominator
    center_y = (
        (x1 * x1 + y1 * y1) * (x3 - x2)
        + (x2 * x2 + y2 * y2) * (x1 - x3)
        + (x3 * x3 + y3 * y3) * (x2 - x1)
    ) / denominator
    radii = [
        math.hypot(x - center_x, y - center_y)
        for x, y in zip(x_values, y_values)
    ]
    return (max(radii) - min(radii)) / (sum(radii) / len(radii))


class TestModuleDiscovery(unittest.TestCase):
    def test_canonical_layout(self):
        self.assertEqual(find_module_dir(Path(__file__).parent), MODULE_DIR)

    def test_flattened_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in CORE_MODULE_FILENAMES:
                (root / name).touch()
            test_file = root / "test_physics_binary.py"
            test_file.touch()
            self.assertEqual(find_module_dir(test_file), root.resolve())

    def test_nearest_matching_ancestor_wins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "outer" / "inner"
            nested.mkdir(parents=True)
            for directory in (root, root / "outer"):
                for name in CORE_MODULE_FILENAMES:
                    (directory / name).touch()
            self.assertEqual(find_module_dir(nested), (root / "outer").resolve())

    def test_missing_set_raises(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "all Binary core modules"):
                find_module_dir(Path(temporary))


class TestVersionAndBuild(unittest.TestCase):
    def test_build_coverage_is_exact(self):
        self.assertEqual(physics.BUILD_ID_COVERS, CORE_MODULE_FILENAMES)

    def test_metadata_formats(self):
        self.assertRegex(physics.MODEL_VERSION, r"^\d+\.\d+\.\d+$")
        self.assertRegex(physics.BUILD_ID, r"^[0-9a-f]{12}$")

    def test_build_id_matches_core_sources(self):
        digest = hashlib.sha256()
        for name in CORE_MODULE_FILENAMES:
            with (MODULE_DIR / name).open(
                "r", encoding="utf-8", newline=None
            ) as source:
                content = source.read().encode("utf-8")
            digest.update(name.encode("utf-8"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        self.assertEqual(physics.BUILD_ID, digest.hexdigest()[:12])

    def test_build_id_fallback_is_unknown_on_source_read_failure(self):
        with mock.patch("builtins.open", side_effect=OSError("unavailable")):
            self.assertEqual(physics._compute_build_id(), "unknown")

    def test_result_carries_metadata(self):
        result = integrate(max_steps=1, stop_after_one_orbit=False)
        self.assertEqual(result.model_version, physics.MODEL_VERSION)
        self.assertEqual(result.build_id, physics.BUILD_ID)

    def test_command_line_version(self):
        completed = subprocess.run(
            [sys.executable, "main.py", "--version"],
            cwd=MODULE_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        expected = f"Binary {physics.MODEL_VERSION} (build {physics.BUILD_ID})"
        self.assertEqual(completed.stdout.strip(), expected)

    def test_python_310_syntax(self):
        for name in CORE_MODULE_FILENAMES:
            with self.subTest(name=name):
                source = (MODULE_DIR / name).read_text(encoding="utf-8")
                ast.parse(source, filename=name, feature_version=(3, 10))


class TestPhysicsFunctions(unittest.TestCase):
    def test_relative_displacement(self):
        self.assertEqual(
            physics.relative_displacement(4.0, 6.0, 1.0, 2.0),
            (3.0, 4.0, 5.0),
        )

    def test_translation_invariance(self):
        first = physics.relative_displacement(4.0, 6.0, 1.0, 2.0)
        second = physics.relative_displacement(14.0, -1.0, 11.0, -5.0)
        self.assertEqual(first, second)

    def test_zero_separation(self):
        with self.assertRaisesRegex(ValueError, "zero separation"):
            physics.relative_displacement(1.0, 2.0, 1.0, 2.0)

    def test_bad_coordinates(self):
        for index in range(4):
            for bad in (math.nan, math.inf, -math.inf, True, "1.0"):
                values = [1.0, 2.0, -3.0, -4.0]
                values[index] = bad
                with self.subTest(index=index, bad=bad):
                    with self.assertRaises(ValueError):
                        physics.relative_displacement(*values)

    def test_unrepresentable_relative_displacement(self):
        cases = (
            (1e308, 0.0, -1e308, 0.0),
            (1.7e308, 1.7e308, 0.0, 0.0),
        )
        for coordinates in cases:
            with self.subTest(coordinates=coordinates):
                with self.assertRaisesRegex(ValueError, "relative displacement"):
                    physics.relative_displacement(*coordinates)

    def test_scaled_acceleration_avoids_r_cubed_overflow(self):
        values = physics.accelerations(
            2e30, 3e30, 1e150, 0.0, 0.0, 0.0
        )
        expected_a = -physics.G * 3e30 / 1e300
        expected_b = physics.G * 2e30 / 1e300
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertTrue(math.isclose(values[0], expected_a, rel_tol=2e-15))
        self.assertTrue(math.isclose(values[2], expected_b, rel_tol=2e-15))
        self.assertEqual((values[1], values[3]), (0.0, 0.0))

    def test_scaled_acceleration_rejects_unrepresentable_near_collision(self):
        with self.assertRaisesRegex(ValueError, "calculated acceleration"):
            physics.accelerations(
                2e30, 3e30, 5e-324, 0.0, 0.0, 0.0
            )

    def test_acceleration_avoids_intermediate_underflow(self):
        tiny_mass = 1e-320
        separation = 1e-100
        expected_tiny = (physics.G / separation) * (tiny_mass / separation)
        first = physics.accelerations(
            1.0, tiny_mass, separation, 0.0, 0.0, 0.0
        )
        swapped = physics.accelerations(
            tiny_mass, 1.0, separation, 0.0, 0.0, 0.0
        )
        self.assertTrue(math.isclose(first[0], -expected_tiny, rel_tol=2e-15))
        self.assertNotEqual(math.hypot(first[0], first[1]), 0.0)
        self.assertTrue(math.isclose(swapped[2], expected_tiny, rel_tol=2e-15))
        self.assertNotEqual(math.hypot(swapped[2], swapped[3]), 0.0)

    def test_scaled_acceleration_preserves_non_axis_direction(self):
        tiny_mass = 1e-320
        displacement_x = 3e-101
        displacement_y = 4e-101
        ax_a, ay_a, _, _ = physics.accelerations(
            1.0, tiny_mass, displacement_x, displacement_y, 0.0, 0.0
        )
        separation = math.hypot(displacement_x, displacement_y)
        expected_magnitude = (
            (physics.G / separation) * (tiny_mass / separation)
        )
        self.assertTrue(
            math.isclose(ax_a, -0.6 * expected_magnitude, rel_tol=2e-15)
        )
        self.assertTrue(
            math.isclose(ay_a, -0.8 * expected_magnitude, rel_tol=2e-15)
        )
        self.assertTrue(
            math.isclose(
                math.hypot(ax_a, ay_a), expected_magnitude, rel_tol=2e-15
            )
        )

    def test_acceleration_rejects_true_float_range_failures(self):
        cases = (
            (1e308, 1e308, 1e-308),
            (1e-308, 1e-308, 1e308),
        )
        for mass_a, mass_b, separation in cases:
            with self.subTest(
                mass_a=mass_a, mass_b=mass_b, separation=separation
            ):
                with self.assertRaisesRegex(ValueError, "calculated acceleration"):
                    physics.accelerations(
                        mass_a, mass_b, separation, 0.0, 0.0, 0.0
                    )

    def test_accelerations_and_signs(self):
        mass_a, mass_b, separation = 2.0e30, 3.0e30, 4.0e10
        ax_a, ay_a, ax_b, ay_b = physics.accelerations(
            mass_a, mass_b, separation / 2, 0.0, -separation / 2, 0.0
        )
        expected_a = -physics.G * mass_b / separation**2
        expected_b = physics.G * mass_a / separation**2
        self.assertAlmostEqual(ax_a, expected_a, delta=abs(expected_a) * 1e-15)
        self.assertAlmostEqual(ax_b, expected_b, delta=abs(expected_b) * 1e-15)
        self.assertEqual((ay_a, ay_b), (0.0, 0.0))

    def test_newtons_third_law(self):
        mass_a, mass_b = 2.3e30, 7.1e29
        ax_a, ay_a, ax_b, ay_b = physics.accelerations(
            mass_a, mass_b, 7e10, -2e10, -1e10, 3e10
        )
        scale = max(abs(mass_a * ax_a), abs(mass_a * ay_a))
        self.assertAlmostEqual(
            mass_a * ax_a + mass_b * ax_b, 0.0, delta=scale * 2e-15
        )
        self.assertAlmostEqual(
            mass_a * ay_a + mass_b * ay_b, 0.0, delta=scale * 2e-15
        )

    def test_inverse_square_scaling(self):
        near = physics.accelerations(2.0, 3.0, 1.0, 0.0, 0.0, 0.0)
        far = physics.accelerations(2.0, 3.0, 2.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(abs(far[0] / near[0]), 0.25, places=15)
        self.assertAlmostEqual(abs(far[2] / near[2]), 0.25, places=15)

    def test_invalid_masses(self):
        for bad in (0.0, -1.0, math.nan, math.inf, True, "2"):
            for name in ("MA", "MB"):
                masses = {"MA": 2.0, "MB": 3.0}
                masses[name] = bad
                with self.subTest(name=name, bad=bad):
                    with self.assertRaises(ValueError):
                        physics.accelerations(
                            masses["MA"], masses["MB"], 1.0, 0.0, 0.0, 0.0
                        )

    def test_energy_components(self):
        potential, kinetic, total = physics.energies(
            2.0, 3.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 4.0
        )
        self.assertAlmostEqual(potential, -physics.G * 6.0 / 5.0)
        self.assertAlmostEqual(kinetic, 49.0)
        self.assertAlmostEqual(total, kinetic + potential)

    def test_energy_symmetries(self):
        first = physics.energies(
            2.0, 3.0, 3.0, 4.0, 5.0, 6.0, -1.0, 2.0, -7.0, 8.0
        )
        swapped = physics.energies(
            3.0, 2.0, -1.0, 2.0, -7.0, 8.0, 3.0, 4.0, 5.0, 6.0
        )
        translated = physics.energies(
            2.0, 3.0, 103.0, -46.0, 5.0, 6.0, 99.0, -48.0, -7.0, 8.0
        )
        self.assertEqual(first, swapped)
        self.assertEqual(first, translated)

    def test_total_energy_preserves_three_term_cancellation_residual(self):
        minimum_subnormal = math.ulp(0.0)
        separation = 2.0 * physics.G
        tiny_velocity = math.sqrt(2.0 * minimum_subnormal)
        potential, kinetic, total = physics.energies(
            2.0, 1.0,
            separation, 0.0, 1.0, 0.0,
            0.0, 0.0, tiny_velocity, 0.0,
        )
        kinetic_a = physics._scaled_kinetic_energy(2.0, 1.0, 0.0)
        kinetic_b = physics._scaled_kinetic_energy(
            1.0, tiny_velocity, 0.0
        )
        self.assertEqual(
            (potential, kinetic_a, kinetic_b),
            (-1.0, 1.0, minimum_subnormal),
        )
        self.assertEqual((kinetic_a + kinetic_b) + potential, 0.0)
        self.assertEqual(kinetic, 1.0)
        self.assertEqual(total, minimum_subnormal)

    def test_exact_zero_total_energy_remains_valid(self):
        separation = 2.0 * physics.G
        potential, kinetic, total = physics.energies(
            2.0, 1.0,
            separation, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        )
        self.assertEqual((potential, kinetic, total), (-1.0, 1.0, 0.0))

    def test_cancellation_residual_is_body_swap_invariant(self):
        minimum_subnormal = math.ulp(0.0)
        separation = 2.0 * physics.G
        tiny_velocity = math.sqrt(2.0 * minimum_subnormal)
        first = physics.energies(
            2.0, 1.0,
            separation, 0.0, 1.0, 0.0,
            0.0, 0.0, tiny_velocity, 0.0,
        )
        swapped = physics.energies(
            1.0, 2.0,
            0.0, 0.0, tiny_velocity, 0.0,
            separation, 0.0, 1.0, 0.0,
        )
        self.assertEqual(first, swapped)
        self.assertEqual(first[2], minimum_subnormal)

    def test_total_kinetic_energy_overflow_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "calculated energy"):
            physics.energies(
                1.0, 1.0,
                1.0, 0.0, 1.4e154, 0.0,
                0.0, 0.0, 1.4e154, 0.0,
            )

    def test_potential_energy_avoids_intermediate_underflow(self):
        expected = -physics.G * 1e-100
        for mass_a, mass_b in ((1e-300, 1e300), (1e300, 1e-300)):
            with self.subTest(mass_a=mass_a, mass_b=mass_b):
                potential, kinetic, total = physics.energies(
                    mass_a, mass_b,
                    1e100, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                )
                self.assertNotEqual(potential, 0.0)
                self.assertTrue(math.isclose(potential, expected, rel_tol=2e-15))
                self.assertEqual(kinetic, 0.0)
                self.assertEqual(total, potential)

    def test_potential_energy_rejects_true_float_range_failures(self):
        cases = (
            (1e308, 1e308, 1.0),
            (1e-300, 1e-300, 1e300),
        )
        for mass_a, mass_b, separation in cases:
            with self.subTest(
                mass_a=mass_a, mass_b=mass_b, separation=separation
            ):
                with self.assertRaisesRegex(ValueError, "calculated energy"):
                    physics.energies(
                        mass_a, mass_b,
                        separation, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                    )

    def test_kinetic_energy_avoids_intermediate_overflow(self):
        cases = (
            (1e200, 0.0, 5e99),
            (0.0, 1e200, 5e99),
            (1e200, 1e200, 1e100),
        )
        for velocity_x, velocity_y, expected in cases:
            with self.subTest(velocity_x=velocity_x, velocity_y=velocity_y):
                _, kinetic, _ = physics.energies(
                    1e-300, 1.0,
                    1.0, 0.0, velocity_x, velocity_y,
                    0.0, 0.0, 0.0, 0.0,
                )
                self.assertTrue(math.isclose(kinetic, expected, rel_tol=2e-15))

    def test_extreme_kinetic_energy_preserves_body_swap_symmetry(self):
        first = physics.energies(
            1e-300, 1.0,
            1.0, 0.0, 1e200, -1e200,
            0.0, 0.0, 0.0, 0.0,
        )
        swapped = physics.energies(
            1.0, 1e-300,
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 1e200, -1e200,
        )
        self.assertEqual(first, swapped)

    def test_kinetic_energy_rejects_true_float_range_failures(self):
        cases = (
            (1e308, 1e-308, 1e308),
            (1e-300, 1e300, 1e-100),
        )
        for mass, other_mass, velocity in cases:
            with self.subTest(
                mass=mass, other_mass=other_mass, velocity=velocity
            ):
                with self.assertRaisesRegex(ValueError, "calculated energy"):
                    physics.energies(
                        mass, other_mass,
                        1.0, 0.0, velocity, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                    )

    def test_velocity_boost_changes_only_kinetic_terms(self):
        first = physics.energies(
            2.0, 3.0, 3.0, 4.0, 5.0, 6.0, -1.0, 2.0, -7.0, 8.0
        )
        boosted = physics.energies(
            2.0, 3.0, 3.0, 4.0, 15.0, -14.0, -1.0, 2.0, 3.0, -12.0
        )
        self.assertEqual(boosted[0], first[0])
        self.assertNotEqual(boosted[1:], first[1:])

    def test_bad_velocity_and_energy_overflow(self):
        for bad in (math.nan, math.inf, -math.inf, True, "3"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    physics.energies(
                        2.0, 3.0, 1.0, 0.0, bad, 0.0, 0.0, 0.0, 0.0, 0.0
                    )
        with self.assertRaisesRegex(ValueError, "calculated energy"):
            physics.energies(
                1e308, 1e308, 1.0, 0.0, 0.0, 0.0,
                -1.0, 0.0, 0.0, 0.0,
            )


class TestDriverValidation(unittest.TestCase):
    REAL_PARAMETERS = (
        "MA", "MB", "xInitA", "yInitA", "vInitA", "uInitA",
        "xInitB", "yInitB", "vInitB", "uInitB", "dt", "eps1", "eps2",
    )

    def test_all_real_inputs_reject_bad_values(self):
        for name in self.REAL_PARAMETERS:
            for bad in (math.nan, math.inf, -math.inf, True, "invalid"):
                with self.subTest(name=name, bad=bad):
                    with self.assertRaisesRegex(ValueError, "finite real number"):
                        integrate(**{name: bad})

    def test_positive_bounds(self):
        for name in ("MA", "MB"):
            for bad in (0.0, -1.0):
                with self.subTest(name=name, bad=bad):
                    with self.assertRaisesRegex(ValueError, "must both be positive"):
                        integrate(**{name: bad})
        for bad in (0.0, -1.0):
            with self.subTest(dt=bad):
                with self.assertRaisesRegex(ValueError, "dt must be positive"):
                    integrate(dt=bad)

    def test_max_steps_validation(self):
        for bad in (0, -1, 1.5, True, "10"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    integrate(max_steps=bad)

    def test_tolerance_validation(self):
        for name in ("eps1", "eps2"):
            for bad in (0.0, -0.1, 1.0, 1.1):
                with self.subTest(name=name, bad=bad):
                    with self.assertRaisesRegex(ValueError, r"0 < .* < 1"):
                        integrate(**{name: bad})

    def test_tolerances_immediately_inside_both_limits_are_accepted(self):
        lower = math.nextafter(0.0, 1.0)
        upper = math.nextafter(1.0, 0.0)
        with mock.patch.object(
            driver, "accelerations", return_value=(0.0, 0.0, 0.0, 0.0)
        ):
            for name in ("eps1", "eps2"):
                for value in (lower, upper):
                    with self.subTest(name=name, value=value):
                        result = integrate(
                            **{
                                name: value,
                                "max_steps": 1,
                                "stop_after_one_orbit": False,
                            }
                        )
                        self.assertEqual(result.accepted_steps, 1)

    def test_stop_flag_validation(self):
        for bad in (0, 1, 0.0, "True", None):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "True or False"):
                    integrate(stop_after_one_orbit=bad)

    def test_initial_collision(self):
        with self.assertRaisesRegex(ValueError, "zero separation"):
            integrate(xInitB=DEFAULTS["xInitA"], yInitB=DEFAULTS["yInitA"])

    def test_internal_numerical_guards(self):
        with self.assertRaisesRegex(RuntimeError, "numerical safety limit"):
            driver._halve_timestep(1.0, 0.75)
        with self.assertRaisesRegex(RuntimeError, "numerical safety limit"):
            driver._halve_timestep(5e-324, 0.0)
        with self.assertRaisesRegex(RuntimeError, "non-finite vector"):
            driver._vector_relative_change(0.0, 0.0, math.inf, 0.0)
        with self.assertRaisesRegex(RuntimeError, "cannot advance time"):
            driver._advance_time(1e308, 1e308)
        with self.assertRaisesRegex(RuntimeError, "cannot advance time"):
            driver._advance_time(1e308, 1.0)


class TestAdaptiveControllerBranches(unittest.TestCase):
    def test_eps1_rejection_retries_without_recording_and_recovers_by_1_1(self):
        original = driver._vector_relative_change
        calls = 0

        def reject_first(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                return 1.0
            return original(*args)

        with mock.patch.object(
            driver, "_vector_relative_change", side_effect=reject_first
        ), mock.patch.object(
            driver,
            "_signed_angle_increment",
            wraps=driver._signed_angle_increment,
        ) as angle:
            result = integrate(max_steps=3, stop_after_one_orbit=False)

        steps = [
            later - earlier
            for earlier, later in zip(result.times, result.times[1:])
        ]
        self.assertEqual(result.accepted_steps, 3)
        self.assertEqual(len(result.times), 4)
        self.assertEqual(angle.call_count, result.accepted_steps)
        self.assertAlmostEqual(steps[0], DEFAULTS["dt"] / 2.0)
        self.assertAlmostEqual(steps[1], steps[0] * 1.1)
        self.assertAlmostEqual(steps[2], steps[1] * 1.1)
        self.assertTrue(all(step <= DEFAULTS["dt"] for step in steps))

    def test_corrector_nonconvergence_halves_and_retries_same_step(self):
        # First trial: acceleration check passes, then both bodies fail all ten
        # corrector iterations. Second trial: all three checks pass.
        changes = [0.0] + [1.0] * 20 + [0.0, 0.0, 0.0]
        with mock.patch.object(
            driver, "_vector_relative_change", side_effect=changes
        ), mock.patch.object(
            driver, "accelerations", return_value=(0.0, 0.0, 0.0, 0.0)
        ):
            result = integrate(max_steps=1, stop_after_one_orbit=False)

        self.assertEqual(result.accepted_steps, 1)
        self.assertEqual(len(result.times), 2)
        self.assertEqual(result.times[1], DEFAULTS["dt"] / 2.0)

    def test_retry_exhaustion_raises_specific_error(self):
        # Hold the mocked halving result constant so the explicit 60-retry
        # ceiling, rather than the separate timestep-floor guard, is exercised.
        with mock.patch.object(
            driver, "_vector_relative_change", return_value=1.0
        ), mock.patch.object(
            driver, "_halve_timestep", side_effect=lambda value, _floor: value
        ) as halve:
            with self.assertRaisesRegex(
                RuntimeError, "could not find a converged timestep after 60 retries"
            ):
                integrate(max_steps=1, stop_after_one_orbit=False)
        self.assertEqual(halve.call_count, 60)

    def test_signed_angle_increment_handles_both_branch_cut_directions(self):
        angle = math.radians(170.0)
        x0, y0 = math.cos(angle), math.sin(angle)
        x1, y1 = math.cos(-angle), math.sin(-angle)
        scale_pairs = (
            (1.0, 1.0),
            (1e200, 1e200),
            (1e-300, 1e-300),
            (1e300, 1e-300),
        )
        for scale0, scale1 in scale_pairs:
            with self.subTest(scale0=scale0, scale1=scale1):
                forward = driver._signed_angle_increment(
                    scale0 * x0, scale0 * y0,
                    scale1 * x1, scale1 * y1,
                )
                reverse = driver._signed_angle_increment(
                    scale1 * x1, scale1 * y1,
                    scale0 * x0, scale0 * y0,
                )
                self.assertAlmostEqual(
                    forward, math.radians(20.0), places=14
                )
                self.assertAlmostEqual(
                    reverse, -math.radians(20.0), places=14
                )

    def test_signed_angle_increment_rejects_invalid_vectors(self):
        cases = (
            (0.0, 0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (math.nan, 0.0, 1.0, 0.0),
            (1.0, 0.0, math.inf, 0.0),
            (1.0, -math.inf, 1.0, 0.0),
        )
        for values in cases:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    driver._signed_angle_increment(*values)

    def test_nonfinite_angle_increment_cannot_poison_integration(self):
        with mock.patch.object(
            driver, "_signed_angle_increment", return_value=math.nan
        ):
            with self.assertRaisesRegex(RuntimeError, "non-finite increment"):
                integrate(max_steps=1, stop_after_one_orbit=True)


class TestIntegrationRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default = integrate()

    def test_lengths_and_initial_record(self):
        result = self.default
        series = (
            result.times, result.xA, result.yA, result.vA, result.uA,
            result.xB, result.yB, result.vB, result.uB,
            result.U, result.K, result.E,
        )
        self.assertTrue(
            all(len(values) == result.accepted_steps + 1 for values in series)
        )
        self.assertEqual(result.times[0], 0.0)
        self.assertEqual(result.xA[0], DEFAULTS["xInitA"])
        self.assertEqual(result.uB[0], DEFAULTS["uInitB"])

    def test_outputs_are_finite_and_time_increases(self):
        result = self.default
        series = (
            result.times, result.xA, result.yA, result.vA, result.uA,
            result.xB, result.yB, result.vB, result.uB,
            result.U, result.K, result.E,
        )
        self.assertTrue(
            all(math.isfinite(value) for values in series for value in values)
        )
        self.assertTrue(all(b > a for a, b in zip(result.times, result.times[1:])))

    def test_default_orbit_and_energy(self):
        self.assertTrue(self.default.completed_orbit)
        self.assertLess(self.default.accepted_steps, DEFAULTS["max_steps"])
        self.assertTrue(4.4e6 < self.default.times[-1] < 4.8e6)
        self.assertLess(energy_drift(self.default), 2e-3)

    def test_center_of_mass_and_momentum(self):
        result = self.default
        total_mass = DEFAULTS["MA"] + DEFAULTS["MB"]
        com = [
            (DEFAULTS["MA"] * a + DEFAULTS["MB"] * b) / total_mass
            for arrays in ((result.xA, result.xB), (result.yA, result.yB))
            for a, b in zip(*arrays)
        ]
        momentum = [
            DEFAULTS["MA"] * a + DEFAULTS["MB"] * b
            for arrays in ((result.vA, result.vB), (result.uA, result.uB))
            for a, b in zip(*arrays)
        ]
        self.assertTrue(all(value == 0.0 for value in com + momentum))

    def test_unequal_mass_moving_com_conservation(self):
        parameters = unequal_moving_case()
        result = driver.integrate_binary(**parameters)
        mass_a = parameters["MA"]
        mass_b = parameters["MB"]
        total_mass = mass_a + mass_b
        initial_com_x = (
            mass_a * parameters["xInitA"] + mass_b * parameters["xInitB"]
        ) / total_mass
        initial_com_y = (
            mass_a * parameters["yInitA"] + mass_b * parameters["yInitB"]
        ) / total_mass
        initial_com_vx = (
            mass_a * parameters["vInitA"] + mass_b * parameters["vInitB"]
        ) / total_mass
        initial_com_vy = (
            mass_a * parameters["uInitA"] + mass_b * parameters["uInitB"]
        ) / total_mass
        initial_px = mass_a * result.vA[0] + mass_b * result.vB[0]
        initial_py = mass_a * result.uA[0] + mass_b * result.uB[0]
        momentum_scale = math.hypot(initial_px, initial_py)
        separation_scale = math.hypot(
            parameters["xInitA"] - parameters["xInitB"],
            parameters["yInitA"] - parameters["yInitB"],
        )

        momentum_error = max(
            math.hypot(
                mass_a * va + mass_b * vb - initial_px,
                mass_a * ua + mass_b * ub - initial_py,
            )
            for va, vb, ua, ub in zip(
                result.vA, result.vB, result.uA, result.uB
            )
        )
        com_error = max(
            math.hypot(
                (mass_a * xa + mass_b * xb) / total_mass
                - (initial_com_x + initial_com_vx * time),
                (mass_a * ya + mass_b * yb) / total_mass
                - (initial_com_y + initial_com_vy * time),
            )
            for time, xa, xb, ya, yb in zip(
                result.times, result.xA, result.xB, result.yA, result.yB
            )
        )
        angular_momentum = [
            mass_a * (xa * ua - ya * va)
            + mass_b * (xb * ub - yb * vb)
            for xa, ya, va, ua, xb, yb, vb, ub in zip(
                result.xA, result.yA, result.vA, result.uA,
                result.xB, result.yB, result.vB, result.uB,
            )
        ]
        angular_drift = max(
            abs(value - angular_momentum[0]) for value in angular_momentum
        ) / abs(angular_momentum[0])

        self.assertLess(momentum_error / momentum_scale, 1e-12)
        self.assertLess(com_error / separation_scale, 1e-12)
        self.assertLess(angular_drift, 1e-8)

    def test_unequal_mass_body_label_swap_preserves_trajectories(self):
        parameters = unequal_moving_case()
        swapped = dict(parameters)
        for name_a, name_b in (
            ("MA", "MB"),
            ("xInitA", "xInitB"),
            ("yInitA", "yInitB"),
            ("vInitA", "vInitB"),
            ("uInitA", "uInitB"),
        ):
            swapped[name_a], swapped[name_b] = swapped[name_b], swapped[name_a]

        original = driver.integrate_binary(**parameters)
        relabeled = driver.integrate_binary(**swapped)
        self.assertEqual(original.accepted_steps, relabeled.accepted_steps)
        for original_values, relabeled_values in (
            (original.xA, relabeled.xB),
            (original.yA, relabeled.yB),
            (original.vA, relabeled.vB),
            (original.uA, relabeled.uB),
            (original.xB, relabeled.xA),
            (original.yB, relabeled.yA),
            (original.vB, relabeled.vA),
            (original.uB, relabeled.uA),
        ):
            self.assertEqual(original_values, relabeled_values)

    def test_angular_momentum_drift(self):
        result = self.default
        values = [
            DEFAULTS["MA"] * (xa * ua - ya * va)
            + DEFAULTS["MB"] * (xb * ub - yb * vb)
            for xa, ya, va, ua, xb, yb, vb, ub in zip(
                result.xA, result.yA, result.vA, result.uA,
                result.xB, result.yB, result.vB, result.uB,
            )
        ]
        initial = values[0]
        drift = max(abs(value - initial) for value in values) / abs(initial)
        self.assertLess(drift, 3e-4)

    def test_analytic_circular_case(self):
        radius = DEFAULTS["xInitA"]
        speed = math.sqrt(physics.G * DEFAULTS["MA"] / (4.0 * radius))
        result = integrate(uInitA=speed, uInitB=-speed)
        separations = [
            math.hypot(xa - xb, ya - yb)
            for xa, xb, ya, yb in zip(result.xA, result.xB, result.yA, result.yB)
        ]
        variation = (max(separations) - min(separations)) / separations[0]
        period = 2 * math.pi * math.sqrt(
            separations[0] ** 3
            / (physics.G * (DEFAULTS["MA"] + DEFAULTS["MB"]))
        )
        self.assertTrue(result.completed_orbit)
        self.assertLess(variation, 2e-5)
        self.assertLess(abs(result.times[-1] - period) / period, 5e-4)

    def test_circular_and_eccentric_velocity_hodographs_are_circular(self):
        radius = DEFAULTS["xInitA"]
        speed = math.sqrt(physics.G * DEFAULTS["MA"] / (4.0 * radius))
        circular = integrate(uInitA=speed, uInitB=-speed)
        self.assertLess(
            fitted_circle_radial_variation(circular.vA, circular.uA), 1e-6
        )
        self.assertLess(
            fitted_circle_radial_variation(self.default.vA, self.default.uA),
            2e-3,
        )

    def test_timestep_convergence(self):
        finer = integrate(dt=1000.0)
        self.assertLess(energy_drift(finer), energy_drift(self.default))

    def test_translation_invariance_of_integrated_relative_motion(self):
        shifted = integrate(
            xInitA=DEFAULTS["xInitA"] + 1e10,
            xInitB=DEFAULTS["xInitB"] + 1e10,
            yInitA=-2e10,
            yInitB=-2e10,
        )
        self.assertEqual(shifted.accepted_steps, self.default.accepted_steps)
        scale = max(
            math.hypot(xa - xb, ya - yb)
            for xa, xb, ya, yb in zip(
                self.default.xA, self.default.xB,
                self.default.yA, self.default.yB,
            )
        )
        error = max(
            math.hypot(
                (xc - xd) - (xa - xb),
                (yc - yd) - (ya - yb),
            )
            for xa, xb, ya, yb, xc, xd, yc, yd in zip(
                self.default.xA, self.default.xB,
                self.default.yA, self.default.yB,
                shifted.xA, shifted.xB, shifted.yA, shifted.yB,
            )
        )
        self.assertLess(error / scale, 1e-12)

    def test_galilean_invariance_of_integrated_relative_motion(self):
        boosted = integrate(
            vInitA=1000.0,
            vInitB=1000.0,
            uInitA=14000.0,
            uInitB=-12000.0,
        )
        self.assertEqual(boosted.accepted_steps, self.default.accepted_steps)
        scale = max(
            math.hypot(xa - xb, ya - yb)
            for xa, xb, ya, yb in zip(
                self.default.xA, self.default.xB,
                self.default.yA, self.default.yB,
            )
        )
        error = max(
            math.hypot(
                (xc - xd) - (xa - xb),
                (yc - yd) - (ya - yb),
            )
            for xa, xb, ya, yb, xc, xd, yc, yd in zip(
                self.default.xA, self.default.xB,
                self.default.yA, self.default.yB,
                boosted.xA, boosted.xB, boosted.yA, boosted.yB,
            )
        )
        self.assertLess(error / scale, 1e-5)

    def test_max_steps_and_unbound_motion(self):
        bound = integrate(max_steps=20, stop_after_one_orbit=False)
        unbound = integrate(
            uInitA=60000.0,
            uInitB=-60000.0,
            max_steps=100,
            stop_after_one_orbit=False,
        )
        self.assertEqual((bound.accepted_steps, bound.completed_orbit), (20, False))
        self.assertEqual(
            (unbound.accepted_steps, unbound.completed_orbit), (100, False)
        )
        self.assertGreater(unbound.E[0], 0.0)

    def test_retrograde_orbit_completes_one_revolution(self):
        result = integrate(uInitA=-13000.0, uInitB=13000.0)
        self.assertTrue(result.completed_orbit)
        self.assertTrue(4.4e6 < result.times[-1] < 4.8e6)

    def test_radial_and_unbound_cases_do_not_false_complete(self):
        radial = integrate(
            uInitA=0.0,
            uInitB=0.0,
            max_steps=100,
            stop_after_one_orbit=True,
        )
        unbound = integrate(
            uInitA=60000.0,
            uInitB=-60000.0,
            max_steps=100,
            stop_after_one_orbit=True,
        )
        self.assertFalse(radial.completed_orbit)
        self.assertFalse(unbound.completed_orbit)
        self.assertEqual((radial.accepted_steps, unbound.accepted_steps), (100, 100))

    def test_head_on_case_fails_instead_of_freezing(self):
        with self.assertRaisesRegex(RuntimeError, "numerical safety limit"):
            integrate(uInitA=0.0, uInitB=0.0, max_steps=100000)

    def test_extreme_trajectory_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "numerical range"):
            integrate(xInitA=1e308, xInitB=-1e308)


class TestPlotting(unittest.TestCase):
    OUTPUT_TYPES = (
        "orbit",
        "orbits",
        "velocity space",
        "position vs. time, body A",
        "position vs. time, body B",
        "velocity vs. time, body A",
        "velocity vs. time, body B",
        "energy vs time",
    )

    @classmethod
    def setUpClass(cls):
        cls.result = integrate(max_steps=2, stop_after_one_orbit=False)

    def tearDown(self):
        plotting.plt.close("all")

    def test_every_output_maps_exact_data_labels_titles_and_legends(self):
        result = self.result
        expectations = {
            "orbit": (
                ((result.xA, result.yA), (result.xB, result.yB)),
                ("Body A", "Body B"), "x (m)", "y (m)", "Binary orbits",
            ),
            "orbits": (
                ((result.xA, result.yA), (result.xB, result.yB)),
                ("Body A", "Body B"), "x (m)", "y (m)", "Binary orbits",
            ),
            "velocity space": (
                ((result.vA, result.uA), (result.vB, result.uB)),
                ("Body A", "Body B"), "v_x (m/s)", "v_y (m/s)",
                "Velocity space",
            ),
            "position vs. time, body A": (
                ((result.times, result.xA), (result.times, result.yA)),
                ("x_A(t)", "y_A(t)"), "t (s)", "position (m)",
                "Position vs time, body A",
            ),
            "position vs. time, body B": (
                ((result.times, result.xB), (result.times, result.yB)),
                ("x_B(t)", "y_B(t)"), "t (s)", "position (m)",
                "Position vs time, body B",
            ),
            "velocity vs. time, body A": (
                ((result.times, result.vA), (result.times, result.uA)),
                ("v_A(t)", "u_A(t)"), "t (s)", "velocity (m/s)",
                "Velocity vs time, body A",
            ),
            "velocity vs. time, body B": (
                ((result.times, result.vB), (result.times, result.uB)),
                ("v_B(t)", "u_B(t)"), "t (s)", "velocity (m/s)",
                "Velocity vs time, body B",
            ),
            "energy vs time": (
                (
                    (result.times, result.U),
                    (result.times, result.K),
                    (result.times, result.E),
                ),
                ("Potential U", "Kinetic K", "Total E"),
                "t (s)", "Energy (J)", "Energy vs time",
            ),
        }

        for output_type, expected in expectations.items():
            with self.subTest(output_type=output_type):
                plotting.plt.close("all")
                with mock.patch.object(plotting.plt, "show") as show, \
                     mock.patch.object(
                         plotting.plt, "tight_layout", wraps=plotting.plt.tight_layout
                     ) as tight_layout:
                    plotting.plot_binary(self.result, output_type)
                show.assert_called_once_with()
                tight_layout.assert_called_once_with()
                axis = plotting.plt.gca()
                data_pairs, labels, xlabel, ylabel, title = expected
                self.assertEqual(len(axis.lines), len(data_pairs))
                for line, (x_values, y_values) in zip(axis.lines, data_pairs):
                    self.assertEqual(list(line.get_xdata()), list(x_values))
                    self.assertEqual(list(line.get_ydata()), list(y_values))
                self.assertEqual(tuple(axis.get_legend_handles_labels()[1]), labels)
                self.assertEqual(axis.get_xlabel(), xlabel)
                self.assertEqual(axis.get_ylabel(), ylabel)
                self.assertEqual(axis.get_title(), title)

    def test_equal_aspect_outputs(self):
        for output_type in ("orbit", "orbits", "velocity space"):
            with self.subTest(output_type=output_type):
                with mock.patch.object(plotting.plt, "show"):
                    plotting.plot_binary(self.result, output_type)
                self.assertEqual(plotting.plt.gca().get_aspect(), 1.0)
                plotting.plt.close("all")

    def test_unknown_output(self):
        with self.assertRaisesRegex(ValueError, "Unknown output_type"):
            plotting.plot_binary(self.result, "not an output")


class HelpContractParser(HTMLParser):
    """Extract contract-bearing Help elements with standard-library HTML parsing."""

    def __init__(self):
        super().__init__()
        self.section = None
        self.capture = None
        self.buffer = []
        self.version_build = ""
        self.output_tags = []
        self.parameter_rows = []
        self.current_row = None
        self.current_cell = None
        self.exercise_headings = []
        self.equation_labels = []
        self.module_names = []

    @staticmethod
    def _classes(attributes):
        value = dict(attributes).get("class", "")
        return set(value.split())

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        classes = self._classes(attributes)
        if tag == "section":
            self.section = attrs.get("id")
        if tag == "p" and attrs.get("id") == "version_build":
            self.capture, self.buffer = "version", []
        elif tag == "span" and "eq-label" in classes:
            self.capture, self.buffer = "equation", []
        elif self.section == "output-types" and tag == "span" and "tag" in classes:
            self.capture, self.buffer = "output", []
        elif self.section == "parameters" and tag == "tr":
            self.current_row = []
        elif self.current_row is not None and tag in ("th", "td"):
            self.current_cell = []
        elif self.section == "experiments" and tag == "h3":
            self.capture, self.buffer = "exercise", []
        elif tag == "div" and "mc-name" in classes:
            self.capture, self.buffer = "module", []

    def handle_data(self, data):
        if self.capture is not None:
            self.buffer.append(data)
        if self.current_cell is not None:
            self.current_cell.append(data)

    def handle_endtag(self, tag):
        if self.current_cell is not None and tag in ("th", "td"):
            text = " ".join("".join(self.current_cell).split())
            self.current_row.append(text)
            self.current_cell = None
        if self.current_row is not None and tag == "tr":
            self.parameter_rows.append(tuple(self.current_row))
            self.current_row = None
        if self.capture == "version" and tag == "p":
            self.version_build = " ".join("".join(self.buffer).split())
            self.capture = None
        elif self.capture == "equation" and tag == "span":
            self.equation_labels.append(" ".join("".join(self.buffer).split()))
            self.capture = None
        elif self.capture == "output" and tag == "span":
            self.output_tags.append(" ".join("".join(self.buffer).split()))
            self.capture = None
        elif self.capture == "exercise" and tag == "h3":
            self.exercise_headings.append(" ".join("".join(self.buffer).split()))
            self.capture = None
        elif self.capture == "module" and tag == "div":
            self.module_names.append(" ".join("".join(self.buffer).split()))
            self.capture = None
        if tag == "section":
            self.section = None


class TestHelpFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HELP_FILE.is_file():
            raise AssertionError(f"Required Help file not found: {HELP_FILE}")
        cls.html = HELP_FILE.read_text(encoding="utf-8")
        cls.prose = re.sub(r"\s+", " ", cls.html)
        cls.contract = HelpContractParser()
        cls.contract.feed(cls.html)
        cls.contract.close()

    def test_version_build_sync(self):
        match = re.fullmatch(
            r"Version\s+([0-9]+\.[0-9]+\.[0-9]+)\s+Build\s+([0-9a-f]{12})",
            self.contract.version_build,
            flags=re.IGNORECASE,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), physics.MODEL_VERSION)
        self.assertEqual(match.group(2), physics.BUILD_ID)

    def test_core_module_card_names_are_exact(self):
        self.assertEqual(
            tuple(self.contract.module_names),
            ("main.py", "physics_binary.py", "driver_binary.py", "plot_binary.py"),
        )

    def test_output_tags_exactly_match_output_type_contract(self):
        self.assertEqual(
            tuple(tag.strip('"') for tag in self.contract.output_tags),
            TestPlotting.OUTPUT_TYPES,
        )

    def test_parameter_table_has_exact_documented_variables_and_defaults(self):
        rows = self.contract.parameter_rows
        self.assertEqual(
            rows[0], ("Variable", "Default", "Unit", "Description")
        )
        self.assertEqual(
            tuple(row[0] for row in rows[1:]),
            (
                "MA, MB",
                "xInitA",
                "xInitB",
                "yInitA, yInitB",
                "vInitA, vInitB",
                "uInitA",
                "uInitB",
                "dt",
                "max_steps",
                "eps1",
                "eps2",
                "stop_after_one_orbit",
            ),
        )
        self.assertEqual(
            tuple(row[1] for row in rows[1:]),
            (
                r"\(2.0\times10^{30}\)",
                r"\(+4.6\times10^{10}\)",
                r"\(-4.6\times10^{10}\)",
                r"\(0\)",
                r"\(0\)",
                r"\(+1.3\times10^4\)",
                r"\(-1.3\times10^4\)",
                r"\(2000\)",
                r"\(10000\)",
                r"\(0.05\)",
                r"\(10^{-4}\)",
                "True",
            ),
        )

    def test_equation_numbers(self):
        labels = [
            int(match.group(1))
            for value in self.contract.equation_labels
            if (match := re.fullmatch(r"\((\d+)\)", value))
        ]
        self.assertEqual(labels, list(range(1, 11)))

    def test_bounds_are_documented(self):
        for phrase in (
            "finite real numbers",
            "strictly between zero and one",
            "0 and 1 are not allowed",
            "double-precision arithmetic",
            "nonzero initial separation",
            "both bodies' accelerations",
            "potential, kinetic, and total energy",
            "round to zero",
        ):
            self.assertIn(phrase, self.prose)

    def test_minimum_runtime_and_matplotlib_versions_are_documented(self):
        self.assertIn("Python 3.10 or later", self.prose)
        self.assertIn("matplotlib</code> 3.5 or later", self.prose)

    def test_plot_count_distinguishes_designs_from_selector_alias(self):
        self.assertIn(
            "seven distinct plots through eight accepted selector strings",
            self.prose,
        )

    def test_termination_and_energy_qualifications(self):
        for phrase in (
            "relative vector",
            "relative revolution",
            "scaled independently",
            "non-finite orbit-angle diagnostic",
            "does not by itself prove",
            "kinetic energy of the centre of mass",
            "subtract its translational kinetic energy",
            "small representable residual",
            "Exact zero remains valid",
            "problem-dependent",
            "There is no universal factor",
        ):
            self.assertIn(phrase, self.prose)

    def test_exercise_order_and_difficulty(self):
        headings = self.contract.exercise_headings
        numbers = [
            int(match.group(1))
            for heading in headings
            if (match := re.match(r"(\d+) · ", heading))
        ]
        self.assertEqual(numbers, list(range(1, 9)))
        joined = " ".join(headings)
        for level in ("Introductory", "Intermediate", "Advanced"):
            self.assertIn(level, joined)

    def test_reflex_setup_is_reproducible(self):
        for phrase in (
            "Jupiter", "Saturn", "Earth",
            r"v_{\rm rel}", r"x_A=-aM_B",
            "precision Solar-System ephemeris",
        ):
            self.assertIn(phrase, self.html)

    def test_relative_displacement_help_matches_three_value_contract(self):
        self.assertIn(
            "relative_displacement()</code> returns the separation vector "
            r"\((\Delta x,\,\Delta y)\) and scalar distance \(r\)",
            self.prose,
        )
        self.assertEqual(
            len(physics.relative_displacement(4.0, 6.0, 1.0, 2.0)), 3
        )

    def test_restore_box_uses_css_classes_not_inline_styles(self):
        match = re.search(
            r'<section class="restore-section".*?</section>',
            self.html,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        self.assertNotIn("style=", match.group(0))

    def test_no_development_history(self):
        for pattern in (
            r"\bAI-generated\b", r"\bChatGPT\b", r"\bClaude\b",
            r"\bCopilot\b", r"\bGemini\b", r"\bported from\b",
            r"\bporting history\b", r"\bbug fix history\b",
        ):
            self.assertIsNone(re.search(pattern, self.html, re.IGNORECASE))

    def test_license_provenance(self):
        for phrase in (
            "Bernard Schutz",
            "Gravity from the Ground Up",
            "Cambridge University Press",
            "CC BY-NC-SA 4.0",
        ):
            self.assertIn(phrase, self.html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
