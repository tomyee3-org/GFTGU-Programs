"""Regression tests for the complete Binary teaching module.

The discovery logic supports both the canonical tests subdirectory and a
review upload in which this file is flattened beside the four core modules.
"""

from __future__ import annotations

import ast
from contextlib import redirect_stdout, redirect_stderr
import functools
import hashlib
import html as html_module
from html.parser import HTMLParser
import io
import math
import os
from pathlib import Path
import re
import shlex
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


def find_help_file(module_dir: Path) -> Path:
    """Find Help in a flattened upload or the GFTGU-Documentation tree.

    Documentation folders no longer use chapter-number prefixes, and the
    Help files live under the sibling ``GFTGU-Documentation`` repository
    rather than beside the program modules inside ``GFTGU-Programs``.
    """
    help_filename = "Binary.html"
    program_name = "Binary"
    candidates = [module_dir / help_filename]
    for ancestor in (module_dir, *module_dir.parents):
        candidates.append(
            ancestor / "GFTGU-Documentation" / program_name / help_filename
        )
        if ancestor.name != program_name:
            candidates.append(ancestor / program_name / help_filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find Binary.html beside the program or in "
        "GFTGU-Documentation/Binary/."
    )


MODULE_DIR = find_module_dir(Path(__file__))
HELP_FILE = find_help_file(MODULE_DIR)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import driver_binary as driver  # noqa: E402
import physics_binary as physics  # noqa: E402
import plot_binary as plotting  # noqa: E402
import main as entry  # noqa: E402


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


class TestCommandLineAndSummary(unittest.TestCase):
    def test_all_driver_defaults_exposed(self):
        args = entry.parse_args([])
        self.assertEqual({name: getattr(args, name) for name in DEFAULTS}, DEFAULTS)
        self.assertEqual(args.output_type, "orbits")

    def test_custom_driver_inputs_and_plot_selector(self):
        sample = integrate(max_steps=2)
        with mock.patch.object(entry, "integrate_binary", return_value=sample) as run, \
             mock.patch.object(entry, "plot_binary") as plot, \
             redirect_stdout(io.StringIO()) as printed:
            entry.main(["--MB", "1e30", "--xInitA", "3e10", "--uInitB", "-9000",
                        "--max_steps", "2", "--no-stop_after_one_orbit",
                        "--output_type", "velocity_space"])
        self.assertEqual(run.call_args.kwargs["MB"], 1e30)
        self.assertEqual(run.call_args.kwargs["xInitA"], 3e10)
        self.assertEqual(run.call_args.kwargs["uInitB"], -9000)
        self.assertFalse(run.call_args.kwargs["stop_after_one_orbit"])
        self.assertEqual(plot.call_args.args[1], "velocity space")
        self.assertIn("Energy at fractions", printed.getvalue())

    def test_invalid_inputs_and_selector(self):
        for args in (["--MA", "0"], ["--dt", "nan"], ["--eps1", "1"],
                     ["--max_steps", "1.5"], ["--output_type", "velocity space"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as raised:
                entry.parse_args(args)
            self.assertEqual(raised.exception.code, 2)

    def test_negative_exponent_form_is_accepted_as_separate_option_value(self):
        args = entry.parse_args(["--xInitB", "-4.2e10", "--uInitB", "-1.3e4"])
        self.assertEqual(args.xInitB, -4.2e10)
        self.assertEqual(args.uInitB, -13000.)

    def test_orbital_elements_circular_two_to_one_mass_ratio_and_boost(self):
        params = dict(unequal_moving_case())
        state = [params[k] for k in ("MA", "MB", "xInitA", "yInitA", "vInitA",
                                     "uInitA", "xInitB", "yInitB", "vInitB", "uInitB")]
        elements = physics.orbital_elements(*state)
        r = math.hypot(params["xInitA"] - params["xInitB"],
                       params["yInitA"] - params["yInitB"])
        self.assertEqual(elements.kind, "elliptic")
        self.assertLess(elements.eccentricity, 1e-12)
        self.assertAlmostEqual(elements.relative_semimajor / r, 1, places=12)
        self.assertAlmostEqual(elements.relative_periapsis / r, 1, places=12)
        self.assertAlmostEqual(elements.relative_apoapsis / r, 1, places=12)
        self.assertAlmostEqual(elements.period / (2 * math.pi * math.sqrt(
            r ** 3 / (physics.G * (params["MA"] + params["MB"])))), 1, places=12)
        state[4] += 12345
        state[8] += 12345
        self.assertAlmostEqual(physics.orbital_elements(*state).relative_semimajor / r,
                               1, places=12)

    def test_open_and_radial_elements_do_not_claim_nonexistent_apsides(self):
        params = dict(DEFAULTS)
        params.update(uInitA=60000., uInitB=-60000.)
        keys = ("MA", "MB", "xInitA", "yInitA", "vInitA", "uInitA",
                "xInitB", "yInitB", "vInitB", "uInitB")
        value = physics.orbital_elements(*(params[k] for k in keys))
        self.assertEqual(value.kind, "hyperbolic")
        self.assertLess(value.relative_semimajor, 0)
        self.assertGreater(value.relative_periapsis, 0)
        self.assertIsNone(value.period)
        self.assertIsNone(value.relative_apoapsis)
        params.update(uInitA=0., uInitB=0.)
        value = physics.orbital_elements(*(params[k] for k in keys))
        self.assertEqual(value.kind, "radial")
        self.assertIsNone(value.relative_speed_periapsis)

    def test_energy_table_has_interpolated_midpoint_and_zero_energy_policy(self):
        result = integrate(max_steps=1)
        output = io.StringIO()
        with redirect_stdout(output):
            entry.print_summary(result, DEFAULTS)
        rows = [line.split() for line in output.getvalue().splitlines()
                if re.match(r"\s*[01]\.\d\s", line)]
        self.assertEqual(len(rows), 11)
        self.assertEqual(rows[0][0], "0.0")
        self.assertEqual(rows[-1][0], "1.0")
        self.assertAlmostEqual(float(rows[5][2]) / ((result.U[0] + result.U[1]) / 2),
                               1, places=4)
        result.E[0] = 0.
        with redirect_stdout(io.StringIO()) as output:
            entry.print_summary(result, DEFAULTS)
        self.assertIn("fractional energy departure is undefined", output.getvalue().lower())


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

    # Until the Release Notes and Sample Outputs Guide are brought up to date
    # in the documentation round, they still carry the previous version and
    # build.  This test is expected to fail for that reason alone; remove the
    # decorator when those two documents are updated (an unexpected pass is
    # reported as a failure, which is the reminder).  The commands they show
    # are still checked by the next test.
    @unittest.expectedFailure
    def test_release_and_sample_output_metadata_and_commands(self):
        docs = HELP_FILE.parent
        for path in (docs / "Binary-ReleaseNotes.html",
                     docs / "SampleOutputs" / "Binary-SampleOutputs_Guide.html"):
            with self.subTest(path=path):
                page = path.read_text(encoding="utf-8")
                self.assertIn(physics.MODEL_VERSION, page)
                self.assertIn(physics.BUILD_ID, page)
                self.assertIn("python main.py", page)
        self.assertIn("--uInitB -31101", (docs / "SampleOutputs" /
                      "Binary-SampleOutputs_Guide.html").read_text())

    def test_release_and_sample_output_commands_and_options(self):
        docs = HELP_FILE.parent
        guide = docs / "SampleOutputs" / "Binary-SampleOutputs_Guide.html"
        notes = docs / "Binary-ReleaseNotes.html"
        if not (guide.is_file() and notes.is_file()):
            self.skipTest("Release Notes and Sample Outputs Guide are not beside the Help file")
        for path in (notes, guide):
            with self.subTest(path=path.name):
                self.assertIn("python main.py", path.read_text(encoding="utf-8"))
        self.assertIn("--uInitB -31101", guide.read_text(encoding="utf-8"))

    def test_core_module_card_names_are_exact(self):
        self.assertEqual(
            tuple(self.contract.module_names),
            ("main.py", "physics_binary.py", "driver_binary.py", "plot_binary.py"),
        )

    def test_output_tags_exactly_match_output_type_contract(self):
        self.assertEqual(
            tuple(self.contract.output_tags), tuple(entry.OUTPUT_TYPES),
        )

    def test_parameter_table_has_exact_documented_variables_and_defaults(self):
        rows = self.contract.parameter_rows
        self.assertEqual(
            rows[0], ("Option", "Default", "Unit", "Description")
        )
        self.assertEqual(
            tuple(row[0] for row in rows[1:]),
            (
                "--MA, --MB", "--xInitA, --xInitB", "--yInitA, --yInitB",
                "--vInitA, --vInitB", "--uInitA, --uInitB", "--dt",
                "--max_steps", "--eps1", "--eps2",
                "--stop_after_one_orbit, --no-stop_after_one_orbit",
                "--output_type",
            ),
        )
        self.assertEqual(
            tuple(row[1] for row in rows[1:]),
            (
                "2e30 each", "+4.6e10, -4.6e10", "0, 0", "0, 0",
                "+13000, -13000", "2000", "10000", "0.05", "1e-4",
                "stop", "orbits",
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
            "seven distinct plots through seven command-line selectors",
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


# ---------------------------------------------------------------------------
# Printed-output helpers used by the summary, Help-command and Beat tests.
# ---------------------------------------------------------------------------

def printed_run(argv):
    """Run ``main.main`` with the plot suppressed and return what it prints."""
    with mock.patch.object(entry, "plot_binary"), redirect_stdout(io.StringIO()) as out:
        entry.main(list(argv))
    return out.getvalue()


SUMMARY_LABELS = (
    "Accepted steps", "Termination", "Total revolutions", "Total time",
    "Minimum separation (sampled)", "Maximum separation (sampled)",
    "Initial Keplerian orbit",
)


def printed_fields(printed):
    """The single-valued summary lines, by label."""
    fields = {}
    for line in printed.splitlines():
        for label in SUMMARY_LABELS:
            if line.startswith(label + ": "):
                fields[label] = line[len(label) + 2:]
    return fields


def printed_rows(printed):
    """The eleven energy-table rows, keyed by fraction, as lists of cells."""
    rows = {}
    for line in printed.splitlines():
        if re.match(r"\s*[01]\.\d\s", line):
            cells = line.split()
            rows[float(cells[0])] = cells
    return rows


def printed_body_block(printed, label):
    """Return the three lines of the ``Body <label>`` block."""
    lines = printed.splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith(f"Body {label} about centre of mass"))
    return lines[start:start + 3]


def printed_number(text):
    """First number in a printed field such as ``4.6e+10 m``."""
    return float(re.search(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text).group(0))


BODY_KEYS = ("MA", "MB", "xInitA", "yInitA", "vInitA", "uInitA",
             "xInitB", "yInitB", "vInitB", "uInitB")


def elements_for(**changes):
    parameters = dict(DEFAULTS)
    parameters.update(changes)
    return physics.orbital_elements(*(parameters[key] for key in BODY_KEYS))


def rotate_case(angle, **changes):
    """The default two-body state with every position and velocity rotated."""
    parameters = dict(DEFAULTS)
    parameters.update(changes)
    cos, sin = math.cos(angle), math.sin(angle)
    for x_key, y_key in (("xInitA", "yInitA"), ("vInitA", "uInitA"),
                         ("xInitB", "yInitB"), ("vInitB", "uInitB")):
        x, y = parameters[x_key], parameters[y_key]
        parameters[x_key], parameters[y_key] = cos * x - sin * y, sin * x + cos * y
    return parameters


def scalar_kepler(parameters):
    """Independent scalar two-body formulas for the relative orbit."""
    total = parameters["MA"] + parameters["MB"]
    mu = physics.G * total
    rx = parameters["xInitA"] - parameters["xInitB"]
    ry = parameters["yInitA"] - parameters["yInitB"]
    vx = parameters["vInitA"] - parameters["vInitB"]
    vy = parameters["uInitA"] - parameters["uInitB"]
    r = math.hypot(rx, ry)
    energy = 0.5 * (vx * vx + vy * vy) - mu / r
    h = abs(rx * vy - ry * vx)
    semi = -mu / (2 * energy)
    e = math.sqrt(1 + 2 * energy * h * h / (mu * mu))
    return {
        "semi": semi, "e": e, "period": 2 * math.pi * math.sqrt(semi ** 3 / mu),
        "peri": semi * (1 - e), "apo": semi * (1 + e),
        "v_peri": h / (semi * (1 - e)), "v_apo": h / (semi * (1 + e)),
    }


class TestCompensatedAngleSum(unittest.TestCase):
    """The orbit angle is summed with Neumaier compensation."""

    def test_helper_matches_exact_summation_where_plain_addition_stalls(self):
        terms = [1.0] + [1e-16] * 200_000
        plain = 0.0
        total, compensation = 0.0, 0.0
        for term in terms:
            plain += term
            total, compensation = driver._compensated_add(total, compensation, term)
        exact = math.fsum(terms)
        self.assertEqual(plain, 1.0)
        self.assertGreater(exact, 1.0 + 1e-11)
        self.assertAlmostEqual((total + compensation) / exact, 1.0, delta=1e-15)

    def test_helper_handles_a_small_term_first_and_a_large_term_first(self):
        for terms in ([1e-16, 1.0], [1.0, 1e-16], [1e-16] * 3 + [1.0] + [-1.0] + [1e-16]):
            with self.subTest(terms=terms):
                total, compensation = 0.0, 0.0
                for term in terms:
                    total, compensation = driver._compensated_add(total, compensation, term)
                self.assertEqual(total + compensation, math.fsum(terms))

    def test_integrator_reports_the_compensated_total(self):
        increments = iter([1.0] + [1e-16] * 1999)
        with mock.patch.object(driver, "_signed_angle_increment",
                               side_effect=lambda *args: next(increments)):
            result = integrate(max_steps=2000, stop_after_one_orbit=False)
        exact = 1.0 + 1999e-16
        self.assertEqual(result.accepted_steps, 2000)
        self.assertGreater(result.total_angle_rad, 1.0 + 1e-13)
        self.assertAlmostEqual(result.total_angle_rad, exact, delta=3e-16)

    def test_stopping_rule_uses_the_compensated_total(self):
        # The plain sum of these increments never reaches 2*pi: each 1e-16 is
        # smaller than half the spacing of doubles near 2*pi and is lost.
        first = 2 * math.pi - 1e-13
        increments = iter([first] + [1e-16] * 4999)
        with mock.patch.object(driver, "_signed_angle_increment",
                               side_effect=lambda *args: next(increments)):
            result = integrate(max_steps=5000, stop_after_one_orbit=True)
        self.assertTrue(result.completed_orbit)
        self.assertTrue(900 < result.accepted_steps < 1200, result.accepted_steps)
        self.assertGreaterEqual(abs(result.total_angle_rad), 2 * math.pi)

    def test_an_accumulated_angle_of_exactly_two_pi_stops_the_run(self):
        # The Help's stopping rule is |Theta| >= 2 pi.
        for sign in (1.0, -1.0):
            increments = iter([sign * 2 * math.pi] + [1e-3] * 9)
            with self.subTest(sign=sign), mock.patch.object(
                    driver, "_signed_angle_increment", side_effect=lambda *args: next(increments)):
                result = integrate(max_steps=10, stop_after_one_orbit=True)
            self.assertTrue(result.completed_orbit)
            self.assertEqual(result.accepted_steps, 1)

    def test_default_orbit_angle_is_one_turn(self):
        result = integrate()
        self.assertTrue(result.completed_orbit)
        self.assertAlmostEqual(result.total_angle_rad / (2 * math.pi), 1.0, delta=1e-4)


class TestHelpOutputOfDefaults(unittest.TestCase):
    def test_help_does_not_print_true_beside_the_stop_off_switch(self):
        with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as raised:
            entry.parse_args(["--help"])
        self.assertEqual(raised.exception.code, 0)
        text = out.getvalue()
        start = text.index("  --stop_after_one_orbit")
        middle = text.index("  --no-stop_after_one_orbit")
        end = text.index("  --output_type")
        self.assertIn("(default: True)", text[start:middle])
        self.assertNotIn("default", text[middle:end])
        flat = " ".join(text.split())
        self.assertIn("(default: 2000.0)", flat)
        self.assertIn("(default: 10000)", flat)

    def test_stop_switches_still_set_the_flag(self):
        self.assertTrue(entry.parse_args([]).stop_after_one_orbit)
        self.assertTrue(entry.parse_args(["--stop_after_one_orbit"]).stop_after_one_orbit)
        self.assertFalse(entry.parse_args(["--no-stop_after_one_orbit"]).stop_after_one_orbit)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            entry.parse_args(["--stop_after_one_orbit", "--no-stop_after_one_orbit"])


class TestOptionValueNormalisation(unittest.TestCase):
    def test_two_negative_values_in_a_row_are_both_bound(self):
        args = entry.parse_args(["--xInitA", "-5e9", "--xInitB", "-4e9", "--uInitA", "-1e4"])
        self.assertEqual((args.xInitA, args.xInitB, args.uInitA), (-5e9, -4e9, -1e4))

    def test_equals_form_and_plain_negative_integers(self):
        args = entry.parse_args(["--xInitB=-5e9", "--vInitA", "-3"])
        self.assertEqual((args.xInitB, args.vInitA), (-5e9, -3.0))

    def test_a_dash_word_is_not_swallowed_as_a_value(self):
        for argv in (["--xInitA", "-abc"], ["--xInitA", "--dt", "5"], ["--output_type", "-1"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as raised:
                entry.parse_args(argv)
            self.assertEqual(raised.exception.code, 2)

    def test_a_flag_that_takes_no_value_is_not_given_one(self):
        with redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            entry.parse_args(["--stop_after_one_orbit", "-5"])
        self.assertIn("unrecognized arguments: -5", err.getvalue())

    def test_negative_value_for_a_positive_option_is_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            entry.parse_args(["--MA", "-1e30"])
        self.assertEqual(raised.exception.code, 2)


class TestInterpolate(unittest.TestCase):
    def test_before_between_at_and_after_the_saved_history(self):
        times, values = [0.0, 1.0, 3.0], [10.0, 20.0, 40.0]
        self.assertEqual(entry.interpolate(times, values, -5.0), 10.0)
        self.assertEqual(entry.interpolate(times, values, 0.0), 10.0)
        self.assertEqual(entry.interpolate(times, values, 0.5), 15.0)
        self.assertEqual(entry.interpolate(times, values, 2.0), 30.0)
        self.assertEqual(entry.interpolate(times, values, 3.0), 40.0)
        self.assertEqual(entry.interpolate(times, values, 9.0), 40.0)

    def test_a_saved_time_returns_the_saved_value_exactly(self):
        # Interpolating from the previous point would give 0.0 here.
        self.assertEqual(entry.interpolate([0.0, 1.0], [1e16, 1.0], 1.0), 1.0)


class TestEnergyDepartureSign(unittest.TestCase):
    def test_departure_is_relative_to_the_magnitude_of_a_negative_initial_energy(self):
        result = integrate(max_steps=3)
        for final, expected in ((-0.5, "0.5"), (-1.5, "-0.5")):
            with self.subTest(final=final):
                result.E[0], result.E[-1] = -1.0, final
                with redirect_stdout(io.StringIO()) as out:
                    entry.print_summary(result, DEFAULTS)
                self.assertEqual(printed_rows(out.getvalue())[1.0][-1], expected)


class TestCorrectorAndStepFloor(unittest.TestCase):
    @staticmethod
    def run_one_step(changes):
        values = iter(changes)
        calls = []
        real = driver.accelerations

        def counting(*args):
            calls.append(args)
            return real(*args)

        with mock.patch.object(driver, "_vector_relative_change",
                               side_effect=lambda *args: next(values)), \
             mock.patch.object(driver, "accelerations", side_effect=counting):
            result = integrate(max_steps=1, stop_after_one_orbit=False)
        return result, len(calls)

    def test_a_converged_first_pass_needs_no_acceleration_recomputation(self):
        # initial check + step start + predicted position
        _, calls = self.run_one_step([0.0, 0.0, 0.0])
        self.assertEqual(calls, 3)

    def test_either_body_failing_the_test_forces_another_pass(self):
        for pair in ((0.0, 1.0), (1.0, 0.0)):
            with self.subTest(pair=pair):
                result, calls = self.run_one_step([0.0, *pair, 0.0, 0.0])
                self.assertEqual(result.accepted_steps, 1)
                self.assertEqual(result.times[1], DEFAULTS["dt"])
                self.assertEqual(calls, 4)

    def test_a_corrector_change_equal_to_eps2_is_not_yet_converged(self):
        # The Help's convergence test is a strict inequality, change < eps2.
        eps2 = DEFAULTS["eps2"]
        result, calls = self.run_one_step([0.0, eps2, 0.0, 0.0, 0.0])
        self.assertEqual(result.accepted_steps, 1)
        self.assertEqual(calls, 4)
        result, calls = self.run_one_step([0.0, math.nextafter(eps2, 0.0), 0.0])
        self.assertEqual(calls, 3)

    def test_a_predictor_change_equal_to_eps1_is_accepted(self):
        # The Help rejects a step only when the change exceeds eps1.
        eps1 = DEFAULTS["eps1"]
        result, calls = self.run_one_step([eps1, 0.0, 0.0])
        self.assertEqual(result.times[1], DEFAULTS["dt"])
        self.assertEqual(calls, 3)
        with self.assertRaises(StopIteration):
            # one step is halved and retried, which needs a second predictor value
            self.run_one_step([math.nextafter(eps1, 1.0), 0.0, 0.0])

    def test_the_predictor_test_on_body_a_speaks_for_body_b(self):
        # Both accelerations point along the line joining the bodies and differ only by
        # a mass factor, so their relative changes agree; testing body A tests both.
        for MA, MB in ((2e30, 2e30), (2e30, 1e30), (1e31, 1e30), (1e28, 3e31)):
            before = physics.accelerations(MA, MB, 3e10, 1e10, -2e10, -4e10)
            after = physics.accelerations(MA, MB, 3.1e10, 0.9e10, -2.2e10, -4.1e10)
            change_a = driver._vector_relative_change(before[0], before[1], after[0], after[1])
            change_b = driver._vector_relative_change(before[2], before[3], after[2], after[3])
            with self.subTest(MA=MA, MB=MB):
                self.assertGreater(change_a, 0.01)
                self.assertAlmostEqual(change_a / change_b, 1.0, delta=1e-12)

    def test_a_strict_corrector_conserves_energy_over_one_orbit(self):
        result = integrate(eps2=1e-8)
        self.assertLess(abs(result.E[-1] - result.E[0]) / abs(result.E[0]), 1e-9)

    def test_loose_corrector_makes_two_acceleration_calls_per_step(self):
        real = driver.accelerations
        with mock.patch.object(driver, "accelerations", side_effect=real) as spy:
            result = integrate(eps2=0.5)
        self.assertEqual(spy.call_count, 1 + 2 * result.accepted_steps)

    @staticmethod
    def force_rejections(count):
        state = {"n": 0}

        def rejecting(*args):
            state["n"] += 1
            return 1.0 if state["n"] <= count else 0.0

        return mock.patch.object(driver, "_vector_relative_change", side_effect=rejecting)

    def test_a_halved_step_exactly_at_the_floor_is_allowed(self):
        self.assertEqual(driver._halve_timestep(2.0, 1.0), 1.0)
        with self.assertRaises(RuntimeError):
            driver._halve_timestep(2.0, math.nextafter(1.0, 2.0))

    def test_thirty_nine_halvings_are_allowed(self):
        with self.force_rejections(39):
            result = integrate(max_steps=1, stop_after_one_orbit=False)
        self.assertEqual(result.accepted_steps, 1)
        self.assertAlmostEqual(result.times[1] / DEFAULTS["dt"], 2.0 ** -39, delta=1e-24)

    def test_the_fortieth_halving_hits_the_safety_floor(self):
        with self.force_rejections(40), self.assertRaisesRegex(
                RuntimeError, "numerical safety limit"):
            integrate(max_steps=1, stop_after_one_orbit=False)


class TestUnequalMassIntegration(unittest.TestCase):
    """Every other integration test uses equal masses, which hide any mix-up of A and B."""

    CASES = {
        "two to one": dict(MA=2e30, MB=1e30, xInitA=3.0666666667e10, xInitB=-6.1333333333e10,
                           uInitA=15550.5265, uInitB=-31101.053),
        "ten to one": dict(MA=1e31, MB=1e30, xInitA=4.6e10 / 11, xInitB=-4.6e10 * 10 / 11,
                           uInitA=10000.0, uInitB=-100000.0),
    }

    def test_stored_energies_are_those_of_the_stored_states(self):
        for label, case in self.CASES.items():
            result = integrate(**case)
            for i in range(0, len(result.times), 50):
                expected = physics.energies(
                    case["MA"], case["MB"], result.xA[i], result.yA[i], result.vA[i], result.uA[i],
                    result.xB[i], result.yB[i], result.vB[i], result.uB[i])
                with self.subTest(case=label, index=i):
                    self.assertEqual((result.U[i], result.K[i], result.E[i]), expected)

    def test_conserved_quantities_stay_conserved(self):
        for label, case in self.CASES.items():
            result = integrate(**case)
            momentum = [case["MA"] * result.vA[i] + case["MB"] * result.vB[i] for i in range(len(result.times))]
            momentum_y = [case["MA"] * result.uA[i] + case["MB"] * result.uB[i] for i in range(len(result.times))]
            scale = case["MA"] * max(abs(u) for u in result.uA)
            angular = [case["MA"] * (result.xA[i] * result.uA[i] - result.yA[i] * result.vA[i])
                       + case["MB"] * (result.xB[i] * result.uB[i] - result.yB[i] * result.vB[i])
                       for i in range(len(result.times))]
            with self.subTest(case=label):
                self.assertLess(max(abs(p) for p in momentum) / scale, 1e-9)
                self.assertLess(max(abs(p - momentum_y[0]) for p in momentum_y) / scale, 1e-9)
                # the integrator's own drift is about 1e-3; a mix-up of A and B is of order one
                self.assertLess(max(abs(x - angular[0]) for x in angular) / abs(angular[0]), 5e-3)
                self.assertLess(max(abs(e - result.E[0]) for e in result.E) / abs(result.E[0]), 1e-3)
                self.assertTrue(result.completed_orbit)

    def test_the_heavier_body_makes_the_smaller_orbit(self):
        for label, case in self.CASES.items():
            result = integrate(**case)
            reach_a = max(math.hypot(x, y) for x, y in zip(result.xA, result.yA))
            reach_b = max(math.hypot(x, y) for x, y in zip(result.xB, result.yB))
            with self.subTest(case=label):
                self.assertAlmostEqual(reach_b / reach_a, case["MA"] / case["MB"], delta=0.02 * case["MA"] / case["MB"])

    def test_each_body_accelerates_toward_the_other_in_proportion_to_the_others_mass(self):
        axA, ayA, axB, ayB = physics.accelerations(2e30, 1e30, 3e10, 0.0, -6e10, 0.0)
        self.assertLess(axA, 0)
        self.assertGreater(axB, 0)
        self.assertAlmostEqual(axA / axB, -0.5, places=12)
        self.assertAlmostEqual(-axA, physics.G * 1e30 / 9e10 ** 2, delta=1e-12 * physics.G * 1e30 / 9e10 ** 2)
        self.assertEqual((ayA, ayB), (0.0, 0.0))


class TestCommandLineWiring(unittest.TestCase):
    """What main.py passes on, in what order, and what it rejects."""

    def test_zero_is_not_a_positive_integer_or_a_positive_number(self):
        for argv in (["--max_steps", "0"], ["--max_steps", "-3"], ["--dt", "0"], ["--MB", "0"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as raised:
                entry.parse_args(argv)
            self.assertEqual(raised.exception.code, 2)

    def test_the_exposed_defaults_are_the_parsed_defaults(self):
        parsed = vars(entry.parse_args([]))
        for key, value in entry.DEFAULTS.items():
            with self.subTest(key=key):
                self.assertEqual(parsed[key], value)
                self.assertIs(type(parsed[key]), type(value))
        self.assertEqual(entry.DEFAULTS, DEFAULTS)
        self.assertIs(entry.DEFAULTS["stop_after_one_orbit"], True)

    def test_each_selector_draws_the_plot_it_names(self):
        titles = {
            "orbits": "Binary orbits", "velocity_space": "Velocity space",
            "position_vs_time_body_a": "Position vs time, body A",
            "position_vs_time_body_b": "Position vs time, body B",
            "velocity_vs_time_body_a": "Velocity vs time, body A",
            "velocity_vs_time_body_b": "Velocity vs time, body B",
            "energy_vs_time": "Energy vs time",
        }
        self.assertEqual(tuple(titles), tuple(entry.OUTPUT_TYPES))
        for selector, title in titles.items():
            with self.subTest(selector=selector), mock.patch.object(plotting.plt, "show"), \
                 redirect_stdout(io.StringIO()):
                entry.main(["--max_steps", "3", "--output_type", selector])
                self.assertEqual(plotting.plt.gcf().axes[0].get_title(), title)
                plotting.plt.close("all")

    def test_the_summary_is_printed_before_the_plot_window_opens(self):
        # plt.show() blocks until the window is closed, so the summary must already be out.
        seen = {}
        with redirect_stdout(io.StringIO()) as out, mock.patch.object(
                entry, "plot_binary", side_effect=lambda *a: seen.update(text=out.getvalue())):
            entry.main(["--max_steps", "3"])
        self.assertIn("Energy at fractions of total run days", seen["text"])

    def test_the_selected_selector_reaches_the_plot_call(self):
        with redirect_stdout(io.StringIO()), mock.patch.object(entry, "plot_binary") as plot:
            entry.main(["--max_steps", "3", "--output_type", "energy_vs_time"])
        self.assertEqual(plot.call_args.args[1], "energy vs time")

    def test_a_clockwise_orbit_reports_a_positive_number_of_revolutions(self):
        clockwise = printed_fields(printed_run(["--uInitA", "-13000", "--uInitB", "13000"]))
        default = printed_fields(printed_run([]))
        self.assertEqual(clockwise["Total revolutions"], default["Total revolutions"])
        self.assertEqual(clockwise["Termination"], "one relative revolution")
        result = integrate(uInitA=-13000.0, uInitB=13000.0)
        self.assertLess(result.total_angle_rad, 0)


class TestOrbitalElementsAnalytic(unittest.TestCase):
    """Elements against independent scalar formulas, in rotated and oblique states."""

    CASES = {
        "default": {},
        "radial velocity component": {"vInitA": 3000.0, "vInitB": -3000.0},
        "past periapsis": {"vInitA": -5000.0, "vInitB": 5000.0, "uInitA": 20000.0,
                           "uInitB": -20000.0},
        "unequal masses": {"MB": 1e30, "uInitA": 9000.0, "uInitB": -18000.0},
    }

    def check(self, parameters):
        expected = scalar_kepler(parameters)
        elements = physics.orbital_elements(*(parameters[key] for key in BODY_KEYS))
        self.assertEqual(elements.kind, "elliptic")
        for name, value in (("eccentricity", expected["e"]),
                            ("relative_semimajor", expected["semi"]),
                            ("period", expected["period"]),
                            ("relative_periapsis", expected["peri"]),
                            ("relative_apoapsis", expected["apo"]),
                            ("relative_speed_periapsis", expected["v_peri"]),
                            ("relative_speed_apoapsis", expected["v_apo"])):
            with self.subTest(field=name):
                self.assertAlmostEqual(getattr(elements, name) / value, 1.0, places=11)

    def test_elements_match_scalar_formulas_in_the_given_frame(self):
        for label, changes in self.CASES.items():
            with self.subTest(case=label):
                parameters = dict(DEFAULTS)
                parameters.update(changes)
                self.check(parameters)

    def test_elements_are_unchanged_by_rotating_the_whole_state(self):
        for label, changes in self.CASES.items():
            for angle in (math.radians(37.0), math.radians(-121.0)):
                with self.subTest(case=label, angle=angle):
                    self.check(rotate_case(angle, **changes))

    def test_default_orbit_numbers(self):
        elements = elements_for()
        self.assertAlmostEqual(elements.eccentricity, 0.76705, places=5)
        self.assertAlmostEqual(elements.relative_periapsis / 1.2128542e10, 1.0, delta=1e-7)
        self.assertAlmostEqual(elements.relative_apoapsis, 9.2e10, delta=1.0)
        self.assertAlmostEqual(elements.period / 86400.0, 52.874, places=3)
        self.assertAlmostEqual(elements.relative_speed_apoapsis, 26000.0, delta=1e-6)

    def test_parabolic_band_is_one_part_in_1e12_of_the_escape_speed(self):
        escape = 0.5 * math.sqrt(2 * physics.G * 4e30 / 9.2e10)
        for factor, kind in ((1 + 4e-13, "parabolic"), (1 - 4e-13, "parabolic"),
                             (1 + 2e-12, "hyperbolic"), (1 - 2e-12, "elliptic")):
            with self.subTest(factor=factor):
                elements = elements_for(uInitA=escape * factor, uInitB=-escape * factor)
                self.assertEqual(elements.kind, kind)
        parabola = elements_for(uInitA=escape, uInitB=-escape)
        self.assertEqual(parabola.eccentricity, 1.0)
        for name in ("relative_semimajor", "period", "relative_apoapsis",
                     "relative_speed_apoapsis"):
            self.assertIsNone(getattr(parabola, name), name)
        self.assertAlmostEqual(parabola.relative_periapsis, 9.2e10, delta=1.0)
        self.assertAlmostEqual(parabola.relative_speed_periapsis, 2 * escape, delta=1e-6)

    def test_radial_start_has_no_apsides(self):
        elements = elements_for(vInitA=60000.0, vInitB=-60000.0, uInitA=0.0, uInitB=0.0)
        self.assertEqual(elements.kind, "radial")
        for name in ("relative_semimajor", "period", "relative_periapsis",
                     "relative_apoapsis", "relative_speed_periapsis",
                     "relative_speed_apoapsis"):
            self.assertIsNone(getattr(elements, name), name)


class TestElementsWithoutFalseRangeErrors(unittest.TestCase):
    """orbital_elements() and the summary's mass fractions must not overflow in an
    intermediate step when the quantity asked for is representable.  The oracle is
    exact decimal arithmetic on the same formulas."""

    @staticmethod
    def _decimal_context():
        import decimal
        context = decimal.getcontext()
        context.prec = 60
        context.Emax = 100000
        context.Emin = -100000
        return decimal

    @staticmethod
    def _circular(mass_a, mass_b, separation):
        """Relative state of a circular orbit, split about the centre of mass."""
        decimal = TestElementsWithoutFalseRangeErrors._decimal_context()
        D = decimal.Decimal
        mu = D(physics.G) * (D(mass_a) + D(mass_b))
        relative_speed = (mu / D(separation)).sqrt()
        period = 2 * D(math.pi) * (D(separation) ** 3 / mu).sqrt()
        return mu, relative_speed, period

    def test_a_circular_period_whose_cube_of_the_radius_overflows(self):
        mass, radius = 5e199, 1e110
        with self.assertRaises(OverflowError):
            radius ** 3
        mu, relative_speed, period = self._circular(mass, mass, radius)
        speed = float(relative_speed)
        elements = physics.orbital_elements(mass, mass, radius / 2, 0.0, 0.0, speed / 2,
                                            -radius / 2, 0.0, 0.0, -speed / 2)
        self.assertEqual(elements.kind, "elliptic")
        self.assertAlmostEqual(elements.period / float(period), 1.0, delta=1e-12)
        self.assertAlmostEqual(elements.relative_semimajor / radius, 1.0, delta=1e-12)
        self.assertAlmostEqual(elements.relative_periapsis / radius, 1.0, delta=1e-9)
        self.assertAlmostEqual(elements.relative_apoapsis / radius, 1.0, delta=1e-9)

    def test_a_mass_sum_that_overflows_although_g_times_it_is_representable(self):
        mass, radius = 1e308, 1e100
        self.assertTrue(math.isinf(mass + mass))
        mu, relative_speed, period = self._circular(mass, mass, radius)
        speed = float(relative_speed)
        elements = physics.orbital_elements(mass, mass, radius / 2, 0.0, 0.0, speed / 2,
                                            -radius / 2, 0.0, 0.0, -speed / 2)
        self.assertEqual(elements.kind, "elliptic")
        self.assertAlmostEqual(elements.period / float(period), 1.0, delta=1e-12)
        self.assertAlmostEqual(elements.relative_semimajor / radius, 1.0, delta=1e-12)

    def test_periapsis_when_the_square_of_the_angular_momentum_overflows(self):
        decimal = self._decimal_context()
        D = decimal.Decimal
        mass, radius, speed = 5e199, 1e110, 1e50
        angular_momentum = radius * speed
        self.assertTrue(math.isinf(angular_momentum * angular_momentum))
        elements = physics.orbital_elements(mass, mass, radius / 2, 0.0, 0.0, speed / 2,
                                            -radius / 2, 0.0, 0.0, -speed / 2)
        mu = D(physics.G) * (D(mass) + D(mass))
        speed_squared = D(speed) ** 2
        e_x = (speed_squared - mu / D(radius)) * D(radius) / mu
        expected_periapsis = (D(radius) * D(speed)) ** 2 / (mu * (1 + abs(e_x)))
        self.assertEqual(elements.kind, "hyperbolic")
        self.assertAlmostEqual(elements.relative_periapsis / float(expected_periapsis), 1.0, delta=1e-9)
        self.assertAlmostEqual(elements.eccentricity / float(abs(e_x)), 1.0, delta=1e-9)

    def test_a_period_that_really_is_out_of_range_is_a_value_error(self):
        mass, radius = 1e-40, 1e200
        mu, relative_speed, period = self._circular(mass, mass, radius)
        self.assertGreater(period, self._decimal_context().Decimal("1e308"))
        speed = float(relative_speed)
        with self.assertRaises(ValueError) as caught:
            physics.orbital_elements(mass, mass, radius / 2, 0.0, 0.0, speed / 2,
                                     -radius / 2, 0.0, 0.0, -speed / 2)
        self.assertIn("numerical range", str(caught.exception))

    def test_ordinary_results_agree_with_the_textbook_forms(self):
        elements = physics.orbital_elements(2e30, 2e30, 4.6e10, 0.0, 0.0, 13000.0,
                                            -4.6e10, 0.0, 0.0, -13000.0)
        mu = physics.G * 4e30
        semi = elements.relative_semimajor
        angular_momentum = 9.2e10 * 26000.0
        self.assertAlmostEqual(elements.period / (2 * math.pi * math.sqrt(semi ** 3 / mu)),
                               1.0, delta=1e-14)
        self.assertAlmostEqual(elements.relative_periapsis
                               / (angular_momentum ** 2 / (mu * (1 + elements.eccentricity))),
                               1.0, delta=1e-14)

    def test_mass_fractions_with_and_without_an_overflowing_sum(self):
        from fractions import Fraction
        cases = ((2e30, 1e30), (1e30, 2e30), (2e30, 2e30), (1.7e308, 1.7e308),
                 (1.7e308, 0.85e308), (0.85e308, 1.7e308), (1.7e308, 1e300), (1e-300, 1.7e308))
        for mass_a, mass_b in cases:
            with self.subTest(MA=mass_a, MB=mass_b):
                total = Fraction(mass_a) + Fraction(mass_b)
                fraction_a, fraction_b = entry._mass_fractions(mass_a, mass_b)
                self.assertAlmostEqual(fraction_a, float(Fraction(mass_b) / total), delta=1e-15)
                self.assertAlmostEqual(fraction_b, float(Fraction(mass_a) / total), delta=1e-15)
                self.assertAlmostEqual(fraction_a + fraction_b, 1.0, delta=2e-16)

class TestPrintedSummaryLines(unittest.TestCase):
    def test_default_summary_lines(self):
        printed = printed_run([])
        fields = printed_fields(printed)
        result = integrate()
        self.assertEqual(fields["Accepted steps"], str(result.accepted_steps))
        self.assertEqual(fields["Termination"], "one relative revolution")
        self.assertEqual(fields["Total revolutions"],
                         f"{abs(result.total_angle_rad) / (2 * math.pi):.5g}")
        self.assertEqual(fields["Total time"], f"{result.times[-1] / 86400:.5g} days")
        separations = [math.hypot(a - b, c - d) for a, b, c, d
                       in zip(result.xA, result.xB, result.yA, result.yB)]
        self.assertEqual(fields["Minimum separation (sampled)"], f"{min(separations):.5g} m")
        self.assertEqual(fields["Maximum separation (sampled)"], f"{max(separations):.5g} m")
        self.assertEqual(fields["Initial Keplerian orbit"], "elliptic; eccentricity: 0.76705")
        self.assertTrue(printed.startswith(f"Binary {physics.MODEL_VERSION} (build {physics.BUILD_ID})\n"))

    def test_safety_ceiling_is_reported_as_max_steps(self):
        fields = printed_fields(printed_run(["--max_steps", "10"]))
        self.assertEqual(fields["Termination"], "max_steps")
        self.assertEqual(fields["Accepted steps"], "10")

    def test_body_blocks_split_the_relative_orbit_by_mass_fraction(self):
        printed = printed_run(["--MA", "2e30", "--MB", "1e30", "--xInitA", "3.0666666667e10",
                               "--xInitB", "-6.1333333333e10", "--uInitA", "15550.5265",
                               "--uInitB", "-31101.053", "--max_steps", "5"])
        block_a, block_b = printed_body_block(printed, "A"), printed_body_block(printed, "B")
        semi_a, semi_b = (printed_number(block[0].split("semi-major axis: ")[1]) for block in (block_a, block_b))
        elements = elements_for(MA=2e30, MB=1e30, xInitA=3.0666666667e10, xInitB=-6.1333333333e10,
                                uInitA=15550.5265, uInitB=-31101.053)
        self.assertAlmostEqual(semi_a / (elements.relative_semimajor * 1e30 / 3e30), 1.0, places=4)
        self.assertAlmostEqual(semi_b / (elements.relative_semimajor * 2e30 / 3e30), 1.0, places=4)
        self.assertAlmostEqual(semi_b / semi_a, 2.0, places=3)
        self.assertEqual(block_a[0].split("period: ")[1], block_b[0].split("period: ")[1])

    def test_parabolic_start_prints_undefined_elements(self):
        escape = 38091.13784870039
        printed = printed_run(["--uInitA", repr(escape), "--uInitB", repr(-escape),
                               "--no-stop_after_one_orbit", "--max_steps", "20"])
        self.assertEqual(printed_fields(printed)["Initial Keplerian orbit"], "parabolic; eccentricity: 1")
        block = printed_body_block(printed, "A")
        self.assertIn("semi-major axis: undefined", block[0])
        self.assertIn("period: undefined", block[0])
        self.assertIn("Periapsis: 4.6e+10 m; apoapsis: undefined", block[1])
        self.assertIn("speed at apoapsis: undefined", block[2])

    def test_radial_start_line_is_true_for_outward_and_inward_motion(self):
        message = "Radial trajectory (zero angular momentum): apsides and period are undefined."
        outward = printed_run(["--vInitA", "60000", "--vInitB", "-60000", "--uInitA", "0",
                               "--uInitB", "0", "--max_steps", "20"])
        # An inward start is a collision course, so only the printing is exercised.
        changes = dict(vInitA=-60000.0, vInitB=60000.0, uInitA=0.0, uInitB=0.0)
        result = integrate(max_steps=2, stop_after_one_orbit=False, **changes)
        with redirect_stdout(io.StringIO()) as out:
            entry.print_summary(result, dict(DEFAULTS, **changes))
        for label, printed in (("outward", outward), ("inward", out.getvalue())):
            with self.subTest(direction=label):
                self.assertEqual(printed_fields(printed)["Initial Keplerian orbit"],
                                 "radial; eccentricity: 1")
                self.assertIn(message, printed)
                self.assertNotIn("collision", printed)


# ---------------------------------------------------------------------------
# Help layout awareness, structure, commands and printed excerpt.
# ---------------------------------------------------------------------------

HELP_HTML = HELP_FILE.read_text(encoding="utf-8")


def help_layout(html):
    """Classify a Help page as ``"beats"``, ``"classic"`` or ``"unrecognised"``."""
    ids = set(re.findall(r'<section id="([^"]+)"', html))
    beat_ids = {name for name in ids if re.fullmatch(r"beat\d+", name)}
    if "beats" in ids and beat_ids:
        return "beats"
    if "two-body" in ids and "predictor-corrector" in ids and not beat_ids and "beats" not in ids:
        return "classic"
    return "unrecognised"


HELP_LAYOUT = help_layout(HELP_HTML)
needs_beats = unittest.skipUnless(
    HELP_LAYOUT == "beats", "the Beats-only assertions apply to the Beats layout"
)


def html_text(fragment):
    """Visible text of an HTML fragment with entities decoded and spaces collapsed."""
    text = re.sub(r"<[^>]+>", " ", fragment)
    return " ".join(html_module.unescape(text).split())


def section_html(html, section_id):
    match = re.search(rf'<section id="{re.escape(section_id)}">(.*?)</section>', html, re.DOTALL)
    if match is None:
        raise AssertionError(f"section {section_id!r} not found in the Help file")
    return match.group(1)


def beat_text(number):
    return html_text(section_html(HELP_HTML, f"beat{number}"))


class HelpStructure:
    """Ids, links, sidebar links and tag balance of a Help page."""

    VOID = {"br", "hr", "img", "meta", "link", "input"}

    def __init__(self, html):
        outer = self
        self.ids, self.hrefs, self.sidebar_hrefs, self.problems = [], [], [], []
        self._stack = []
        self._in_sidebar = 0

        class Parser(HTMLParser):
            def handle_starttag(self, tag, attrs):
                attributes = dict(attrs)
                if "id" in attributes:
                    outer.ids.append(attributes["id"])
                if tag == "a" and attributes.get("href", "").startswith("#"):
                    outer.hrefs.append(attributes["href"][1:])
                    if outer._in_sidebar:
                        outer.sidebar_hrefs.append(attributes["href"][1:])
                if tag not in outer.VOID:
                    outer._stack.append(tag)
                    if tag == "nav" and attributes.get("id") == "sidebar":
                        outer._in_sidebar = len(outer._stack)

            def handle_startendtag(self, tag, attrs):
                if "id" in dict(attrs):
                    outer.ids.append(dict(attrs)["id"])

            def handle_endtag(self, tag):
                if tag in outer.VOID:
                    return
                if not outer._stack or outer._stack[-1] != tag:
                    outer.problems.append(f"unexpected </{tag}> (open: {outer._stack[-3:]})")
                    if tag in outer._stack:
                        while outer._stack and outer._stack.pop() != tag:
                            pass
                    return
                if outer._in_sidebar == len(outer._stack):
                    outer._in_sidebar = 0
                outer._stack.pop()

        parser = Parser(convert_charrefs=True)
        parser.feed(html)
        parser.close()
        if self._stack:
            self.problems.append(f"unclosed tags at end of file: {self._stack}")


def documented_commands(html):
    """Every ``python main.py ...`` command shown in a code block, as argv lists."""
    commands = []
    for block in re.findall(r"<pre[^>]*>(.*?)</pre>", html, re.DOTALL):
        text = html_module.unescape(re.sub(r"<[^>]+>", "", block))
        text = re.sub(r"\\\n\s*", " ", text)
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("python main.py"):
                argv = shlex.split(line)
                assert argv[:2] == ["python", "main.py"], line
                if argv[2:] and argv[2] in ("--help", "-h", "--version"):
                    continue
                commands.append(argv[2:])
    return commands


class HelpLayoutRecogniserTests(unittest.TestCase):
    def test_recogniser_classifies_synthetic_pages(self):
        beats = '<section id="overview"></section><section id="beats"></section><section id="beat0"></section>'
        classic = ('<section id="overview"></section><section id="two-body"></section>'
                   '<section id="predictor-corrector"></section>')
        self.assertEqual(help_layout(beats), "beats")
        self.assertEqual(help_layout(classic), "classic")
        self.assertEqual(help_layout(classic + '<section id="beat0"></section>'), "unrecognised")
        self.assertEqual(help_layout('<section id="overview"></section>'), "unrecognised")
        self.assertEqual(help_layout(""), "unrecognised")

    def test_shipped_help_has_a_recognised_layout(self):
        self.assertIn(HELP_LAYOUT, ("beats", "classic"))


class HelpStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structure = HelpStructure(HELP_HTML)

    def test_tags_are_balanced(self):
        self.assertEqual(self.structure.problems, [])

    def test_ids_are_unique(self):
        duplicates = sorted({name for name in self.structure.ids if self.structure.ids.count(name) > 1})
        self.assertEqual(duplicates, [])

    def test_every_in_page_link_resolves(self):
        unresolved = sorted({name for name in self.structure.hrefs if name not in self.structure.ids})
        self.assertEqual(unresolved, [])

    def test_sidebar_lists_every_named_section_in_document_order(self):
        section_ids = re.findall(r'<section id="([^"]+)"', HELP_HTML)
        self.assertEqual(self.structure.sidebar_hrefs, section_ids)

    def test_mathjax_delimiters_open_and_close_in_matching_pairs(self):
        body = re.sub(r"<script.*?</script>", "", HELP_HTML, flags=re.DOTALL)
        body = re.sub(r"\\\\\[[^\]]*\]", "", body)          # the line-break spacing \\[2pt] is not a delimiter
        pair = {"\\(": "\\)", "\\[": "\\]"}
        open_delimiter = None
        for token in re.findall(r"\\[()\[\]]", body):
            if open_delimiter is None:
                self.assertIn(token, pair)
                open_delimiter = token
            else:
                self.assertEqual(token, pair[open_delimiter])
                open_delimiter = None
        self.assertIsNone(open_delimiter)

    def test_page_loads_mathjax_from_the_cdn_and_no_other_external_script(self):
        sources = re.findall(r'<script[^>]*\ssrc="([^"]+)"', HELP_HTML)
        self.assertEqual(sources, ["https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"])

    def test_experiment_headings_have_anchors_in_the_beats_layout(self):
        titles = re.findall(r"<h3[^>]*>(\d+) · ", section_html(HELP_HTML, "experiments"))
        self.assertEqual(titles, [str(n) for n in range(1, 9)])
        if HELP_LAYOUT == "beats":
            anchors = [name for name in self.structure.ids if re.fullmatch(r"exp\d+", name)]
            self.assertEqual(anchors, [f"exp{n}" for n in range(1, 9)])

    def test_equation_references_name_defined_equations(self):
        text = html_text(HELP_HTML)
        cited = set()
        for match in re.finditer(r"\b(?:Eqs?\.|Equations?)\s+((?:\(\d+\)(?:\s*(?:,|and|to|–)\s*)?)+)", text):
            cited |= {int(n) for n in re.findall(r"\((\d+)\)", match.group(1))}
        if HELP_LAYOUT == "beats":
            self.assertTrue(cited)          # the classic layout labels its equations differently
        self.assertTrue(cited <= set(range(1, 11)), sorted(cited))


class HelpCommandTests(unittest.TestCase):
    """Every command printed in the Help must parse and run."""

    @classmethod
    def setUpClass(cls):
        cls.commands = documented_commands(HELP_HTML)

    def test_help_shows_a_substantial_number_of_commands(self):
        self.assertGreaterEqual(len(self.commands), 8 if HELP_LAYOUT == "classic" else 30)

    def test_every_documented_command_parses(self):
        for argv in self.commands:
            with self.subTest(argv=argv):
                entry.parse_args(argv)

    def test_every_distinct_documented_command_runs_and_prints_its_summary(self):
        seen = []
        for argv in self.commands:
            if argv in seen:
                continue
            seen.append(argv)
            with self.subTest(argv=argv):
                printed = cached_printed(tuple(argv))
                self.assertIn("Energy at fractions of total run days", printed)
                self.assertEqual(len(printed_rows(printed)), 11)
        self.assertGreaterEqual(len(seen), 8 if HELP_LAYOUT == "classic" else 25)

    def test_every_output_type_command_selects_a_real_plot(self):
        for argv in self.commands:
            if "--output_type" in argv:
                self.assertIn(argv[argv.index("--output_type") + 1], entry.OUTPUT_TYPES)


@needs_beats
class PrintedSummaryExcerptTests(unittest.TestCase):
    def test_excerpt_is_the_real_default_output_after_the_version_line(self):
        block = re.search(r"<pre><code>(Accepted steps.*?)</code></pre>",
                          section_html(HELP_HTML, "summary"), re.DOTALL)
        self.assertIsNotNone(block)
        shown = html_module.unescape(block.group(1)).rstrip("\n").splitlines()
        actual = printed_run([]).splitlines()
        self.assertTrue(actual[0].startswith("Binary "))
        self.assertEqual(shown, actual[1:])

    def test_summary_section_describes_every_printed_line_kind(self):
        text = html_text(section_html(HELP_HTML, "summary"))
        for phrase in ("Accepted steps", "Termination", "one relative revolution", "max_steps",
                       "Total revolutions", "Total time", "Minimum separation (sampled)",
                       "Maximum separation (sampled)", "Initial Keplerian orbit", "elliptic",
                       "parabolic", "hyperbolic", "radial", "Body A about centre of mass",
                       "Body B about centre of mass", "undefined", "(E-E0)/|E0|",
                       "eleven rows", "exactly zero"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)


@needs_beats
class BeatStructureTests(unittest.TestCase):
    def test_beats_are_numbered_from_zero_in_document_order(self):
        ids = re.findall(r'<section id="(beat\d+)"', HELP_HTML)
        self.assertEqual(ids, [f"beat{n}" for n in range(8)])
        self.assertLess(HELP_HTML.index('id="beats"'), HELP_HTML.index('id="beat0"'))
        self.assertLess(HELP_HTML.index('id="beat7"'), HELP_HTML.index('id="equations"'))

    def test_section_order_is_the_documented_reading_order(self):
        ids = re.findall(r'<section id="([^"]+)"', HELP_HTML)
        self.assertEqual(ids, ["overview", "beats", *[f"beat{n}" for n in range(8)], "equations",
                               "algorithm", "modules", "quickstart", "parameters", "output-types",
                               "summary", "experiments", "related", "license"])

    def test_every_beat_has_the_same_parts_in_the_same_order(self):
        for number in range(8):
            with self.subTest(beat=number):
                body = section_html(HELP_HTML, f"beat{number}")
                self.assertRegex(body, rf"<h2>Beat {number} · ")
                self.assertEqual(body.count("<pre>"), 1)
                self.assertEqual(body.count("Three tasks, in order."), 1)
                self.assertEqual(body.count("<em>Then</em>"), 1)
                self.assertEqual(body.count("Experiments that go with this beat"), 1)
                self.assertRegex(body, r'<a href="#exp\d">')
                between = body[body.index("</pre>"):body.index("Three tasks, in order.")]
                self.assertRegex(between, r"\b[Ll]ook\b[^.]*? at\b")
                self.assertLess(body.index("<pre>"), body.index("Three tasks, in order."))
                self.assertLess(body.index("Three tasks, in order."), body.index("<em>Then</em>"))
                self.assertLess(body.index("<em>Then</em>"), body.index("Experiments that go with"))

    def test_sidebar_labels_carry_the_beat_numbers(self):
        for number in range(8):
            with self.subTest(beat=number):
                self.assertRegex(HELP_HTML, rf'<a href="#beat{number}">{number} · ')

    def test_every_experiment_is_pointed_to_by_at_least_one_beat(self):
        cited = set()
        for number in range(8):
            cited |= set(re.findall(r'href="#(exp\d+)"', section_html(HELP_HTML, f"beat{number}")))
        every = set(re.findall(r'id="(exp\d+)"', HELP_HTML))
        self.assertEqual(cited, every)
        self.assertEqual(len(every), 8)

    def test_each_beat_command_block_holds_only_commands(self):
        for number in range(8):
            with self.subTest(beat=number):
                body = section_html(HELP_HTML, f"beat{number}")
                block = html_module.unescape(re.search(r"<pre>(.*?)</pre>", body, re.DOTALL).group(1))
                block = re.sub(r"\\\n\s*", " ", block)
                lines = [line for line in block.splitlines() if line.strip()]
                self.assertTrue(lines)
                for line in lines:
                    self.assertTrue(line.startswith("python main.py"), line)

    def test_equation_index_lists_every_numbered_equation(self):
        index = section_html(HELP_HTML, "equations")
        for number in range(1, 11):
            with self.subTest(equation=number):
                self.assertRegex(index, rf"<td>\({number}\)</td>")
        self.assertNotIn("eq-label", index)

    def test_numbered_equations_appear_in_the_beats_that_the_index_names(self):
        index = section_html(HELP_HTML, "equations")
        beat_of = {int(number): int(beat) for number, beat in re.findall(
            r"<td>\((\d+)\)</td><td>.*?</td><td>.*?</td><td>(\d)</td>", index, re.DOTALL)}
        self.assertEqual(sorted(beat_of), list(range(1, 11)))
        for number, beat in beat_of.items():
            with self.subTest(equation=number):
                self.assertIn(f'<span class="eq-label">({number})</span>',
                              section_html(HELP_HTML, f"beat{beat}"))

    def test_every_function_the_equation_index_names_exists_in_the_module_it_names(self):
        index = section_html(HELP_HTML, "equations")
        modules = {"physics_binary.py": physics, "driver_binary.py": driver, "main.py": entry,
                   "plot_binary.py": plotting}
        named = 0
        for cell in re.findall(r"<td>((?:(?!</td>).)*?<code>\w+\(\)</code>(?:(?!</td>).)*?)</td>", index, re.DOTALL):
            names = re.findall(r"<code>(\w+)\(\)</code>", cell)
            files = re.findall(r"<code>(\w+\.py)</code>", cell)
            for name in names:
                named += 1
                with self.subTest(function=name, files=files):
                    searched = [modules[f] for f in files] if files else list(modules.values())
                    self.assertTrue(any(callable(getattr(module, name, None)) for module in searched))
        self.assertGreaterEqual(named, 12)
        self.assertIn("<code>_compensated_add()</code>", index)

    def test_equation_kind_tags_match_the_role_of_each_equation(self):
        blocks = re.findall(
            r'<div class="eq">(?:<span class="eq-label">\((\d+)\)</span>)?\s*'
            r'<div class="eq-kind">(.*?)</div>', HELP_HTML, re.DOTALL)
        kinds = [(int(number) if number else None, re.findall(r'kind kind-(\w+)', tag))
                 for number, tag in blocks]
        expected = [
            (1, ["law"]), (2, ["ode"]), (3, ["ode"]), (4, ["def"]),
            (None, ["der"]),                       # kinetic energy of a moving pair
            (5, ["def"]), (6, ["der"]), (7, ["def"]),
            (None, ["der"]), (None, ["der"]), (None, ["der"]),   # apsides, Kepler III, hodograph
            (None, ["alg"]),                       # one-revolution stopping rule
            (8, ["alg"]), (9, ["alg"]),
            (None, ["alg"]),                       # corrector convergence test
            (10, ["alg"]),
        ]
        self.assertEqual(kinds, expected)

    def test_legend_names_exactly_the_five_kinds(self):
        legend = section_html(HELP_HTML, "beats")
        self.assertEqual(sorted(set(re.findall(r"kind kind-(\w+)", legend))),
                         ["alg", "def", "der", "law", "ode"])


@needs_beats
class ProgramFilesCardTests(unittest.TestCase):
    def test_physics_card_names_all_four_public_functions_and_the_dataclasses(self):
        card = html_text(section_html(HELP_HTML, "modules"))
        self.assertIn("Provides four functions", card)
        for name in ("relative_displacement()", "accelerations()", "energies()",
                     "orbital_elements()", "OrbitalElements", "BinaryState", "BinaryResult"):
            with self.subTest(name=name):
                self.assertIn(name, card)
        for name in ("relative_displacement", "accelerations", "energies", "orbital_elements"):
            self.assertTrue(callable(getattr(physics, name)))
        self.assertTrue(hasattr(physics, "OrbitalElements"))

    def test_driver_card_lists_the_result_fields_that_exist(self):
        fields = set(driver.BinaryResult.__dataclass_fields__)
        for name in ("completed_orbit", "accepted_steps", "total_angle_rad", "model_version", "build_id"):
            self.assertIn(name, fields)
        card = html_text(section_html(HELP_HTML, "modules"))
        for phrase in ("number of accepted steps", "whether the orbit was completed",
                       "accumulated orbit angle", "version and build"):
            self.assertIn(phrase, card)


# ---------------------------------------------------------------------------
# Numbers quoted in the Beats, recomputed from real runs.
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def cached_printed(argv):
    return printed_run(argv)


@functools.lru_cache(maxsize=None)
def cached_integration(items):
    return integrate(**dict(items))


def run_result(**changes):
    return cached_integration(tuple(sorted(changes.items())))


def P(*argv):
    return cached_printed(tuple(argv))


def F(*argv):
    return printed_fields(P(*argv))


def R(*argv):
    return printed_rows(P(*argv))


def departure(rows, fraction):
    return float(rows[fraction][-1])


def sci(value, plus=False):
    """Scientific notation as the Help writes it: minus sign, five figures, trailing zeros dropped."""
    mantissa, exponent = f"{abs(value):.4e}".split("e")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    sign = "−" if value < 0 else ("+" if plus else "")
    return f"{sign}{mantissa}e{exponent}"


def um(cell):
    """A printed number with a typographic minus sign."""
    return cell.replace("-", "−")


def acceleration_profile(**changes):
    """Cached ``_acceleration_profile``; the result is read, never modified."""
    return _acceleration_profile(tuple(sorted(changes.items())))


@functools.lru_cache(maxsize=None)
def _acceleration_profile(items):
    """Return (result, total calls to accelerations, steps that repeated the corrector,
    total repeated passes)."""
    changes = dict(items)
    calls = []
    real = driver.accelerations

    def spy(*args):
        calls.append(tuple(args[2:]))
        return real(*args)

    with mock.patch.object(driver, "accelerations", side_effect=spy):
        result = integrate(**changes)
    index = {(result.xA[i], result.yA[i], result.xB[i], result.yB[i]): i
             for i in range(len(result.times))}
    per_step = []
    for args in calls[1:]:
        found = index.get(args)
        if found is not None and found == len(per_step):
            per_step.append(0)
        else:
            per_step[-1] += 1
    repeated = [number + 1 for number, extra in enumerate(x - 1 for x in per_step) if extra > 0]
    return result, len(calls), repeated, sum(x - 1 for x in per_step)


def separation_list(result):
    return [math.hypot(a - b, c - d) for a, b, c, d in zip(result.xA, result.xB, result.yA, result.yB)]


def largest_departure(result):
    """Largest |E - E0| / |E0| over every stored state, with its sign."""
    initial = result.E[0]
    worst = max(result.E, key=lambda value: abs(value - initial))
    return (worst - initial) / abs(initial)


def relative_path_deviation(boost):
    base = run_result()
    moved = run_result(vInitA=boost, vInitB=boost)
    count = min(len(base.times), len(moved.times))
    return max(math.hypot((moved.xA[i] - moved.xB[i]) - (base.xA[i] - base.xB[i]),
                          (moved.yA[i] - moved.yB[i]) - (base.yA[i] - base.yB[i]))
               for i in range(count)) / 9.2e10


def predictor_acceleration_changes(result, parameters=DEFAULTS):
    """Eq. (10) evaluated at every accepted step of a stored run."""
    changes = []
    for i in range(len(result.times) - 1):
        step = result.times[i + 1] - result.times[i]
        a0 = physics.accelerations(parameters["MA"], parameters["MB"], result.xA[i], result.yA[i],
                                   result.xB[i], result.yB[i])
        a1 = physics.accelerations(parameters["MA"], parameters["MB"],
                                   result.xA[i] + result.vA[i] * step, result.yA[i] + result.uA[i] * step,
                                   result.xB[i] + result.vB[i] * step, result.yB[i] + result.uB[i] * step)
        changes.append(math.hypot(a1[0] - a0[0], a1[1] - a0[1])
                       / max(math.hypot(a0[0], a0[1]), math.hypot(a1[0], a1[1])))
    return changes


@needs_beats
class BeatQuotedNumberTests(unittest.TestCase):
    """Every number quoted in a Beat is pinned to the phrase it sits in."""

    def assertPhrase(self, number, *phrases):
        text = beat_text(number)
        for phrase in phrases:
            with self.subTest(beat=number, phrase=phrase):
                self.assertIn(phrase, text)

    # -------------------------------------------------------------- Beat 0
    def test_beat_0_two_bodies_one_force(self):
        fields, result = F(), run_result()
        smallest, largest = (float(fields[key].split()[0]) for key in
                             ("Minimum separation (sampled)", "Maximum separation (sampled)"))
        percent = round(100 * smallest / largest)
        self.assertEqual(percent, 13)
        self.assertAlmostEqual(smallest / largest, 1 / 8, delta=0.01)
        self.assertEqual((DEFAULTS["MA"], DEFAULTS["MB"], physics.G), (2e30, 2e30, 6.6743e-11))
        self.assertEqual(largest, 2 * DEFAULTS["xInitA"])
        steps = int(fields["Accepted steps"])
        self.assertEqual(result.times[-1], steps * 2000.0)
        self.assertPhrase(
            0,
            "start 9.2×10 10 m apart",
            "sideways speed of 13 km s −1",
            f"The maximum, {fields['Maximum separation (sampled)']}, is the separation the bodies "
            "started with (2 × 4.6×10 10 m)",
            f"The minimum, {fields['Minimum separation (sampled)']}, is {percent}% of that",
            f"The line Termination: {fields['Termination']} says that the program stopped",
            f"Accepted steps: {steps} steps of 2000 s is {steps * 2000 / 1e6:.3f}×10 6 s, "
            f"which is the Total time of {fields['Total time']}",
            "The program uses \\(G=6.67430\\times10^{-11}",
        )

    # -------------------------------------------------------------- Beat 1
    BEAT_1 = ("--MA", "2e30", "--MB", "1e30", "--xInitA", "3.0666666667e10", "--xInitB",
              "-6.1333333333e10", "--uInitA", "15550.5265", "--uInitB", "-31101.053")

    def test_beat_1_who_moves_and_how_much(self):
        printed = P(*self.BEAT_1)
        block_a, block_b = printed_body_block(printed, "A"), printed_body_block(printed, "B")
        semi_a = block_a[0].split("semi-major axis: ")[1].split(";")[0]
        semi_b = block_b[0].split("semi-major axis: ")[1].split(";")[0]
        speed_a = block_a[2].split("periapsis: ")[1].split(" m/s")[0]
        speed_b = block_b[2].split("periapsis: ")[1].split(" m/s")[0]
        period_a = block_a[0].split("period: ")[1]
        self.assertEqual(period_a, block_b[0].split("period: ")[1])
        ratio = printed_number(semi_b) / printed_number(semi_a)
        self.assertEqual(f"{ratio:.4f}", "2.0000")
        self.assertAlmostEqual(float(speed_b) / float(speed_a), 2.0, places=3)
        product_a, product_b = 2e30 * printed_number(semi_a), 1e30 * printed_number(semi_b)
        self.assertEqual(f"{product_a:.4e}", f"{product_b:.4e}")
        centre_x = (2e30 * 3.0666666667e10 + 1e30 * -6.1333333333e10) / 3e30
        self.assertAlmostEqual(centre_x, 0.0, delta=1.0)
        self.assertAlmostEqual(3.0666666667e10 / 9.2e10, 1 / 3, places=8)
        self.assertPhrase(
            1,
            f"compare the semi-major axes: {semi_a} for body A and {semi_b} for body B. "
            "Their ratio is 2.0000",
            f"both come to {product_a / 1e40:.4f}×10 40 kg m",
            f"the speeds on the Speed at periapsis lines, {speed_a} m/s and {speed_b} m/s",
            f"the periods, which are the same, {period_a}",
            "Here those are 1/3 and 2/3 of 9.2×10 10 m",
        )
        self.assertIn("2×10 30 × 3.0666×10 10 and 1×10 30 × 6.1332×10 10", beat_text(1))

    # -------------------------------------------------------------- Beat 2
    def test_beat_2_only_the_separation_matters(self):
        default = P()
        shifted = P("--xInitA", "5e10", "--xInitB", "-4.2e10", "--yInitA", "1e10", "--yInitB", "1e10")
        boosted = P("--vInitA", "60000", "--vInitB", "60000")
        head = lambda text: text[text.index("\n") + 1: text.index("Energy at fractions")]
        self.assertEqual(head(shifted), head(default))
        self.assertEqual(head(boosted), head(default))
        centre = ((2e30 * 5e10 + 2e30 * -4.2e10) / 4e30, (2e30 * 1e10 + 2e30 * 1e10) / 4e30)
        self.assertAlmostEqual(centre[0], 4e9, delta=1.0)
        self.assertAlmostEqual(centre[1], 1e10, delta=1.0)
        fields = printed_fields(default)
        rows_default, rows_boost = printed_rows(default), printed_rows(boosted)
        kinetic_default, kinetic_boost = rows_default[0.0][3], rows_boost[0.0][3]
        self.assertEqual(rows_boost[0.0][2], rows_default[0.0][2])
        extra = float(kinetic_boost) - float(kinetic_default)
        self.assertAlmostEqual(extra / (0.5 * 4e30 * 6e4 ** 2), 1.0, places=3)
        self.assertEqual(f"{extra / 1e39:.1f}", "7.2")
        self.assertGreater(float(rows_boost[0.0][4]), 0)
        self.assertEqual(printed_fields(boosted)["Initial Keplerian orbit"], fields["Initial Keplerian orbit"])
        change_default = float(rows_default[0.5][4]) - float(rows_default[0.0][4])
        change_boost = float(rows_boost[0.5][4]) - float(rows_boost[0.0][4])
        for change in (change_default, change_boost):
            self.assertAlmostEqual(abs(change), 3e36, delta=0.5e36)
        self.assertPhrase(
            2,
            f"which is {fields['Accepted steps']} steps, {fields['Total time'].replace(' days', '')} days, "
            f"a minimum separation of {fields['Minimum separation (sampled)'].replace(' m', '')} m and eccentricity "
            f"{fields['Initial Keplerian orbit'].split(': ')[1]}",
            "the centre of mass is now at (4×10 9 , 1×10 10 ) m",
            f"the kinetic energy at row 0.0 is {kinetic_boost} J instead of {kinetic_default} J",
            "The difference, 7.2×10 39 J",
            f"the potential energy, {um(rows_boost[0.0][2])} J, has not changed",
            f"it is +{rows_boost[0.0][4]} J, positive",
            f"row 0.5 of the last column reads {sci(departure(rows_boost, 0.5))} in this run and "
            f"{sci(departure(rows_default, 0.5))} in the first",
        )

    def test_beat_2_boost_changes_corrector_decisions_and_relative_paths_slightly(self):
        _, _, repeated_default, _ = acceleration_profile()
        _, _, repeated_small, _ = acceleration_profile(vInitA=500.0, vInitB=500.0)
        _, _, repeated_boost, _ = acceleration_profile(vInitA=60000.0, vInitB=60000.0)
        self.assertEqual((len(repeated_default), len(repeated_small), len(repeated_boost)), (127, 127, 192))
        self.assertEqual(repeated_small, repeated_default)
        self.assertLess(relative_path_deviation(500.0), 5e-15)
        self.assertTrue(4e-6 < relative_path_deviation(60000.0) < 6e-6)
        self.assertTrue(6e-4 < relative_path_deviation(1e6) < 8e-4)
        self.assertPhrase(
            2,
            "the corrector repeats in 192 steps instead of 127",
            "agree to about 5 parts in 10 6 of the starting separation",
            "differ by about 7 parts in 10 4",
            "the corrector makes the same decisions as without it, and the two paths differ by about "
            "2 parts in 10 15",
        )

    # -------------------------------------------------------------- Beat 3
    BEAT_3 = (
        ("--output_type", "energy_vs_time"),
        ("--uInitA", "38091.14", "--uInitB", "-38091.14", "--no-stop_after_one_orbit", "--max_steps", "1000",
         "--output_type", "energy_vs_time"),
        ("--uInitA", "60000", "--uInitB", "-60000", "--no-stop_after_one_orbit", "--max_steps", "1000",
         "--output_type", "energy_vs_time"),
    )

    def test_beat_3_energy_separates_bound_from_unbound(self):
        rows = [R(*argv) for argv in self.BEAT_3]
        kinds = [F(*argv)["Initial Keplerian orbit"] for argv in self.BEAT_3]
        potential = {row[0.0][2] for row in rows}
        self.assertEqual(potential, {"-2.9019e+39"})
        self.assertAlmostEqual(-physics.G * (2e30) ** 2 / 9.2e10 / -2.9019e39, 1.0, places=4)
        kinetic = [row[0.0][3] for row in rows]
        total = [row[0.0][4] for row in rows]
        self.assertEqual(kinds[0], "elliptic; eccentricity: 0.76705")
        self.assertEqual(kinds[1], "hyperbolic; eccentricity: 1")
        self.assertEqual(kinds[2], "hyperbolic; eccentricity: 3.9623")
        ratio = float(total[1]) / float(kinetic[1])
        self.assertEqual(f"{ratio * 1e7:.1f}", "1.1")
        precise = elements_for(uInitA=38091.14, uInitB=-38091.14).eccentricity
        self.assertEqual(f"{precise:.7f}", "1.0000002")
        bound_end = departure(rows[0], 1.0)
        middle_end = departure(rows[1], 1.0)
        self.assertEqual(f"{abs(bound_end) * 1e6:.1f}", "3.2")
        self.assertEqual(round(100 * middle_end), 406)
        change = float(rows[1][1.0][4]) - float(rows[1][0.0][4])
        self.assertEqual(f"{change / 1e33:.1f}", "1.3")
        self.assertEqual(f"{change / float(kinetic[1]) * 1e7:.1f}", "4.6")
        self.assertPhrase(
            3,
            f"The potential energy is {um(rows[0][0.0][2])} J in each",
            f"The kinetic energy differs: {kinetic[0]}, {kinetic[1]} and {kinetic[2]} J, for speeds of "
            "13000, 38091.14 and 60000 m s −1 for each body",
            f"the totals are {um(total[0])}, +{total[1]} and +{total[2]} J",
            "the negative total is elliptic with eccentricity 0.76705, and the large positive total is "
            "hyperbolic with eccentricity 3.9623",
            "a total of only 1.1×10 −7 of its kinetic energy",
            "with an eccentricity that prints as 1 (it is 1.0000002)",
            "In the bound run it is 3.2 parts in 10 6 at the end",
            f"In the middle run it grows to {rows[1][1.0][-1]}, that is 406%",
            "Total E has changed by only 1.3×10 33 J, which is 4.6×10 −7 of the kinetic energy",
        )

    def test_beat_3_the_energy_plot_cannot_show_the_departures(self):
        result = run_result()
        span = max(result.K) - min(result.K)
        self.assertLess(abs(largest_departure(result)) * abs(result.E[0]) / span, 1e-3)
        self.assertPhrase(3, "the printed (E-E0)/|E0| column is the quantitative diagnostic")

    # -------------------------------------------------------------- Beat 4
    def test_beat_4_the_shape_of_an_orbit(self):
        circle_args = ("--uInitA", "26934.5", "--uInitB", "-26934.5", "--output_type", "velocity_space")
        circle, default = F(*circle_args), F("--output_type", "velocity_space")
        circle_block = printed_body_block(P(*circle_args), "A")
        default_block = printed_body_block(P("--output_type", "velocity_space"), "A")
        self.assertEqual(circle["Minimum separation (sampled)"], circle["Maximum separation (sampled)"])
        self.assertEqual(circle["Initial Keplerian orbit"], "elliptic; eccentricity: 1.393e-07")
        self.assertIn("Periapsis: 4.6e+10 m; apoapsis: 4.6e+10 m", circle_block[1])
        self.assertIn("Speed at periapsis: 26935 m/s; speed at apoapsis: 26934 m/s", circle_block[2])
        self.assertIn("period: 124.2 days", circle_block[0])
        self.assertEqual((circle["Accepted steps"], circle["Total time"], circle["Total revolutions"]),
                         ("5366", "124.21 days", "1.0001"))
        self.assertEqual(5366 * 2000.0, run_result(uInitA=26934.5, uInitB=-26934.5).times[-1])
        self.assertIn("Periapsis: 6.0643e+09 m; apoapsis: 4.6e+10 m", default_block[1])
        self.assertIn("Speed at periapsis: 98610 m/s; speed at apoapsis: 13000 m/s", default_block[2])
        elements = elements_for()
        peri, apo = elements.relative_periapsis / 2, elements.relative_apoapsis / 2
        v_peri, v_apo = elements.relative_speed_periapsis / 2, elements.relative_speed_apoapsis / 2
        self.assertAlmostEqual(peri * v_peri, 5.98e14, delta=1e10)
        self.assertAlmostEqual(apo * v_apo, 5.98e14, delta=1e10)
        self.assertEqual((f"{apo / peri:.3f}", f"{v_peri / v_apo:.3f}"), ("7.585", "7.585"))
        self.assertEqual((f"{(v_peri + v_apo) / 2:.0f}", f"{(v_peri - v_apo) / 2:.0f}"), ("55805", "42805"))
        self.assertEqual(f"{(v_peri - v_apo) / (v_peri + v_apo):.5f}", "0.76705")
        mass_total = 4e30
        for label, semi_sum, days in (("default", 2 * 2.6032e10, 52.874), ("circle", 9.2e10, 124.198)):
            kepler = 2 * math.pi * math.sqrt(semi_sum ** 3 / (physics.G * mass_total)) / 86400
            self.assertAlmostEqual(kepler, days, places=2, msg=label)
        run = run_result()
        below = 100 * (2 * peri - min(separation_list(run))) / (2 * peri)
        self.assertEqual(f"{below:.3f}", "0.015")
        # the rounded digits printed in the Help give the smaller figure the Help quotes beside it
        printed_min = float(default["Minimum separation (sampled)"].replace(" m", ""))
        printed_below = 100 * (2 * 6.0643e9 - printed_min) / (2 * 6.0643e9)
        self.assertEqual(f"{printed_below:.3f}", "0.013")
        self.assertIn("the rounded digits printed here give about 0.013%", beat_text(4))
        self.assertPhrase(
            4,
            "The minimum and maximum separations are both 9.2e+10 m",
            "the eccentricity is 1.393e-07",
            "the two speeds, 26935 and 26934 m/s",
            "The period is 124.2 days, and the run took 5366 steps of 2000 s, which is the Total time of 124.21 days",
            f"The separation swings between {default['Minimum separation (sampled)'].replace(' m', '')} and "
            f"{default['Maximum separation (sampled)'].replace(' m', '')} m and the eccentricity is 0.76705",
            "Body A's periapsis is 6.0643e+09 m, where it moves at 98610 m/s, and its apoapsis is 4.6e+10 m, "
            "where it moves at 13000 m/s",
            "the product is 5.980×10 14 m 2 s −1 at periapsis and 5.980×10 14 at apoapsis, where the body is "
            "7.585 times farther away and 7.585 times slower",
            "The Keplerian period is 52.874 days and the run stopped after 52.87 days",
            "the bodies are 2 × 6.0643×10 9 ≈ 1.2129×10 10 m apart",
            f"{default['Minimum separation (sampled)'].replace(' m', '')} m, is about 0.015% below that",
            "one circle of radius 26934.5 m/s about the origin",
            "Each has a radius of 55805 m/s, half of 98610 + 13000, and its centre is 42805 m/s from the origin, "
            "half of 98610 − 13000",
            "42805/55805 = 0.76705, the eccentricity",
            "2.6032e+10 + 2.6032e+10 = 5.2064e+10 m for the default orbit and 9.2e+10 m for the circle",
            "which is 52.874 days for the default orbit and 124.198 days for the circle, which the block prints as 124.2",
            "the circular run reports 1.0001 revolutions",
        )

    def test_beat_4_the_velocity_space_circles_have_the_stated_radius_and_offset(self):
        for label, argv, radius, offset in (
                ("default", {}, 55805.3, 42805.3),
                ("circle", {"uInitA": 26934.5, "uInitB": -26934.5}, 26934.5, 0.0)):
            result = run_result(**argv)
            speeds = [math.hypot(v, u - (-offset)) for v, u in zip(result.vA, result.uA)]
            with self.subTest(case=label):
                # the default-step integration error in the speeds is about 3 parts in 10^4
                self.assertAlmostEqual(max(speeds) / radius, 1.0, delta=5e-4)
                self.assertAlmostEqual(min(speeds) / radius, 1.0, delta=5e-4)
                # body B traces the mirror-image circle
                mirror = [math.hypot(v, u - offset) for v, u in zip(result.vB, result.uB)]
                self.assertAlmostEqual(max(mirror) / radius, 1.0, delta=5e-4)

    def test_beat_4_origin_lies_inside_on_or_outside_the_circle_as_e_is_below_one_or_above(self):
        for label, speed, inside in (("bound", 13000.0, True), ("hyperbolic", 60000.0, False)):
            elements = elements_for(uInitA=speed, uInitB=-speed)
            with self.subTest(case=label):
                self.assertEqual(elements.eccentricity < 1, inside)

    # -------------------------------------------------------------- Beat 5
    def test_beat_5_the_corrector(self):
        default = R()
        strict, loose = R("--eps2", "1e-8"), R("--eps2", "0.5")
        fields = {name: F(*argv) for name, argv in (("default", ()), ("strict", ("--eps2", "1e-8")),
                                                     ("loose", ("--eps2", "0.5")))}
        self.assertEqual([fields[k]["Accepted steps"] for k in ("strict", "loose", "default")],
                         ["2284", "2285", "2284"])
        self.assertEqual((fields["strict"]["Total time"], fields["loose"]["Total time"]),
                         ("52.87 days", "52.894 days"))
        dips = [departure(rows, 0.5) for rows in (strict, default, loose)]
        ends = [departure(rows, 1.0) for rows in (strict, default, loose)]
        self.assertEqual([sci(v) for v in dips], ["−1.1854e-03", "−1.2814e-03", "−1.784e-03"])
        self.assertEqual([sci(v) for v in ends], ["3.2091e-12", "−3.1646e-06", "8.1292e-05"])
        self.assertEqual(f"{dips[2] / dips[0]:.1f}", "1.5")
        self.assertEqual(f"{1e-8 and 0.5 / 1e-8:.0e}", "5e+07")
        profiles = {name: acceleration_profile(**changes) for name, changes in
                    (("default", {}), ("strict", {"eps2": 1e-8}), ("loose", {"eps2": 0.5}))}
        repeated = profiles["default"][2]
        self.assertEqual((len(repeated), min(repeated), max(repeated)), (127, 1080, 1206))
        self.assertEqual(len(repeated), max(repeated) - min(repeated) + 1)
        self.assertEqual((len(profiles["strict"][2]), profiles["strict"][3]), (2284, 2878))
        self.assertEqual(profiles["loose"][2], [])
        self.assertEqual([profiles[k][1] for k in ("default", "strict", "loose")], [4696, 7447, 4571])
        self.assertEqual(f"{profiles['strict'][1] / profiles['default'][1]:.1f}", "1.6")
        self.assertPhrase(
            5,
            "The default run, eps2 = 1e-4, gives −1.2814e-03 in row 0.5 and −3.1646e-06 in row 1.0",
            "2284 for the strict run and 2285 for the loose one, against 2284 for the default",
            "The tolerance changed by a factor of 5×10 7",
            "The strict run's Total time is 52.87 days, which is 2284 steps of 2000 s, and the loose run's is "
            "52.894 days, which is 2285 steps of 2000 s",
            "row 0.5: −1.1854e-03 for the strict run, −1.2814e-03 for the default and −1.784e-03 for the loose run",
            "the loosest corrector makes the dip only 1.5 times as deep as the strictest",
            "row 1.0, the end of the orbit: 3.2091e-12 for the strict run, −3.1646e-06 for the default and "
            "8.1292e-05 for the loose one",
            "within 3 parts in 10 12 of its starting value, the default run's to 3 parts in 10 6 , and the loose "
            "run ends 8 parts in 10 5 above it",
            "The default run repeats the correction in 127 of its 2284 steps, the steps 1080 to 1206 around closest "
            "approach and no others",
            "The strict run repeats it in every step, 2878 repetitions in all, and the loose run never does",
            "the default run makes 4696, the strict run 7447 and the loose run 4571",
            "A strict corrector costs about 1.6 times as much here",
            "returns to 3 parts in 10 12",
        )

    def test_beat_5_first_corrector_pass_usually_satisfies_the_default_test(self):
        result = run_result()
        changes = []
        real = driver._vector_relative_change
        with mock.patch.object(driver, "_vector_relative_change",
                               side_effect=lambda *a: changes.append(real(*a)) or changes[-1]):
            integrate()
        first_passes = changes[1::3][:5]   # acceleration check, pass-1 change A, change B, ...
        self.assertTrue(all(value < 1e-4 for value in first_passes))
        self.assertEqual(result.accepted_steps, 2284)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def halving_sources(commands):
        """Count where the step is halved: after a corrector failure, or after a predictor rejection."""
        source = Path(driver.__file__).read_text(encoding="utf-8").splitlines()
        marker = next(i for i, line in enumerate(source) if line.strip() == "if not converged:")
        failure_line = marker + 2      # 1-based line of the halving call after the marker
        seen = {"corrector": 0, "predictor": 0}
        real = driver._halve_timestep

        def spy(dt_work, dt_min):
            caller = sys._getframe(1).f_lineno
            seen["corrector" if caller == failure_line else "predictor"] += 1
            return real(dt_work, dt_min)

        with mock.patch.object(driver, "_halve_timestep", side_effect=spy):
            for argv in commands:
                args = entry.parse_args(list(argv))
                driver.integrate_binary(**{key: getattr(args, key) for key in DEFAULTS})
        return seen

    def test_beat_5_the_corrector_always_converges_within_ten_passes_in_beats_5_to_7(self):
        commands = tuple(tuple(argv) for number in (5, 6, 7)
                         for argv in documented_commands(section_html(HELP_HTML, f"beat{number}")))
        self.assertGreaterEqual(len(commands), 7)
        seen = self.halving_sources(commands)
        self.assertEqual(seen["corrector"], 0)
        self.assertGreater(seen["predictor"], 0)

    # -------------------------------------------------------------- Beat 6
    def test_beat_6_the_step_control(self):
        wide, loose = ("--dt", "20000"), ("--dt", "20000", "--eps1", "0.5")
        rows_wide, rows_loose, rows_default = R(*wide), R(*loose), R()
        fields_wide, fields_loose = F(*wide), F(*loose)
        result_wide = run_result(dt=20000.0)
        result_loose = run_result(dt=20000.0, eps1=0.5)
        steps = [b - a for a, b in zip(result_wide.times, result_wide.times[1:])]
        short = [i + 1 for i, step in enumerate(steps) if step < 20000.0 * (1 - 1e-9)]
        self.assertEqual((fields_wide["Accepted steps"], len(short)), ("387", 211))
        self.assertEqual((short[0], short[-1]), (89, 299))
        self.assertEqual(f"{result_wide.times[short[0] - 1] / 86400:.1f}", "20.4")
        self.assertEqual(f"{result_wide.times[short[-1]] / 86400:.1f}", "32.6")
        self.assertEqual(f"{min(steps):.0f}", "1539")
        self.assertEqual(set(steps[:88]), {20000.0})
        self.assertEqual([round(step) for step in steps[88:95]],
                         [10000, 11000, 12100, 13310, 14641, 16105, 8858])
        self.assertEqual(f"{4.568e6 / 20000:.0f}", "228")
        self.assertEqual(f"{departure(rows_wide, 0.5) / departure(rows_default, 0.5):.1f}", "1.8")
        kepler_minimum = elements_for().relative_periapsis
        self.assertEqual(f"{100 * (kepler_minimum - min(separation_list(result_wide))) / kepler_minimum:.2f}", "0.13")
        self.assertEqual((fields_loose["Accepted steps"], fields_loose["Total time"]), ("222", "51.389 days"))
        self.assertEqual(result_loose.times[-1], 222 * 20000.0)
        self.assertTrue(all(step == 20000.0 for step in (b - a for a, b in
                                                          zip(result_loose.times, result_loose.times[1:]))))
        self.assertEqual(f"{100 * (kepler_minimum - min(separation_list(result_loose))) / kepler_minimum:.1f}", "1.6")
        self.assertEqual(f"{largest_departure(result_loose) * 100:.1f}", "-12.5")
        self.assertEqual(sci(departure(rows_loose, 0.5)), "−4.0345e-02")
        largest_default = max(predictor_acceleration_changes(run_result()))
        largest_loose = max(predictor_acceleration_changes(result_loose))
        self.assertEqual((f"{largest_default:.4f}", f"{largest_loose:.3f}"), ("0.0325", "0.318"))
        self.assertLess(largest_default, DEFAULTS["eps1"])
        self.assertGreater(largest_loose, DEFAULTS["eps1"])
        self.assertEqual(2 ** -40 < 1e-12 < 2 ** -39, True)
        self.assertPhrase(
            6,
            "The default run gave 2284 steps, a minimum separation of 1.2127e+10 m and −1.2814e-03 in row 0.5",
            "4.568×10 6 s divided by 20000 s is 228. The run took 387 steps. Of those, 211 were shorter than "
            "20000 s, from step 89 (day 20.4) to step 299 (day 32.6), around closest approach, and the shortest "
            "was 1539 s",
            f"row 0.5 is {sci(departure(rows_wide, 0.5))}, only 1.8 times the default run's dip",
            f"{fields_wide['Minimum separation (sampled)'].replace(' m', '')} m, is 0.13% below the Keplerian "
            "1.2129×10 10 m",
            "222 steps, and 222 × 20000 s = 4.44×10 6 s is the Total time of 51.389 days, so no step was shortened",
            f"Row 0.5 is now {sci(departure(rows_loose, 0.5))}, a dip of 4%",
            f"{fields_loose['Minimum separation (sampled)'].replace(' m', '')} m, 1.6% below the Keplerian value",
            "the largest departure among the 222 accepted states is −12.5%",
            "the largest \\(\\Delta_a\\) over the whole orbit is 0.0325",
            "are 20000 s for the first 88 steps and then 10000, 11000, 12100, 13310, 14641, 16105 and 8858 s",
            "the largest \\(\\Delta_a\\) among the accepted steps is 0.318",
            "dt × 10 −12 , which a step falls below at the fortieth halving in a row",
        )

    # -------------------------------------------------------------- Beat 7
    GRID = {  # (dt, eps2) -> command
        (2000, "1e-4"): (), (1000, "1e-4"): ("--dt", "1000"), (500, "1e-4"): ("--dt", "500"),
        (2000, "1e-8"): ("--eps2", "1e-8"), (1000, "1e-8"): ("--dt", "1000", "--eps2", "1e-8"),
        (500, "1e-8"): ("--dt", "500", "--eps2", "1e-8"),
    }

    def test_beat_7_how_far_to_trust_it(self):
        rows = {key: R(*argv) for key, argv in self.GRID.items()}
        fields = {key: F(*argv) for key, argv in self.GRID.items()}
        dip = {key: departure(rows[key], 0.5) for key in rows}
        end = {key: departure(rows[key], 1.0) for key in rows}
        ratios = [dip[(dt, "1e-4")] / dip[(dt // 2, "1e-4")] for dt in (2000, 1000)]
        strict_ratios = [dip[(dt, "1e-8")] / dip[(dt // 2, "1e-8")] for dt in (2000, 1000)]
        self.assertEqual([f"{r:.2f}" for r in ratios], ["2.84", "3.98"])
        self.assertEqual([f"{r:.2f}" for r in strict_ratios], ["4.00", "4.00"])
        self.assertEqual(max(dip[(dt, "1e-4")] / dip[(dt, "1e-8")] for dt in (2000, 1000, 500)) < 1.6, True)
        self.assertEqual(f"{max(dip[(dt, '1e-4')] / dip[(dt, '1e-8')] for dt in (2000, 1000, 500)):.1f}", "1.5")
        self.assertEqual(sorted(abs(end[(dt, "1e-8")]) for dt in (2000, 1000, 500))[-1] < 3.3e-12, True)
        self.assertTrue(all(1e-6 < abs(end[(dt, "1e-4")]) < 1.1e-5 for dt in (2000, 1000, 500)))
        self.assertGreater(end[(1000, "1e-4")], 0)
        self.assertLess(end[(2000, "1e-4")], 0)
        table = section_html(HELP_HTML, "beat7")
        body = re.search(r"<tbody>(.*?)</tbody>", table, re.DOTALL).group(1)
        shown = [[html_text(cell) for cell in re.findall(r"<td>(.*?)</td>", row, re.DOTALL)]
                 for row in re.findall(r"<tr>(.*?)</tr>", body, re.DOTALL)]
        expected = [[str(dt), f"{fields[(dt, '1e-4')]['Accepted steps']} / {fields[(dt, '1e-8')]['Accepted steps']}",
                     sci(dip[(dt, "1e-4")]), sci(dip[(dt, "1e-8")]), sci(end[(dt, "1e-4")]), sci(end[(dt, "1e-8")])]
                    for dt in (2000, 1000, 500)]
        self.assertEqual(shown, expected)
        profiles = {dt: acceleration_profile(dt=float(dt)) for dt in (2000, 1000, 500)}
        self.assertEqual([len(profiles[dt][2]) for dt in (2000, 1000, 500)], [127, 0, 0])
        separation = {dt: fields[(dt, "1e-4")]["Minimum separation (sampled)"] for dt in (1000, 500)}
        self.assertEqual(separation, {1000: "1.2128e+10 m", 500: "1.2129e+10 m"})
        self.assertAlmostEqual(elements_for().relative_periapsis, 1.2128542115e10, delta=1.0)
        self.assertPhrase(
            7,
            "The default run, dt = 2000, gave 2284 steps, 1.2127e+10 m, −1.2814e-03 and −3.1646e-06",
            f"The first two runs give {fields[(1000, '1e-4')]['Accepted steps']} and "
            f"{fields[(500, '1e-4')]['Accepted steps']} steps, and {sci(dip[(1000, '1e-4')])} and "
            f"{sci(dip[(500, '1e-4')])}",
            "divided the dip by 2.84, and then by 3.98, which is close to 4",
            "the sampled minimum separation, 1.2128e+10 and 1.2129e+10 m",
            f"The first two runs give {sci(end[(1000, '1e-4')], plus=True)} and "
            f"{sci(end[(500, '1e-4')], plus=True)} and the default run gave {sci(end[(2000, '1e-4')])}",
            f"gives {sci(dip[(1000, '1e-8')])} in row 0.5 and {sci(end[(1000, '1e-8')])} in row 1.0",
            "trims it by at most a factor of 1.5 in the table below",
            "the dip falls by a factor of 4.00 for each halving of dt",
            "the end-of-orbit departure is 3 parts in 10 12 or less",
            "At the default eps2 the factors are 2.84 and 3.98 and the end-of-orbit departure is 10 −6 to 10 −5",
            "the corrector repeated at dt = 2000, in the steps around closest approach, and never repeats in the "
            "whole run at dt = 1000 or 500",
            "cost 1.6 times as many acceleration evaluations",
        )

    def test_beat_7_a_very_eccentric_orbit_and_a_head_on_start(self):
        argv = ("--uInitA", "1000", "--uInitB", "-1000", "--max_steps", "400000")
        fields, rows = F(*argv), R(*argv)
        block = printed_body_block(P(*argv), "A")
        self.assertEqual((fields["Accepted steps"], fields["Total time"]), ("2456", "43.231 days"))
        self.assertEqual(fields["Initial Keplerian orbit"], "elliptic; eccentricity: 0.99862")
        self.assertIn("period: 43.956 days", block[0])
        self.assertEqual(f"{elements_for(uInitA=1000.0, uInitB=-1000.0).relative_periapsis:.4e}", "6.3451e+07")
        self.assertEqual(sci(departure(rows, 1.0)), "−4.2836e-03")
        self.assertEqual(f"{-100 * departure(rows, 1.0):.2f}", "0.43")
        self.assertPhrase(
            7,
            "an orbit with eccentricity 0.99862 and a closest approach of 6.3451×10 7 m",
            "the default settings finish after 2456 steps and 43.231 days, although the Keplerian period is 43.956 days",
            "row 1.0 of the last column is −4.2836e-03: the energy dropped by 0.43%",
        )
        for head_on in (("--uInitA", "0", "--uInitB", "0"),):
            with self.subTest(argv=head_on), self.assertRaises(SystemExit) as raised, \
                 mock.patch.object(entry, "plot_binary"), redirect_stdout(io.StringIO()):
                entry.main(list(head_on))
            self.assertIn("numerical safety limit", str(raised.exception))


# ---------------------------------------------------------------------------
# Numbers quoted in the Suggested Experiments, recomputed from real runs.
# ---------------------------------------------------------------------------

def experiment_html(number):
    experiments = section_html(HELP_HTML, "experiments")
    match = re.search(rf'<h3 id="exp{number}">.*?(?=<h3 id="exp|\Z)', experiments, re.DOTALL)
    if match is None:
        raise AssertionError(f"experiment {number} not found in the Help file")
    return match.group(0)


def experiment_text(number):
    return html_text(experiment_html(number))


def experiment_commands(number):
    return documented_commands(experiment_html(number))


def hodograph_arc(result, centre_v, centre_u, body="A"):
    """Distances of one body's velocity points from a centre in velocity space."""
    v_values = result.vA if body == "A" else result.vB
    u_values = result.uA if body == "A" else result.uB
    return [math.hypot(v - centre_v, u - centre_u) for v, u in zip(v_values, u_values)]


@needs_beats
class ExperimentQuotedNumberTests(unittest.TestCase):
    """Every number quoted in a Suggested Experiment is pinned to the phrase it sits in."""

    def assertPhrase(self, number, *phrases):
        text = experiment_text(number)
        for phrase in phrases:
            with self.subTest(experiment=number, phrase=phrase):
                self.assertIn(phrase, text)

    # ---------------------------------------------------------- Experiment 1
    def test_experiment_1_eccentricity_against_starting_speed(self):
        speeds = (13000, 20000, 26934.5, 30000)
        self.assertEqual([tuple(argv) for argv in experiment_commands(1)],
                         [(), ("--uInitA", "26934.5", "--uInitB", "-26934.5"),
                          ("--uInitA", "20000", "--uInitB", "-20000"),
                          ("--uInitA", "30000", "--uInitB", "-30000")])
        eccentricity = {speed: elements_for(uInitA=speed, uInitB=-speed).eccentricity for speed in speeds}
        self.assertEqual({speed: f"{value:.5g}" for speed, value in eccentricity.items()},
                         {13000: "0.76705", 20000: "0.44863", 26934.5: "1.393e-07", 30000: "0.24058"})
        printed = {speed: F("--uInitA", str(speed), "--uInitB", str(-speed))["Initial Keplerian orbit"]
                   for speed in speeds}
        self.assertEqual(printed[26934.5], "elliptic; eccentricity: 1.393e-07")
        v_circular = math.sqrt(physics.G * 2e30 / (4 * 4.6e10))
        self.assertEqual(f"{v_circular / 1e4:.2f}", "2.69")
        # zero at the circular speed, growing on both sides of it
        self.assertTrue(eccentricity[26934.5] < eccentricity[30000] < eccentricity[20000] < eccentricity[13000])
        self.assertLess(eccentricity[26934.5], 1e-6)
        block = printed_body_block(P("--uInitA", "30000", "--uInitB", "-30000"), "A")
        self.assertIn("Periapsis: 4.6e+10 m; apoapsis: 7.5145e+10 m", block[1])
        self.assertPhrase(
            1,
            "With the default values this is about \\(2.69\\times10^4\\) m/s",
            "a speed of 13000 m/s gives an eccentricity of 0.76705, 20000 gives 0.44863, 26934.5 gives "
            "1.393e-07 (circular) and 30000 gives 0.24058",
            "at 30000 m/s the periapsis is the starting distance, 4.6e+10 m, and the apoapsis is 7.5145e+10 m",
        )

    # ---------------------------------------------------------- Experiments 2 and 3
    def test_experiments_2_and_3_commands_change_nothing_before_the_energy_table(self):
        head = lambda text: text[text.index("\n") + 1: text.index("Energy at fractions")]
        reference = head(P())
        for number in (2, 3):
            for argv in experiment_commands(number):
                # the plot selector does not affect the printed text
                shown = [item for index, item in enumerate(argv)
                         if item != "--output_type" and argv[max(index - 1, 0)] != "--output_type"]
                with self.subTest(experiment=number, argv=argv):
                    self.assertEqual(head(P(*shown)), reference)
        self.assertEqual(len(experiment_commands(2)), 1)
        self.assertEqual(len(experiment_commands(3)), 2)

    def test_experiment_3_the_small_boost_hardly_moves_the_hodograph_and_the_large_one_does(self):
        base = run_result()
        extent_v = max(max(base.vA), max(base.vB)) - min(min(base.vA), min(base.vB))
        extent_u = max(max(base.uA), max(base.uB)) - min(min(base.uA), min(base.uB))
        self.assertLess(500.0, 0.01 * max(extent_v, extent_u))
        self.assertGreater(30000.0, 0.2 * extent_v)
        moved = run_result(vInitA=30000.0, vInitB=30000.0)
        for old, new in ((base.vA, moved.vA), (base.vB, moved.vB)):
            self.assertAlmostEqual(new[0] - old[0], 30000.0, delta=1e-9)
            self.assertAlmostEqual(min(new) - min(old), 30000.0, delta=1.0)
        self.assertPhrase(
            3,
            "A boost of 500 m/s moves the curves by less than 1% of the width of the plot",
            "whose boost moves each curve 30000 m/s to the right",
        )

    # ---------------------------------------------------------- Experiment 4
    def test_experiment_4_hodograph_circles_and_the_unbound_arc(self):
        commands = experiment_commands(4)
        self.assertEqual(len(commands), 3)
        self.assertEqual(commands[2], ["--uInitA", "60000", "--uInitB", "-60000", "--no-stop_after_one_orbit",
                                       "--max_steps", "1000", "--output_type", "velocity_space"])
        elements = elements_for(uInitA=60000.0, uInitB=-60000.0)
        mu = physics.G * 4e30
        h = 9.2e10 * 120000.0
        radius, offset = 0.5 * mu / h, 0.5 * mu / h * elements.eccentricity
        self.assertEqual((f"{radius:.0f}", f"{offset:.0f}", f"{elements.eccentricity:.4f}"),
                         ("12091", "47909", "3.9623"))
        self.assertGreater(offset, radius)          # the origin is outside the circle
        result = run_result(uInitA=60000.0, uInitB=-60000.0, stop_after_one_orbit=False, max_steps=1000)
        distances = hodograph_arc(result, 0.0, 60000.0 - radius)
        self.assertAlmostEqual(min(distances) / radius, 1.0, delta=1e-3)
        self.assertAlmostEqual(max(distances) / radius, 1.0, delta=1e-3)
        # only an arc: the velocity turns through less than half of the circle
        turned = math.atan2(result.vA[-1], result.uA[-1] - (60000.0 - radius))
        self.assertLess(abs(turned), math.pi / 2)
        self.assertPhrase(
            4,
            "the circular run gives one circle of radius 26934.5 m/s about the origin, and the default run gives "
            "two circles of radius 55805 m/s whose centres are 42805 m/s from the origin, a ratio of 0.76705, "
            "the eccentricity",
            "Each of its curves is only an arc of a circle whose centre, 47909 m/s from the origin, lies farther "
            "from it than the circle's radius, 12091 m/s, as an eccentricity of 3.9623 requires",
        )

    # ---------------------------------------------------------- Experiment 5
    def test_experiment_5_check_numbers_for_the_two_to_one_mass_ratio(self):
        printed = P(*ExperimentQuotedNumberTests.EXP_5)
        block_a, block_b = printed_body_block(printed, "A"), printed_body_block(printed, "B")
        self.assertEqual(experiment_commands(5), [list(ExperimentQuotedNumberTests.EXP_5)])
        self.assertIn("semi-major axis: 3.0666e+10 m", block_a[0])
        self.assertIn("semi-major axis: 6.1332e+10 m", block_b[0])
        self.assertIn("period: 143.41 days", block_a[0])
        self.assertIn("period: 143.41 days", block_b[0])
        self.assertPhrase(
            5,
            "the semi-major axes printed under Body A and Body B are 3.0666e+10 and 6.1332e+10 m, in the ratio "
            "2.0000, the inverse of the mass ratio, and the two periods are equal, 143.41 days",
        )

    EXP_5 = ("--MA", "2e30", "--MB", "1e30", "--xInitA", "3.0666666667e10", "--xInitB", "-6.1333333333e10",
             "--uInitA", "15550.5265", "--uInitB", "-31101.053")

    # ---------------------------------------------------------- Experiment 6
    def test_experiment_6_bound_parabolic_and_unbound_check_numbers(self):
        escape = 38091.13784870039
        commands = experiment_commands(6)
        self.assertEqual([command[1] for command in commands], ["38091.14", "60000", "38091.13784870039"])
        self.assertEqual(f"{escape:.7g}", "38091.14")
        self.assertAlmostEqual(escape / (0.5 * math.sqrt(2 * physics.G * 4e30 / 9.2e10)), 1.0, delta=1e-15)
        default_rows = R()
        self.assertEqual(um(default_rows[0.0][4]), "−2.5639e+39")
        rounded = P("--uInitA", "38091.14", "--uInitB", "-38091.14", "--no-stop_after_one_orbit",
                    "--max_steps", "1000")
        self.assertIn("semi-major axis: -2.0362e+17 m", printed_body_block(rounded, "A")[0])
        self.assertEqual(printed_fields(rounded)["Initial Keplerian orbit"], "hyperbolic; eccentricity: 1")
        parabolic_argv = ("--uInitA", repr(escape), "--uInitB", repr(-escape), "--no-stop_after_one_orbit",
                          "--max_steps", "1000")
        parabolic = P(*parabolic_argv)
        block = printed_body_block(parabolic, "A")
        self.assertEqual(printed_fields(parabolic)["Initial Keplerian orbit"], "parabolic; eccentricity: 1")
        self.assertIn("Periapsis: 4.6e+10 m; apoapsis: undefined", block[1])
        row = printed_rows(parabolic)[0.0]
        self.assertEqual(um(row[4]), "−6.0446e+23")
        self.assertEqual(f"{-float(row[4]) / float(row[3]) * 1e16:.0f}", "2")
        self.assertEqual(printed_fields(P("--uInitA", "60000", "--uInitB", "-60000", "--no-stop_after_one_orbit",
                                          "--max_steps", "1000"))["Initial Keplerian orbit"],
                         "hyperbolic; eccentricity: 3.9623")
        self.assertPhrase(
            6,
            "The bound case is the default run, with total energy −2.5639e+39 J",
            "The speed 38091.14 is the escape speed rounded to seven digits and lies 1.1×10 −7 of the kinetic "
            "energy above it",
            "although its eccentricity prints as 1 (it is 1.0000002), and it prints a semi-major axis of −2.0362e+17 m",
            "Its total energy, −6.0446e+23 J, is 2×10 −16 of the kinetic energy",
            "eccentricity 3.9623",
        )

    # ---------------------------------------------------------- Experiment 7
    PLANETS = {
        "Jupiter": dict(mass=1.898e27, separation=7.785e11, dt=20000.0),
        "Saturn": dict(mass=5.683e26, separation=1.434e12, dt=50000.0),
        "Earth": dict(mass=5.972e24, separation=1.496e11, dt=20000.0),
    }

    @staticmethod
    def reflex_setup(mass, separation, sun=1.989e30):
        total = sun + mass
        v_rel = math.sqrt(physics.G * total / separation)
        return dict(MA=sun, MB=mass, xInitA=-separation * mass / total, xInitB=separation * sun / total,
                    yInitA=0.0, yInitB=0.0, vInitA=0.0, vInitB=0.0,
                    uInitA=-v_rel * mass / total, uInitB=v_rel * sun / total)

    def test_experiment_7_setup_formulas_reproduce_the_command_shown(self):
        (argv,) = experiment_commands(7)
        args = entry.parse_args(argv)
        setup = self.reflex_setup(**{k: v for k, v in self.PLANETS["Jupiter"].items() if k != "dt"})
        for key in ("xInitA", "xInitB", "uInitA", "uInitB"):
            with self.subTest(key=key):
                self.assertAlmostEqual(getattr(args, key) / setup[key], 1.0, delta=1e-9)
        self.assertEqual((args.MA, args.MB, args.dt, args.max_steps), (1.989e30, 1.898e27, 20000.0, 30000))

    def test_experiment_7_reflex_values_read_from_the_printed_summary(self):
        values = {}
        for name, planet in self.PLANETS.items():
            setup = self.reflex_setup(planet["mass"], planet["separation"])
            argv = [item for key, value in setup.items() for item in (f"--{key}", repr(value))]
            argv += ["--dt", repr(planet["dt"]), "--max_steps", "30000"]
            printed = P(*argv)
            block = printed_body_block(printed, "A")
            values[name] = (block[0].split("semi-major axis: ")[1].split(";")[0],
                            block[2].split("periapsis: ")[1].split(";")[0],
                            block[0].split("period: ")[1], printed_fields(printed))
        self.assertEqual(values["Jupiter"][:2], ("7.4217e+08 m", "12.455 m/s"))
        self.assertEqual(values["Saturn"][:2], ("4.0961e+08 m", "2.7487 m/s"))
        self.assertEqual(values["Earth"][:2], ("4.4917e+05 m", "0.089441 m/s"))
        self.assertEqual(values["Saturn"][2], "10837 days")
        self.assertEqual(values["Saturn"][3]["Termination"], "one relative revolution")
        self.assertEqual(f"{7.4217e8 / 6.96e8:.3f}", "1.066")
        # Jupiter's reflex orbit is the largest in radius and in speed, Earth's the smallest
        self.assertTrue(printed_number(values["Jupiter"][0]) > printed_number(values["Saturn"][0])
                        > printed_number(values["Earth"][0]))
        # the mass-ratio scaling: radius ∝ M_planet × separation^(1/2) at fixed central mass
        # (the reflex speed scales with M_planet / sqrt(separation))
        self.assertAlmostEqual(printed_number(values["Jupiter"][1]) / printed_number(values["Saturn"][1]),
                               (1.898e27 / 5.683e26) * math.sqrt(1.434e12 / 7.785e11), delta=0.01)
        # dt 20000 with max_steps 30000 cannot finish Saturn's orbit
        setup = self.reflex_setup(5.683e26, 1.434e12)
        short = run_result(dt=20000.0, max_steps=30000, **setup)
        self.assertFalse(short.completed_orbit)
        self.assertEqual(f"{abs(short.total_angle_rad) / (2 * math.pi):.2f}", "0.64")
        self.assertPhrase(
            7,
            "For Jupiter they are 7.4217e+08 m, which is 1.066 solar radii, and 12.455 m/s",
            "For Saturn they are 4.0961e+08 m and 2.7487 m/s",
            "Saturn's orbit takes 10837 days, so use --dt 50000",
            "the run ends at max_steps after 0.64 of a revolution",
            "For Earth, with --dt 20000 , they are 4.4917e+05 m and 0.089441 m/s",
        )

    # ---------------------------------------------------------- Experiment 8
    def snippet_source(self):
        code = html_module.unescape(re.sub(r"<[^>]+>", "", next(
            block for block in re.findall(r"<pre[^>]*>(.*?)</pre>", experiment_html(8), re.DOTALL)
            if "integrate_binary" in block)))
        self.assertIn("from driver_binary import integrate_binary", code)
        return code

    def test_experiment_8_snippet_runs_and_prints_the_quoted_departures(self):
        code = self.snippet_source()
        calls = []

        def cached_integrate_binary(**settings):
            # The same integration, served from the suite's cache; the arguments
            # must all be keywords that the real function accepts.
            calls.append(settings)
            return run_result(**settings)

        stand_in = type(sys)("driver_binary")
        stand_in.integrate_binary = cached_integrate_binary
        with mock.patch.dict(sys.modules, {"driver_binary": stand_in}):
            with redirect_stdout(io.StringIO()) as out:
                exec(compile(code, "experiment 8 snippet", "exec"), {"__name__": "snippet"})
        printed = [float(line) for line in out.getvalue().split()]
        self.assertEqual(len(printed), 6)
        self.assertEqual(len(calls), 6)
        self.assertEqual([f"{abs(value):.4e}" for value in printed[:3]], ["1.2853e-03", "4.5163e-04", "2.9623e-04"])
        self.assertEqual(f"{100 * abs(printed[3]):.1f}", "25.9")
        self.assertEqual([f"{abs(value):.4e}" for value in printed[4:]], ["2.0184e-04", "1.0371e-04"])
        # each printed value is the suite's own largest departure, measured from the first stored E
        for settings, value in zip(calls, printed):
            with self.subTest(settings=settings):
                # exact equality: the same subtraction and division, so a different reference
                # energy (a change of about 4e-13 in relative terms) would show
                self.assertEqual(abs(value), abs(largest_departure(run_result(**settings))))
        self.assertIn("so the first line begins 0.00128", experiment_text(8))

    def test_experiment_8_snippet_first_line_runs_against_the_real_integrator(self):
        code = self.snippet_source().split("print(largest_departure())")[0] + "print(largest_departure())"
        with redirect_stdout(io.StringIO()) as out:
            exec(compile(code, "experiment 8 snippet", "exec"), {"__name__": "snippet"})
        self.assertEqual(f"{float(out.getvalue()):.4e}", "1.2853e-03")
        self.assertTrue(out.getvalue().startswith("0.00128"))

    def test_experiment_8_three_orbit_comparison(self):
        def worst(**changes):
            result = run_result(stop_after_one_orbit=False, **changes)
            return result, largest_departure(result)

        first, first_departure = worst(dt=2000.0, max_steps=7000)
        second, second_departure = worst(dt=1000.0, max_steps=14000)
        _, strict_departure = worst(dt=1000.0, max_steps=14000, eps2=1e-8)
        self.assertEqual((f"{abs(first_departure):.4e}", f"{abs(second_departure):.4e}",
                          f"{abs(strict_departure):.4e}"), ("1.2853e-03", "4.5163e-04", "2.9623e-04"))
        self.assertEqual((f"{abs(first.total_angle_rad) / (2 * math.pi):.4f}",
                          f"{abs(second.total_angle_rad) / (2 * math.pi):.4f}"), ("3.0136", "3.0133"))
        self.assertAlmostEqual(first.times[-1], 1.4e7)
        self.assertAlmostEqual(second.times[-1], 1.4e7)
        self.assertEqual(f"{first.times[-1] / 86400 / 10:.1f}", "16.2")
        # the largest departure is at the third closest approach
        separations = separation_list(first)
        minima = [i for i in range(1, len(separations) - 1)
                  if separations[i] < separations[i - 1] and separations[i] <= separations[i + 1]]
        self.assertEqual([f"{first.times[i] / 86400:.1f}" for i in minima], ["26.4", "79.3", "132.2"])
        worst_index = max(range(len(first.E)), key=lambda i: abs(first.E[i] - first.E[0]))
        self.assertLess(abs(first.times[worst_index] - first.times[minima[2]]), 6000.0)
        self.assertGreater(abs(first_departure) / abs(second_departure), 2.5)
        self.assertPhrase(
            8,
            "Neither the plot nor the printed table can give that maximum",
            "the eleven rows of the table fall at multiples of 16.2 days, while the closest approaches are at "
            "days 26.4, 79.3 and 132.2",
            "The first, for the default settings, is 1.2853e-03, reached at the third closest approach",
            "With dt 1000 and max_steps 14000 it is 4.5163e-04, and with eps2 1e-8 as well it is 2.9623e-04",
            "The printed Total revolutions is 3.0136 for the first run and 3.0133 for the second",
        )

    def test_experiment_8_very_eccentric_orbit(self):
        argv = ("--uInitA", "1000", "--uInitB", "-1000", "--max_steps", "400000")
        tight = argv + ("--eps1", "0.001")
        default_fields, tight_fields = F(*argv), F(*tight)
        self.assertEqual((default_fields["Accepted steps"], default_fields["Total time"]), ("2456", "43.231 days"))
        self.assertEqual((tight_fields["Accepted steps"], tight_fields["Total time"]), ("43303", "43.97 days"))
        default_worst = largest_departure(run_result(uInitA=1000.0, uInitB=-1000.0, max_steps=400000))
        tight_result = run_result(uInitA=1000.0, uInitB=-1000.0, max_steps=400000, eps1=0.001)
        strict_result = run_result(uInitA=1000.0, uInitB=-1000.0, max_steps=400000, eps1=0.001, eps2=1e-8)
        self.assertEqual(f"{100 * abs(default_worst):.1f}", "25.9")
        self.assertEqual((f"{abs(largest_departure(tight_result)):.4e}",
                          f"{abs(largest_departure(strict_result)):.4e}"), ("2.0184e-04", "1.0371e-04"))
        self.assertEqual(f"{elements_for(uInitA=1000.0, uInitB=-1000.0).period / 86400:.3f}", "43.956")
        self.assertEqual([command for command in experiment_commands(8)[-2:]],
                         [list(argv), list(tight)])
        self.assertPhrase(
            8,
            "The eccentricity is 0.99862 and the Keplerian period is 43.956 days",
            "the run takes 2456 steps and stops after 43.231 days",
            "reports a largest departure of 25.9%",
            "With eps1 = 0.001 the run takes 43303 steps, stops after 43.97 days, and the largest departure, on the "
            "fifth line, is 2.0184e-04",
            "Add --eps2 1e-8 as well and it falls to 1.0371e-04, the sixth line",
        )



# ---------------------------------------------------------------------------
# Reference-type content that no run can confirm: the commands a student is told
# to type, table headings, row labels, cross-references and the equations
# themselves.  These are frozen so that an edit anywhere shows up as a failure
# with a readable difference.  A deliberate edit means updating the table here
# in the same change; the run-derived numbers are checked separately.
# ---------------------------------------------------------------------------

EXPECTED_COMMANDS = {'beat0': ['python main.py'],
 'beat1': ['python main.py --MA 2e30 --MB 1e30 --xInitA 3.0666666667e10 --xInitB -6.1333333333e10 --uInitA '
           '15550.5265 --uInitB -31101.053'],
 'beat2': ['python main.py --xInitA 5e10 --xInitB -4.2e10 --yInitA 1e10 --yInitB 1e10',
           'python main.py --vInitA 60000 --vInitB 60000'],
 'beat3': ['python main.py --output_type energy_vs_time',
           'python main.py --uInitA 38091.14 --uInitB -38091.14 --no-stop_after_one_orbit --max_steps 1000 '
           '--output_type energy_vs_time',
           'python main.py --uInitA 60000 --uInitB -60000 --no-stop_after_one_orbit --max_steps 1000 '
           '--output_type energy_vs_time'],
 'beat4': ['python main.py --uInitA 26934.5 --uInitB -26934.5 --output_type velocity_space',
           'python main.py --output_type velocity_space'],
 'beat5': ['python main.py --eps2 1e-8', 'python main.py --eps2 0.5'],
 'beat6': ['python main.py --dt 20000', 'python main.py --dt 20000 --eps1 0.5'],
 'beat7': ['python main.py --dt 1000', 'python main.py --dt 500', 'python main.py --dt 1000 --eps2 1e-8'],
 'quickstart': ['python main.py', 'python main.py --output_type velocity_space', 'python main.py --help'],
 'exp1': ['python main.py',
          'python main.py --uInitA 26934.5 --uInitB -26934.5',
          'python main.py --uInitA 20000 --uInitB -20000',
          'python main.py --uInitA 30000 --uInitB -30000'],
 'exp2': ['python main.py --xInitA 5e10 --xInitB -4.2e10 --yInitA 1e10 --yInitB 1e10'],
 'exp3': ['python main.py --vInitA 500 --vInitB 500 --uInitA 13500 --uInitB -12500 --output_type '
          'velocity_space',
          'python main.py --vInitA 30000 --vInitB 30000 --output_type velocity_space'],
 'exp4': ['python main.py --uInitA 26934.5 --uInitB -26934.5 --output_type velocity_space',
          'python main.py --output_type velocity_space',
          'python main.py --uInitA 60000 --uInitB -60000 --no-stop_after_one_orbit --max_steps 1000 '
          '--output_type velocity_space'],
 'exp5': ['python main.py --MA 2e30 --MB 1e30 --xInitA 3.0666666667e10 --xInitB -6.1333333333e10 --uInitA '
          '15550.5265 --uInitB -31101.053'],
 'exp6': ['python main.py --uInitA 38091.14 --uInitB -38091.14 --no-stop_after_one_orbit --max_steps 1000 '
          '--output_type energy_vs_time',
          'python main.py --uInitA 60000 --uInitB -60000 --no-stop_after_one_orbit --max_steps 1000 '
          '--output_type energy_vs_time',
          'python main.py --uInitA 38091.13784870039 --uInitB -38091.13784870039 --no-stop_after_one_orbit '
          '--max_steps 1000 --output_type energy_vs_time'],
 'exp7': ['python main.py --MA 1.989e30 --MB 1.898e27 --xInitA -7.42174134486e8 --xInitB 7.77757825866e11 '
          '--uInitA -12.4550437189 --uInitB 13052.2033493 --dt 20000 --max_steps 30000'],
 'exp8': ['python main.py --no-stop_after_one_orbit --dt 2000 --max_steps 7000 --output_type energy_vs_time',
          'python main.py --no-stop_after_one_orbit --dt 1000 --max_steps 14000 --output_type energy_vs_time',
          'python main.py --uInitA 1000 --uInitB -1000 --max_steps 400000',
          'python main.py --uInitA 1000 --uInitB -1000 --max_steps 400000 --eps1 0.001']}

EXPECTED_TABLE_HEADINGS = {'beat7': ['dt (s)',
           'Accepted steps eps2 = 1e-4 / 1e-8',
           'Row 0.5 eps2 = 1e-4',
           'Row 0.5 eps2 = 1e-8',
           'Row 1.0 eps2 = 1e-4',
           'Row 1.0 eps2 = 1e-8'],
 'equations': ['Equation', 'Kind', 'What it says', 'Beat', 'Where in the code'],
 'algorithm': ['Parameter', 'Default', 'Role'],
 'parameters': ['Option', 'Default', 'Unit', 'Description'],
 'experiments': ['Body', 'Mass (kg)', 'Mean Sun–planet separation (m)']}

EXPECTED_CROSS_REFERENCES = {'overview': ['Beats 5 to 7'],
 'beats': ['Beats 4 and 7',
           'Beat 0',
           'Beat 1',
           'Beat 2',
           'Beat 3',
           'Beat 4',
           'Beats 5 to 7',
           'Eq. (1)',
           'Eqs. (8) to (10)'],
 'beat0': ['Beat 0', 'Eq. (1)', 'Eq. (1)', 'Beat 2', 'Beat 1', 'Experiment 1'],
 'beat1': ['Beat 1',
           'Equation (1)',
           'Beat 5)',
           'Eq. (2)',
           'Eq. (3)',
           'Beat 2',
           'Experiment 5',
           'Experiment 7'],
 'beat2': ['Beat 2',
           'Equation (1)',
           'Eqs. (2) and (3)',
           'Beat 0',
           'Beat 0',
           'Eqs. (2) and (3)',
           'Beat 3)',
           'Beat 5)',
           'Experiment 3',
           'Experiment 2',
           'Experiment 3'],
 'beat3': ['Beat 3', 'Eq. (1)', 'Beat 4)', 'Equation (6)', 'Eq. (1)', 'Beats 5 to 7', 'Experiment 6'],
 'beat4': ['Beat 4', 'Beat 3', 'Eq. (1)', 'Beat 2', 'Experiment 3)', 'Experiment 1', 'Experiment 4'],
 'beat5': ['Beat 5',
           'Eqs. (2) and (3)',
           'Eq. (9)',
           'Eq. (9)',
           'Eq. (9)',
           'Eq. (9)',
           'Beats 6 and 7',
           'Beats 5 to 7',
           'Experiment 8'],
 'beat6': ['Beat 6', 'Beat 0', 'Eq. (8)', 'Beat 1)', 'Experiment 8'],
 'beat7': ['Beat 7',
           'Beats 5 and 6',
           'Beat 4',
           'Beat 5)',
           'Beat 5',
           'Experiment 8',
           'Eq. (1)',
           'Beat 6',
           'Experiment 8'],
 'equations': ['Beat 2', 'Beat 4', 'Beat 4', 'Beat 5'],
 'algorithm': ['Beats 4 to 7',
               'Equations (8) and (9)',
               'Beat 5',
               'Eq. (10)',
               'Beat 6',
               'Beat 4',
               'Eqs. (2) and (3)',
               'Eq. (8)',
               'Eq. (10)',
               'Eq. (9)',
               'Eqs. (5) to (7)',
               'Beat 6)',
               'Beat 7)',
               'Beat 7)',
               'Beats 5 and 7',
               'Beat 7)',
               'Beat 7',
               'Experiment 8)'],
 'output-types': ['Beat 3)'],
 'summary': ['Beat 6)',
             'Beat 4',
             'Beat 7)',
             'Beat 6)',
             'Beat 2)',
             'Experiment 6)',
             'Beat 1)',
             'Beat 3',
             'Experiment 8)'],
 'experiments': ['Experiments 1',
                 'Beat 2)',
                 'Beat 2)',
                 'Experiment 1',
                 'Beat 4',
                 'Experiment 6',
                 'Beat 1)',
                 'Beat 3)',
                 'Beat 7',
                 'Beat 7)'],
 'related': ['Experiments 5 and 7)']}

EXPECTED_ROW_LABELS = {'beat2': ['row 0.0', 'row 0.0', 'row 0.5'],
 'beat3': ['row 0.0', 'row 0.0'],
 'beat5': ['rows 0.5 and 1.0',
           'Row 0.5',
           'row 0.5',
           'row 1.0',
           'Row 0.5',
           'row 0.5',
           'row 1.0',
           'row 1.0',
           'row 0.5'],
 'beat6': ['row 0.5', 'row 0.5', 'row 0.5', 'Row 0.5'],
 'beat7': ['rows 0.5 and 1.0',
           'row 0.5',
           'row 1.0',
           'row 0.5',
           'row 1.0',
           'row 0.5',
           'row 1.0',
           'Row 0.5',
           'Row 0.5',
           'Row 1.0',
           'Row 1.0',
           'row 1.0'],
 'summary': ['fraction 0.0 through 1.0']}

EXPECTED_DISPLAY_EQUATIONS = [('beat0', ['F=\\frac{GM_AM_B}{r^2}, \\qquad r=\\sqrt{(x_A-x_B)^2+(y_A-y_B)^2}.']),
 ('beat1',
  ['\\ddot x_A \\;=\\; -\\frac{G\\,M_B\\,\\Delta x}{r^3}, \\qquad \\ddot y_A \\;=\\; '
   '-\\frac{G\\,M_B\\,\\Delta y}{r^3}',
   '\\ddot x_B \\;=\\; +\\frac{G\\,M_A\\,\\Delta x}{r^3}, \\qquad \\ddot y_B \\;=\\; '
   '+\\frac{G\\,M_A\\,\\Delta y}{r^3}',
   '\\mathbf R_{\\rm CM}= \\frac{M_A\\mathbf r_A+M_B\\mathbf r_B}{M_A+M_B}, \\qquad \\mathbf V_{\\rm CM}= '
   '\\frac{M_A\\mathbf v_A+M_B\\mathbf v_B}{M_A+M_B}.']),
 ('beat2', ['K \\;=\\; \\tfrac{1}{2}\\,(M_A+M_B)\\,V_{\\rm CM}^{\\,2} \\;+\\; K_{\\rm CM\\text{-}frame}']),
 ('beat3',
  ['K \\;=\\; \\tfrac{1}{2}\\,M_A\\!\\left(v_A^2 + u_A^2\\right) + \\tfrac{1}{2}\\,M_B\\!\\left(v_B^2 + '
   'u_B^2\\right)',
   'U \\;=\\; -\\frac{G\\,M_A\\,M_B}{r}',
   'E \\;=\\; K + U']),
 ('beat4',
  ['\\ell_A \\;=\\; (\\mathbf r_A-\\mathbf R_{\\rm CM})\\times(\\mathbf v_A-\\mathbf V_{\\rm CM}) \\;=\\; '
   '\\text{constant}, \\qquad r_p\\,v_p \\;=\\; r_a\\,v_a \\;=\\; \\ell_A',
   'T \\;=\\; 2\\pi\\sqrt{\\frac{a^{3}}{G\\,(M_A+M_B)}}, \\qquad a \\;=\\; a_A + a_B',
   'R_v \\;=\\; \\tfrac12\\,(v_p+v_a), \\qquad d_v \\;=\\; \\tfrac12\\,(v_p-v_a), \\qquad e \\;=\\; '
   '\\frac{d_v}{R_v} \\;=\\; \\frac{v_p-v_a}{v_p+v_a}',
   '\\Theta \\;=\\; \\sum_n \\Delta\\theta_n, \\qquad \\Delta\\theta_n \\;=\\; '
   '\\operatorname{atan2}\\!\\left(\\mathbf r_n\\times\\mathbf r_{n+1},\\;\\mathbf r_n\\cdot\\mathbf '
   'r_{n+1}\\right), \\qquad \\text{stop when } |\\Theta|\\ge 2\\pi']),
 ('beat5',
  ['\\tilde{\\mathbf r}_A=\\mathbf r_A^n+\\mathbf v_A^n\\Delta t, \\qquad \\tilde{\\mathbf v}_A=\\mathbf '
   'v_A^n+\\mathbf a_A^n\\Delta t,',
   '\\begin{aligned} \\mathbf v_A^{n+1} &= \\mathbf v_A^n + \\frac{\\mathbf a_A^n+\\mathbf '
   'a_A^{n+1}}{2}\\,\\Delta t,\\\\[2pt] \\mathbf r_A^{n+1} &= \\mathbf r_A^n + \\frac{\\mathbf '
   'v_A^n+\\mathbf v_A^{n+1}}{2}\\,\\Delta t. \\end{aligned}',
   '\\delta_A^{(k)} \\;=\\; \\frac{\\left\\lVert\\mathbf v_A^{(k)}-\\mathbf v_A^{(k-1)}\\right\\rVert} '
   '{\\max\\!\\left(\\left\\lVert\\mathbf v_A^{(k)}\\right\\rVert,\\left\\lVert\\mathbf '
   'v_A^{(k-1)}\\right\\rVert\\right)}, \\qquad \\text{accept when } '
   '\\max\\!\\left(\\delta_A^{(k)},\\delta_B^{(k)}\\right)\\lt\\varepsilon_2']),
 ('beat6',
  ['\\Delta_a= \\frac{\\left\\lVert\\tilde{\\mathbf a}_A-\\mathbf a_A^n\\right\\rVert} '
   '{\\max\\!\\left(\\left\\lVert\\tilde{\\mathbf a}_A\\right\\rVert, \\left\\lVert\\mathbf '
   'a_A^n\\right\\rVert\\right)}.'])]

EXPECTED_INLINE_MATH_WITH_NUMBERS = {'beat0': ['G=6.67430\\times10^{-11}\\,\\mathrm{m^3\\,kg^{-1}\\,s^{-2}}'],
 'beat1': ['M_A/M_B=2', 'M_A\\mathbf a_A+M_B\\mathbf a_B=0'],
 'beat2': ['\\tfrac12(M_A+M_B)V^2',
           '\\tfrac12\\times4\\times10^{30}\\times(6\\times10^4)^2',
           '3\\times10^{36}',
           '|E_0|',
           '\\mathbf V_0',
           '\\mathbf r_i(t)\\to\\mathbf r_i(t)+\\mathbf V_0t',
           '\\mathbf V_0t'],
 'beat3': ['-6.6743\\times10^{-11}\\times(2\\times10^{30})^2',
           '9.2\\times10^{10}',
           'E<0',
           'E>0',
           'E=0',
           'E<0',
           '|E_0|',
           'E_0'],
 'beat4': ['a(1-e)',
           'a(1+e)',
           'v_p/v_a=(1+e)/(1-e)',
           '(v_p-v_a)/2',
           'e<1',
           'e=1',
           'e>1',
           '\\mathbf V_0',
           '\\mathbf V_0',
           '2\\pi'],
 'beat5': ['\\mathbf a^{n+1}', '\\mathbf r^{n+1}', '\\mathbf v^{(0)}'],
 'beat6': ['\\Delta_a>\\varepsilon_1', '\\varepsilon_1'],
 'algorithm': ['\\Delta_a>\\varepsilon_1', '2\\pi', '0.05', '10^{-4}'],
 'quickstart': ['M_A = M_B = 2\\times10^{30}\\,\\mathrm{kg}',
                'x = \\pm\\,4.6\\times10^{10}\\,\\mathrm{m}',
                '\\pm\\,1.3\\times10^{4}\\,\\mathrm{m\\,s^{-1}}'],
 'summary': ['E_0'],
 'experiments': ['2a',
                 'v_{\\rm circ}=\\sqrt{GM/(4a)}',
                 '2.69\\times10^4',
                 'u_A=+2.69\\times10^4',
                 'u_B=-2.69\\times10^4',
                 'M_A\\mathbf v_A+M_B\\mathbf v_B=0',
                 'E<0',
                 'E\\approx0',
                 'E>0',
                 'E_0',
                 'M_A\\mathbf v_A+M_B\\mathbf v_B=0',
                 '1.989\\times10^{30}',
                 '1.898\\times10^{27}',
                 '7.785\\times10^{11}',
                 '5.683\\times10^{26}',
                 '1.434\\times10^{12}',
                 '5.972\\times10^{24}',
                 '1.496\\times10^{11}',
                 '6.96\\times10^8',
                 '1.4\\times10^7',
                 '\\max|E(t)-E(0)|/|E(0)|']}


ALL_SECTIONS = (["overview", "beats"] + [f"beat{n}" for n in range(8)]
                + ["equations", "algorithm", "modules", "quickstart", "parameters", "output-types",
                   "summary", "experiments", "related"])
CROSS_REFERENCE = re.compile(
    r"\b(Beats?|Experiments?|Eqs?\.|Equations?)\s+(\(?\d+\)?(?:\s*(?:,|and|to)\s*\(?\d+\)?)*)")
ROW_LABEL = re.compile(r"\b(?:[Rr]ows?|fraction)\s+(\d\.\d(?:\s*(?:,|and|through)\s*\d\.\d)*)")


def nonempty(mapping):
    return {key: value for key, value in mapping.items() if value}


def frozen_commands():
    found = {}
    for section in ALL_SECTIONS:
        body = section_html(HELP_HTML, section)
        if section == "experiments":
            for number in range(1, 9):
                found[f"exp{number}"] = experiment_lines(experiment_html(number))
        else:
            found[section] = experiment_lines(body)
    return nonempty(found)


def experiment_lines(fragment):
    """Every ``python main.py`` line in the code blocks of a fragment, whitespace collapsed."""
    lines = []
    for block in re.findall(r"<pre[^>]*>(.*?)</pre>", fragment, re.DOTALL):
        text = re.sub(r"\\\n\s*", " ", html_module.unescape(re.sub(r"<[^>]+>", "", block)))
        lines += [" ".join(line.split()) for line in text.splitlines()
                  if line.strip().startswith("python main.py")]
    return lines


@needs_beats
class FrozenReferenceContentTests(unittest.TestCase):
    """Content that no run can confirm: what the student is told to type, headings, labels,
    cross-references and the equations.  Frozen so that any edit is seen (see the tables)."""

    maxDiff = None

    def test_the_commands_shown_are_the_commands_the_quoted_numbers_came_from(self):
        self.assertEqual(frozen_commands(), EXPECTED_COMMANDS)

    def test_table_headings(self):
        found = {section: [html_text(cell) for cell in
                           re.findall(r"<th[^>]*>(.*?)</th>", section_html(HELP_HTML, section), re.DOTALL)]
                 for section in ALL_SECTIONS}
        self.assertEqual(nonempty(found), EXPECTED_TABLE_HEADINGS)

    def test_cross_references(self):
        found = {section: [match.group(0) for match in
                           CROSS_REFERENCE.finditer(html_text(section_html(HELP_HTML, section)))]
                 for section in ALL_SECTIONS}
        self.assertEqual(nonempty(found), EXPECTED_CROSS_REFERENCES)

    def test_row_labels_of_the_energy_table(self):
        found = {section: [match.group(0) for match in
                           ROW_LABEL.finditer(html_text(section_html(HELP_HTML, section)))]
                 for section in ALL_SECTIONS}
        self.assertEqual(nonempty(found), EXPECTED_ROW_LABELS)
        for labels in found.values():
            for label in labels:
                for fraction in re.findall(r"\d\.\d", label):
                    self.assertIn(fraction, {f"{j / 10:.1f}" for j in range(11)})

    def test_display_equations(self):
        found = []
        for section in ALL_SECTIONS:
            blocks = [" ".join(block.split()) for block in
                      re.findall(r"\\\[(.*?)\\\]", section_html(HELP_HTML, section), re.DOTALL)]
            if blocks:
                found.append((section, blocks))
        self.assertEqual(found, EXPECTED_DISPLAY_EQUATIONS)

    def test_inline_mathematics_that_contains_a_number(self):
        found = {}
        for section in ALL_SECTIONS:
            body = re.sub(r"\\\[(.*?)\\\]", "", section_html(HELP_HTML, section), flags=re.DOTALL)
            found[section] = [" ".join(html_module.unescape(text).split())
                              for text in re.findall(r"\\\((.*?)\\\)", body, re.DOTALL)
                              if re.search(r"\d", text)]
        self.assertEqual(nonempty(found), EXPECTED_INLINE_MATH_WITH_NUMBERS)

    def test_every_reference_names_something_that_exists(self):
        for section in ALL_SECTIONS:
            for match in CROSS_REFERENCE.finditer(html_text(section_html(HELP_HTML, section))):
                numbers = [int(n) for n in re.findall(r"\d+", match.group(2))]
                limit = {"Beat": 7, "Experiment": 8}.get(match.group(1).rstrip("s."), 10)
                lowest = 1 if match.group(1).startswith(("Exp", "Eq")) else 0
                with self.subTest(section=section, reference=match.group(0)):
                    self.assertTrue(all(lowest <= n <= limit for n in numbers), numbers)

    def test_an_equation_is_never_cited_before_the_beat_that_introduces_it(self):
        index = section_html(HELP_HTML, "equations")
        introduced = {int(number): int(beat) for number, beat in re.findall(
            r"<td>\((\d+)\)</td><td>.*?</td><td>.*?</td><td>(\d)</td>", index, re.DOTALL)}
        for beat in range(8):
            for match in CROSS_REFERENCE.finditer(html_text(section_html(HELP_HTML, f"beat{beat}"))):
                if match.group(1).startswith("Eq"):
                    for number in (int(n) for n in re.findall(r"\d+", match.group(2))):
                        with self.subTest(beat=beat, equation=number):
                            self.assertLessEqual(introduced[number], beat)



# ---------------------------------------------------------------------------
# Remaining quoted numbers: second mentions, settings, and statements about
# the method that a run can confirm.
# ---------------------------------------------------------------------------

@needs_beats
class ProsePinTests(unittest.TestCase):
    """Small statements of number and setting, each tied to the run or constant behind it."""

    def assertIn_section(self, section, *phrases):
        text = html_text(section_html(HELP_HTML, section))
        for phrase in phrases:
            with self.subTest(section=section, phrase=phrase):
                self.assertIn(phrase, text)

    def test_the_default_masses_and_settings_the_text_quotes(self):
        self.assertEqual((DEFAULTS["MA"], DEFAULTS["MB"]), (2e30, 2e30))
        self.assertEqual((DEFAULTS["eps1"], DEFAULTS["eps2"], DEFAULTS["dt"]), (0.05, 1e-4, 2000.0))
        self.assertIn_section("beat0", "each with about the mass of the Sun (2×10 30 kg), start 9.2×10 10 m apart")
        self.assertIn_section("beat6", "the run with the default eps1 , which is 0.05",
                              "With eps1 = 0.5 the largest \\(\\Delta_a\\) among the accepted steps is 0.318",
                              "At dt = 20000 the controller has to act")
        self.assertIn_section("algorithm", "below the default eps1 of 0.05")

    def test_beat_1_the_speeds_are_in_the_ratio_of_the_semi_major_axes(self):
        block_a = printed_body_block(P(*BeatQuotedNumberTests.BEAT_1), "A")
        block_b = printed_body_block(P(*BeatQuotedNumberTests.BEAT_1), "B")
        speed_a = float(block_a[2].split("periapsis: ")[1].split(" m/s")[0])
        speed_b = float(block_b[2].split("periapsis: ")[1].split(" m/s")[0])
        self.assertEqual((speed_a, speed_b), (15551.0, 31102.0))
        self.assertEqual(round(speed_b / speed_a), 2)
        self.assertIn_section("beat1", "15551 m/s and 31102 m/s, which are also in the ratio 1 to 2, and the periods")

    def test_beat_2_the_energy_sign_and_the_boost_sizes_quoted(self):
        self.assertEqual(F()["Initial Keplerian orbit"], "elliptic; eccentricity: 0.76705")
        self.assertEqual(F("--vInitA", "60000", "--vInitB", "60000")["Initial Keplerian orbit"],
                         "elliptic; eccentricity: 0.76705")
        self.assertGreater(float(R("--vInitA", "60000", "--vInitB", "60000")[0.0][4]), 0)
        self.assertIn_section(
            "beat2",
            "the same ellipse with eccentricity 0.76705, which is bound",
            "For a boost of 10 6 m s −1 , far faster than any orbital speed here, they differ by about 7 parts in 10 4",
            "For a boost of 500 m s −1 , as in Experiment 3, the corrector makes the same decisions as without it",
        )
        # the 10^6 m/s boost is far above the relative orbital speed of about 1.2e5 m/s at periapsis
        self.assertGreater(1e6, 5 * elements_for().relative_speed_periapsis)

    def test_beat_4_the_circular_run_overshoots_by_part_of_one_step(self):
        result = run_result(uInitA=26934.5, uInitB=-26934.5)
        turn = 2 * math.pi
        self.assertGreater(result.total_angle_rad, turn)
        self.assertLess(result.total_angle_rad, turn * 1.0002)
        step_angle = result.total_angle_rad / result.accepted_steps
        self.assertLess(result.total_angle_rad - step_angle, turn)      # one step earlier it had not turned once
        self.assertEqual(set(round(b - a, 6) for a, b in zip(result.times, result.times[1:])), {2000.0})
        self.assertIn_section(
            "beat4",
            "the two speeds, 26935 and 26934 m/s",
            "apoapsis are both 4.6e+10 m",
            "That is why the circular run reports 1.0001 revolutions, the last 2000 s step having carried it past the turn",
            "Each body's Keplerian periapsis is 6.0643×10 9 m from the centre of mass",
        )
        default_block = printed_body_block(P(), "A")
        self.assertIn("Periapsis: 6.0643e+09 m", default_block[1])

    def test_beat_5_the_dip_is_a_tenth_to_a_fifth_of_a_percent_and_first_passes_are_within_eps2(self):
        for argv in ((), ("--eps2", "1e-8"), ("--eps2", "0.5")):
            dip = abs(departure(R(*argv), 0.5))
            with self.subTest(argv=argv):
                self.assertTrue(0.001 <= dip <= 0.002, dip)
        self.assertIn_section(
            "beat5",
            "In all three the total energy has dipped by 0.1% to 0.2% at closest approach",
            "the predicted and corrected velocities differ by less than 1 part in 10 4",
        )

    def test_beat_6_the_default_run_never_shortens_a_step_and_the_controller_grows_it_by_ten_percent(self):
        default = run_result()
        steps = [b - a for a, b in zip(default.times, default.times[1:])]
        self.assertEqual(len(steps), 2284)
        self.assertTrue(all(abs(step - 2000.0) < 1e-6 for step in steps))
        wide = run_result(dt=20000.0)
        steps = [b - a for a, b in zip(wide.times, wide.times[1:])]
        self.assertEqual(max(steps), 20000.0)
        for earlier, later in zip(steps[88:93], steps[89:94]):
            self.assertAlmostEqual(later / earlier, 1.1, places=9)
        self.assertEqual(len(steps), 387)
        self.assertIn_section(
            "beat6",
            "With the default dt of 2000 s this never happens: the 52.87 days of Beat 0 are 2284 steps of exactly 2000 s",
            "Then the run with eps1 = 0.5: 222 steps",
            "which is below \\(\\varepsilon_1\\)=0.05, so no trial is ever rejected",
            "the working timestep is allowed to grow by 10% toward the user-specified ceiling",
            "Both use steps of up to 20000 s",
            "The table interpolates linearly between accepted states 20000 s apart",
            "the step is halved, grows by 10% a step until the test fails again, and is halved again",
            "The step sizes in the 387-step run",
        )
        self.assertIn_section("algorithm", "the working timestep is allowed to grow by 10% toward the user-specified ceiling",
                              "let the working step grow by 10%, up to dt")

    def test_beat_7_the_eccentric_and_head_on_commands_and_the_default_eccentricity(self):
        self.assertEqual(f"{elements_for().eccentricity:.3f}", "0.767")
        self.assertEqual(F("--uInitA", "1000", "--uInitB", "-1000", "--max_steps", "400000")
                         ["Initial Keplerian orbit"], "elliptic; eccentricity: 0.99862")
        self.assertIn_section(
            "beat7",
            "its eccentricity is 0.767",
            "With --uInitA 1000 --uInitB -1000 , an orbit with eccentricity 0.99862",
            "a head-on start, such as --uInitA 0 --uInitB 0 , falls to zero separation",
            "The third run, with dt = 1000 and eps2 = 1e-8, gives −2.9623e-04 in row 0.5",
            "which approaches the Keplerian 1.2129×10 10 m of Beat 4 as dt shrinks",
            "the dip falls by a factor of 4.00 for each halving of dt",
            "In the columns for eps2 = 1e-8 the dip falls by a factor of 4.00 for each halving of dt",
        )
        self.assertAlmostEqual(elements_for().relative_periapsis / 1.2129e10, 1.0, delta=1e-4)

    def test_the_algorithm_section_quotes_the_stopping_rule_the_floor_and_the_largest_change(self):
        largest = max(predictor_acceleration_changes(run_result()))
        self.assertEqual(f"{largest:.4f}", "0.0325")
        self.assertTrue(2 ** -40 < 1e-12 < 2 ** -39)
        self.assertIn_section(
            "algorithm",
            "the largest Δ a over the orbit is 0.0325, below the default eps1 of 0.05",
            "The safety floor is dt × 10 −12",
            "the running total, which is summed with compensation so that rounding error does not build up over many steps; "
            "if its magnitude has reached 2π and stop_after_one_orbit is on, stop",
            "halve Δt and return to step 2",
        )
        items = [html_text(item) for item in
                 re.findall(r"<li>(.*?)</li>", re.search(r"<ol>(.*?)</ol>", section_html(HELP_HTML, "algorithm"),
                                                         re.DOTALL).group(1), re.DOTALL)]
        self.assertEqual(len(items), 5)
        self.assertTrue(items[1].startswith("Take a trial step of the working length"))
        self.assertTrue(items[2].endswith("halve Δt and return to step 2."))
        self.assertTrue(items[3].endswith("If ten passes do not suffice, halve Δt and return to step 2."))
        self.assertTrue(items[4].startswith("Accept the step and record the state"))

    def test_the_summary_section_quotes_the_parabolic_band_and_the_overshoot(self):
        circular = F("--uInitA", "26934.5", "--uInitB", "-26934.5")["Total revolutions"]
        self.assertEqual(circular, "1.0001")      # a completed run can read more than 1
        self.assertEqual(F()["Total revolutions"], "1")
        rounded = F("--uInitA", "38091.14", "--uInitB", "-38091.14", "--no-stop_after_one_orbit", "--max_steps", "5")
        self.assertEqual(rounded["Initial Keplerian orbit"], "hyperbolic; eccentricity: 1")
        self.assertIn_section(
            "summary",
            "divided by 2π",
            "so a completed run can read slightly more than 1",
            "matches the escape speed to about one part in 10 12",
            "hyperbolic even though its eccentricity prints as 1 (Experiment 6)",
        )

    def test_the_quick_start_states_the_requirements_that_were_tested(self):
        self.assertIn_section("quickstart", 'Python 3.10 or later with matplotlib 3.5 or later: pip install "matplotlib>=3.5"')

    def test_experiment_6_and_7_and_8_settings_and_repeated_numbers(self):
        escape = 38091.13784870039
        self.assertEqual(len(repr(escape).replace(".", "")), 16)
        self.assertEqual(printed_fields(P("--uInitA", repr(escape), "--uInitB", repr(-escape),
                                          "--no-stop_after_one_orbit", "--max_steps", "1000"))
                         ["Initial Keplerian orbit"], "parabolic; eccentricity: 1")
        self.assertEqual(len(str(38091.14).replace(".", "")), 7)
        for number, phrases in (
                (6, ("The third command starts at the escape speed itself, to 16 digits",
                     "prints parabolic with an eccentricity of 1 and a periapsis of 4.6e+10 m",
                     "Compare the 60000 m/s run with the Keplerian line: eccentricity 3.9623")),
                (7, ("with --dt 20000 and --max_steps 30000 the run ends at max_steps after 0.64 of a revolution",)),
                (8, ("first use --dt 2000 --max_steps 7000 ; then repeat with --dt 1000 --max_steps 14000",
                     "which is Beat 7's factor of 4 per halving of dt again")),
        ):
            for phrase in phrases:
                with self.subTest(experiment=number, phrase=phrase):
                    self.assertIn(phrase, experiment_text(number))
        self.assertEqual(P("--uInitA", repr(escape), "--uInitB", repr(-escape), "--no-stop_after_one_orbit",
                           "--max_steps", "1000").count("Periapsis: 4.6e+10 m"), 2)

    def test_the_experiment_8_snippet_is_the_call_whose_result_is_quoted(self):
        block = next(block for block in re.findall(r"<pre[^>]*>(.*?)</pre>", experiment_html(8), re.DOTALL)
                     if "integrate_binary" in block)
        source = html_module.unescape(re.sub(r"<[^>]+>", "", block))
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        self.assertEqual(function.name, "largest_departure")
        settings = next(node.value for node in function.body
                        if isinstance(node, ast.Assign) and node.targets[0].id == "settings")
        defaults = {keyword.arg: ast.literal_eval(keyword.value) for keyword in settings.keywords}
        expected = dict(DEFAULTS, dt=2000.0, max_steps=7000, stop_after_one_orbit=False)
        self.assertEqual(defaults, expected)
        self.assertEqual(len(defaults), 15)
        printed = [{keyword.arg: ast.literal_eval(keyword.value) for keyword in node.value.args[0].keywords}
                   for node in tree.body if isinstance(node, ast.Expr)]
        self.assertEqual(printed, [
            {},
            {"dt": 1000, "max_steps": 14000},
            {"dt": 1000, "max_steps": 14000, "eps2": 1e-8},
            {"uInitA": 1000, "uInitB": -1000, "max_steps": 400000, "stop_after_one_orbit": True},
            {"uInitA": 1000, "uInitB": -1000, "max_steps": 400000, "stop_after_one_orbit": True, "eps1": 0.001},
            {"uInitA": 1000, "uInitB": -1000, "max_steps": 400000, "stop_after_one_orbit": True, "eps1": 0.001,
             "eps2": 1e-8},
        ])


@needs_beats
class TerminationClaimTests(unittest.TestCase):
    """What the Help says about how a run ends must match what the program does, and the
    places that say it must not contradict one another."""

    def test_an_inward_radial_start_stops_early_with_the_safety_error(self):
        with self.assertRaisesRegex(RuntimeError, "numerical safety limit"):
            driver.integrate_binary(**dict(DEFAULTS, uInitA=0.0, uInitB=0.0, max_steps=100000))

    def test_an_outward_radial_start_runs_to_the_step_ceiling(self):
        result = driver.integrate_binary(**dict(DEFAULTS, uInitA=0.0, uInitB=0.0, vInitA=4000.0,
                                                vInitB=-4000.0, max_steps=300))
        self.assertFalse(result.completed_orbit)
        self.assertEqual(result.accepted_steps, 300)

    def test_an_unbound_orbit_runs_to_the_step_ceiling(self):
        result = run_result(uInitA=60000.0, uInitB=-60000.0, max_steps=1000)
        self.assertFalse(result.completed_orbit)
        self.assertEqual(result.accepted_steps, 1000)

    def test_the_printed_summary_gives_both_outcomes(self):
        summary = html_text(section_html(HELP_HTML, "summary"))
        self.assertIn("the one-revolution rule cannot stop it; if the integration stays numerically viable "
                      "it runs to max_steps", summary)
        self.assertIn("An inward radial start can instead stop early with the close-approach safety error "
                      "(Beat 7)", summary)

    def test_beat_7_and_the_algorithm_section_agree_that_a_head_on_start_stops_early(self):
        beat = html_text(section_html(HELP_HTML, "beat7"))
        self.assertIn("a head-on start", beat)
        self.assertIn("the program stops with a message that the required timestep has fallen below the "
                      "numerical safety limit", beat)
        algorithm = html_text(section_html(HELP_HTML, "algorithm"))
        self.assertIn("A run can also end earlier with an error, for example when an inward radial start "
                      "reaches the close-approach safety limit (Beat 7)", algorithm)

    def test_no_sentence_says_that_an_unbound_or_radial_run_always_ends_at_max_steps(self):
        for section in ALL_SECTIONS:
            text = html_text(section_html(HELP_HTML, section))
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if "max_steps" in sentence and re.search(r"radial|unbound|head-on", sentence):
                    with self.subTest(section=section, sentence=sentence[:80]):
                        self.assertNotRegex(sentence, r"\b(always|invariably|must)\b")


@needs_beats
class FrameAndWordingTests(unittest.TestCase):
    """Statements about frames, closed-form solutions and energy diagnostics, each tied to
    the computation that makes it true (or to the wording that keeps it from over-reaching)."""

    def test_beat_4_angular_momentum_is_stated_about_the_centre_of_mass(self):
        beat = html_text(section_html(HELP_HTML, "beat4"))
        self.assertIn("exerts no torque about the centre of mass", beat)
        self.assertNotIn("cannot turn it", beat)
        self.assertIn("In the two commands above the centre of mass is at the origin and at rest", beat)
        html = section_html(HELP_HTML, "beat4")
        self.assertIn(r"\ell_A \;=\; (\mathbf r_A-\mathbf R_{\rm CM})\times(\mathbf v_A-\mathbf V_{\rm CM})", html)

    def test_the_centre_of_mass_form_is_conserved_and_the_origin_form_is_not_when_the_pair_drifts(self):
        # equal masses: R_CM is the mean position and V_CM the mean velocity
        def series(result, about_centre_of_mass):
            values = []
            for xa, ya, va, ua, xb, yb, vb, ub in zip(result.xA, result.yA, result.vA, result.uA,
                                                      result.xB, result.yB, result.vB, result.uB):
                if about_centre_of_mass:
                    rx, ry = xa - (xa + xb) / 2, ya - (ya + yb) / 2
                    wx, wy = va - (va + vb) / 2, ua - (ua + ub) / 2
                else:
                    rx, ry, wx, wy = xa, ya, va, ua
                values.append(rx * wy - ry * wx)
            return values

        at_rest = run_result()
        moving = run_result(vInitA=60000.0, vInitB=60000.0)
        first = series(at_rest, False)[0]
        self.assertAlmostEqual(first, 4.6e10 * 13000.0, delta=1e-9 * first)
        for label, result, about_centre_of_mass in (("at rest, origin", at_rest, False),
                                                   ("at rest, centre of mass", at_rest, True),
                                                   ("moving, centre of mass", moving, True)):
            values = series(result, about_centre_of_mass)
            with self.subTest(case=label):
                self.assertLess(max(values) - min(values), 5e-4 * abs(values[0]))
        spread = series(moving, False)
        self.assertGreater(max(spread) - min(spread), 0.1 * abs(spread[0]))

    def test_the_hodograph_origin_test_needs_a_centre_of_mass_at_rest(self):
        def winding(result):
            """Total turning of body A's inertial velocity vector, in units of a full turn."""
            angle = 0.0
            for i in range(len(result.vA) - 1):
                cross = result.vA[i] * result.uA[i + 1] - result.uA[i] * result.vA[i + 1]
                dot = result.vA[i] * result.vA[i + 1] + result.uA[i] * result.uA[i + 1]
                angle += math.atan2(cross, dot)
            return angle / (2 * math.pi)

        self.assertAlmostEqual(abs(winding(run_result())), 1.0, delta=0.01)
        self.assertLess(abs(winding(run_result(vInitA=60000.0, vInitB=60000.0))), 0.25)
        text = html_text(section_html(HELP_HTML, "beat4"))
        for phrase in ("which are measured relative to the centre of mass",
                       "when the centre of mass is at rest, a glance at the velocity-space plot tells you "
                       "whether an orbit is bound",
                       "so with a moving centre of mass the origin test applies only after \\(\\mathbf V_{\\rm CM}\\) has "
                       "been subtracted, and a large enough boost puts the origin outside the circle of a bound orbit"):
            with self.subTest(phrase=phrase[:50]):
                self.assertIn(phrase, text)

    def test_beat_5_does_not_claim_that_the_two_body_problem_has_no_closed_form(self):
        beat = html_text(section_html(HELP_HTML, "beat5"))
        self.assertNotIn("cannot be solved in closed form", beat)
        self.assertIn("do have an exact solution for two point masses", beat)
        self.assertIn("Binary deliberately integrates the equations step by step", beat)
        self.assertIn("the Keplerian block then serves as an outside check on the integration", beat)

    def test_energy_error_wording_does_not_claim_more_than_the_evidence(self):
        beat3 = html_text(section_html(HELP_HTML, "beat3"))
        self.assertNotIn("direct measure of accumulated numerical error", beat3)
        self.assertIn("one important diagnostic of accumulated numerical error, although a small energy error "
                      "does not by itself bound the error in the positions, the period or the phase", beat3)
        beat7 = html_text(section_html(HELP_HTML, "beat7"))
        self.assertIn("shows in these runs what the corrector contributes", beat7)
        self.assertIn("not a universal separation of two sources of error", beat7)
        algorithm = html_text(section_html(HELP_HTML, "algorithm"))
        self.assertIn("in the runs of Beats 5 and 7", algorithm)
        self.assertIn("In the runs of Beat 7, a departure that remains at the end of the orbit", algorithm)

    def test_row_0_5_is_at_or_near_closest_approach_in_the_runs_that_beat_5_describes(self):
        beat5 = html_text(section_html(HELP_HTML, "beat5"))
        self.assertIn("Row 0.5 is half way through the run; every run here starts at apoapsis, so that row "
                      "falls at or near closest approach", beat5)
        for changes in ({}, {"eps2": 1e-8}, {"eps2": 0.5}):
            with self.subTest(changes=changes):
                result = run_result(**changes)
                separations = separation_list(result)
                closest = min(range(len(separations)), key=separations.__getitem__)
                self.assertAlmostEqual(result.times[closest] / result.times[-1], 0.5, delta=0.01)
                self.assertEqual(separations[0], 9.2e10)
                self.assertLessEqual(max(separations), 9.2e10 * (1 + 1e-3))

    def test_beat_3_gives_the_real_reason_the_unbound_runs_stop_at_max_steps(self):
        beat3 = html_text(section_html(HELP_HTML, "beat3"))
        self.assertIn("because their commands turn the one-revolution stop off", beat3)
        for command in experiment_commands(6)[:2]:
            self.assertIn("--no-stop_after_one_orbit", command)

    def test_every_table_scrolls_inside_a_labelled_region(self):
        html = HELP_HTML
        tables = re.findall(r"<table\b", html)
        wrappers = re.findall(r'<div class="table-scroll" role="region" tabindex="0" aria-label="([^"]+)">'
                              r"\s*<table\b", html)
        self.assertEqual(len(tables), len(wrappers))
        self.assertGreaterEqual(len(tables), 5)
        self.assertEqual(len(set(wrappers)), len(wrappers))
        self.assertRegex(html, r"\.table-scroll\s*\{\s*overflow-x:\s*auto;")
        self.assertRegex(html, r"main\s*\{\s*min-width:\s*0;")

    def test_no_inline_formula_is_too_long_to_wrap_on_a_phone(self):
        for section in ALL_SECTIONS:
            body = re.sub(r"\\\[(.*?)\\\]", "", section_html(HELP_HTML, section), flags=re.DOTALL)
            for formula in re.findall(r"\\\((.*?)\\\)", body, re.DOTALL):
                with self.subTest(section=section, formula=formula[:40]):
                    self.assertLessEqual(len(" ".join(formula.split())), 55)


if __name__ == "__main__":
    unittest.main(verbosity=2)
