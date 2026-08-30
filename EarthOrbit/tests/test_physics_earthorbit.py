"""Regression and unit tests for the EarthOrbit program.

Run from the program directory with:

    python -m unittest discover -s tests -v

The module-location helper intentionally supports both the canonical tests
layout and an uploaded/flattened copy beside the four program modules.
"""

import ast
import hashlib
from html import unescape
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CORE_MODULE_FILENAMES = (
    "physics_earthorbit.py",
    "driver_earthorbit.py",
    "main.py",
    "plot_earthorbit.py",
)


def find_module_dir(start):
    """Find the nearest ancestor containing all four EarthOrbit modules."""
    start_path = Path(start).resolve()
    candidate = start_path if start_path.is_dir() else start_path.parent
    for directory in (candidate, *candidate.parents):
        if all((directory / name).is_file() for name in CORE_MODULE_FILENAMES):
            return directory
    names = ", ".join(CORE_MODULE_FILENAMES)
    raise FileNotFoundError(
        f"Could not find a directory containing all EarthOrbit modules: {names}"
    )


MODULE_DIR = find_module_dir(Path(__file__).resolve().parent)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import driver_earthorbit as driver
import physics_earthorbit as physics
import plot_earthorbit as plotter


HELP_FILE = MODULE_DIR / "EarthOrbit.html"


class _IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name == "id":
                self.ids.append(value)


def _independent_build_id(directory):
    digest = hashlib.sha256()
    for name in CORE_MODULE_FILENAMES:
        with open(directory / name, "r", encoding="utf-8", newline=None) as source:
            content = source.read().encode("utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()[:12]


def _module_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _exercise_fragment(document, number):
    """Return the HTML for one numbered exercise card."""
    marker = f'<div class="ec-num">EXP-{number} ·'
    start = document.index(marker)
    next_card = document.find('<div class="exp-card">', start + len(marker))
    if next_card == -1:
        next_card = document.index("</div><!-- /exp-grid -->", start)
    return document[start:next_card]


def _exercise_code(document, number):
    """Extract and HTML-unescape the Python block from an exercise card."""
    fragment = _exercise_fragment(document, number)
    match = re.search(r'<div class="ep">(.*?)</div>', fragment, flags=re.DOTALL)
    if match is None:
        raise AssertionError(f"EXP-{number} has no Python code block")
    return unescape(match.group(1)).strip()


class ModuleDiscoveryTests(unittest.TestCase):
    def test_canonical_tests_directory_is_supported(self):
        self.assertEqual(find_module_dir(MODULE_DIR / "tests"), MODULE_DIR)

    def test_flattened_start_beside_modules_is_supported(self):
        self.assertEqual(find_module_dir(MODULE_DIR), MODULE_DIR)
        self.assertEqual(find_module_dir(MODULE_DIR / "main.py"), MODULE_DIR)

    def test_nearest_matching_ancestor_wins(self):
        with tempfile.TemporaryDirectory() as temp_name:
            outer = Path(temp_name)
            for name in CORE_MODULE_FILENAMES:
                (outer / name).write_text("# outer\n", encoding="utf-8")
            inner = outer / "inner"
            inner.mkdir()
            for name in CORE_MODULE_FILENAMES:
                (inner / name).write_text("# inner\n", encoding="utf-8")
            nested = inner / "tests"
            nested.mkdir()
            self.assertEqual(find_module_dir(nested), inner)

    def test_missing_modules_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_name:
            with self.assertRaisesRegex(FileNotFoundError, "all EarthOrbit modules"):
                find_module_dir(temp_name)


class VersionAndBuildTests(unittest.TestCase):
    """Compatibility-contract tests for released version/build metadata."""

    def test_version_is_semantic(self):
        self.assertRegex(physics.MODEL_VERSION, r"^\d+\.\d+\.\d+$")

    def test_build_coverage_is_exactly_the_four_core_modules(self):
        self.assertEqual(physics.BUILD_ID_COVERS, CORE_MODULE_FILENAMES)

    def test_build_id_matches_independent_calculation(self):
        """Lock the documented filename/length/content hash framing."""
        self.assertEqual(physics.BUILD_ID, _independent_build_id(MODULE_DIR))
        self.assertRegex(physics.BUILD_ID, r"^[0-9a-f]{12}$")

    def test_build_id_is_line_ending_independent(self):
        with tempfile.TemporaryDirectory() as first_name, tempfile.TemporaryDirectory() as second_name:
            first = Path(first_name)
            second = Path(second_name)
            for index, name in enumerate(CORE_MODULE_FILENAMES):
                lines = [f"# file {index}", "value = 1", ""]
                (first / name).write_bytes("\n".join(lines).encode("utf-8"))
                (second / name).write_bytes("\r\n".join(lines).encode("utf-8"))
            with mock.patch.object(physics, "__file__", str(first / "physics_earthorbit.py")):
                first_id = physics._compute_build_id()
            with mock.patch.object(physics, "__file__", str(second / "physics_earthorbit.py")):
                second_id = physics._compute_build_id()
            self.assertEqual(first_id, second_id)

    def test_missing_source_returns_unknown(self):
        with tempfile.TemporaryDirectory() as temp_name:
            fake_file = Path(temp_name) / "physics_earthorbit.py"
            fake_file.write_text("# incomplete package\n", encoding="utf-8")
            with mock.patch.object(physics, "__file__", str(fake_file)):
                self.assertEqual(physics._compute_build_id(), "unknown")

    def test_driver_version_info_matches_physics(self):
        self.assertEqual(
            driver.version_info(),
            {
                "model_version": physics.MODEL_VERSION,
                "build_id": physics.BUILD_ID,
            },
        )


class ArchitectureAndCompatibilityTests(unittest.TestCase):
    def test_all_core_modules_parse_as_python_310(self):
        for name in CORE_MODULE_FILENAMES:
            with self.subTest(module=name):
                source = (MODULE_DIR / name).read_text(encoding="utf-8")
                ast.parse(source, filename=name, feature_version=(3, 10))

    def test_physics_layer_has_no_numpy_or_plotting_dependency(self):
        imports = _module_imports(MODULE_DIR / "physics_earthorbit.py")
        self.assertNotIn("numpy", imports)
        self.assertNotIn("matplotlib", imports)
        self.assertNotIn("driver_earthorbit", imports)
        self.assertNotIn("plot_earthorbit", imports)

    def test_driver_layer_has_no_plotting_dependency(self):
        imports = _module_imports(MODULE_DIR / "driver_earthorbit.py")
        self.assertNotIn("matplotlib", imports)
        self.assertNotIn("plot_earthorbit", imports)

    def test_plot_layer_has_no_physics_or_driver_dependency(self):
        imports = _module_imports(MODULE_DIR / "plot_earthorbit.py")
        self.assertNotIn("physics_earthorbit", imports)
        self.assertNotIn("driver_earthorbit", imports)


class PhysicalConstantTests(unittest.TestCase):
    def test_textbook_constants_are_preserved(self):
        self.assertEqual(physics.G_SURFACE, 9.8)
        self.assertEqual(physics.R_EARTH, 6_378_200.0)

    def test_modern_earth_gravitational_parameter(self):
        self.assertEqual(physics.MU_EARTH, 3.986_004_355_07e14)

    def test_textbook_derived_parameter_is_available_for_comparison(self):
        self.assertEqual(
            physics.K_APPROX,
            physics.G_SURFACE * physics.R_EARTH * physics.R_EARTH,
        )
        relative_difference = abs(physics.K_APPROX / physics.MU_EARTH - 1.0)
        self.assertLess(relative_difference, 3.0e-4)
        self.assertGreater(relative_difference, 1.0e-5)


class AccelerationTests(unittest.TestCase):
    """Scientific invariants and defensive bounds for the physics layer."""

    def test_simplified_axis_components(self):
        self.assertEqual(
            physics.compute_acceleration(physics.R_EARTH, 0.0),
            (-physics.G_SURFACE, -0.0),
        )
        self.assertEqual(
            physics.compute_acceleration(0.0, physics.R_EARTH),
            (-0.0, -physics.G_SURFACE),
        )

    def test_simplified_magnitude_is_constant(self):
        for x, y in ((3.0, 4.0), (-7.0, 11.0), (1.0e100, -2.0e100)):
            with self.subTest(x=x, y=y):
                ax, ay = physics.compute_acceleration(x, y, "simplified")
                self.assertAlmostEqual(math.hypot(ax, ay), physics.G_SURFACE, places=13)
                self.assertLess(ax * x + ay * y, 0.0)

    def test_inverse_square_magnitude_and_direction(self):
        x = 3.0e6
        y = 4.0e6
        r = 5.0e6
        ax, ay = physics.compute_acceleration(x, y, "inverse_square")
        self.assertAlmostEqual(
            math.hypot(ax, ay), physics.MU_EARTH / (r * r), places=13
        )
        self.assertLess(ax * x + ay * y, 0.0)
        self.assertAlmostEqual(ax / ay, x / y, places=14)

    def test_inverse_square_falls_as_inverse_radius_squared(self):
        a1 = math.hypot(
            *physics.compute_acceleration(physics.R_EARTH, 0.0, "inverse_square")
        )
        a2 = math.hypot(
            *physics.compute_acceleration(2.0 * physics.R_EARTH, 0.0, "inverse_square")
        )
        self.assertAlmostEqual(a2 / a1, 0.25, places=14)

    def test_acceleration_is_odd_under_position_reversal(self):
        for law in ("simplified", "inverse_square"):
            with self.subTest(force_law=law):
                forward = physics.compute_acceleration(2.0e6, -3.0e6, law)
                reverse = physics.compute_acceleration(-2.0e6, 3.0e6, law)
                np.testing.assert_allclose(reverse, -np.asarray(forward), rtol=1e-15)

    def test_huge_finite_positions_do_not_overflow(self):
        for law in ("simplified", "inverse_square"):
            with self.subTest(force_law=law):
                acceleration = physics.compute_acceleration(1.0e308, 1.0e308, law)
                self.assertTrue(all(math.isfinite(value) for value in acceleration))

    def test_unrepresentably_large_acceleration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "too large to represent"):
            physics.compute_acceleration(1.0e-200, 0.0, "inverse_square")

    def test_native_overflow_is_translated_to_value_error(self):
        class OverflowingParameter:
            def __truediv__(self, other):
                raise OverflowError("simulated platform overflow")

        with mock.patch.object(physics, "MU_EARTH", OverflowingParameter()):
            with self.assertRaisesRegex(ValueError, "too large to represent"):
                physics.compute_acceleration(
                    physics.R_EARTH, 0.0, "inverse_square"
                )

    def test_centre_is_rejected(self):
        for law in ("simplified", "inverse_square"):
            with self.subTest(force_law=law):
                with self.assertRaisesRegex(ValueError, "undefined at Earth's centre"):
                    physics.compute_acceleration(0.0, 0.0, law)

    def test_invalid_positions_are_rejected(self):
        cases = (
            (float("nan"), 1.0),
            (float("inf"), 1.0),
            (1.0, float("-inf")),
            ("1", 1.0),
            (None, 1.0),
            (True, 1.0),
        )
        for x, y in cases:
            with self.subTest(x=x, y=y):
                with self.assertRaisesRegex(ValueError, "finite real position"):
                    physics.compute_acceleration(x, y)

    def test_unknown_force_law_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown force_law"):
            physics.compute_acceleration(physics.R_EARTH, 0.0, "constant")


class DriverValidationTests(unittest.TestCase):
    """Public input-validation and error-reporting contracts."""

    def test_invalid_scalar_inputs_are_rejected(self):
        cases = (
            ("h0", {"h0": -1.0}),
            ("h0", {"h0": float("nan")}),
            ("h0", {"h0": True}),
            ("uInit", {"uInit": float("inf")}),
            ("uInit", {"uInit": "7900"}),
            ("vInit", {"vInit": None}),
            ("dt", {"dt": 0.0}),
            ("dt", {"dt": -0.1}),
            ("dt", {"dt": float("nan")}),
            ("dt", {"dt": False}),
        )
        for expected_name, kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, expected_name):
                    driver.run_earth_orbit(maxSteps=2, **kwargs)

    def test_invalid_max_steps_values_are_rejected(self):
        for value in (1, 0, -2, 2.5, True, "100"):
            with self.subTest(maxSteps=value):
                with self.assertRaisesRegex(ValueError, "maxSteps"):
                    driver.run_earth_orbit(maxSteps=value)

    def test_numpy_integer_max_steps_is_accepted(self):
        result = driver.run_earth_orbit(maxSteps=np.int64(2))
        self.assertEqual(len(result[0]), 2)

    def test_unknown_force_law_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "force_law"):
            driver.run_earth_orbit(maxSteps=2, force_law="constant")

    def test_return_diagnostics_must_be_boolean(self):
        for value in (1, 0, "yes", None, np.bool_(True)):
            with self.subTest(return_diagnostics=value):
                with self.assertRaisesRegex(ValueError, "return_diagnostics"):
                    driver.run_earth_orbit(
                        maxSteps=2, return_diagnostics=value
                    )

    def test_impossible_allocation_has_clear_error(self):
        with mock.patch.object(driver.np, "zeros", side_effect=MemoryError):
            with self.assertRaisesRegex(ValueError, "too large"):
                driver.run_earth_orbit(maxSteps=2)

    def test_platform_size_overflow_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "too large"):
            driver.run_earth_orbit(maxSteps=10**100)

    def test_nonfinite_integrated_state_is_rejected(self):
        with self.assertRaisesRegex(FloatingPointError, "non-finite state"):
            driver.run_earth_orbit(uInit=1.0e308, dt=10.0, maxSteps=2)


class DriverBehaviorTests(unittest.TestCase):
    """Scientific behavior plus explicitly named legacy regression contracts."""

    def test_initial_conditions_and_first_simplified_step(self):
        xs, ys, x_earth, y_earth, ts, us, vs = driver.run_earth_orbit(
            maxSteps=2, return_diagnostics=True
        )
        self.assertEqual(xs[0], 0.0)
        self.assertEqual(ys[0], physics.R_EARTH + 300.0)
        self.assertEqual(us[0], 7900.0)
        self.assertEqual(vs[0], 0.0)
        self.assertEqual(ts.tolist(), [0.0, 0.4])
        self.assertAlmostEqual(us[1], 7900.0, places=12)
        self.assertAlmostEqual(vs[1], -3.92, places=12)
        self.assertAlmostEqual(xs[1], 3160.0, places=12)
        self.assertAlmostEqual(ys[1], physics.R_EARTH + 299.216, places=9)
        self.assertEqual(len(x_earth), 401)
        self.assertEqual(len(y_earth), 401)

    def test_first_inverse_square_step_matches_hand_calculation(self):
        h0 = 300_000.0
        dt = 0.5
        r0 = physics.R_EARTH + h0
        expected_ay = -physics.MU_EARTH / (r0 * r0)
        result = driver.run_earth_orbit(
            h0=h0,
            uInit=1000.0,
            vInit=50.0,
            dt=dt,
            maxSteps=2,
            force_law="inverse_square",
            return_diagnostics=True,
        )
        xs, ys, _, _, ts, us, vs = result
        self.assertAlmostEqual(us[1], 1000.0, places=12)
        self.assertAlmostEqual(vs[1], 50.0 + expected_ay * dt, places=12)
        self.assertAlmostEqual(xs[1], 500.0, places=12)
        expected_y = r0 + (50.0 + vs[1]) * 0.5 * dt
        self.assertAlmostEqual(ys[1], expected_y, places=9)
        self.assertEqual(ts[-1], dt)

    def test_legacy_contract_exact_default_trajectory_regression(self):
        """Protect the current non-interpolated Schutz-style default loop."""
        xs, ys, x_earth, y_earth = driver.run_earth_orbit()
        self.assertEqual(len(xs), 503)
        self.assertEqual(len(ys), 503)
        self.assertAlmostEqual(xs[-1], 1_570_040.3158075272, places=6)
        self.assertAlmostEqual(ys[-1], 6_181_941.119307943, places=6)
        self.assertLess(math.hypot(xs[-1], ys[-1]), physics.R_EARTH)
        self.assertGreaterEqual(
            math.hypot(xs[-2], ys[-2]), physics.R_EARTH
        )
        self.assertTrue(np.all(np.isfinite(xs)))
        self.assertTrue(np.all(np.isfinite(ys)))
        self.assertEqual(x_earth[0], x_earth[-1])
        self.assertEqual(y_earth[0], y_earth[-1])

    def test_surface_curve_is_closed_and_has_correct_radius(self):
        _, _, x_earth, y_earth = driver.run_earth_orbit(maxSteps=2)
        radii = np.hypot(x_earth[:-1], y_earth[:-1])
        np.testing.assert_allclose(radii, physics.R_EARTH, rtol=2e-16)
        self.assertAlmostEqual(x_earth[100], 0.0, delta=2.0e-9)
        self.assertAlmostEqual(y_earth[100], physics.R_EARTH, places=8)
        self.assertAlmostEqual(x_earth[200], -physics.R_EARTH, places=8)
        self.assertAlmostEqual(y_earth[200], 0.0, delta=1.0e-8)

    def test_max_steps_counts_stored_points(self):
        for count in (2, 3, 17):
            with self.subTest(maxSteps=count):
                xs, ys, _, _ = driver.run_earth_orbit(
                    h0=1_000_000.0,
                    uInit=0.0,
                    vInit=1000.0,
                    dt=0.01,
                    maxSteps=count,
                )
                self.assertEqual(len(xs), count)
                self.assertEqual(len(ys), count)

    def test_stationary_launch_impacts_and_retains_below_surface_endpoint(self):
        xs, ys, _, _ = driver.run_earth_orbit(
            h0=0.0, uInit=0.0, vInit=0.0, dt=0.1, maxSteps=100
        )
        self.assertEqual(len(xs), 2)
        self.assertEqual(xs[-1], 0.0)
        self.assertLess(ys[-1], physics.R_EARTH)

    def test_diagnostic_arrays_align_with_ordinary_output(self):
        ordinary = driver.run_earth_orbit(maxSteps=30)
        diagnostic = driver.run_earth_orbit(
            maxSteps=30, return_diagnostics=True
        )
        for ordinary_array, diagnostic_array in zip(ordinary, diagnostic[:4]):
            np.testing.assert_array_equal(ordinary_array, diagnostic_array)
        xs, ys, _, _, ts, us, vs = diagnostic
        self.assertEqual(len(xs), len(ts))
        self.assertEqual(len(xs), len(us))
        self.assertEqual(len(xs), len(vs))
        np.testing.assert_allclose(np.diff(ts), 0.4, rtol=0.0, atol=1e-14)
        self.assertEqual(len(xs), len(ys))

    def test_force_law_is_forwarded_on_every_update(self):
        original = physics.compute_acceleration
        with mock.patch.object(
            driver, "compute_acceleration", wraps=original
        ) as acceleration:
            driver.run_earth_orbit(
                h0=1_000.0,
                maxSteps=4,
                force_law="inverse_square",
            )
        self.assertEqual(acceleration.call_count, 3)
        self.assertTrue(
            all(call.args[2] == "inverse_square" for call in acceleration.call_args_list)
        )

    def test_near_circular_inverse_square_run_remains_above_surface(self):
        h0 = 300_000.0
        r0 = physics.R_EARTH + h0
        circular_speed = math.sqrt(physics.MU_EARTH / r0)
        xs, ys, _, _ = driver.run_earth_orbit(
            h0=h0,
            uInit=circular_speed,
            vInit=0.0,
            dt=1.0,
            maxSteps=1000,
            force_law="inverse_square",
        )
        self.assertEqual(len(xs), 1000)
        self.assertGreater(np.min(np.hypot(xs, ys)), physics.R_EARTH)

    def test_circular_inverse_square_energy_error_shows_first_order_convergence(self):
        h0 = 300_000.0
        r0 = physics.R_EARTH + h0
        circular_speed = math.sqrt(physics.MU_EARTH / r0)

        def maximum_error(dt):
            points = int(2000.0 / dt) + 1
            xs, ys, _, _, _, us, vs = driver.run_earth_orbit(
                h0=h0,
                uInit=circular_speed,
                dt=dt,
                maxSteps=points,
                force_law="inverse_square",
                return_diagnostics=True,
            )
            radius = np.hypot(xs, ys)
            energy = 0.5 * (us * us + vs * vs) - physics.MU_EARTH / radius
            return float(np.max(np.abs(energy - energy[0])))

        errors = [maximum_error(dt) for dt in (4.0, 2.0, 1.0)]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        observed_orders = [
            math.log(errors[index] / errors[index + 1], 2.0)
            for index in range(2)
        ]
        for order in observed_orders:
            with self.subTest(observed_order=order):
                self.assertGreater(order, 0.8)
                self.assertLess(order, 1.2)

    def test_full_period_orbit_recovery_converges_toward_initial_state(self):
        h0 = 300_000.0
        initial_radius = physics.R_EARTH + h0
        initial_speed = math.sqrt(physics.MU_EARTH / initial_radius)
        period = 2.0 * math.pi * math.sqrt(
            initial_radius**3 / physics.MU_EARTH
        )

        errors = []
        for update_count in (720, 1440, 2880):
            dt = period / update_count
            xs, ys, _, _, _, us, vs = driver.run_earth_orbit(
                h0=h0,
                uInit=initial_speed,
                vInit=0.0,
                dt=dt,
                maxSteps=update_count + 1,
                force_law="inverse_square",
                return_diagnostics=True,
            )
            angle = np.unwrap(np.arctan2(ys, xs))
            angle_error = abs(abs(angle[-1] - angle[0]) - 2.0 * math.pi)
            radius_error = abs(math.hypot(xs[-1], ys[-1]) - initial_radius)
            position_error = math.hypot(xs[-1], ys[-1] - initial_radius)
            velocity_error = math.hypot(us[-1] - initial_speed, vs[-1])
            errors.append(
                (angle_error, radius_error, position_error, velocity_error)
            )

        for metric_index in range(4):
            metric_errors = [row[metric_index] for row in errors]
            self.assertGreater(metric_errors[0], metric_errors[1])
            self.assertGreater(metric_errors[1], metric_errors[2])

        # Version 1.1.1 baselines at 2880 updates were approximately:
        # 0.0636 rad angular error, 1.37% radial error, 6.54% position
        # error, and 6.37% velocity error.  These ceilings retain a modest
        # cross-platform margin while detecting a meaningful degradation.
        fine_angle, fine_radius, fine_position, fine_velocity = errors[-1]
        self.assertLess(fine_angle, 0.068)
        self.assertLess(fine_radius / initial_radius, 0.015)
        self.assertLess(fine_position / initial_radius, 0.068)
        self.assertLess(fine_velocity / initial_speed, 0.068)

        metric_names = ("angle", "radius", "position", "velocity")
        for metric_index, metric_name in enumerate(metric_names):
            observed_orders = [
                math.log(
                    errors[index][metric_index]
                    / errors[index + 1][metric_index],
                    2.0,
                )
                for index in range(2)
            ]
            for order in observed_orders:
                with self.subTest(metric=metric_name, observed_order=order):
                    self.assertGreater(order, 0.8)
                    self.assertLess(order, 1.2)


class PlotTests(unittest.TestCase):
    """Presentation-contract tests for the documented matplotlib output."""

    def tearDown(self):
        plt.close("all")

    def test_plot_properties_and_return_value(self):
        with mock.patch.object(plt, "show") as show:
            figure, axes = plotter.plot_earth_orbit(
                [0.0, 1.0],
                [2.0, 3.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            )
        show.assert_called_once_with()
        self.assertIs(figure, axes.figure)
        self.assertEqual(len(axes.lines), 2)
        earth_line, trajectory_line = axes.lines
        self.assertEqual(earth_line.get_color(), "blue")
        self.assertEqual(earth_line.get_linestyle(), "--")
        self.assertEqual(earth_line.get_label(), "Earth surface")
        self.assertEqual(trajectory_line.get_color(), "red")
        self.assertEqual(trajectory_line.get_linewidth(), 2.0)
        self.assertEqual(trajectory_line.get_label(), "Projectile trajectory")
        self.assertEqual(axes.get_xlabel(), "x (meters)")
        self.assertEqual(axes.get_ylabel(), "y (meters)")
        self.assertEqual(
            axes.get_title(), "EarthOrbit — Attempting to Achieve Orbit"
        )
        self.assertEqual(axes.get_aspect(), 1.0)

    def test_invalid_plot_inputs_are_rejected(self):
        cases = (
            ([], [], [0], [0]),
            ([0, 1], [0], [0], [0]),
            ([0], [0], [], []),
            ([0], [0], [0, 1], [0]),
            ([float("nan")], [0], [0], [0]),
            ([0], [0], [float("inf")], [0]),
            (["x"], [0], [0], [0]),
            ([[0]], [[0]], [0], [0]),
        )
        for values in cases:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    plotter.plot_earth_orbit(*values)


class MainProgramTests(unittest.TestCase):
    """Command-line and documented console-interface contract tests."""

    def test_version_option(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--version"],
            cwd=MODULE_DIR,
            text=True,
            capture_output=True,
            timeout=20,
            check=True,
        )
        self.assertEqual(
            result.stdout.strip(),
            f"EarthOrbit {physics.MODEL_VERSION} (build {physics.BUILD_ID})",
        )
        self.assertEqual(result.stderr, "")

    def test_documented_console_contract_and_headless_smoke_run(self):
        environment = os.environ.copy()
        environment["MPLBACKEND"] = "Agg"
        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=MODULE_DIR,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        self.assertIn(
            f"EarthOrbit {physics.MODEL_VERSION} (build {physics.BUILD_ID})",
            result.stdout,
        )
        expected_samples = len(driver.run_earth_orbit()[0])
        self.assertIn(
            f"{expected_samples:,} trajectory samples", result.stdout
        )
        self.assertEqual(result.stderr, "")


class HelpFileTests(unittest.TestCase):
    """Documentation-interface, scientific wording, and presentation contracts."""

    @classmethod
    def setUpClass(cls):
        cls.html = HELP_FILE.read_text(encoding="utf-8")

    def test_help_file_exists(self):
        self.assertTrue(HELP_FILE.is_file())

    def test_version_and_build_match_program(self):
        pattern = (
            r'<p id="version_build"[^>]*>\s*'
            r"Version\s+([^&<\s]+)&nbsp;(?:&nbsp;){3}Build\s+([0-9a-f]+)"
        )
        match = re.search(pattern, self.html)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), physics.MODEL_VERSION)
        self.assertEqual(match.group(2), physics.BUILD_ID)

    def test_version_build_id_is_unique(self):
        parser = _IdCollector()
        parser.feed(self.html)
        self.assertEqual(parser.ids.count("version_build"), 1)
        self.assertEqual(len(parser.ids), len(set(parser.ids)))

    def test_help_describes_core_defaults_and_interfaces(self):
        required_text = (
            "h0=300.0",
            "uInit=7900.0",
            "vInit=0.0",
            "dt=0.4",
            "maxSteps=15000",
            'force_law="simplified"',
            "return_diagnostics=True",
            "MU_EARTH",
            "3.986_004_355_07e14",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, self.html)
        self.assertIn("number of trajectory samples", self.html)

    def test_help_has_no_stale_inverse_square_energy_formula(self):
        self.assertNotIn("K_APPROX/r", self.html)
        self.assertIn("MU_EARTH/r", self.html)

    def test_help_contains_mathjax_offline_explanation(self):
        self.assertIn("MathJax", self.html)
        self.assertIn("loaded from a public CDN", self.html)
        self.assertIn("internet connection is needed", self.html)
        self.assertIn("program itself, which needs no internet access", self.html)

    def test_help_contains_exactly_ten_ranked_exercises(self):
        labels = re.findall(r'<div class="ec-num">EXP-(\d+) · ([^<]+)</div>', self.html)
        self.assertEqual([int(number) for number, _ in labels], list(range(1, 11)))
        expected_levels = (
            "INTRODUCTORY",
            "INTRODUCTORY",
            "INTRODUCTORY–INTERMEDIATE",
            "INTERMEDIATE · SCHUTZ SUGGESTION",
            "INTERMEDIATE",
            "INTERMEDIATE",
            "INTERMEDIATE",
            "INTERMEDIATE–ADVANCED",
            "ADVANCED",
            "ADVANCED",
        )
        self.assertEqual(tuple(level for _, level in labels), expected_levels)

    def test_exercises_distinguish_the_two_force_laws(self):
        self.assertIn(r"v_c=\sqrt{gr}", self.html)
        self.assertIn("Requires <code>force_law=\"inverse_square\"</code>", self.html)
        self.assertIn("it has no finite escape speed", self.html)
        self.assertIn(r"T^2\propto r", self.html)
        self.assertIn(r"T^2/r^3", self.html)

    def test_help_documents_validation_and_impact_endpoint(self):
        self.assertIn("Accepted parameter values", self.html)
        self.assertIn("must be an integer of at least 2", self.html)
        self.assertIn("return_diagnostics", self.html)
        self.assertIn("final plotted point may lie slightly below", self.html)
        self.assertIn("Interpolate the Impact Point", self.html)

    def test_runnable_diagnostic_blocks_parse_and_execute(self):
        namespaces = {}
        for number in (4, 9, 10):
            with self.subTest(experiment=number):
                code = _exercise_code(self.html, number)
                ast.parse(
                    code,
                    filename=f"EarthOrbit.html EXP-{number}",
                    feature_version=(3, 10),
                )
                namespace = {}
                exec(compile(code, f"EXP-{number}", "exec"), namespace)
                namespaces[number] = namespace

        impact_radius = math.hypot(
            namespaces[4]["x_hit"], namespaces[4]["y_hit"]
        )
        self.assertAlmostEqual(impact_radius, physics.R_EARTH, delta=1.0)

        angle_travelled = namespaces[9]["angle_travelled"]
        self.assertGreater(float(angle_travelled[-1]), 2.0 * math.pi)
        self.assertEqual(len(angle_travelled), len(namespaces[9]["ts"]))

        self.assertEqual(
            len(namespaces[10]["r"]), len(namespaces[10]["speed_squared"])
        )
        self.assertEqual(len(namespaces[10]["r"]), len(namespaces[10]["ts"]))

    def test_advanced_blocks_are_explicitly_runnable_starter_code(self):
        for number in (9, 10):
            with self.subTest(experiment=number):
                fragment = _exercise_fragment(self.html, number)
                self.assertIn("Runnable starter code", fragment)
                self.assertNotIn("run_earth_orbit(...", fragment)

        exp9_tree = ast.parse(_exercise_code(self.html, 9))
        top_level_targets = {
            target.id
            for node in exp9_tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertNotIn("force_law", top_level_targets)
        self.assertNotIn("G_SURFACE", _exercise_code(self.html, 10))

    def test_cannon_trajectory_is_listed_as_direct_predecessor(self):
        related = self.html.split('<section id="related">', 1)[1]
        self.assertIn("<strong>CannonTrajectory</strong>", related)
        self.assertIn("Investigations 4.1 and 4.2", related)

    def test_development_history_is_confined_to_license_provenance(self):
        student_content = self.html.split('<section id="license">', 1)[0]
        self.assertNotIn("Triana", student_content)
        self.assertNotIn("original Java", student_content)
        self.assertNotIn("porting", student_content.lower())

    def test_no_known_malformed_paragraph_nesting(self):
        self.assertIsNone(re.search(r"<p(?:\s[^>]*)?>\s*<p(?:\s[^>]*)?>", self.html))
        self.assertIsNone(re.search(r"</p>\s*</p>", self.html))

    def test_all_core_module_names_appear(self):
        for name in CORE_MODULE_FILENAMES:
            with self.subTest(module=name):
                self.assertIn(name, self.html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
