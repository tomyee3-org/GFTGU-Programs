"""Regression tests for the CannonTrajectory program module.

The discovery helper deliberately supports both the repository layout
(``tests/test_physics_cannon.py``) and an upload layout in which this file is
flattened beside the four program modules.
"""

import ast
from collections import Counter
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np


CORE_MODULE_FILES = (
    "physics_cannon.py",
    "driver_cannon.py",
    "main.py",
    "plot_cannon.py",
)
HELP_FILE = "CannonTrajectory.html"


def find_module_dir(start):
    """Find the nearest ancestor containing all four core program modules."""
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        if all((directory / name).is_file() for name in CORE_MODULE_FILES):
            return directory

    required = ", ".join(CORE_MODULE_FILES)
    raise FileNotFoundError(
        f"could not find a directory containing all core modules: {required}"
    )


MODULE_DIR = find_module_dir(Path(__file__))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import driver_cannon as driver  # noqa: E402
import physics_cannon as physics  # noqa: E402
import plot_cannon as plotting  # noqa: E402


def analytic_state(speed, angle_deg, time):
    """Return the analytic [x, h, u, v] state for the program's model."""
    theta = math.radians(angle_deg)
    u0 = speed * math.cos(theta)
    v0 = speed * math.sin(theta)
    return np.array(
        [
            u0 * time,
            v0 * time - 0.5 * physics.g * time**2,
            u0,
            v0 - physics.g * time,
        ]
    )


def interpolated_range(xs, hs):
    """Linearly interpolate x between the final above/below-ground samples."""
    return xs[-2] + (xs[-1] - xs[-2]) * hs[-2] / (hs[-2] - hs[-1])


def recompute_build_id(directory):
    """Independently reproduce the documented normalized source hash."""
    digest = hashlib.sha256()
    for name in physics.BUILD_ID_COVERS:
        with (directory / name).open(
            "r", encoding="utf-8", newline=None
        ) as source:
            content = source.read().encode("utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()[:12]


class TestModuleDiscovery(unittest.TestCase):
    def test_finds_canonical_tests_layout(self):
        self.assertEqual(find_module_dir(Path(__file__)), MODULE_DIR)

    def test_finds_flattened_layout(self):
        self.assertEqual(find_module_dir(MODULE_DIR / "main.py"), MODULE_DIR)

    def test_uses_nearest_matching_ancestor(self):
        self.assertEqual(find_module_dir(MODULE_DIR / "tests"), MODULE_DIR)

    def test_missing_module_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            find_module_dir(Path(MODULE_DIR.anchor))


class TestMetadataAndCompatibility(unittest.TestCase):
    def test_model_version(self):
        self.assertEqual(physics.MODEL_VERSION, "1.1.0")

    def test_build_coverage_is_exactly_the_executable_core(self):
        self.assertEqual(tuple(physics.BUILD_ID_COVERS), CORE_MODULE_FILES)
        self.assertNotIn(HELP_FILE, physics.BUILD_ID_COVERS)
        self.assertFalse(any("test" in name for name in physics.BUILD_ID_COVERS))

    def test_build_id_matches_independent_calculation(self):
        self.assertRegex(physics.BUILD_ID, r"^[0-9a-f]{12}$")
        self.assertEqual(physics.BUILD_ID, recompute_build_id(MODULE_DIR))

    def test_driver_reports_same_metadata(self):
        self.assertEqual(
            driver.version_info(),
            {
                "model_version": physics.MODEL_VERSION,
                "build_id": physics.BUILD_ID,
            },
        )

    def test_all_core_sources_parse_as_python_3_10(self):
        for name in CORE_MODULE_FILES:
            with self.subTest(name=name):
                source = (MODULE_DIR / name).read_text(encoding="utf-8")
                ast.parse(source, filename=name, feature_version=(3, 10))

    def test_version_command(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--version"],
            cwd=MODULE_DIR,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            f"CannonTrajectory {physics.MODEL_VERSION} (build {physics.BUILD_ID})",
        )


class TestPhysics(unittest.TestCase):
    def test_uses_conventional_standard_gravity(self):
        self.assertEqual(physics.g, 9.80665)

    def test_derivatives(self):
        state = np.array([12.0, 34.0, 5.5, -2.25])
        result = physics.derivs_cannon(state)
        np.testing.assert_array_equal(result, [5.5, -2.25, 0.0, -physics.g])
        self.assertEqual(result.dtype, float)

    def test_derivatives_do_not_mutate_state(self):
        state = np.array([1.0, 2.0, 3.0, 4.0])
        original = state.copy()
        physics.derivs_cannon(state)
        np.testing.assert_array_equal(state, original)

    def test_forward_euler_step(self):
        state = np.array([1.0, 2.0, 3.0, 4.0])
        result = physics.euler_step(state, 0.25)
        np.testing.assert_allclose(
            result,
            [1.75, 3.0, 3.0, 4.0 - 0.25 * physics.g],
            rtol=0.0,
            atol=1e-14,
        )

    def test_improved_euler_step(self):
        state = np.array([1.0, 2.0, 3.0, 4.0])
        dt = 0.25
        result = physics.improved_euler_step(state, dt)
        np.testing.assert_allclose(
            result,
            [
                1.0 + 3.0 * dt,
                2.0 + 4.0 * dt - 0.5 * physics.g * dt**2,
                3.0,
                4.0 - physics.g * dt,
            ],
            rtol=0.0,
            atol=1e-14,
        )

    def test_steppers_do_not_mutate_state(self):
        for stepper in (physics.euler_step, physics.improved_euler_step):
            with self.subTest(stepper=stepper.__name__):
                state = np.array([1.0, 2.0, 3.0, 4.0])
                original = state.copy()
                stepper(state, 0.1)
                np.testing.assert_array_equal(state, original)

    def test_heun_matches_analytic_state_after_many_steps(self):
        initial = analytic_state(83.0, 37.0, 0.0)
        for dt, count in ((0.5, 7), (0.1, 53), (0.025, 211)):
            with self.subTest(dt=dt):
                state = initial.copy()
                for _ in range(count):
                    state = physics.improved_euler_step(state, dt)
                np.testing.assert_allclose(
                    state,
                    analytic_state(83.0, 37.0, count * dt),
                    rtol=2e-14,
                    atol=2e-12,
                )

    def test_euler_vertical_position_error_is_first_order(self):
        errors = []
        initial = analytic_state(40.0, 60.0, 0.0)
        for dt in (0.2, 0.1, 0.05):
            state = initial.copy()
            count = round(2.0 / dt)
            for _ in range(count):
                state = physics.euler_step(state, dt)
            errors.append(abs(state[1] - analytic_state(40.0, 60.0, 2.0)[1]))
        self.assertAlmostEqual(errors[0] / errors[1], 2.0, places=12)
        self.assertAlmostEqual(errors[1] / errors[2], 2.0, places=11)


class TestDriverNominalBehavior(unittest.TestCase):
    def test_default_trajectory_structure_and_landing(self):
        xs, hs = driver.run_cannon_trajectory()
        self.assertEqual(len(xs), 146)
        self.assertEqual(xs.shape, hs.shape)
        self.assertEqual(xs.ndim, 1)
        self.assertTrue(np.all(np.isfinite(xs)))
        self.assertTrue(np.all(np.isfinite(hs)))
        self.assertEqual(xs[0], 0.0)
        self.assertEqual(hs[0], 0.0)
        self.assertTrue(np.all(np.diff(xs) >= 0.0))
        self.assertGreaterEqual(hs[-2], 0.0)
        self.assertLess(hs[-1], 0.0)

    def test_default_point_count_follows_time_of_flight(self):
        speed = 100.0
        angle = 45.0
        dt = 0.1
        flight_time = 2.0 * speed * math.sin(math.radians(angle)) / physics.g
        expected_points = math.floor(flight_time / dt) + 2
        xs, _ = driver.run_cannon_trajectory(speed, angle, dt)
        self.assertEqual(len(xs), expected_points)

    def test_improved_trajectory_matches_analytic_samples(self):
        speed = 73.0
        angle = 28.0
        for dt in (0.5, 0.1, 0.03):
            with self.subTest(dt=dt):
                xs, hs = driver.run_cannon_trajectory(
                    speed=speed, angle_deg=angle, dt=dt, method="improved"
                )
                times = np.arange(len(xs)) * dt
                theta = math.radians(angle)
                expected_x = speed * math.cos(theta) * times
                expected_h = (
                    speed * math.sin(theta) * times
                    - 0.5 * physics.g * times**2
                )
                np.testing.assert_allclose(xs, expected_x, rtol=2e-14, atol=2e-11)
                np.testing.assert_allclose(hs, expected_h, rtol=2e-14, atol=2e-11)

    def test_euler_trajectory_matches_its_discrete_formula(self):
        speed = 73.0
        angle = 28.0
        dt = 0.13
        xs, hs = driver.run_cannon_trajectory(
            speed=speed, angle_deg=angle, dt=dt, method="euler"
        )
        times = np.arange(len(xs)) * dt
        theta = math.radians(angle)
        expected_x = speed * math.cos(theta) * times
        expected_h = (
            speed * math.sin(theta) * times
            - 0.5 * physics.g * (times**2 - times * dt)
        )
        np.testing.assert_allclose(xs, expected_x, rtol=2e-14, atol=2e-11)
        np.testing.assert_allclose(hs, expected_h, rtol=2e-14, atol=2e-11)

    def test_linear_landing_interpolation_improves_range(self):
        speed = 100.0
        angle = 45.0
        xs, hs = driver.run_cannon_trajectory(speed, angle, 0.5, method="improved")
        exact = speed**2 * math.sin(math.radians(2.0 * angle)) / physics.g
        self.assertLess(
            abs(interpolated_range(xs, hs) - exact),
            abs(xs[-1] - exact),
        )

    def test_complementary_angles_have_same_interpolated_range(self):
        ranges = []
        for angle in (30.0, 60.0):
            xs, hs = driver.run_cannon_trajectory(100.0, angle, 0.01)
            ranges.append(interpolated_range(xs, hs))
        self.assertAlmostEqual(ranges[0], ranges[1], delta=2e-3)

    def test_range_scales_as_speed_squared(self):
        ranges = []
        for speed in (50.0, 100.0):
            xs, hs = driver.run_cannon_trajectory(speed, 45.0, 0.005)
            ranges.append(interpolated_range(xs, hs))
        self.assertAlmostEqual(ranges[1] / ranges[0], 4.0, delta=2e-5)

    def test_45_degrees_is_numerically_optimal_in_angle_sweep(self):
        ranges = {}
        for angle in range(1, 90):
            xs, hs = driver.run_cannon_trajectory(100.0, angle, 0.02)
            ranges[angle] = interpolated_range(xs, hs)
        self.assertEqual(max(ranges, key=ranges.get), 45)

    def test_zero_degree_boundary_lands(self):
        improved_x, improved_h = driver.run_cannon_trajectory(1.0, 0.0, 0.1)
        euler_x, euler_h = driver.run_cannon_trajectory(
            1.0, 0.0, 0.1, method="euler"
        )
        self.assertEqual(len(improved_x), 2)
        self.assertEqual(len(euler_x), 3)
        self.assertLess(improved_h[-1], 0.0)
        self.assertLess(euler_h[-1], 0.0)

    def test_ninety_degree_boundary_has_negligible_horizontal_drift(self):
        xs, hs = driver.run_cannon_trajectory(100.0, 90.0, 0.1)
        self.assertLess(np.max(np.abs(xs)), 1e-10)
        self.assertLess(hs[-1], 0.0)

    def test_accepts_numpy_integer_max_steps(self):
        xs, hs = driver.run_cannon_trajectory(
            1.0, 0.0, 0.1, max_steps=np.int64(3)
        )
        self.assertEqual(len(xs), len(hs))


class TestDriverValidation(unittest.TestCase):
    def assert_invalid(self, keyword, values, exception):
        for value in values:
            with self.subTest(keyword=keyword, value=value):
                with self.assertRaises(exception):
                    driver.run_cannon_trajectory(**{keyword: value})

    def test_speed_value_validation(self):
        self.assert_invalid(
            "speed", [0.0, -1.0, math.nan, math.inf, -math.inf], ValueError
        )

    def test_speed_type_validation(self):
        self.assert_invalid("speed", [True, "100", None, 1 + 2j], TypeError)

    def test_angle_value_validation(self):
        self.assert_invalid(
            "angle_deg",
            [-0.001, 90.001, math.nan, math.inf, -math.inf],
            ValueError,
        )

    def test_angle_type_validation(self):
        self.assert_invalid("angle_deg", [True, "45", None, 1 + 2j], TypeError)

    def test_timestep_value_validation(self):
        self.assert_invalid(
            "dt", [0.0, -0.1, math.nan, math.inf, -math.inf], ValueError
        )

    def test_timestep_type_validation(self):
        self.assert_invalid("dt", [True, "0.1", None, 1 + 2j], TypeError)

    def test_max_steps_value_validation(self):
        self.assert_invalid("max_steps", [-1, 0, 1], ValueError)

    def test_max_steps_type_validation(self):
        self.assert_invalid(
            "max_steps", [True, 2.0, "100", None, 2 + 0j], TypeError
        )

    def test_method_validation(self):
        self.assert_invalid(
            "method", ["Euler", "heun", "imprved", "", None, []], ValueError
        )

    def test_step_ceiling_raises_instead_of_returning_truncation(self):
        with self.assertRaisesRegex(RuntimeError, "before the projectile landed"):
            driver.run_cannon_trajectory(max_steps=2)

    def test_non_finite_computed_state_raises(self):
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            driver.run_cannon_trajectory(speed=1e308, dt=1e308)


class TestPlottingAndMain(unittest.TestCase):
    def tearDown(self):
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_contents(self):
        import matplotlib.pyplot as plt

        xs = np.array([0.0, 1.0, 2.0])
        hs = np.array([0.0, 1.0, -0.2])
        with mock.patch.object(plt, "show") as show:
            plotting.plot_cannon(xs, hs)
        show.assert_called_once_with()
        figure = plt.gcf()
        self.assertEqual(len(figure.axes), 1)
        axes = figure.axes[0]
        self.assertEqual(
            axes.get_title(), "CannonTrajectory — Newtonian Projectile Motion"
        )
        self.assertEqual(axes.get_xlabel(), "Horizontal distance (m)")
        self.assertEqual(axes.get_ylabel(), "Vertical distance (m)")
        self.assertEqual(axes.get_aspect(), 1.0)
        self.assertEqual(axes.lines[0].get_label(), "Projectile trajectory")
        self.assertIsNotNone(axes.get_legend())

    def test_main_smoke_run_with_noninteractive_backend(self):
        environment = os.environ.copy()
        environment["MPLBACKEND"] = "Agg"
        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=MODULE_DIR,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"CannonTrajectory {physics.MODEL_VERSION}", result.stdout)
        self.assertIn(f"(build {physics.BUILD_ID})", result.stdout)
        self.assertIn("146 trajectory samples", result.stdout)


class TestHelpFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = MODULE_DIR / HELP_FILE
        cls.html = cls.path.read_text(encoding="utf-8")

    def test_help_file_exists(self):
        self.assertTrue(self.path.is_file())

    def test_version_and_build_match_program(self):
        expected = (
            f"Version {physics.MODEL_VERSION}&nbsp;&nbsp;&nbsp;&nbsp;"
            f"Build {physics.BUILD_ID}"
        )
        self.assertIn('<p id="version_build"', self.html)
        self.assertIn(expected, self.html)

    def test_help_uses_current_constant_and_default_results(self):
        for required in ("9.80665", "1019.72", "14.4210", "146 stored points"):
            with self.subTest(required=required):
                self.assertIn(required, self.html)

    def test_help_documents_parameters_and_error_behavior(self):
        for required in (
            "speed",
            "angle_deg",
            "dt",
            "max_steps",
            "method",
            "RuntimeError",
            "FloatingPointError",
            "Python 3.10 or",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.html)

    def test_help_documents_mathjax_connectivity_plainly(self):
        self.assertIn("cdn.jsdelivr.net/npm/mathjax@3", self.html)
        self.assertIn("an internet connection is needed", self.html)
        self.assertNotIn("navigator.onLine", self.html)

    def test_help_contains_seven_ranked_exercises(self):
        labels = re.findall(r'Experiment ([1-7]) · ([^<]+)</div>', self.html)
        self.assertEqual([number for number, _ in labels], list("1234567"))
        self.assertIn("Introductory", labels[1][1])
        self.assertIn("Intermediate", labels[4][1])
        self.assertIn("Advanced", labels[6][1])

    def test_schutz_degraded_update_is_translated_correctly(self):
        self.assertIn("state[1] + dt * ds2[1]", self.html)
        self.assertIn("h\\leftarrow h+w\\,dt", self.html)

    def test_range_equation_has_uncorrupted_theta(self):
        self.assertIn(r"R=v_0^2\sin(2\theta)/g", self.html)
        self.assertNotIn("\t", self.html)

    def test_all_internal_navigation_targets_exist_and_ids_are_unique(self):
        ids = re.findall(r'\bid="([^"]+)"', self.html)
        counts = Counter(ids)
        self.assertFalse({name: count for name, count in counts.items() if count > 1})
        targets = [
            target
            for target in re.findall(r'href="#([^"]+)"', self.html)
            if not target.startswith("$")
        ]
        self.assertTrue(targets)
        for target in targets:
            with self.subTest(target=target):
                self.assertIn(target, counts)

    def test_no_review_or_audit_history_leaked_into_student_help(self):
        for phrase in ("Claude", "Copilot", "Gemini", "Critique", "Audit1"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, self.html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
