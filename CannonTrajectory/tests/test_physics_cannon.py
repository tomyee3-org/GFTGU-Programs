"""Regression tests for the CannonTrajectory program module.

The discovery helper deliberately supports both the repository layout
(``tests/test_physics_cannon.py``) and an upload layout in which this file is
flattened beside the four program modules.
"""

import ast
from collections import Counter
import contextlib
from fractions import Fraction
import hashlib
from html.parser import HTMLParser
import inspect
import io
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import warnings

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


def find_help_file(module_dir):
    """Find Help in a flattened upload or the GFTGU-Documentation tree.

    Documentation folders no longer use chapter-number prefixes, and the
    Help files live under the sibling ``GFTGU-Documentation`` repository
    rather than beside the program modules inside ``GFTGU-Programs``.
    The previous fallback ``module_dir.parent / HELP_FILE`` pointed at
    ``GFTGU-Programs/CannonTrajectory.html``, which is not a real layout.
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

import driver_cannon as driver  # noqa: E402
import main as entrypoint  # noqa: E402
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


class HtmlNode:
    """Small dependency-free HTML tree node used for structural Help tests."""

    def __init__(self, tag, attrs=()):
        self.tag = tag
        self.attrs = dict(attrs)
        self.content = []

    def text(self):
        return "".join(
            item.text() if isinstance(item, HtmlNode) else item
            for item in self.content
        )


class HtmlTreeParser(HTMLParser):
    """Build just enough of a DOM to test sections, tables, and cards."""

    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
                 "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = HtmlNode("document")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = HtmlNode(tag, attrs)
        self.stack[-1].content.append(node)
        if tag not in self.VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].content.append(HtmlNode(tag, attrs))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data):
        self.stack[-1].content.append(data)


def descendants(node, predicate=lambda item: True):
    """Return descendant nodes matching predicate, in document order."""
    matches = []
    for item in node.content:
        if isinstance(item, HtmlNode):
            if predicate(item):
                matches.append(item)
            matches.extend(descendants(item, predicate))
    return matches


def normalized_text(node):
    return " ".join(node.text().split())


def has_class(node, class_name):
    return class_name in node.attrs.get("class", "").split()


def nodes_by_id(root, element_id):
    return descendants(root, lambda node: node.attrs.get("id") == element_id)


def main_trajectory_settings(directory):
    """Return the CLI defaults passed to the trajectory driver."""
    if Path(directory).resolve() != MODULE_DIR:
        raise AssertionError("settings must be read from the active module directory")
    with mock.patch.object(sys, "argv", ["main.py"]):
        args = entrypoint.parse_args()
    return {
        name: getattr(args, name)
        for name in ("speed", "angle_deg", "dt", "max_steps", "method")
    }


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

    def test_complete_suite_runs_from_a_flattened_layout(self):
        if os.environ.get("CANNON_FLATTENED_TEST_CHILD") == "1":
            return

        with tempfile.TemporaryDirectory() as temporary:
            flat_dir = Path(temporary)
            for name in CORE_MODULE_FILES:
                shutil.copy2(MODULE_DIR / name, flat_dir / name)
            shutil.copy2(HELP_PATH, flat_dir / HELP_FILE)
            flat_test = flat_dir / "test_physics_cannon.py"
            shutil.copy2(Path(__file__), flat_test)

            environment = os.environ.copy()
            environment["CANNON_FLATTENED_TEST_CHILD"] = "1"
            environment["MPLBACKEND"] = "Agg"
            result = subprocess.run(
                [sys.executable, str(flat_test)],
                cwd=flat_dir,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("OK", result.stdout + result.stderr)


class TestMetadataAndCompatibility(unittest.TestCase):
    def test_model_version(self):
        self.assertEqual(physics.MODEL_VERSION, "1.3.1")

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

    def test_driver_summary_helpers_match_default_printout_rounding(self):
        xs, hs = driver.run_cannon_trajectory()
        range_m = driver.interpolated_landing_range(xs, hs)
        height_m = driver.interpolated_maximum_height(hs)
        flight_time_s = driver.interpolated_flight_time(hs, 0.1)
        self.assertAlmostEqual(range_m, interpolated_range(xs, hs), delta=1e-12)
        self.assertEqual(f"{range_m:.5g}", "1019.7")
        self.assertEqual(f"{height_m:.5g}", "254.93")
        self.assertEqual(f"{flight_time_s:.5g}", "14.421")
        self.assertGreaterEqual(height_m, max(0.0, float(hs[-2])))

    def test_parabolic_maximum_recovers_quadratic_vertex(self):
        hs = np.array([1.0, 3.0, 4.0, 4.5, 4.0, 3.0])
        self.assertEqual(driver.interpolated_maximum_height(hs), 4.5)

        offset_hs = np.array([-0.0625, 3.4375, 4.9375, 4.4375, 1.9375, -2.5625])
        self.assertEqual(driver.interpolated_maximum_height(offset_hs), 5.0)

    def test_interpolated_flight_time_handles_exact_and_between_sample_landings(self):
        self.assertAlmostEqual(
            driver.interpolated_flight_time([0.0, 0.5, -0.5], 0.2), 0.3
        )
        self.assertEqual(
            driver.interpolated_flight_time([0.0, 1.0, 0.0, -1.0], 0.25),
            0.5,
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

    def test_exact_ground_sample_is_followed_by_negative_sample(self):
        xs, hs = driver.run_cannon_trajectory(
            speed=physics.g,
            angle_deg=90.0,
            dt=1.0,
            max_steps=10,
            method="improved",
        )
        self.assertEqual(len(xs), 4)
        self.assertEqual(hs[-2], 0.0)
        self.assertLess(hs[-1], 0.0)


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

    def test_enormous_step_ceiling_does_not_trigger_enormous_allocation(self):
        # The enormous ceiling proves that nothing is allocated up front.  If a
        # defect kept the ball above the ground, the append-only loop would
        # grow until memory ran out, so the stepper is wrapped and the test
        # fails after 100 calls instead.  The correct program needs one call.
        calls = []
        real_step = driver.improved_euler_step

        class RanAway(Exception):
            pass

        def counting_step(state, dt):
            calls.append(dt)
            if len(calls) > 100:
                raise RanAway("the projectile had not landed after 100 steps")
            return real_step(state, dt)

        with mock.patch.object(driver, "improved_euler_step", counting_step):
            xs, hs = driver.run_cannon_trajectory(
                speed=1.0,
                angle_deg=0.0,
                dt=0.1,
                max_steps=10**100,
                method="improved",
            )
        np.testing.assert_allclose(xs, [0.0, 0.1])
        self.assertEqual(len(hs), 2)
        self.assertLess(hs[-1], 0.0)
        self.assertEqual(len(calls), 1)

    def test_non_finite_computed_state_raises(self):
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            driver.run_cannon_trajectory(speed=1e308, dt=1e308)

    def test_unrepresentable_finite_reals_raise_explanatory_value_error(self):
        for keyword in ("speed", "dt"):
            for value in (10**1000, Fraction(10**1000, 1)):
                with self.subTest(keyword=keyword, value_type=type(value).__name__):
                    with self.assertRaisesRegex(ValueError, "representable as a float"):
                        driver.run_cannon_trajectory(**{keyword: value})

    def test_representable_fractions_are_normalized_and_accepted(self):
        xs, hs = driver.run_cannon_trajectory(
            speed=Fraction(100, 1),
            angle_deg=Fraction(45, 1),
            dt=Fraction(1, 10),
        )
        self.assertEqual(len(xs), 146)
        self.assertEqual(xs.shape, hs.shape)
        self.assertEqual(xs.dtype, float)


class TestPlottingAndMain(unittest.TestCase):
    def tearDown(self):
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_contents(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba

        xs = np.array([0.0, 1.0, 2.0])
        hs = np.array([0.0, 1.0, -0.2])
        with mock.patch.object(plt, "show") as show:
            returned_figure, returned_axes = plotting.plot_cannon(xs, hs)
        show.assert_called_once_with()
        figure = plt.gcf()
        self.assertIs(returned_figure, figure)
        self.assertEqual(len(figure.axes), 1)
        axes = figure.axes[0]
        self.assertIs(returned_axes, axes)
        np.testing.assert_allclose(figure.get_size_inches(), [8.0, 6.0])
        self.assertEqual(
            axes.get_title(), "CannonTrajectory — Newtonian Projectile Motion"
        )
        self.assertEqual(axes.get_xlabel(), "Horizontal distance (m)")
        self.assertEqual(axes.get_ylabel(), "Vertical distance (m)")
        self.assertEqual(axes.get_aspect(), 1.0)
        self.assertEqual(len(axes.lines), 1)
        self.assertEqual(axes.lines[0].get_label(), "Projectile trajectory")
        np.testing.assert_array_equal(axes.lines[0].get_xdata(), xs)
        np.testing.assert_array_equal(axes.lines[0].get_ydata(), hs)
        self.assertEqual(len(axes.collections), 1)
        marker = axes.collections[0]
        self.assertEqual(marker.get_label(), "Launch point")
        np.testing.assert_allclose(marker.get_offsets(), [[0.0, 0.0]])
        np.testing.assert_allclose(marker.get_facecolors()[0], to_rgba("orange"))
        self.assertTrue(any(line.get_visible() for line in axes.get_xgridlines()))
        self.assertTrue(any(line.get_visible() for line in axes.get_ygridlines()))
        legend_labels = [text.get_text() for text in axes.get_legend().get_texts()]
        self.assertEqual(legend_labels, ["Projectile trajectory", "Launch point"])

    def test_overlay_plots_all_curves_on_one_figure(self):
        import matplotlib.pyplot as plt

        trajectories = [
            ("30 degrees", np.array([0.0, 2.0]), np.array([0.0, -0.1])),
            ("60 degrees", np.array([0.0, 2.0]), np.array([0.0, -0.2])),
        ]
        with mock.patch.object(plt, "show") as show:
            figure, axes = plotting.plot_cannon_overlay(trajectories)
        show.assert_called_once_with()
        self.assertEqual(len(figure.axes), 1)
        self.assertIs(axes, figure.axes[0])
        self.assertEqual(len(axes.lines), 2)
        self.assertEqual(len(axes.collections), 1)
        self.assertEqual(
            [line.get_label() for line in axes.lines],
            ["30 degrees", "60 degrees"],
        )
        np.testing.assert_array_equal(axes.lines[1].get_xdata(), [0.0, 2.0])
        np.testing.assert_array_equal(axes.lines[1].get_ydata(), [0.0, -0.2])
        legend_labels = [text.get_text() for text in axes.get_legend().get_texts()]
        self.assertEqual(
            legend_labels, ["30 degrees", "60 degrees", "Launch point"]
        )

    def test_overlay_rejects_an_empty_collection(self):
        import matplotlib.pyplot as plt

        with mock.patch.object(plt, "show") as show:
            with self.assertRaisesRegex(ValueError, "at least one trajectory"):
                plotting.plot_cannon_overlay([])
        show.assert_not_called()

    def test_overlay_closes_figure_after_malformed_input(self):
        import matplotlib.pyplot as plt

        baseline = set(plt.get_fignums())
        malformed = (
            [("missing heights", [0.0, 1.0])],
            [("unequal lengths", [0.0, 1.0], [0.0])],
        )
        for trajectories in malformed:
            with self.subTest(trajectories=trajectories):
                with self.assertRaises(ValueError):
                    plotting.plot_cannon_overlay(trajectories)
                self.assertEqual(set(plt.get_fignums()), baseline)

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
        self.assertIn("Range (interpolated ground crossing): 1019.7 m", result.stdout)
        self.assertIn(
            "Maximum height (parabolic interpolation): 254.93 m", result.stdout
        )
        self.assertIn(
            "Flight time (interpolated ground crossing): 14.421 s", result.stdout
        )

    def test_main_accepts_all_driver_parameters_at_command_line(self):
        environment = os.environ.copy()
        environment["MPLBACKEND"] = "Agg"
        result = subprocess.run(
            [
                sys.executable, "main.py", "--speed", "50", "--angle_deg", "30",
                "--dt", "0.2", "--max_steps", "10000", "--method", "euler",
            ],
            cwd=MODULE_DIR,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("28 trajectory samples", result.stdout)
        self.assertIn("Range (interpolated ground crossing): 229.35 m", result.stdout)
        self.assertIn("Maximum height (parabolic interpolation): 34.415 m", result.stdout)
        self.assertIn("Flight time (interpolated ground crossing): 5.2967 s", result.stdout)

    def test_command_help_describes_both_method_choices(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            cwd=MODULE_DIR,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        normalized_help = " ".join(result.stdout.split())
        for required in ("--speed", "--angle_deg", "--dt", "--max_steps",
                         "--method", "first-order forward Euler",
                         "second-order improved Euler"):
            with self.subTest(required=required):
                self.assertIn(required, normalized_help)


class TestHelpFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = HELP_PATH
        cls.html = cls.path.read_text(encoding="utf-8")
        parser = HtmlTreeParser()
        parser.feed(cls.html)
        parser.close()
        cls.root = parser.root

    def test_help_file_exists(self):
        self.assertTrue(self.path.is_file())

    def test_version_and_build_match_program(self):
        version_nodes = nodes_by_id(self.root, "version_build")
        self.assertEqual(len(version_nodes), 1)
        self.assertEqual(version_nodes[0].tag, "p")
        self.assertEqual(
            normalized_text(version_nodes[0]),
            f"Version {physics.MODEL_VERSION} Build {physics.BUILD_ID}",
        )

    def test_help_uses_current_constant_and_default_results(self):
        for required in ("9.80665", "1019.72", "14.4210", "146 stored points"):
            with self.subTest(required=required):
                self.assertIn(required, self.html)

    def test_parameter_table_defaults_match_driver_and_main(self):
        parameter_sections = nodes_by_id(self.root, "parameters")
        self.assertEqual(len(parameter_sections), 1)
        tables = descendants(
            parameter_sections[0], lambda node: has_class(node, "param-table")
        )
        self.assertEqual(len(tables), 1)
        rows = {}
        for row in descendants(tables[0], lambda node: node.tag == "tr"):
            cells = [
                normalized_text(cell)
                for cell in descendants(row, lambda node: node.tag == "td")
            ]
            if cells:
                self.assertEqual(len(cells), 4)
                rows[cells[0]] = cells[1:]

        self.assertEqual(set(rows), {
            "--speed", "--angle_deg", "--dt", "--max_steps", "--method", "g"
        })
        signature = inspect.signature(driver.run_cannon_trajectory)
        defaults = {
            name: parameter.default
            for name, parameter in signature.parameters.items()
        }
        self.assertEqual(float(rows["--speed"][0]), defaults["speed"])
        self.assertEqual(float(rows["--angle_deg"][0]), defaults["angle_deg"])
        self.assertEqual(float(rows["--dt"][0]), defaults["dt"])
        self.assertEqual(int(rows["--max_steps"][0]), defaults["max_steps"])
        self.assertEqual(rows["--method"][0], defaults["method"])
        self.assertEqual(float(rows["g"][0]), physics.g)
        self.assertEqual(
            main_trajectory_settings(MODULE_DIR),
            {name: defaults[name] for name in defaults},
        )

    def test_errors_and_runtime_requirements_are_in_relevant_sections(self):
        parameter_text = normalized_text(nodes_by_id(self.root, "parameters")[0])
        for required in (
            "FloatingPointError",
            "Python 3.10 or later",
            "representable as finite Python floats",
        ):
            with self.subTest(required=required):
                self.assertIn(required, parameter_text)
        algorithm_text = normalized_text(nodes_by_id(self.root, "algorithm")[0])
        self.assertIn("RuntimeError", algorithm_text)

    def test_exact_ground_landing_needs_no_interpolation(self):
        algorithm_text = normalized_text(nodes_by_id(self.root, "algorithm")[0])
        self.assertIn(
            "it is already the landing point and no interpolation is required",
            algorithm_text,
        )
        experiment_section = nodes_by_id(self.root, "experiments")[0]
        cards = descendants(
            experiment_section, lambda node: has_class(node, "experiment-card")
        )
        experiment_five = normalized_text(cards[4])
        self.assertIn("that stored sample is already the landing point", experiment_five)
        self.assertIn("no interpolation is needed", experiment_five)

    def test_help_documents_mathjax_connectivity_plainly(self):
        self.assertIn("cdn.jsdelivr.net/npm/mathjax@3", self.html)
        self.assertIn("an internet connection is needed", self.html)
        self.assertNotIn("navigator.onLine", self.html)

    def test_help_contains_exact_rank_and_title_for_every_exercise(self):
        experiment_section = nodes_by_id(self.root, "experiments")
        self.assertEqual(len(experiment_section), 1)
        cards = descendants(
            experiment_section[0], lambda node: has_class(node, "experiment-card")
        )
        actual = []
        for card in cards:
            number = descendants(card, lambda node: has_class(node, "exp-num"))
            title = descendants(card, lambda node: has_class(node, "exp-title"))
            self.assertEqual(len(number), 1)
            self.assertEqual(len(title), 1)
            actual.append((normalized_text(number[0]), normalized_text(title[0])))

        expected = [
            ("Experiment 1 · Introductory–Intermediate", "Optimal firing angle"),
            ("Experiment 2 · Introductory", "What does the timestep change?"),
            ("Experiment 3 · Intermediate", "Effect of muzzle speed"),
            ("Experiment 4 · Intermediate", "Overlay several trajectories"),
            (
                "Experiment 5 · Intermediate · Schutz suggestion",
                "Interpolate the landing point",
            ),
            (
                "Experiment 6 · Intermediate–Advanced",
                "Accuracy of the two integrators",
            ),
            (
                "Experiment 7 · Advanced · Schutz suggestion",
                "Deliberately degrade the vertical-position update",
            ),
        ]
        self.assertEqual(actual, expected)

    def test_overlay_exercise_uses_the_supported_helper(self):
        experiment_section = nodes_by_id(self.root, "experiments")[0]
        cards = descendants(
            experiment_section, lambda node: has_class(node, "experiment-card")
        )
        experiment_four = normalized_text(cards[3])
        self.assertIn("plot_cannon_overlay", experiment_four)
        self.assertIn("trajectories.append", experiment_four)
        output_text = normalized_text(nodes_by_id(self.root, "output")[0])
        self.assertIn("(label, xs, hs) triple", output_text)

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


# ---------------------------------------------------------------------------
# Additions for the Beats-layout Help file and for the overlay interface.
# ---------------------------------------------------------------------------

def printed_summary(*args):
    """Run ``main.main()`` in-process with plotting stubbed out.

    Returns the four values on the printed summary as the strings printed.
    """
    with mock.patch.object(sys, "argv", ["main.py", *args]), \
            mock.patch.object(entrypoint, "plot_cannon") as plot, \
            contextlib.redirect_stdout(io.StringIO()) as captured:
        entrypoint.main()
    plot.assert_called_once()
    lines = captured.getvalue().splitlines()
    assert len(lines) == 4, lines
    first = re.fullmatch(
        r"CannonTrajectory \S+ \(build [0-9a-f]{12}\) — ([\d,]+) trajectory samples",
        lines[0],
    )
    values = [
        re.fullmatch(pattern, line)
        for pattern, line in zip(
            (
                r"Range \(interpolated ground crossing\): (\S+) m",
                r"Maximum height \(parabolic interpolation\): (\S+) m",
                r"Flight time \(interpolated ground crossing\): (\S+) s",
            ),
            lines[1:],
        )
    ]
    assert first and all(values), lines
    return {
        "n": first.group(1),
        "R": values[0].group(1),
        "H": values[1].group(1),
        "T": values[2].group(1),
    }


def exact_launch(speed=100.0, angle_deg=45.0):
    """Return the closed-form u0, v0y, T, H and R of Eqs. 3 to 5 of the Help."""
    theta = math.radians(angle_deg)
    u0 = speed * math.cos(theta)
    v0y = speed * math.sin(theta)
    g = physics.g
    return {
        "u0": u0,
        "v0y": v0y,
        "T": 2.0 * v0y / g,
        "H": v0y ** 2 / (2.0 * g),
        "R": speed ** 2 * math.sin(2.0 * theta) / g,
    }


def full_run(speed=100.0, angle_deg=45.0, dt=0.1, method="improved"):
    """Return unrounded (n, R, H, T, xs, hs) for one launch."""
    xs, hs = driver.run_cannon_trajectory(
        speed=speed, angle_deg=angle_deg, dt=dt, method=method
    )
    return {
        "n": len(xs),
        "R": driver.interpolated_landing_range(xs, hs),
        "H": driver.interpolated_maximum_height(hs),
        "T": driver.interpolated_flight_time(hs, dt),
        "xs": xs,
        "hs": hs,
    }


def degraded_improved_step(state, dt):
    """Experiment 7 of the Help: height advanced with the end-of-step velocity."""
    ds1 = physics.derivs_cannon(state)
    predictor = state + dt * ds1
    ds2 = physics.derivs_cannon(predictor)
    new_state = state + 0.5 * dt * (ds1 + ds2)
    new_state[1] = state[1] + dt * ds2[1]
    return new_state


def minus_to_hyphen(text):
    return text.replace("−", "-").replace("&minus;", "-")


_TEX_TOKEN = re.compile(
    r"\\t?frac|\\varphi|\\Delta\s*t|\\[;,!]|\\quad|u_0|v_\{0y\}|\d+(?:\.\d+)?|[A-Za-z]|[-+^(){}]|\s+"
)


def evaluate_tex(tex, **variables):
    """Evaluate the right-hand side of a displayed formula in the small TeX
    subset the Help's error equations use (fractions, powers, products).

    Supported names: u_0, v_{0y}, \\varphi, \\Delta t, g, T (from ``variables``
    as u0, v0y, phi, dt, g, T).  Anything else raises ValueError, so an edit
    that introduces an unknown symbol fails loudly instead of evaluating to
    something plausible.  The text after the last "=" or "\\approx" is used.
    """
    right = re.split(r"=|\\approx", tex)[-1].strip().rstrip(".")
    names = {"u_0": "u0", "v_{0y}": "v0y", "\\varphi": "phi", "\\Delta t": "dt", "g": "g", "T": "T"}
    tokens, position = [], 0
    while position < len(right):
        match = _TEX_TOKEN.match(right, position)
        if not match:
            raise ValueError(f"cannot read {right[position:position + 12]!r}")
        position = match.end()
        token = re.sub(r"\s+", " ", match.group(0))
        if token.strip() == "" or token.startswith(("\\;", "\\,", "\\!", "\\quad")):
            continue
        tokens.append("\\Delta t" if token.startswith("\\Delta") else token)
    state = {"i": 0}

    def peek():
        return tokens[state["i"]] if state["i"] < len(tokens) else None

    def take():
        state["i"] += 1
        return tokens[state["i"] - 1]

    def group():
        if take() != "{":
            raise ValueError("expected {")
        value = expression()
        if take() != "}":
            raise ValueError("expected }")
        return value

    def atom():
        token = take()
        if token in ("\\frac", "\\tfrac"):
            numerator = group()
            return numerator / group()
        if token == "(":
            value = expression()
            if take() != ")":
                raise ValueError("expected )")
            return value
        if token == "{":
            state["i"] -= 1
            return group()
        if token in names:
            return variables[names[token]]
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            return float(token)
        raise ValueError(f"unknown symbol {token!r}")

    def power():
        base = atom()
        if peek() == "^":
            take()
            exponent = group() if peek() == "{" else atom()
            return base ** exponent
        return base

    def term():
        value = power()
        while peek() not in (None, "+", "-", ")", "}"):
            value *= power()
        return value

    def expression():
        sign = 1.0
        if peek() == "-":
            take()
            sign = -1.0
        value = sign * term()
        while peek() in ("+", "-"):
            value = value + term() if take() == "+" else value - term()
        return value

    result = expression()
    if state["i"] != len(tokens):
        raise ValueError(f"unread tokens {tokens[state['i']:]}")
    return result


def help_layout(root):
    """Recognise which Help layout a parsed file uses: 'beats', 'classic' or None."""
    beats = all(nodes_by_id(root, f"beat{number}") for number in range(8))
    classic = all(
        nodes_by_id(root, name) for name in ("description", "background")
    )
    if beats and not classic:
        return "beats"
    if classic and not beats:
        return "classic"
    return None


class TestHelpLayoutRecogniser(unittest.TestCase):
    @staticmethod
    def parse(html):
        parser = HtmlTreeParser()
        parser.feed(html)
        parser.close()
        return parser.root

    def test_recognises_synthetic_layouts(self):
        classic = self.parse(
            '<section id="description"></section><section id="background"></section>'
        )
        beats = self.parse(
            "".join(f'<section id="beat{n}"></section>' for n in range(8))
        )
        both = self.parse(
            '<section id="description"></section><section id="background"></section>'
            + "".join(f'<section id="beat{n}"></section>' for n in range(8))
        )
        partial = self.parse('<section id="beat0"></section>')
        self.assertEqual(help_layout(classic), "classic")
        self.assertEqual(help_layout(beats), "beats")
        self.assertIsNone(help_layout(both))
        self.assertIsNone(help_layout(partial))

    def test_shipped_help_is_exactly_one_known_layout(self):
        root = self.parse(HELP_PATH.read_text(encoding="utf-8"))
        self.assertIn(help_layout(root), {"classic", "beats"})


# Every printed summary that the Beats Help quotes, frozen from live runs of
# this build.  A change in the program's numbers therefore fails here, and the
# Help must then be rewritten, instead of drifting silently.
FROZEN_RUNS = {
    "default": ((), "146", "1019.7", "254.93", "14.421"),
    "a30": (("--angle_deg", "30"), "103", "883.1", "127.46", "10.197"),
    "a60": (("--angle_deg", "60"), "178", "883.09", "382.39", "17.662"),
    "a0": (("--angle_deg", "0"), "2", "0", "0", "0"),
    "a90": (("--angle_deg", "90"), "205", "1.2488e-13", "509.86", "20.394"),
    "s200": (("--speed", "200"), "290", "4078.9", "1019.7", "28.842"),
    "s50": (("--speed", "50"), "74", "254.92", "63.732", "7.2104"),
    "s1000": (("--speed", "1000"), "1,444", "1.0197e+05", "25493", "144.21"),
    "e05": (("--method", "euler", "--dt", "0.5"), "31", "1054.9", "272.91", "14.919"),
    "e1": (("--method", "euler", "--dt", "1"), "17", "1089.3", "291.51", "15.405"),
    "i1": (("--dt", "1"), "16", "1018.5", "254.93", "14.404"),
    "e0deg": (("--method", "euler", "--angle_deg", "0"), "3", "10", "0", "0.1"),
    "a15": (("--angle_deg", "15"), "54", "509.83", "34.154", "5.2781"),
    "a75": (("--angle_deg", "75"), "198", "509.86", "475.7", "19.699"),
    "a40": (("--angle_deg", "40"), "133", "1004.2", "210.66", "13.109"),
    "a50": (("--angle_deg", "50"), "158", "1004.2", "299.2", "15.623"),
}

# For each beat: which frozen runs it quotes, and which of n, R, H, T it quotes.
BEAT_QUOTES = {
    "beat0": (("default", "nRHT"),),
    "beat1": (("a30", "RHT"), ("a60", "RHT"), ("a0", "n"), ("a90", "nRHT"), ("default", "RHT")),
    "beat2": (("s200", "RHT"), ("s50", "RHT"), ("default", "RHT")),
    "beat3": (("e05", "nRHT"), ("e0deg", "nRT")),
    "beat4": (("i1", "RHT"), ("e1", "RHT")),
    "beat5": (("i1", "RHT"),),
    "beat7": (("s1000", "nRHT"),),
    "summary": (("default", "RHT"),),
}


CARRIED_SECTION_SHA256 = {
    "parameters": "f50de08513258e8a8e1cc0228ab276786910f7fb2ea5923b2fc484b4beb43322",
    "related": "ebb26ad08d44fefca5cb0aa58e630b7668882941fc86b4dce21e8eb5f0c56578",
    "license": "a1e3040a10e833a6b6cf84912c769c0460134487b8a3e3089745527364ec6e4a",
}
CARRIED_EXPERIMENT_SHA256 = (
    "ee978a6779b22aaf6c1c52f57391bd5c780bb3dad469b381fa2565cfc417886e",
    "24f348174cea7f0a0498f9316e59ad0309c62b35497bbafa6dcc4db2cd8214c0",
    "25e5905297c5126aadea17f3d00c8ef1b5070f4ae7b79bd95147fb794eecc0da",
    "234c93d3de38fddbfffe26967fa8ee85f848ed5b91ac34590ba06331394b59a0",
    "1872f2dd13d9fbb330e7eab3f373eadbaec032c27ef239bb70612be4a111b49d",
    "dc0abcc37d14bf6de92959e659e18d828ea67d3cf7a4c6022c511866d0f5c729",
    "bdaafc5a6be324c37d771dd72df9cad6d2a1e2a20f2bd2cdc1ac1e9a72d2c921",
)


class TestBeatsHelp(unittest.TestCase):
    """Tests specific to the Beats layout of the Help file.

    They are skipped, not failed, when the shipped Help uses the classic
    layout, whose structure is covered by ``TestHelpFile``.
    """

    SECTION_ORDER = (
        ["overview", "beats"]
        + [f"beat{n}" for n in range(8)]
        + ["equations", "algorithm", "modules", "quickstart", "parameters",
           "output", "summary", "experiments", "related", "license"]
    )

    @classmethod
    def setUpClass(cls):
        cls.html = HELP_PATH.read_text(encoding="utf-8")
        parser = HtmlTreeParser()
        parser.feed(cls.html)
        parser.close()
        cls.root = parser.root
        if help_layout(cls.root) != "beats":
            raise unittest.SkipTest("the shipped Help uses the classic layout")

    def section(self, section_id):
        found = nodes_by_id(self.root, section_id)
        self.assertEqual(len(found), 1, section_id)
        return found[0]

    def text(self, section_id):
        return normalized_text(self.section(section_id))

    def code_blocks(self, section_id):
        return [
            block.text()
            for block in descendants(
                self.section(section_id), lambda node: node.tag == "pre"
            )
        ]

    # -- structure ---------------------------------------------------------
    def test_sections_are_present_once_and_in_order(self):
        ids = [
            node.attrs["id"]
            for node in descendants(
                self.root, lambda node: node.tag == "section" and "id" in node.attrs
            )
        ]
        self.assertEqual(ids, self.SECTION_ORDER)

    def test_sidebar_lists_every_section_in_order(self):
        nav = nodes_by_id(self.root, "sidebar")
        self.assertEqual(len(nav), 1)
        targets = [
            link.attrs["href"][1:]
            for link in descendants(nav[0], lambda node: node.tag == "a")
        ]
        self.assertEqual(targets, self.SECTION_ORDER)

    def test_every_beat_has_the_agreed_parts(self):
        for number in range(8):
            with self.subTest(beat=number):
                node = self.section(f"beat{number}")
                heading = normalized_text(
                    descendants(node, lambda item: item.tag == "h2")[0]
                )
                self.assertTrue(heading.startswith(f"Beat {number} · "), heading)
                text = normalized_text(node)
                self.assertIn("Three tasks, in order.", text)
                self.assertIn("Experiments that go with this beat:", text)
                self.assertGreaterEqual(len(self.code_blocks(f"beat{number}")), 1)
                then = [
                    item
                    for item in descendants(node, lambda n: n.tag == "em")
                    if normalized_text(item) == "Then"
                ]
                self.assertEqual(len(then), 0 if number == 6 else 1)
                self.assertEqual(
                    bool(descendants(node, lambda item: has_class(item, "eq-block"))),
                    number <= 5,
                )

    def test_beat_links_point_to_experiment_anchors(self):
        cards = descendants(
            self.section("experiments"), lambda node: has_class(node, "experiment-card")
        )
        self.assertEqual(
            [card.attrs.get("id") for card in cards], [f"exp{n}" for n in range(1, 8)]
        )
        for number in range(8):
            node = self.section(f"beat{number}")
            for link in descendants(
                node, lambda item: item.tag == "a" and item.attrs["href"].startswith("#exp")
            ):
                if link.attrs["href"] != "#experiments":
                    self.assertRegex(link.attrs["href"], r"^#exp[1-7]$")
        for card in cards:
            beat_links = [
                link.attrs["href"]
                for link in descendants(
                    card, lambda item: item.tag == "a" and item.attrs["href"].startswith("#beat")
                )
            ]
            self.assertEqual(len(beat_links), 1, card.attrs["id"])

    def test_notation_and_scope_statements(self):
        overview = self.text("overview")
        self.assertIn("gravity pulls only downward", overview)
        self.assertIn("What this program is, and what it does not model.", overview)
        self.assertIn("no air resistance", overview)
        beat0 = self.text("beat0")
        self.assertIn("Notation.", beat0)
        self.assertIn("\\(v\\) is the vertical velocity component", beat0)
        self.assertIn("is written \\(v_0\\) with a subscript", beat0)

    # -- equations ---------------------------------------------------------
    EXPECTED_EQUATIONS = {
        1: ("ODE", "beat0"), 2: ("ODE", "beat0"), 3: ("DEFINITION", "beat1"),
        4: ("DERIVED", "beat2"), 5: ("DERIVED", "beat2"), 6: ("ALGORITHM", "beat3"),
        7: ("ALGORITHM", "beat4"), 8: ("ALGORITHM", "beat4"),
        9: ("ALGORITHM", "beat5"), 10: ("ALGORITHM", "beat5"),
        11: ("ALGORITHM", "beat5"),
    }

    def test_numbered_equations_appear_once_in_order_with_kind_and_beat(self):
        found = []
        for number in range(8):
            for label in descendants(
                self.section(f"beat{number}"), lambda node: has_class(node, "eq-label")
            ):
                match = re.match(r"Eq\. (\d+) — ", normalized_text(label))
                if match:
                    kind = descendants(label, lambda node: has_class(node, "kind"))
                    self.assertEqual(len(kind), 1)
                    found.append((int(match.group(1)), normalized_text(kind[0]), f"beat{number}"))
        self.assertEqual(
            found,
            [(n, kind, beat) for n, (kind, beat) in sorted(self.EXPECTED_EQUATIONS.items())],
        )

    def test_unnumbered_derivations_are_tagged_as_derived(self):
        labels = [
            normalized_text(label)
            for label in descendants(self.root, lambda node: has_class(node, "eq-label"))
            if "(unnumbered)" in normalized_text(label)
        ]
        self.assertEqual(len(labels), 5, labels)
        for label in labels:
            self.assertTrue(label.endswith("DERIVED"), label)

    def test_equation_index_matches_the_beats_and_names_real_code(self):
        table = descendants(
            self.section("equations"), lambda node: node.tag == "table"
        )
        self.assertEqual(len(table), 1)
        rows = []
        for row in descendants(table[0], lambda node: node.tag == "tr"):
            cells = descendants(row, lambda node: node.tag == "td")
            if cells:
                rows.append((cells, [normalized_text(cell) for cell in cells]))
        self.assertEqual(len(rows), 11)
        modules = {
            "physics_cannon.py": physics,
            "driver_cannon.py": driver,
        }
        for (cells, texts), (number, (kind, beat)) in zip(
            rows, sorted(self.EXPECTED_EQUATIONS.items())
        ):
            with self.subTest(equation=number):
                self.assertEqual(texts[0], f"Eq. {number}")
                self.assertEqual(texts[1], kind)
                self.assertEqual(texts[3], beat[len("beat"):])
                names = [normalized_text(code) for code in descendants(cells[4], lambda n: n.tag == "code")]
                if number in (4, 5):
                    self.assertEqual(names, [])
                    continue
                self.assertEqual(len(names), 2)
                function_name = names[0].rstrip("()")
                self.assertTrue(
                    callable(getattr(modules[names[1]], function_name, None)),
                    names,
                )

    def test_kind_legend_names_every_kind_used(self):
        legend = normalized_text(
            descendants(
                self.section("beats"),
                lambda node: has_class(node, "note-box")
                and "What kind of statement" in normalized_text(node),
            )[0]
        )
        for kind in ("ODE", "DEFINITION", "DERIVED", "ALGORITHM"):
            self.assertIn(kind, legend)
        self.assertIn("Eleven equations are numbered", legend)

    # -- commands shown ----------------------------------------------------
    def shown_commands(self):
        commands = []
        for number in range(8):
            for block in self.code_blocks(f"beat{number}"):
                for line in block.splitlines():
                    if line.startswith("python main.py"):
                        commands.append((f"beat{number}", shlex.split(line)[2:]))
        return commands

    def test_every_command_shown_in_a_beat_is_accepted_by_the_program(self):
        commands = self.shown_commands()
        self.assertGreaterEqual(len(commands), 14)
        for beat, args in commands:
            with self.subTest(beat=beat, args=args):
                with mock.patch.object(sys, "argv", ["main.py", *args]):
                    parsed = entrypoint.parse_args()
                self.assertTrue(parsed.speed > 0)

    def test_frozen_runs_match_the_live_program(self):
        for name, (args, n, r, h, t) in FROZEN_RUNS.items():
            with self.subTest(run=name):
                self.assertEqual(
                    printed_summary(*args), {"n": n, "R": r, "H": h, "T": t}
                )

    def test_each_beat_quotes_the_printed_values_of_its_runs(self):
        fields = {"n": 1, "R": 2, "H": 3, "T": 4}
        units = {"R": " m", "H": " m", "T": " s"}
        for section_id, quotes in BEAT_QUOTES.items():
            text = self.text(section_id)
            for name, wanted in quotes:
                frozen = FROZEN_RUNS[name]
                for letter in wanted:
                    value = frozen[fields[letter]]
                    with self.subTest(section=section_id, run=name, value=letter):
                        if letter == "n":
                            self.assertRegex(
                                text, rf"\b{re.escape(value)} (trajectory )?samples"
                            )
                        else:
                            self.assertIn(value + units[letter], text)

    def test_commands_shown_are_the_commands_whose_numbers_are_quoted(self):
        wanted = {
            "beat1": [["--angle_deg", "30"], ["--angle_deg", "60"], ["--angle_deg", "0"], ["--angle_deg", "90"]],
            "beat2": [["--speed", "200"], ["--speed", "50"]],
            "beat3": [["--method", "euler", "--dt", "0.5"]],
            "beat4": [["--dt", "1"], ["--method", "euler", "--dt", "1"]],
            "beat7": [["--max_steps", "145"], ["--max_steps", "146"], ["--speed", "1000"]],
        }
        shown = {}
        for beat, args in self.shown_commands():
            shown.setdefault(beat, []).append(args)
        self.assertEqual(shown["beat0"], [[]])
        for beat, expected in wanted.items():
            self.assertEqual(shown[beat], expected, beat)

    # -- derived statements, checked against the program --------------------
    def test_exact_values_quoted_in_beat_two_follow_the_formulas(self):
        exact = exact_launch()
        text = self.text("beat2")
        for expected in (
            f"{exact['u0']:.3f}", f"{exact['T']:.4f}", f"{exact['H']:.3f}",
            f"{exact['R']:.2f}", f"{exact['R']:.4f}",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, text)
        self.assertIn(f"{exact['H']:.2f}", text)
        # Speed scaling: R and H as v0 squared, T as v0.
        base, double = full_run(), full_run(speed=200.0)
        self.assertAlmostEqual(double["R"] / base["R"], 4.0, places=3)
        self.assertAlmostEqual(double["H"] / base["H"], 4.0, places=9)
        self.assertAlmostEqual(double["T"] / base["T"], 2.0, places=3)
        # Complementary angles have the same exact range.
        self.assertAlmostEqual(
            exact_launch(angle_deg=30.0)["R"], exact_launch(angle_deg=60.0)["R"], places=9
        )

    def test_beat_one_explanations_of_limiting_angles(self):
        text = self.text("beat1")
        drift = 100.0 * math.cos(math.radians(90.0))
        self.assertIn(f"{drift:.4g} m/s", text)
        self.assertIn(f"{exact_launch()['u0']:.3f}", text)
        run = full_run(angle_deg=90.0)
        self.assertEqual(f"{run['R']:.5g}", "1.2488e-13")

    def test_euler_samples_lie_on_the_shifted_parabola(self):
        g = physics.g
        exact = exact_launch()
        for dt in (1.0, 0.5, 0.2, 0.1, 0.05):
            with self.subTest(dt=dt):
                run = full_run(dt=dt, method="euler")
                times = np.arange(len(run["hs"])) * dt
                shifted = (exact["v0y"] + 0.5 * g * dt) * times - 0.5 * g * times ** 2
                np.testing.assert_allclose(run["hs"], shifted, rtol=1e-12, atol=1e-9)
                self.assertAlmostEqual(
                    run["H"],
                    exact["H"] + exact["v0y"] * dt / 2 + g * dt * dt / 8,
                    places=8,
                )
                # Flight time and range are T + dt and R + u0 dt, less the
                # small straight-line interpolation error of Beat 5.
                self.assertLess(run["T"], exact["T"] + dt)
                self.assertGreater(run["T"], exact["T"] + dt - 0.02 * dt - 0.03)
                self.assertLess(run["R"], exact["R"] + exact["u0"] * dt)
        run = full_run(dt=0.5, method="euler")
        text = self.text("beat3")
        self.assertIn(f"There are {run['n']} samples instead of {full_run()['n']}.", text)
        for value in (
            f"{exact['R'] + exact['u0'] * 0.5:.2f}",
            f"{exact['T'] + 0.5:.3f}",
            f"{exact['H'] + exact['v0y'] * 0.5 / 2 + g * 0.25 / 8:.2f}",
            f"{run['R'] - exact['R']:.1f}",
            f"{100 * (run['R'] - exact['R']) / exact['R']:.1f}%",
            f"{run['H'] - exact['H']:.1f}",
            f"{run['T'] - exact['T']:.2f}",
        ):
            with self.subTest(value=value):
                self.assertIn(value, text)

    def test_euler_error_lists_quoted_in_beat_three(self):
        exact = exact_launch()
        timesteps = (0.5, 0.2, 0.1, 0.05, 0.02)
        errors = [full_run(dt=dt, method="euler")["R"] - exact["R"] for dt in timesteps]
        quotients = [error / dt for error, dt in zip(errors, timesteps)]
        text = self.text("beat3")
        for values in (errors, quotients):
            words = [f"{value:.3g}" for value in values]
            self.assertIn(", ".join(words[:-1]) + " and " + words[-1], text)
        for quotient in quotients:
            self.assertLess(abs(quotient - exact["u0"]), 0.4)

    def test_euler_at_zero_degrees_keeps_the_ball_up_for_one_step(self):
        run = full_run(angle_deg=0.0, method="euler")
        self.assertEqual(run["n"], 3)
        self.assertAlmostEqual(run["R"], 100.0 * 0.1, places=9)
        self.assertAlmostEqual(run["T"], 0.1, places=12)
        text = self.text("beat3")
        self.assertIn(f"={0.5 * physics.g * 0.1:.2f}\\) m/s", text)

    def test_heun_samples_are_exact_at_every_timestep(self):
        exact = exact_launch()
        for dt in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
            with self.subTest(dt=dt):
                run = full_run(dt=dt)
                times = np.arange(len(run["hs"])) * dt
                analytic = exact["v0y"] * times - 0.5 * physics.g * times ** 2
                self.assertLess(np.max(np.abs(run["hs"] - analytic)), 1e-9)
                self.assertEqual(f"{run['H']:.5g}", "254.93")
                self.assertLess(abs(run["H"] - exact["H"]), 1e-9)
        text = self.text("beat4")
        self.assertIn("by less than 10−9 m", text)

    def test_beat_four_and_five_hand_calculations(self):
        run = full_run(dt=1.0)
        xs, hs = run["xs"], run["hs"]
        text5 = self.text("beat5")
        fraction = hs[-2] / (hs[-2] - hs[-1])
        for value in (
            f"{xs[-2]:.2f}", f"{xs[-1]:.2f}", f"{hs[-2]:.2f}", f"{abs(hs[-1]):.2f}",
            f"{xs[-1] - exact_launch()['R']:.1f}", f"{fraction:.3f}",
        ):
            with self.subTest(value=value):
                self.assertIn(value, text5)
        peak = int(np.argmax(hs))
        self.assertEqual(peak, 7)
        for value in (hs[peak], hs[peak - 1], hs[peak + 1]):
            self.assertIn(f"{value:.2f} m", text5)
        # The parabola through three exact samples gives the exact maximum.
        self.assertAlmostEqual(run["H"], exact_launch()["H"], places=9)
        # The interpolated range and time are those of the hand calculation.
        self.assertAlmostEqual(run["R"], xs[-2] + fraction * (xs[-1] - xs[-2]), places=9)
        self.assertAlmostEqual(run["T"], 14.0 + fraction, places=12)

    def test_beat_five_python_command_prints_the_quoted_samples(self):
        command = [
            line for block in self.code_blocks("beat5") for line in block.splitlines()
            if line.startswith("python -c ")
        ]
        self.assertEqual(len(command), 1)
        source = shlex.split(command[0])[2]
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            exec(compile(source, "<beat5>", "exec"), {})
        run = full_run(dt=1.0)
        expected = f"{run['xs'][-2:]} {run['hs'][-2:]}"
        self.assertEqual(captured.getvalue().strip(), expected)

    def test_interpolated_range_error_follows_the_leading_order_formula(self):
        # The formula of Beat 5 is a leading-order approximation, so the
        # tolerances below are those of an approximation: 12% of the predicted
        # error, and 2% beyond the "at most about" figure.  The exact
        # expression is checked to rounding error in the next test.
        g = physics.g
        worst = 0.0
        for speed in (20.0, 50.0, 100.0, 300.0):
            for angle in (10.0, 25.0, 45.0, 60.0, 80.0):
                for dt in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02):
                    exact = exact_launch(speed, angle)
                    if exact["T"] / dt < 10.0:
                        continue
                    error = exact["R"] - full_run(speed, angle, dt)["R"]
                    phi = (exact["T"] / dt) % 1.0
                    predicted = (g * dt * dt / 2) * (exact["u0"] / exact["v0y"]) * phi * (1 - phi)
                    bound = g * dt * dt * exact["u0"] / (8 * exact["v0y"])
                    with self.subTest(speed=speed, angle=angle, dt=dt):
                        self.assertGreater(error, 0.0)
                        self.assertLessEqual(error, 1.02 * bound)
                        self.assertLessEqual(abs(error - predicted), 0.12 * predicted + 1e-9)
                    worst = max(worst, error / bound)
        self.assertGreater(worst, 0.9)
        text = self.text("beat5")
        exact = exact_launch()
        phi = (exact["T"] / 1.0) % 1.0
        self.assertIn(f"{g * 1.0 / 8:.2f} m for", text)
        self.assertIn(f"{g * 0.01 / 8:.4f} m for", text)
        self.assertIn(f"\\varphi={phi:.3f}\\)", text)
        measured = full_run(dt=1.0)["R"] - exact["R"]
        exact_error = -exact["u0"] * phi * (1 - phi) / (exact["T"] + (1 - 2 * phi))
        self.assertIn(
            f"predicts \u2212{g / 2 * phi * (1 - phi):.3f} m, the exact expression "
            f"\u2212{abs(exact_error):.3f} m, and the measured error is \u2212{abs(measured):.3f} m",
            text,
        )
        for speed, angle, dt, key in ((100.0, 30.0, 0.1, "30"), (100.0, 60.0, 0.1, "60"), (50.0, 45.0, 0.1, "50")):
            with self.subTest(case=key):
                exact = exact_launch(speed, angle)
                phi = (exact["T"] / dt) % 1.0
                self.assertIn(f"{phi:.3f}", text)
                self.assertIn(f"{(g * dt * dt / 2) * (exact['u0'] / exact['v0y']) * phi * (1 - phi):.4f} m", text)

    @staticmethod
    def exact_interpolation_errors(speed, angle, dt):
        """Return (phi, T* - T, R* - R) from the closed form of Beat 5.

        With T/dt = N + phi and every stored sample on the exact parabola,
        Eq. 9 gives T* - T = -phi (1 - phi) dt^2 / (T + (1 - 2 phi) dt), and
        the range error is u0 times that.  This is derived here from the
        sample heights, not read from the Help or from the program.
        """
        exact = exact_launch(speed, angle)
        g = physics.g
        T = exact["T"]
        phi = (T / dt) % 1.0
        whole = int(round(T / dt - phi))
        t_before, t_after = whole * dt, (whole + 1) * dt
        h_before = exact["v0y"] * t_before - 0.5 * g * t_before ** 2
        h_after = exact["v0y"] * t_after - 0.5 * g * t_after ** 2
        fraction = h_before / (h_before - h_after)
        time_error = (whole + fraction) * dt - T
        closed_form = -phi * (1 - phi) * dt * dt / (T + (1 - 2 * phi) * dt)
        return phi, time_error, closed_form, exact["u0"]

    def test_exact_interpolation_error_expression_matches_the_program(self):
        # An independent oracle for the unnumbered "exact error" equation of
        # Beat 5: for every improved-Euler run whose landing is not on a
        # sample, the printed time and range errors equal the closed form to
        # rounding error, with no allowance for an approximation.  The grid
        # includes flights of fewer than ten steps and of less than one step.
        checked = 0
        separated = 0
        for speed in (5.0, 20.0, 50.0, 100.0, 300.0):
            for angle in (10.0, 25.0, 45.0, 60.0, 80.0):
                for dt in (2.0, 1.37, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02):
                    phi, from_samples, closed_form, u0 = self.exact_interpolation_errors(speed, angle, dt)
                    if min(phi, 1.0 - phi) < 1e-6:
                        continue
                    exact = exact_launch(speed, angle)
                    run = full_run(speed, angle, dt)
                    with self.subTest(speed=speed, angle=angle, dt=dt):
                        self.assertAlmostEqual(from_samples, closed_form, delta=1e-12 + 1e-9 * abs(closed_form))
                        self.assertAlmostEqual(run["T"] - exact["T"], closed_form, delta=1e-10)
                        self.assertAlmostEqual(run["R"] - exact["R"], u0 * closed_form, delta=1e-7)
                    leading = -(physics.g * dt * dt / 2) * (u0 / exact["v0y"]) * phi * (1 - phi)
                    if abs(leading - u0 * closed_form) > 0.05 * abs(u0 * closed_form):
                        separated += 1
                    checked += 1
        self.assertGreaterEqual(checked, 200)
        # The oracle can tell the exact expression from the leading-order one.
        self.assertGreaterEqual(separated, 20)

    def test_leading_order_bound_is_not_a_strict_bound(self):
        # Default launch, dt = 1.37 s: the exact error is larger than the
        # "at most about" figure g dt^2 u0 / (8 v0y), so a strict bound is false.
        g = physics.g
        dt = 1.37
        exact = exact_launch()
        error = exact["R"] - full_run(dt=dt)["R"]
        bound = g * dt * dt * exact["u0"] / (8 * exact["v0y"])
        self.assertAlmostEqual(error, 2.3059219701, places=8)
        self.assertAlmostEqual(bound, 2.3007626731, places=8)
        self.assertGreater(error, bound)
        phi, _, closed_form, u0 = self.exact_interpolation_errors(100.0, 45.0, dt)
        self.assertGreater(phi, 0.5)
        self.assertAlmostEqual(-u0 * closed_form, error, places=8)
        text = self.text("beat5")
        self.assertIn(f"\\Delta t={dt}\\) s, where \\(\\varphi={phi:.3f}\\)", text)
        self.assertIn(f"the exact error is \u2212{error:.4f} m", text)
        self.assertIn(f"slightly larger than the figure {bound:.4f} m", text)

    def test_displayed_error_equations_evaluate_to_the_closed_forms(self):
        # The displayed TeX itself is evaluated (not compared as text), so a
        # wrong sign, power or denominator in either equation fails here even
        # if the ledgers were regenerated over it.
        displayed = {}
        for block in descendants(self.section("beat5"), lambda node: has_class(node, "eq-block")):
            text = normalized_text(block)
            if "(unnumbered)" in text:
                displayed[text.split(" (unnumbered)")[0]] = re.search(r"\\\[(.*)\\\]", text, re.S).group(1)
        self.assertEqual(
            sorted(displayed),
            ["Exact error of the interpolated range and time",
             "Leading-order error of the interpolated range"],
        )
        exact_tex = displayed["Exact error of the interpolated range and time"]
        leading_tex = displayed["Leading-order error of the interpolated range"]
        g = physics.g
        differing = 0
        for speed, angle, dt in ((100.0, 45.0, 1.0), (100.0, 45.0, 1.37), (100.0, 45.0, 0.1),
                                 (50.0, 30.0, 0.7), (300.0, 80.0, 0.25), (20.0, 60.0, 2.0)):
            with self.subTest(speed=speed, angle=angle, dt=dt):
                launch = exact_launch(speed, angle)
                T = launch["T"]
                phi = (T / dt) % 1.0
                variables = dict(u0=launch["u0"], v0y=launch["v0y"], phi=phi, dt=dt, g=g, T=T)
                closed_exact = -launch["u0"] * phi * (1 - phi) * dt * dt / (T + (1 - 2 * phi) * dt)
                closed_leading = -(g * dt * dt / 2) * (launch["u0"] / launch["v0y"]) * phi * (1 - phi)
                self.assertAlmostEqual(evaluate_tex(exact_tex, **variables), closed_exact, delta=1e-12 * (1 + abs(closed_exact)))
                self.assertAlmostEqual(evaluate_tex(leading_tex, **variables), closed_leading, delta=1e-12 * (1 + abs(closed_leading)))
                if abs(closed_exact - closed_leading) > 0.01 * abs(closed_exact):
                    differing += 1
                # An edited equation would be noticed: flip one sign in each.
                self.assertNotAlmostEqual(evaluate_tex(exact_tex.replace("1-2", "1+2"), **variables), closed_exact, delta=1e-9)
                self.assertNotAlmostEqual(evaluate_tex(leading_tex.replace("\\frac{g", "\\frac{2g"), **variables), closed_leading, delta=1e-9)
        self.assertGreaterEqual(differing, 3)
        with self.assertRaises(ValueError):
            evaluate_tex(r"x = 2 \\alpha")

    def test_beat_five_calls_the_formula_leading_order_and_gives_the_exact_error(self):
        text = self.text("beat5")
        html = self.section("beat5")
        labels = [
            normalized_text(label)
            for label in descendants(html, lambda node: has_class(node, "eq-label"))
            if "(unnumbered)" in normalized_text(label)
        ]
        self.assertEqual(len(labels), 2, labels)
        self.assertTrue(labels[0].startswith("Exact error of the interpolated range and time"), labels)
        self.assertTrue(labels[1].startswith("Leading-order error of the interpolated range"), labels)
        self.assertIn("This is a leading-order approximation, not an exact result.", text)
        self.assertIn("The error is second order in", text)
        self.assertNotIn("first-order estimate", text)
        self.assertNotIn("first-order", text)
        self.assertIn("short by at most about", text)
        self.assertIn("never too long; they are a little short unless the landing falls exactly on a sample", text)
        self.assertNotIn("always a little short", text)
        self.assertIn("The word \u201cabout\u201d matters", text)

    def test_beat_six_states_the_error_bound_only_approximately(self):
        text = self.text("beat6")
        self.assertIn("lies between 0 and about", text)
        self.assertNotIn("short by between 0 and", text)
        self.assertIn("the exact error of Beat 5", text)

    def test_beat_six_refinement_check_is_stated_with_its_limits(self):
        # Timestep refinement is the standard practical check, not a test that
        # always works, and it says nothing about the model or the program.
        text = self.text("beat6")
        for required in (
            "The standard practical check is the one used here",
            "Halve \\(\\Delta t\\) at least three times",
            "settle into a steady trend",
            "It shows how sensitive the numerical answer is to the timestep.",
            "It does not show that the model is right",
            "it does not show that the program solves the equations it claims to solve",
            "the refinement check described above is the practical one to use there",
            "a tenfold reduction in the leading error takes about ten times as many steps",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)
        for forbidden in ("always works", "you must take ten times as many steps", "halving test"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)

    def test_small_gold_text_uses_a_colour_that_meets_wcag_aa(self):
        style = re.search(r"<style>(.*?)</style>", self.html, re.S).group(1)

        def variable(name):
            return re.search(rf"--{name}:\s*(#[0-9a-fA-F]{{6}})", style).group(1)

        def luminance(colour):
            channels = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
            linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
            return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

        def contrast(one, other):
            high, low = sorted((luminance(one), luminance(other)), reverse=True)
            return (high + 0.05) / (low + 0.05)

        for background in ("eq-bg", "surface", "bg", "note-bg"):
            with self.subTest(background=background):
                self.assertGreaterEqual(contrast(variable("gold-text"), variable(background)), 4.5)
        # The decorative gold stays for borders; it is too light for small text.
        self.assertLess(contrast(variable("gold"), variable("eq-bg")), 4.5)
        allowed = {".eq-label", ".experiment-card .exp-num", ".related-card .rc-ch"}
        gold_text_rules = set()
        for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", style):
            if re.search(r"(^|;)\s*color:\s*var\(--gold\)", body.strip()):
                gold_text_rules.add(" ".join(selector.split()))
        self.assertEqual(gold_text_rules, allowed)
        self.assertIn(
            ".eq-label, .experiment-card .exp-num, .related-card .rc-ch { color: var(--gold-text); }",
            style,
        )

    def test_links_and_signature_boxes_pass_the_automated_accessibility_rules_found_failing(self):
        # axe-core reported a footer link distinguishable only by colour and a
        # scrollable signature box that keyboard users cannot reach.
        style = re.search(r"<style>(.*?)</style>", self.html, re.S).group(1)
        self.assertIn("footer a { text-decoration: underline; }", style)
        self.assertIn(
            ".module-card .sig { white-space: pre-wrap; overflow-wrap: anywhere; overflow-x: visible; }",
            style,
        )

    def test_skip_link_is_the_first_link_and_leads_to_the_main_content(self):
        links = descendants(self.root, lambda node: node.tag == "a" and has_class(node, "skip-link"))
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].attrs["href"], "#content")
        target = nodes_by_id(self.root, "content")
        self.assertEqual([node.tag for node in target], ["main"])
        self.assertEqual(target[0].attrs.get("tabindex"), "-1")
        first = descendants(self.root, lambda node: node.tag == "a" and "href" in node.attrs)[0]
        self.assertIs(first, links[0])

    def test_degraded_update_mirrors_forward_euler(self):
        g = physics.g
        exact = exact_launch()
        with mock.patch.object(driver, "improved_euler_step", degraded_improved_step):
            for dt in (0.5, 0.2, 0.1, 0.05):
                with self.subTest(dt=dt):
                    run = full_run(dt=dt)
                    times = np.arange(len(run["hs"])) * dt
                    reduced = (exact["v0y"] - 0.5 * g * dt) * times - 0.5 * g * times ** 2
                    np.testing.assert_allclose(run["hs"], reduced, rtol=1e-12, atol=1e-9)
                    self.assertLess(run["R"], exact["R"])
                    self.assertAlmostEqual(
                        (run["R"] - exact["R"]) / dt, -exact["u0"], delta=0.6
                    )
            degraded = full_run(dt=0.1)
        values = {key: f"{degraded[key]:.5g}" for key in ("R", "H", "T")}
        card = normalized_text(
            descendants(self.section("experiments"), lambda n: n.attrs.get("id") == "exp7")[0]
        )
        for key in ("R", "H", "T"):
            self.assertIn(values[key], card)

    # -- the tables and the script of Beat 6 -------------------------------
    def table_rows(self, aria_label):
        tables = descendants(
            self.section("beat6"),
            lambda node: node.tag == "div" and node.attrs.get("aria-label") == aria_label,
        )
        self.assertEqual(len(tables), 1, aria_label)
        rows = []
        for row in descendants(tables[0], lambda node: node.tag == "tr"):
            cells = [normalized_text(c) for c in descendants(row, lambda n: n.tag == "td")]
            if cells:
                rows.append(cells)
        return rows

    def test_tables_are_the_live_errors_of_both_methods(self):
        g = physics.g
        exact = exact_launch()
        timesteps = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
        euler = self.table_rows("Forward Euler errors against the exact solution")
        improved = self.table_rows("Improved Euler errors against the exact solution")
        self.assertEqual(len(euler), 7)
        self.assertEqual(len(improved), 7)
        for dt, row_e, row_i in zip(timesteps, euler, improved):
            with self.subTest(dt=dt):
                run = full_run(dt=dt, method="euler")
                self.assertEqual(
                    [row_e[0], row_e[1], row_e[2]],
                    [f"{dt:g}", f"{run['n']:,}", f"{run['R']:.5g}"],
                )
                self.assertEqual(
                    [float(minus_to_hyphen(cell)) for cell in row_e[3:]],
                    [
                        float(f"{run['R'] - exact['R']:+.4g}"),
                        float(f"{(run['R'] - exact['R']) / dt:.4g}"),
                        float(f"{run['H'] - exact['H']:+.4g}"),
                        float(f"{run['T'] - exact['T']:+.4g}"),
                    ],
                )
                run = full_run(dt=dt)
                phi = (exact["T"] / dt) % 1.0
                self.assertEqual(
                    [row_i[0], row_i[1], row_i[2], row_i[5]],
                    [f"{dt:g}", f"{run['n']:,}", f"{run['R']:.5g}", f"{run['H']:.5g}"],
                )
                self.assertEqual(
                    [float(minus_to_hyphen(row_i[k])) for k in (3, 4, 6)],
                    [
                        float(f"{run['R'] - exact['R']:+.4g}"),
                        float(f"{-(g / 2) * dt * dt * phi * (1 - phi):+.4g}"),
                        float(f"{run['T'] - exact['T']:+.4g}"),
                    ],
                )

    def test_beat_six_ratios_and_statements(self):
        exact = exact_launch()
        errors = {dt: full_run(dt=dt)["R"] - exact["R"] for dt in (0.1, 0.05, 0.02)}
        text = self.text("beat6")
        self.assertIn(f"about {errors[0.1] / errors[0.05]:.1f}", text)
        self.assertIn(f"about {errors[0.05] / errors[0.02]:.0f}", text)
        self.assertIn("close to 70.7 in every row", text)
        for dt in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
            run = full_run(dt=dt, method="euler")
            self.assertLess(abs((run["R"] - exact["R"]) / dt - 70.7), 1.5)
            self.assertLess(abs(run["T"] - exact["T"] - dt), 0.02 * dt + 0.02)

    def test_beat_six_script_reproduces_the_tables(self):
        blocks = [b for b in self.code_blocks("beat6") if "from driver_cannon import" in b]
        self.assertEqual(len(blocks), 1)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            exec(compile(blocks[0], "convergence.py", "exec"), {"__name__": "convergence"})
        lines = captured.getvalue().splitlines()
        self.assertEqual(len(lines), 15)
        tables = {
            "euler": self.table_rows("Forward Euler errors against the exact solution"),
            "improved": self.table_rows("Improved Euler errors against the exact solution"),
        }
        seen = {"euler": 0, "improved": 0}
        for line in lines[1:]:
            method, dt, points, range_error, height_error, time_error = line.split()
            row = tables[method][seen[method]]
            seen[method] += 1
            with self.subTest(method=method, dt=dt):
                self.assertEqual(float(dt), float(row[0]))
                self.assertEqual(f"{int(points):,}", row[1])
                self.assertAlmostEqual(
                    float(range_error), float(minus_to_hyphen(row[3])),
                    delta=abs(float(range_error)) * 1e-3,
                )
                self.assertAlmostEqual(
                    float(time_error), float(minus_to_hyphen(row[-1])),
                    delta=abs(float(time_error)) * 1e-3,
                )
                if method == "euler":
                    self.assertAlmostEqual(
                        float(height_error), float(minus_to_hyphen(row[5])),
                        delta=abs(float(height_error)) * 1e-3,
                    )
                else:
                    self.assertLess(abs(float(height_error)), 1e-9)
        self.assertEqual(seen, {"euler": 7, "improved": 7})

    # -- experiments -------------------------------------------------------
    def experiment_cards(self):
        return descendants(
            self.section("experiments"), lambda node: has_class(node, "experiment-card")
        )

    def carried_text(self, card):
        """Text of an experiment card without the Beats additions."""
        parts = []
        for item in card.content:
            if not isinstance(item, HtmlNode):
                parts.append(item)
            elif has_class(item, "exp-beat"):
                continue
            elif item.tag == "p" and normalized_text(item).startswith("Check:"):
                continue
            else:
                parts.append(item.text())
        return " ".join("".join(parts).split())

    def test_experiment_cards_keep_the_classic_text(self):
        digests = [
            hashlib.sha256(self.carried_text(card).encode("utf-8")).hexdigest()
            for card in self.experiment_cards()
        ]
        self.assertEqual(digests, list(CARRIED_EXPERIMENT_SHA256))

    def test_carried_sections_keep_the_classic_text(self):
        digests = {
            name: hashlib.sha256(self.text(name).encode("utf-8")).hexdigest()
            for name in CARRIED_SECTION_SHA256
        }
        self.assertEqual(digests, CARRIED_SECTION_SHA256)

    def test_experiment_checks_quote_the_live_runs(self):
        cards = [normalized_text(card) for card in self.experiment_cards()]
        run = {name: FROZEN_RUNS[name] for name in FROZEN_RUNS}
        self.assertIn(
            f"30° prints a range of {run['a30'][2]} m and 60° prints {run['a60'][2]} m", cards[0]
        )
        self.assertIn(f"40° and 50° both print {run['a40'][2]} m", cards[0])
        self.assertEqual(run["a40"][2], run["a50"][2])
        counts = {
            method: [f"{full_run(dt=dt, method=method)['n']:,}" for dt in (0.5, 0.2, 0.1, 0.05, 0.02)]
            for method in ("euler", "improved")
        }
        self.assertEqual(counts["euler"], ["31", "75", "147", "291", "724"])
        self.assertEqual(counts["improved"], ["30", "74", "146", "290", "723"])
        self.assertIn("31, 75, 147, 291 and 724 with --method euler", cards[1])
        self.assertIn("30, 74, 146, 290 and 723 with --method improved", cards[1])
        self.assertIn(f"--speed 50 prints a range of {run['s50'][2]} m", cards[2])
        self.assertIn(f"--speed 200 prints {run['s200'][2]} m", cards[2])
        self.assertIn(f"--speed 1000 prints {run['s1000'][2]} m", cards[2])

    def test_overlay_exercise_snippet_runs_and_its_quoted_numbers_are_live(self):
        import matplotlib.pyplot as plt

        card = self.experiment_cards()[3]
        blocks = [block.text() for block in descendants(card, lambda n: n.tag == "pre")]
        self.assertEqual(len(blocks), 1)
        with mock.patch.object(plt, "show"):
            try:
                exec(compile(blocks[0], "<experiment4>", "exec"), {})
                axes = plt.gcf().axes[0]
                labels = [line.get_label() for line in axes.lines]
            finally:
                plt.close("all")
        self.assertEqual(labels, [f"{a} degrees" for a in (15, 30, 45, 60, 75)])
        text = normalized_text(card)
        for angle in (15, 30, 60, 75):
            run = full_run(angle_deg=float(angle))
            for key in ("R", "H", "T"):
                self.assertIn(f"{run[key]:.5g}", text)
        self.assertIn("509.83 m at 15° and 509.86 m at 75°", text)
        self.assertIn("883.1 m at 30° and 883.09 m at 60°", text)

    def test_landing_exercise_quotes_raw_and_interpolated_numbers(self):
        card = normalized_text(self.experiment_cards()[4])
        exact = exact_launch()
        run = full_run(dt=1.0)
        self.assertIn(f"printed range is {run['R']:.5g} m", card)
        self.assertIn(f"which is {abs(run['R'] - exact['R']):.2f} m short of the exact {exact['R']:.2f} m", card)
        self.assertIn(f"xs[-1] is {run['xs'][-1]:.2f} m", card)
        self.assertIn(f"which is {run['xs'][-1] - exact['R']:.1f} m too long", card)
        first = full_run(dt=0.5)
        second = full_run(dt=0.1)
        self.assertAlmostEqual(first["xs"][-1], second["xs"][-1], places=9)
        self.assertAlmostEqual(first["xs"][-1], 14.5 * exact["u0"], places=9)
        self.assertIn(f"{second['xs'][-1]:.1f} m for both --dt 0.5 and --dt 0.1", card)

    def test_degraded_update_exercise_keeps_the_translation_and_the_mirror_claim(self):
        card = normalized_text(self.experiment_cards()[6])
        self.assertIn("state[1] + dt * ds2[1]", card)
        self.assertIn("reduced by", card)
        with mock.patch.object(driver, "improved_euler_step", degraded_improved_step):
            run = full_run(dt=0.1)
        euler = full_run(dt=0.1, method="euler")
        for value in (
            f"{run['R']:.5g} m", f"{run['H']:.5g} m", f"{run['T']:.5g} s",
            f"{euler['R']:.5g} m", f"{euler['H']:.5g} m", f"{euler['T']:.5g} s",
        ):
            self.assertIn(value, card)

    # -- limits, output and summary -------------------------------------
    def test_beat_seven_errors_ceiling_and_flat_earth_numbers(self):
        text = self.text("beat7")
        for args in (["--max_steps", "145"], ["--speed", "1e308", "--dt", "1e-3"]):
            with self.subTest(args=args):
                with mock.patch.object(sys, "argv", ["main.py", *args]):
                    with self.assertRaises(SystemExit) as caught:
                        entrypoint.main()
                self.assertIsInstance(caught.exception.code, str)
                self.assertIn(caught.exception.code, text)
        self.assertEqual(len(driver.run_cannon_trajectory(max_steps=146)[0]), 146)
        with self.assertRaises(RuntimeError):
            driver.run_cannon_trajectory(max_steps=145)
        self.assertIn("146 is the smallest ceiling that works", text)
        distance = full_run(speed=1000.0)["R"]
        sagitta = distance ** 2 / (2 * 6.371e6)
        self.assertIn(f"\\approx{sagitta:.0f}\\) m", text)
        self.assertIn(f"by about {math.degrees(distance / 6.371e6):.1f}°", text)

    def test_summary_section_shows_the_live_default_lines(self):
        block = self.code_blocks("summary")
        self.assertEqual(len(block), 1)
        values = printed_summary()
        self.assertEqual(
            block[0].strip().splitlines(),
            [
                f"Range (interpolated ground crossing): {values['R']} m",
                f"Maximum height (parabolic interpolation): {values['H']} m",
                f"Flight time (interpolated ground crossing): {values['T']} s",
            ],
        )
        self.assertIn(f"for the default run is {values['n']}", self.text("summary"))

    def test_overlay_note_quotes_the_error_the_helper_raises(self):
        import matplotlib.pyplot as plt

        text = self.text("output")
        self.assertIn("(label, xs, hs) triple", text)
        with self.assertRaises(ValueError) as caught:
            plotting.plot_cannon_overlay([("only two", [0.0, 1.0])])
        plt.close("all")
        self.assertIn("each trajectory must be a (label, xs, hs) triple", str(caught.exception))
        self.assertIn("each trajectory must be a (label, xs, hs) triple", text)
        self.assertIn("Code that extends or replaces the helper should keep accepting this form", text)

    def test_build_id_statement_matches_what_the_build_id_covers(self):
        text = self.text("modules")
        self.assertIn("The Build ID is a short hash of the four program modules above and of nothing else.", text)
        self.assertIn("are not part of it", text)
        self.assertEqual(tuple(physics.BUILD_ID_COVERS), CORE_MODULE_FILES)
        for name in CORE_MODULE_FILES:
            self.assertIn(name, text)

    def test_quick_start_commands_run(self):
        commands = [
            line for block in self.code_blocks("quickstart") for line in block.splitlines()
            if line.startswith("python main.py")
        ]
        self.assertEqual(len(commands), 4)
        for line in commands:
            args = shlex.split(line)[2:]
            with self.subTest(args=args):
                if args and args[0] in ("--help", "--version"):
                    with mock.patch.object(sys, "argv", ["main.py", *args]), \
                            contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(SystemExit) as caught:
                            entrypoint.parse_args()
                    self.assertEqual(caught.exception.code, 0)
                else:
                    printed_summary(*args)

    def test_algorithm_section_lists_the_five_stages_and_keeps_the_stop_rule(self):
        node = self.section("algorithm")
        first_list = descendants(node, lambda item: item.tag == "ol")[0]
        steps = descendants(first_list, lambda item: item.tag == "li")
        self.assertEqual(len(steps), 5)
        text = self.text("algorithm")
        self.assertIn("Eqs. 9, 11 and 10", text)
        self.assertIn("raise a RuntimeError", text)
        self.assertIn("first stored point with \\(h&lt;0\\)".replace("&lt;", "<"), text)

    def test_equation_numbers_stay_within_the_eleven_defined(self):
        for number in range(8):
            text = self.text(f"beat{number}")
            for match in re.finditer(r"\bEqs?\. (\d+)", text):
                self.assertTrue(1 <= int(match.group(1)) <= 11, match.group(0))
        self.assertNotIn("TBD", self.html)


class TestMaintenanceItems(unittest.TestCase):
    """The three maintenance items of the 1.3.0 Release Notes, made executable."""

    def tearDown(self):
        import matplotlib.pyplot as plt

        plt.close("all")

    def build_id_in_copy(self, directory):
        result = subprocess.run(
            [sys.executable, "main.py", "--version"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1"),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        match = re.fullmatch(r"CannonTrajectory (\S+) \(build ([0-9a-f]{12})\)", result.stdout.strip())
        self.assertIsNotNone(match, result.stdout)
        return match.group(2)

    def test_item_one_build_id_ignores_documentation_and_tests_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary)
            for name in CORE_MODULE_FILES:
                shutil.copy2(MODULE_DIR / name, copy / name)
            shutil.copy2(HELP_PATH, copy / HELP_FILE)
            (copy / "tests").mkdir()
            shutil.copy2(Path(__file__), copy / "tests" / Path(__file__).name)
            (copy / "CannonTrajectory-ReleaseNotes.html").write_text("<p>notes</p>", encoding="utf-8")
            baseline = self.build_id_in_copy(copy)
            self.assertEqual(baseline, physics.BUILD_ID)
            with (copy / HELP_FILE).open("a", encoding="utf-8") as handle:
                handle.write("\n<!-- documentation-only edit -->\n")
            with (copy / "tests" / Path(__file__).name).open("a", encoding="utf-8") as handle:
                handle.write("\n# test-only edit\n")
            (copy / "CannonTrajectory-ReleaseNotes.html").write_text("<p>other notes</p>", encoding="utf-8")
            self.assertEqual(self.build_id_in_copy(copy), baseline)
            for name in CORE_MODULE_FILES:
                original = (copy / name).read_text(encoding="utf-8")
                with self.subTest(module=name):
                    (copy / name).write_text(original + "\n# edit\n", encoding="utf-8")
                    self.assertNotEqual(self.build_id_in_copy(copy), baseline)
                    (copy / name).write_text(original, encoding="utf-8")
            self.assertEqual(self.build_id_in_copy(copy), baseline)

    def test_item_three_overlay_accepts_exactly_the_label_xs_hs_triple(self):
        import matplotlib.pyplot as plt

        xs, hs = np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, -0.5])
        with mock.patch.object(plt, "show"):
            for container in (list, tuple, iter):
                with self.subTest(container=container.__name__):
                    figure, axes = plotting.plot_cannon_overlay(
                        container([("a", xs, hs), ["b", xs, hs], ("c", xs.tolist(), hs.tolist())])
                    )
                    self.assertEqual([line.get_label() for line in axes.lines], ["a", "b", "c"])
                    plt.close(figure)

    def test_item_three_malformed_items_raise_one_clear_value_error(self):
        import matplotlib.pyplot as plt

        xs, hs = np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, -0.5, -1.0])
        malformed = {
            "pair": [(xs, hs)],
            "pair in list of two arrays": [xs, hs],
            "four values": [("a", xs, hs, "extra")],
            "single triple not in a list": ("a", xs, hs),
            "dictionary as the whole collection": {"a": (xs, hs)},
            "three-key integer dictionary as an item": [{7: 1, 8: 2, 9: 3}],
            "three-key string dictionary as an item": [{"a": 1, "b": 2, "c": 3}],
            "dictionary with the right names as an item": [{"label": "a", "xs": xs, "hs": hs}],
            "three-member set as an item": [{1, 2, 3}],
            "three-member frozenset as an item": [frozenset({1, 2, 3})],
            "number": [5],
            "three-character string": ["abc"],
            "bytes": [b"abc"],
            "empty tuple": [()],
        }
        baseline = set(plt.get_fignums())
        with mock.patch.object(plt, "show") as show:
            for name, trajectories in malformed.items():
                with self.subTest(case=name):
                    with self.assertRaisesRegex(
                        ValueError, r"^each trajectory must be a \(label, xs, hs\) triple"
                    ):
                        plotting.plot_cannon_overlay(trajectories)
                    self.assertEqual(set(plt.get_fignums()), baseline)
        show.assert_not_called()

    def test_item_three_plot_cannon_is_the_same_interface_with_one_triple(self):
        xs, hs = np.array([0.0, 1.0]), np.array([0.0, -1.0])
        with mock.patch.object(plotting, "plot_cannon_overlay", return_value=("f", "a")) as overlay:
            self.assertEqual(plotting.plot_cannon(xs, hs), ("f", "a"))
        overlay.assert_called_once()
        (trajectories,), _ = overlay.call_args
        self.assertEqual(len(trajectories), 1)
        label, sent_xs, sent_hs = trajectories[0]
        self.assertEqual(label, "Projectile trajectory")
        self.assertIs(sent_xs, xs)
        self.assertIs(sent_hs, hs)

    def test_run_docstring_lists_every_exception_the_driver_raises(self):
        documented = inspect.getdoc(driver.run_cannon_trajectory)
        for name in ("TypeError", "ValueError", "RuntimeError", "FloatingPointError"):
            self.assertIn(name, documented)
        with self.assertRaises(TypeError):
            driver.run_cannon_trajectory(speed="fast")
        with self.assertRaises(FloatingPointError):
            driver.run_cannon_trajectory(speed=1e308, dt=1e-3)

# ---------------------------------------------------------------------------
# Edge behaviour that the mutation check of Kickoff2 showed was not pinned:
# the three interpolation helpers, the ceiling boundary, the command-line
# error text, and the documented "unknown" build-identifier fallback.
# ---------------------------------------------------------------------------

class TestDriverHelperEdges(unittest.TestCase):
    """The interpolation helpers accept exactly the documented inputs."""

    def test_landing_range_rejects_arrays_that_are_not_a_landing_pair(self):
        rejected = {
            "unequal lengths": ([0.0, 1.0, 2.0], [1.0, -1.0]),
            "two-dimensional": ([[0.0, 1.0]], [[1.0, -1.0]]),
            "one sample": ([0.0], [-1.0]),
            "no samples": ([], []),
            "non-finite x": ([0.0, 1.0, float("nan")], [0.5, 0.5, -0.5]),
            "infinite x": ([0.0, float("inf"), 2.0], [0.5, 0.5, -0.5]),
            "non-finite h": ([0.0, 1.0, 2.0], [0.5, float("nan"), -0.5]),
            "last sample on the ground": ([0.0, 1.0], [1.0, 0.0]),
            "last sample above the ground": ([0.0, 1.0], [1.0, 0.5]),
            "sample before landing below ground": ([0.0, 1.0, 2.0], [1.0, -0.5, -2.0]),
        }
        for name, (xs, hs) in rejected.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    driver.interpolated_landing_range(xs, hs)

    def test_landing_range_values_at_and_between_samples(self):
        self.assertEqual(driver.interpolated_landing_range([0.0, 4.0, 8.0], [1.0, 0.0, -1.0]), 4.0)
        self.assertEqual(driver.interpolated_landing_range([0.0, 4.0], [3.0, -1.0]), 3.0)
        self.assertEqual(driver.interpolated_landing_range([0.0, 4.0], [1.0, -3.0]), 1.0)
        self.assertIsInstance(driver.interpolated_landing_range([0.0, 4.0], [1.0, -3.0]), float)

    def test_maximum_height_rejects_empty_multidimensional_and_non_finite(self):
        for name, hs in {
            "empty": [],
            "two-dimensional": [[1.0, 2.0]],
            "nan": [1.0, float("nan")],
            "infinite": [1.0, float("inf")],
        }.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, "hs must be|height samples must be"):
                    driver.maximum_height(hs)
                with self.assertRaisesRegex(ValueError, "hs must be|height samples must be"):
                    driver.interpolated_maximum_height(hs)
        self.assertEqual(driver.maximum_height([-3.0, -1.0, -2.0]), -1.0)

    def test_parabolic_maximum_uses_the_sampled_value_at_either_end(self):
        self.assertEqual(driver.interpolated_maximum_height([0.0, 1.0, 2.0, 3.0]), 3.0)
        self.assertEqual(driver.interpolated_maximum_height([3.0, 2.0, 1.0]), 3.0)
        self.assertEqual(driver.interpolated_maximum_height([7.0]), 7.0)

    def test_parabolic_maximum_with_a_tied_top_uses_the_first_maximum(self):
        # argmax returns the first of two equal samples, so the parabola
        # through (1, 2, 2) has its vertex 1/8 above the tied top.
        self.assertEqual(driver.interpolated_maximum_height([1.0, 2.0, 2.0, 2.0, 1.0]), 2.125)

    def test_parabolic_maximum_is_never_below_the_largest_sample(self):
        for hs in ([0.0, 9.0, 10.0, 0.0], [0.0, 10.0, 9.0, 0.0], [0.0, 3.0, 4.0, 3.0, 0.0]):
            with self.subTest(hs=hs):
                self.assertGreaterEqual(driver.interpolated_maximum_height(hs), max(hs))

    def test_flight_time_rejects_bad_arrays_and_timesteps(self):
        for name, hs in {
            "one sample": [-1.0],
            "two-dimensional": [[1.0, -1.0]],
            "non-finite": [1.0, float("nan"), -1.0],
            "last sample on the ground": [1.0, 0.0],
            "sample before landing below ground": [1.0, -0.5, -2.0],
        }.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    driver.interpolated_flight_time(hs, 0.1)
        for dt in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(dt=dt):
                with self.assertRaises(ValueError):
                    driver.interpolated_flight_time([1.0, -1.0], dt)
        for dt in ("0.1", True, None):
            with self.subTest(dt=dt):
                with self.assertRaises(TypeError):
                    driver.interpolated_flight_time([1.0, -1.0], dt)

    def test_helpers_refuse_a_non_finite_result_from_finite_samples(self):
        # Every supplied value is finite; the arithmetic is what overflows.
        overflowing = {
            "range: sample spacing overflows": (
                driver.interpolated_landing_range, ([1e308, -1e308], [1.0, -1.0]),
                "the interpolated landing range is not finite for these samples"),
            "range: height difference overflows": (
                driver.interpolated_landing_range, ([0.0, 1.0], [1e308, -1e308]),
                "the interpolated landing range is not finite for these samples"),
            "time: step times count overflows": (
                driver.interpolated_flight_time, ([1.0, 1.0, 1.0, -1.0], 1e308),
                "the interpolated flight time is not finite for these samples"),
            "time: height difference overflows": (
                driver.interpolated_flight_time, ([1e308, -1e308], 1.0),
                "the interpolated flight time is not finite for these samples"),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for name, (helper, arguments, message) in overflowing.items():
                with self.subTest(case=name):
                    with self.assertRaisesRegex(FloatingPointError, f"^{message}$"):
                        helper(*arguments)
            # Large but representable results are still returned.
            self.assertEqual(driver.interpolated_landing_range([0.0, 1e300], [1.0, -1.0]), 5e299)
            self.assertEqual(driver.interpolated_flight_time([1.0, -1.0], 1e300), 5e299)
            self.assertEqual(driver.interpolated_flight_time([0.0, -1.0], 2.0), 0.0)

    def test_run_raises_when_the_ceiling_is_reached_exactly_on_the_ground(self):
        # With g launched straight up at speed g and dt = 1, improved Euler
        # samples h = 0, g/2, 0 exactly, then -g/2.  Three stored points end
        # on h == 0 exactly: the ball has not yet gone below the ground.
        speed = physics.g
        with self.assertRaises(RuntimeError):
            driver.run_cannon_trajectory(speed, 90.0, 1.0, max_steps=3)
        xs, hs = driver.run_cannon_trajectory(speed, 90.0, 1.0, max_steps=4)
        self.assertEqual(len(hs), 4)
        self.assertLess(hs[-1], 0.0)

    def test_overflow_is_reported_as_floating_point_error_for_both_methods(self):
        # The driver has two independent guards against a non-finite state:
        # NumPy's error state and an explicit finiteness test.  Either one
        # alone reports these runs, so this checks the observable result
        # (one FloatingPointError, with its message) for both steppers.
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            driver.run_cannon_trajectory(speed=1e308, dt=1e-3)
        for method in ("euler", "improved"):
            with self.subTest(method=method):
                with self.assertRaises(FloatingPointError):
                    driver.run_cannon_trajectory(speed=1.7e308, angle_deg=0.0, dt=1e300, method=method)


class TestEntryPointMessages(unittest.TestCase):
    """The command-line text that the Help quotes."""

    def error_text(self, *args):
        with mock.patch.object(sys, "argv", ["main.py", *args]), \
                mock.patch.object(entrypoint, "plot_cannon"):
            with self.assertRaises(SystemExit) as caught:
                entrypoint.main()
        return caught.exception.code

    def test_run_errors_are_prefixed_with_the_program_name(self):
        message = self.error_text("--dt", "-1")
        self.assertEqual(message, "CannonTrajectory: dt must be a finite positive number")
        message = self.error_text("--max_steps", "5")
        self.assertRegex(message, r"^CannonTrajectory: max_steps was reached before the projectile landed")

    def test_usage_line_uses_the_program_name(self):
        buffer = io.StringIO()
        with mock.patch.object(sys, "argv", ["anything.py", "--help"]), \
                contextlib.redirect_stdout(buffer):
            with self.assertRaises(SystemExit) as caught:
                entrypoint.main()
        self.assertEqual(caught.exception.code, 0)
        self.assertTrue(buffer.getvalue().startswith("usage: CannonTrajectory "), buffer.getvalue()[:60])


class TestBuildIdFallback(unittest.TestCase):
    """BUILD_ID is the text "unknown" when the sources cannot be read."""

    def build_id_of(self, directory):
        result = subprocess.run(
            [sys.executable, "-c", "import physics_cannon; print(physics_cannon.BUILD_ID)"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_missing_or_undecodable_source_gives_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary)
            shutil.copy2(MODULE_DIR / "physics_cannon.py", copy / "physics_cannon.py")
            self.assertEqual(self.build_id_of(copy), "unknown")
            for name in CORE_MODULE_FILES[1:]:
                shutil.copy2(MODULE_DIR / name, copy / name)
            self.assertRegex(self.build_id_of(copy), r"^[0-9a-f]{12}$")
            (copy / "plot_cannon.py").write_bytes(b"\xff\xfe not utf-8 \x80")
            self.assertEqual(self.build_id_of(copy), "unknown")


# ---------------------------------------------------------------------------
# Ledgers of the numbers and of the key words in the Beats Help.
#
# THESE LEDGERS DETECT CHANGE; THEY DO NOT VERIFY.  Regenerating them after an
# edit records whatever the edited text says, including a mistake, so a green
# ledger test says only that the Help still reads as it did when the ledger
# was last reviewed.  The claims themselves are verified by tests that compute
# them independently of the Help's wording: live command-line output
# (FROZEN_RUNS), the closed-form Eqs. 3 to 5, the exact expression for the
# interpolation error (test_exact_interpolation_error_expression_matches_the_
# program and test_leading_order_bound_is_not_a_strict_bound), the convergence
# tables and script of Beat 6, the equation-to-code mapping, the deliberate
# wording guards on scope and on exact-versus-approximate statements, and the
# digests of the carried sections.
#
# CLAIMS: every number in the prose, code, output and equation text is
# recorded with the two words in front of it and the word after it.
# DIRECTIONS: every word that states a direction (higher, later, shorter, ...),
# names a method, an equation kind or a quantity (range, height, time, speed),
# or gives an order or position (first, last, above, below, before, after,
# more, less, half) is recorded in order.  If an edit changes any entry the
# test fails and shows the entries that moved.  After an intended edit, check
# each moved entry against a live run or an independent calculation, then
# regenerate the data with
#     python -c "import test_physics_cannon as t; t.print_ledgers()"
# and paste the output over the LEDGER_* constants below.
#
# The carried sections (parameters, related, license) are pinned by digest
# elsewhere and are left out here.  The experiments section is included in
# full, carried cards and Beats additions alike.
# ---------------------------------------------------------------------------

LEDGER_SECTIONS = (
    "overview", "beats", "beat0", "beat1", "beat2", "beat3", "beat4", "beat5",
    "beat6", "beat7", "equations", "algorithm", "modules", "quickstart",
    "output", "summary", "experiments",
)
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:e[+-]?\d+)?")
_DIRECTION = re.compile(
    r"(?<![\w-])(?:higher|lower|larger|smaller|longer|shorter|increases|"
    r"decreases|later|earlier|halves|doubles|twice|shrinks|grows|falls|"
    r"rises|exactly|roughly|never|always|does not|peak|landing|Euler|Heun|"
    r"improved|forward|ODE|DEFINITION|DERIVED|ALGORITHM|range|height|time|"
    r"speed|first|last|below|above|more|less|after|before|half)(?![\w-])"
)


def claims_ledger(text):
    """Every number in ``text`` with its two words before and one after."""
    entries = []
    for match in _NUMBER.finditer(text):
        before = text[max(0, match.start() - 120):match.start()].split()[-2:]
        after = text[match.end():match.end() + 120].split()[:1]
        entries.append(" ".join(before) + " [" + match.group(0) + "] " + " ".join(after))
    return entries


def direction_ledger(text):
    """Every direction, method and kind word in ``text``, in order."""
    return [match.group(0).replace(" ", "_") for match in _DIRECTION.finditer(text)]


def print_ledgers():
    """Print paste-ready LEDGER_CLAIMS and LEDGER_DIRECTIONS constants."""
    parser = HtmlTreeParser()
    parser.feed(HELP_PATH.read_text(encoding="utf-8"))
    parser.close()
    claims, words = {}, {}
    for name in LEDGER_SECTIONS:
        (node,) = nodes_by_id(parser.root, name)
        text = normalized_text(node)
        claims[name] = claims_ledger(text)
        words[name] = direction_ledger(text)
    print("LEDGER_CLAIMS = {")
    for name in LEDGER_SECTIONS:
        print(f'    "{name}": r"""')
        for entry in claims[name]:
            assert '"""' not in entry and "\n" not in entry
            print(entry)
        print('""",')
    print("}\n")
    print("LEDGER_DIRECTIONS = {")
    for name in LEDGER_SECTIONS:
        print(f'    "{name}": "{" ".join(words[name])}",')
    print("}")


class TestHelpLedgers(unittest.TestCase):
    """No number or direction word of the Beats Help changes unnoticed."""

    @classmethod
    def setUpClass(cls):
        parser = HtmlTreeParser()
        parser.feed(HELP_PATH.read_text(encoding="utf-8"))
        parser.close()
        cls.root = parser.root
        if help_layout(cls.root) != "beats":
            raise unittest.SkipTest("the shipped Help uses the classic layout")

    def section_text(self, name):
        found = nodes_by_id(self.root, name)
        self.assertEqual(len(found), 1, name)
        return normalized_text(found[0])

    def assert_same(self, name, expected, actual, what):
        if expected != actual:
            import difflib

            diff = "\n".join(
                list(difflib.unified_diff(expected, actual, "recorded", "current", lineterm="", n=0))[:12]
            )
            self.fail(
                f"the {what} of section {name!r} changed; check each moved entry "
                f"against a live run, then regenerate the ledgers:\n{diff}"
            )

    def test_ledgers_cover_exactly_the_uncarried_sections(self):
        self.assertEqual(set(LEDGER_CLAIMS), set(LEDGER_SECTIONS))
        self.assertEqual(set(LEDGER_DIRECTIONS), set(LEDGER_SECTIONS))
        self.assertTrue(set(LEDGER_SECTIONS).isdisjoint(CARRIED_SECTION_SHA256))

    def test_every_number_is_where_it_was_verified(self):
        for name in LEDGER_SECTIONS:
            with self.subTest(section=name):
                self.assert_same(
                    name,
                    LEDGER_CLAIMS[name].strip().split("\n") if LEDGER_CLAIMS[name].strip() else [],
                    claims_ledger(self.section_text(name)),
                    "numbers",
                )

    def test_every_direction_and_method_word_is_where_it_was_verified(self):
        for name in LEDGER_SECTIONS:
            with self.subTest(section=name):
                self.assert_same(
                    name,
                    LEDGER_DIRECTIONS[name].split(),
                    direction_ledger(self.section_text(name)),
                    "direction and method words",
                )


# ---------------------------------------------------------------------------
# Markup and cross-reference checks that the structural mutants of Kickoff2
# showed were not pinned.  TestHelpMarkup runs on either layout;
# TestBeatsHelpReferences is skipped for a classic Help file.
# ---------------------------------------------------------------------------

class TestHelpMarkup(unittest.TestCase):
    VOID = {"meta", "link", "br", "hr", "img", "input", "base", "col", "area", "source", "wbr"}

    @classmethod
    def setUpClass(cls):
        cls.html = HELP_PATH.read_text(encoding="utf-8")

    def test_tags_are_balanced_and_properly_nested(self):
        problems = []
        stack = []

        class Checker(HTMLParser):
            def handle_starttag(inner, tag, attrs):
                if tag not in TestHelpMarkup.VOID:
                    stack.append((tag, inner.getpos()[0]))

            def handle_startendtag(inner, tag, attrs):
                pass

            def handle_endtag(inner, tag):
                if tag in TestHelpMarkup.VOID:
                    return
                if not stack or stack[-1][0] != tag:
                    problems.append(f"</{tag}> at line {inner.getpos()[0]} closes {stack[-1] if stack else 'nothing'}")
                    if any(open_tag == tag for open_tag, _ in stack):
                        while stack and stack[-1][0] != tag:
                            stack.pop()
                        stack.pop()
                else:
                    stack.pop()

        checker = Checker(convert_charrefs=True)
        checker.feed(self.html)
        checker.close()
        problems.extend(f"<{tag}> at line {line} never closed" for tag, line in stack)
        self.assertEqual(problems, [])

    def test_the_only_external_script_is_the_mathjax_bundle(self):
        sources = re.findall(r"<script\b[^>]*\bsrc=\"([^\"]+)\"", self.html)
        self.assertEqual(sources, ["https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"])
        self.assertNotRegex(self.html, r"<link\b[^>]*\bhref=\"https?://")
        self.assertNotRegex(self.html, r"<img\b")


class TestBeatsHelpReferences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        html = HELP_PATH.read_text(encoding="utf-8")
        parser = HtmlTreeParser()
        parser.feed(html)
        parser.close()
        cls.root = parser.root
        if help_layout(cls.root) != "beats":
            raise unittest.SkipTest("the shipped Help uses the classic layout")

    def test_sidebar_entries_carry_the_number_of_the_beat_they_open(self):
        sidebar = nodes_by_id(self.root, "sidebar")[0]
        links = {
            link.attrs["href"]: normalized_text(link)
            for link in descendants(sidebar, lambda node: node.tag == "a")
        }
        for number in range(8):
            with self.subTest(beat=number):
                self.assertTrue(links[f"#beat{number}"].startswith(f"{number} · "), links[f"#beat{number}"])
                heading = descendants(nodes_by_id(self.root, f"beat{number}")[0], lambda n: n.tag == "h2")[0]
                self.assertTrue(normalized_text(heading).startswith(f"Beat {number} · "))

    def test_each_kind_tag_has_the_class_of_its_word(self):
        classes = {"ODE": "kind-ode", "DEFINITION": "kind-def", "DERIVED": "kind-der", "ALGORITHM": "kind-alg"}
        tags = descendants(self.root, lambda node: has_class(node, "kind"))
        self.assertEqual(len(tags), 31)
        for tag in tags:
            with self.subTest(word=normalized_text(tag)):
                self.assertIn(classes[normalized_text(tag)], tag.attrs["class"].split())
                self.assertEqual(len([c for c in tag.attrs["class"].split() if c.startswith("kind-")]), 1)

    EXPECTED_CODE = {
        1: ("derivs_cannon", "physics_cannon.py"), 2: ("derivs_cannon", "physics_cannon.py"),
        3: ("run_cannon_trajectory", "driver_cannon.py"),
        6: ("euler_step", "physics_cannon.py"),
        7: ("improved_euler_step", "physics_cannon.py"),
        8: ("improved_euler_step", "physics_cannon.py"),
        9: ("interpolated_landing_range", "driver_cannon.py"),
        10: ("interpolated_flight_time", "driver_cannon.py"),
        11: ("interpolated_maximum_height", "driver_cannon.py"),
    }

    def test_equation_index_names_the_function_that_implements_each_equation(self):
        table = descendants(nodes_by_id(self.root, "equations")[0], lambda node: node.tag == "table")[0]
        found = {}
        for row in descendants(table, lambda node: node.tag == "tr"):
            cells = descendants(row, lambda node: node.tag == "td")
            if not cells:
                continue
            number = int(normalized_text(cells[0]).split()[1])
            names = [normalized_text(code) for code in descendants(cells[4], lambda n: n.tag == "code")]
            found[number] = (names[0].rstrip("()"), names[1]) if names else None
        expected = dict(self.EXPECTED_CODE)
        expected[4] = expected[5] = None
        self.assertEqual(found, expected)
        for name, module_file in self.EXPECTED_CODE.values():
            module = {"physics_cannon.py": physics, "driver_cannon.py": driver}[module_file]
            self.assertTrue(callable(getattr(module, name)), name)

    def test_every_function_or_file_named_in_a_code_element_exists(self):
        modules = (physics, driver, plotting, entrypoint)
        settings = set(inspect.signature(driver.run_cannon_trajectory).parameters)
        allowed_files = set(CORE_MODULE_FILES) | {"convergence.py"}  # a script the exercise asks for
        for code in descendants(self.root, lambda node: node.tag == "code"):
            text = normalized_text(code)
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*(\(.*\))?", text) or "_" not in text and "." not in text:
                continue
            with self.subTest(code=text):
                name = text.split("(")[0]
                if name.endswith(".py"):
                    self.assertIn(name, allowed_files)
                elif name in settings:
                    continue
                else:
                    self.assertTrue(
                        any(callable(getattr(module, name, None)) for module in modules), name
                    )


LEDGER_CLAIMS = {
    "overview": r"""
in Investigation [1.3] of
(\(R_\oplus \approx [6.4] \times10^6\)
\approx 6.4\times10^ [6] \)
launched at [100] m/s
acceleration \(g= [9.80665] \;\text{m
\(g=9.80665\;\text{m s}^{- [2] }\).
result. Beats [3] to
3 to [6] show
and Beat [7] collects
""",
    "beats": r"""
of Beat [2] ,
hand. Beats [3] to
3 to [6] compare
error (Beat [7] ),
command. Beat [0] runs
launch, Beat [1] changes
angle, Beat [2] changes
solution, Beat [3] changes
timestep, Beat [4] uses
method, Beat [5] looks
samples, Beat [6] tabulates
and Beat [7] collects
of Eqs. [1] and
1 and [2] is
physics: Eqs. [6] to
6 to [11] .
""",
    "beat0": r"""
Beat [0] ·
ground at [100] m/s,
100 m/s, [45] °
the numbers: [146] trajectory
range of [1019.7] m,
height of [254.93] m
time of [14.421] s.
at about [509.9] m
samples are [0.1] s
there are [146] of
one is [145] steps
launch, at [14.5] s.
time of [14.421] s
it (Beat [5] ).
\(-g\): Eq. [1] —
\frac{du}{dt} = [0] .
\] Eq. [2] —
\([u,\; v,\; [0] ,\;
is \(g= [9.80665] \;\text{m
\(g=9.80665\;\text{m s}^{- [2] }\),
of Eqs. [1] and
1 and [2] hold
is \(v_{ [0] y}\)
\(v_{0y}\) (Eq. [3] ).
in Eq. [1] mentions
in Eq. [2] mentions
\(u\). Beat [1] asks
beat: Experiment [1] in
""",
    "beat1": r"""
Beat [1] ·
main.py --angle_deg [30] python
main.py --angle_deg [60] python
main.py --angle_deg [0] python
main.py --angle_deg [90] In
of Beat [0] beside
beside you: [45] °
45° gave [1019.7] m,
1019.7 m, [254.93] m
m and [14.421] s.
First, compare [30] °
30° and [60] °.
ranges are [883.1] m
m and [883.09] m,
different: at [30] °
ball reaches [127.46] m
flight of [10.197] s,
and at [60] °
it reaches [382.39] m
flight of [17.662] s.
physics; Beat [5] explains
angles. At [0] °
first step: [2] samples,
time of [0] .
0. At [90] °
back down: [205] samples,
height of [509.86] m
time of [20.394] s.
prints as [1.2488e-13] m,
m, not [0] .
is \( [100] \cos
is \(100\cos [90] °\)
floating point, [6.123e-15] m/s
the default: [45] °
the five, [1019.7] m.
better (Experiment [1] ).
origin: Eq. [3] —
x_0 = [0] ,
h_0 = [0] ,
\quad v_{ [0] y}
lie between [0] °
0° and [90] °,
launch \(u_0=v_{ [0] y}=70.711\)
launch \(u_0=v_{0y}= [70.711] \)
equal because [45] °
equally. Beat [2] solves
solves Eqs. [1] to
1 to [3] exactly.
beat: Experiment [1] and
and Experiment [4] in
""",
    "beat2": r"""
Beat [2] ·
main.py --speed [200] python
main.py --speed [50] Compare
of Beat [0] :
Beat 0: [1019.7] m,
1019.7 m, [254.93] m
m and [14.421] s.
range of [4078.9] m,
height of [1019.7] m
time of [28.842] s.
the reverse: [254.92] m,
254.92 m, [63.732] m
m and [7.2104] s,
prints as [254.92] and
not as [254.93] ,
effect, Beat [5] .)
They give [1019.72] m,
1019.72 m, [254.93] m
m and [14.4210] s,
Integrating Eq. [1] once
Integrating Eq. [2] once
gives \(v=v_{ [0] y}-gt\),
height: Eq. [4] —
= v_{ [0] y}\,
- \tfrac{ [1] }{2}
- \tfrac{1}{ [2] }
g\, t^ [2] .
when \(h= [0] \)
when \(v= [0] \).
from Eq. [4] :
= \frac{ [2] \,v_{0y}}{g},
= \frac{2\,v_{ [0] y}}{g},
= \frac{v_{ [0] y}^{\,2}}{2g}.
= \frac{v_{0y}^{\, [2] }}{2g}.
= \frac{v_{0y}^{\,2}}{ [2] g}.
with Eq. [3] is
is Eq. [5] —
= \frac{v_0^ [2] \sin
\frac{v_0^2 \sin [2] \theta}{g}.
as \(v_0^ [2] \)
Because \(\sin [2] \theta\)
when \( [2] \theta=90°\),
when \(2\theta= [90] °\),
at \(\theta= [45] °\),
angle is [45] °.
because \(\sin [2] \theta=\sin(180°-2\theta)\),
\(\sin 2\theta=\sin( [180] °-2\theta)\),
\(\sin 2\theta=\sin(180°- [2] \theta)\),
and \( [90] °-\theta\)
saw at [30] °
30° and [60] °.
For \(v_0= [100] \)
m/s at [45] °
are \(u_0=v_{ [0] y}=70.711\)
are \(u_0=v_{0y}= [70.711] \)
m/s, \(T= [14.4210] \)
s, \(H= [254.929] \)
and \(R= [1019.72] \)
precisely, \( [1019.7162] \)
beat: Experiment [1] and
and Experiment [3] in
""",
    "beat3": r"""
Beat [3] ·
of Beat [2] need
of Eqs. [1] and
1 and [2] ,
of Beat [2] .
euler --dt [0.5] Look
with Beat [0] ,
There are [31] samples
instead of [146] .
range is [1054.9] m
m against [1019.72] m,
m, so [35.2] m,
m, or [3.5] %,
height is [272.91] m
m against [254.93] m,
254.93 m, [18.0] m
time is [14.919] s
s against [14.4210] s,
14.4210 s, [0.50] s
with --dt [0.2] ,
--dt 0.2, [0.1] ,
0.2, 0.1, [0.05] and
0.05 and [0.02] .
ranges are [1054.9] ,
are 1054.9, [1033.8] ,
1054.9, 1033.8, [1026.8] ,
1033.8, 1026.8, [1023.2] and
1023.2 and [1021.1] m
\(\Delta t=\) [0.5] ,
t=\) 0.5, [0.2] ,
0.5, 0.2, [0.1] ,
0.2, 0.1, [0.05] and
0.05 and [0.02] s,
errors are [35.2] ,
are 35.2, [14.1] ,
35.2, 14.1, [7.06] ,
14.1, 7.06, [3.53] and
3.53 and [1.41] m.
quotients are [70.4] ,
are 70.4, [70.6] ,
70.4, 70.6, [70.6] ,
70.6, 70.6, [70.7] and
70.7 and [70.7] m/s.
speed \(u_0= [70.711] \)
f(\mathbf s)=[u,\,v,\, [0] ,\,{-g}]\)
of Eqs. [1] and
1 and [2] :
2: Eq. [6] —
\[ \mathbf{s}_{n+ [1] }
out, Eq. [6] says
says \(x_{n+ [1] }=x_n+u_0\Delta
t\), \(v_{n+ [1] }=v_n-g\Delta
and \(h_{n+ [1] }=h_n+v_n\Delta
\(h_n=n\Delta t\,v_{ [0] y}-g\Delta
t\,v_{0y}-g\Delta t^ [2] \,n(n-1)/2\),
t\,v_{0y}-g\Delta t^2\,n(n- [1] )/2\),
t\,v_{0y}-g\Delta t^2\,n(n-1)/ [2] \),
= \bigl(v_{ [0] y}
\tfrac12 g\,t_n^{\, [2] }.
by \(v_{ [0] y}\Delta
\(v_{0y}\Delta t/ [2] +g\Delta
t/2+g\Delta t^ [2] /8\).
t/2+g\Delta t^2/ [8] \).
\(\Delta t= [0.5] \)
range of [1055.07] m,
time of [14.921] s
height of [272.91] m;
program printed [1054.9] m,
1054.9 m, [14.919] s
s and [272.91] m.
line (Beat [5] ).
euler --angle_deg [0] .
range of [0] ,
range of [10] m
time of [0.1] s,
s, from [3] samples.
g\Delta t= [0.49] \)
\(u_0\Delta t= [10.0] \)
beat: Experiment [2] in
""",
    "beat4": r"""
Beat [4] ·
\(\Delta t= [1] \)
main.py --dt [1] python
euler --dt [1] In
run prints [254.93] m,
at \(t= [7] \)
is only [254.71] m:
arch, at [7.210] s,
it (Beat [5] ).
Euler prints [291.51] m,
which is [36.6] m
run prints [1018.5] m
m and [14.404] s,
which are [1.18] m
m and [0.017] s
the exact [1019.72] m
m and [14.4210] s.
Euler prints [1089.3] m
m and [15.405] s,
15.405 s, [69.6] m
m and [0.98] s
of Eq. [6] ,
ends: Eq. [7] —
\] Eq. [8] —
\[ \mathbf{s}_{n+ [1] }
\frac{\Delta t}{ [2] }\bigl[\mathbf{f}(\mathbf{s}_{n})
f(\mathbf s_n)+[ [0] ,\,-g\Delta
s_n)+[0,\,-g\Delta t,\, [0] ,\,0]^T\).
s_n)+[0,\,-g\Delta t,\,0,\, [0] ]^T\).
into Eq. [8] gives
gives \(x_{n+ [1] }=x_n+u_0\Delta
t\), \(v_{n+ [1] }=v_n-g\Delta
\[ h_{n+ [1] }
\tfrac{\Delta t}{ [2] }\bigl[v_n
g\,\Delta t^{ [2] }.
that Eq. [4] gives
and \(t_{n+ [1] }=t_n+\Delta
from Eq. [4] by
less than [10] −9
than 10− [9] m.
\(\Delta t^ [2] \),
exact. Experiment [7] breaks
beat: Experiment [7] in
""",
    "beat5": r"""
Beat [5] ·
\(\Delta t= [1] \)
main.py --dt [1] python
= run(dt= [1.0] );
run(dt=1.0); print(xs[- [2] :],
print(xs[-2:], hs[- [2] :])"
at \(x= [1060.66] \)
and \(h=- [42.59] \)
ball is [42.59] m
\(x\) is [40.9] m
ball was [28.90] m
at \(x= [989.95] \)
m and [42.59] m
at \(x= [1060.66] \)
a fraction [28.90] /
/ ( [28.90] +
(28.90 + [42.59] )
42.59) = [0.404] of
x = [989.95] +
989.95 + [0.404] ×
× ( [1060.66] −
(1060.66 − [989.95] )
989.95) ≈ [1018.5] m,
same way: [14] s
s + [0.404] ×
0.404 × [1] s
s = [14.404] s,
sample is [254.71] m,
m, at [7] s,
samples of [247.74] m
m and [251.87] m,
height is [254.93] m.
maximum (Beat [4] ).
Let \(h_{n- [1] }\ge0\)
and \(h_n< [0] \)
at Eq. [9] —
= x_{n- [1] }
- x_{n- [1] })\,\frac{h_{n-1}}{h_{n-1}-h_n}.
- x_{n-1})\,\frac{h_{n- [1] }}{h_{n-1}-h_n}.
- x_{n-1})\,\frac{h_{n-1}}{h_{n- [1] }-h_n}.
\] Eq. [10] —
= t_{n- [1] }
+ \frac{h_{n- [1] }}{h_{n-1}-h_n}\,\Delta
+ \frac{h_{n-1}}{h_{n- [1] }-h_n}\,\Delta
\] Eq. [9] is
and Eq. [10] the
where \(t_{n- [1] }=(n-1)\Delta
where \(t_{n-1}=(n- [1] )\Delta
If \(h_{n- [1] }=0\)
If \(h_{n-1}= [0] \)
fraction is [0] and
neighbours \(h_{k- [1] }\)
and \(h_{k+ [1] }\),
let \(\delta=h_{k- [1] }-2h_k+h_{k+1}\).
let \(\delta=h_{k-1}- [2] h_k+h_{k+1}\).
let \(\delta=h_{k-1}-2h_k+h_{k+ [1] }\).
If \(\delta< [0] \),
at Eq. [11] —
- \frac{(h_{k+ [1] }-h_{k-1})^2}{8\,\delta}.
- \frac{(h_{k+1}-h_{k- [1] })^2}{8\,\delta}.
- \frac{(h_{k+1}-h_{k-1})^ [2] }{8\,\delta}.
- \frac{(h_{k+1}-h_{k-1})^2}{ [8] \,\delta}.
of Beat [4] lie
g\Delta t^ [2] \,s(1-s)\)
g\Delta t^2\,s( [1] -s)\)
and \( [0] \le\varphi<1\).
and \(0\le\varphi< [1] \).
of Beat [4] ,
parabola, Eqs. [9] and
9 and [10] give
\;=\; -\,\frac{u_0\,\varphi\,( [1] -\varphi)\,\Delta
-\,\frac{u_0\,\varphi\,(1-\varphi)\,\Delta t^{ [2] }}{T
+ ( [1] -2\varphi)\,\Delta
+ (1- [2] \varphi)\,\Delta
denominator \(T+( [1] -2\varphi)\Delta
denominator \(T+(1- [2] \varphi)\Delta
to \(T= [2] v_{0y}/g\),
to \(T=2v_{ [0] y}/g\),
-\,\frac{g\,\Delta t^{ [2] }}{2}\,\frac{u_0}{v_{0y}}\;\varphi\,(1-\varphi).
-\,\frac{g\,\Delta t^{2}}{ [2] }\,\frac{u_0}{v_{0y}}\;\varphi\,(1-\varphi).
-\,\frac{g\,\Delta t^{2}}{2}\,\frac{u_0}{v_{ [0] y}}\;\varphi\,(1-\varphi).
-\,\frac{g\,\Delta t^{2}}{2}\,\frac{u_0}{v_{0y}}\;\varphi\,( [1] -\varphi).
with \(u_0/v_{ [0] y}=1\)
with \(u_0/v_{0y}= [1] \)
for a [45] °
factor \(\varphi( [1] -\varphi)\)
\(g\Delta t^ [2] u_0/(8v_{0y})\),
\(g\Delta t^2u_0/( [8] v_{0y})\),
\(g\Delta t^2u_0/(8v_{ [0] y})\),
which is [1.23] m
\(\Delta t= [1] \)
s and [0.0123] m
\(\Delta t= [0.1] \)
\(\Delta t= [1.37] \)
where \(\varphi= [0.526] \),
is − [2.3059] m,
the figure [2.3008] m.
above \(\varphi= [0.421] \),
predicts − [1.195] m,
expression − [1.182] m,
is − [1.182] m.
beats. The [30] °
30° and [60] °
of Beat [1] have
exact range, [883.10] m.
flights last [10.197] s
s and [17.662] s,
at \(\varphi= [0.972] \)
and \( [0.620] \)
factor \(\varphi( [1] -\varphi)\)
\(\varphi(1-\varphi)\) is [0.028] and
0.028 and [0.236] ,
short by [0.0023] m
m and [0.0067] m,
print as [883.1] and
883.1 and [883.09] .
883.09. The [50] m/s
of Beat [2] has
has \(\varphi= [0.105] \)
short by [0.0046] m,
an exact [254.929] m
the printed [254.92] m.
beat: Experiment [5] in
""",
    "beat6": r"""
Beat [6] ·
of Beat [2] lets
of Eqs. [4] and
4 and [5] ,
g = [9.80665] T
T = [2] *
* ( [100] *
(100 * [0.5] **
0.5 ** [0.5] )
= ( [100] *
(100 * [0.5] **
0.5 ** [0.5] )
0.5) ** [2] /
/ ( [2] *
R = [100.0] **
100.0 ** [2] /
in ( [1.0] ,
in (1.0, [0.5] ,
(1.0, 0.5, [0.2] ,
0.5, 0.2, [0.1] ,
0.2, 0.1, [0.05] ,
0.1, 0.05, [0.02] ,
0.05, 0.02, [0.01] ):
method=method) print(f"{method: [9] s}{dt:<6}{len(xs):<8}"
method=method) print(f"{method:9s}{dt:< [6] }{len(xs):<8}"
method=method) print(f"{method:9s}{dt:<6}{len(xs):< [8] }"
- R:< [14.6] g}"
- H:< [15.6] g}"
error (s) [1171089.3] +69.669.6+36.58+0.9844
(s) 1171089.3+ [69.669] .6+36.58+0.9844
(s) 1171089.3+69.669.6+ [36.58] +0.9844
(s) 1171089.3+69.669.6+36.58+ [0.9844] 0.5311054.9+35.1970.39+17.98+0.4977
(s) 1171089.3+69.669.6+36.58+0.9844 [0.5311054] .9+35.1970.39+17.98+0.4977
1171089.3+69.669.6+36.58+0.9844 0.5311054.9+ [35.1970] .39+17.98+0.4977
1171089.3+69.669.6+36.58+0.9844 0.5311054.9+35.1970.39+ [17.98] +0.4977
1171089.3+69.669.6+36.58+0.9844 0.5311054.9+35.1970.39+17.98+ [0.4977] 0.2751033.8+14.1270.62+7.12+0.1997
1171089.3+69.669.6+36.58+0.9844 0.5311054.9+35.1970.39+17.98+0.4977 [0.2751033] .8+14.1270.62+7.12+0.1997
0.5311054.9+35.1970.39+17.98+0.4977 0.2751033.8+ [14.1270] .62+7.12+0.1997
0.5311054.9+35.1970.39+17.98+0.4977 0.2751033.8+14.1270.62+ [7.12] +0.1997
0.5311054.9+35.1970.39+17.98+0.4977 0.2751033.8+14.1270.62+7.12+ [0.1997] 0.11471026.8+7.06370.63+3.548+0.09989
0.5311054.9+35.1970.39+17.98+0.4977 0.2751033.8+14.1270.62+7.12+0.1997 [0.11471026] .8+7.06370.63+3.548+0.09989
0.2751033.8+14.1270.62+7.12+0.1997 0.11471026.8+ [7.06370] .63+3.548+0.09989
0.2751033.8+14.1270.62+7.12+0.1997 0.11471026.8+7.06370.63+ [3.548] +0.09989
0.2751033.8+14.1270.62+7.12+0.1997 0.11471026.8+7.06370.63+3.548+ [0.09989] 0.052911023.2+3.53370.65+1.771+0.04996
0.2751033.8+14.1270.62+7.12+0.1997 0.11471026.8+7.06370.63+3.548+0.09989 [0.052911023] .2+3.53370.65+1.771+0.04996
0.11471026.8+7.06370.63+3.548+0.09989 0.052911023.2+ [3.53370] .65+1.771+0.04996
0.11471026.8+7.06370.63+3.548+0.09989 0.052911023.2+3.53370.65+ [1.771] +0.04996
0.11471026.8+7.06370.63+3.548+0.09989 0.052911023.2+3.53370.65+1.771+ [0.04996] 0.027241021.1+1.41470.71+0.7076+0.02
0.11471026.8+7.06370.63+3.548+0.09989 0.052911023.2+3.53370.65+1.771+0.04996 [0.027241021] .1+1.41470.71+0.7076+0.02
0.052911023.2+3.53370.65+1.771+0.04996 0.027241021.1+ [1.41470] .71+0.7076+0.02
0.052911023.2+3.53370.65+1.771+0.04996 0.027241021.1+1.41470.71+ [0.7076] +0.02
0.052911023.2+3.53370.65+1.771+0.04996 0.027241021.1+1.41470.71+0.7076+ [0.02] 0.011,4451020.4+0.707170.71+0.3537+0.009999
0.052911023.2+3.53370.65+1.771+0.04996 0.027241021.1+1.41470.71+0.7076+0.02 [0.011] ,4451020.4+0.707170.71+0.3537+0.009999
0.027241021.1+1.41470.71+0.7076+0.02 0.011, [4451020.4] +0.707170.71+0.3537+0.009999
0.027241021.1+1.41470.71+0.7076+0.02 0.011,4451020.4+ [0.707170] .71+0.3537+0.009999
0.027241021.1+1.41470.71+0.7076+0.02 0.011,4451020.4+0.707170.71+ [0.3537] +0.009999
0.027241021.1+1.41470.71+0.7076+0.02 0.011,4451020.4+0.707170.71+0.3537+ [0.009999] Improved
error (s) [1161018.5] −1.182−1.195254.93−0.01672
(s) 1161018.5− [1.182] −1.195254.93−0.01672
(s) 1161018.5−1.182− [1.195254] .93−0.01672
(s) 1161018.5−1.182−1.195254.93− [0.01672] 0.5301019.5−0.1671−0.1631254.93−0.002363
(s) 1161018.5−1.182−1.195254.93−0.01672 [0.5301019] .5−0.1671−0.1631254.93−0.002363
1161018.5−1.182−1.195254.93−0.01672 0.5301019.5− [0.1671] −0.1631254.93−0.002363
1161018.5−1.182−1.195254.93−0.01672 0.5301019.5−0.1671− [0.1631254] .93−0.002363
1161018.5−1.182−1.195254.93−0.01672 0.5301019.5−0.1671−0.1631254.93− [0.002363] 0.2741019.7−0.0182−0.0184254.93−0.0002575
1161018.5−1.182−1.195254.93−0.01672 0.5301019.5−0.1671−0.1631254.93−0.002363 [0.2741019] .7−0.0182−0.0184254.93−0.0002575
0.5301019.5−0.1671−0.1631254.93−0.002363 0.2741019.7− [0.0182] −0.0184254.93−0.0002575
0.5301019.5−0.1671−0.1631254.93−0.002363 0.2741019.7−0.0182− [0.0184254] .93−0.0002575
0.5301019.5−0.1671−0.1631254.93−0.002363 0.2741019.7−0.0182−0.0184254.93− [0.0002575] 0.11461019.7−0.008092−0.008125254.93−0.0001144
0.5301019.5−0.1671−0.1631254.93−0.002363 0.2741019.7−0.0182−0.0184254.93−0.0002575 [0.11461019] .7−0.008092−0.008125254.93−0.0001144
0.2741019.7−0.0182−0.0184254.93−0.0002575 0.11461019.7− [0.008092] −0.008125254.93−0.0001144
0.2741019.7−0.0182−0.0184254.93−0.0002575 0.11461019.7−0.008092− [0.008125254] .93−0.0001144
0.2741019.7−0.0182−0.0184254.93−0.0002575 0.11461019.7−0.008092−0.008125254.93− [0.0001144] 0.052901019.7−0.002983−0.002985254.93−4.219e−05
0.2741019.7−0.0182−0.0184254.93−0.0002575 0.11461019.7−0.008092−0.008125254.93−0.0001144 [0.052901019] .7−0.002983−0.002985254.93−4.219e−05
0.11461019.7−0.008092−0.008125254.93−0.0001144 0.052901019.7− [0.002983] −0.002985254.93−4.219e−05
0.11461019.7−0.008092−0.008125254.93−0.0001144 0.052901019.7−0.002983− [0.002985254] .93−4.219e−05
0.11461019.7−0.008092−0.008125254.93−0.0001144 0.052901019.7−0.002983−0.002985254.93− [4.219] e−05
0.11461019.7−0.008092−0.008125254.93−0.0001144 0.052901019.7−0.002983−0.002985254.93−4.219e− [05] 0.027231019.7−8.995e−05−9.007e−05254.93−1.272e−06
0.11461019.7−0.008092−0.008125254.93−0.0001144 0.052901019.7−0.002983−0.002985254.93−4.219e−05 [0.027231019] .7−8.995e−05−9.007e−05254.93−1.272e−06
0.052901019.7−0.002983−0.002985254.93−4.219e−05 0.027231019.7− [8.995] e−05−9.007e−05254.93−1.272e−06
0.052901019.7−0.002983−0.002985254.93−4.219e−05 0.027231019.7−8.995e− [05] −9.007e−05254.93−1.272e−06
0.052901019.7−0.002983−0.002985254.93−4.219e−05 0.027231019.7−8.995e−05− [9.007] e−05254.93−1.272e−06
0.052901019.7−0.002983−0.002985254.93−4.219e−05 0.027231019.7−8.995e−05−9.007e− [05254.93] −1.272e−06
0.052901019.7−0.002983−0.002985254.93−4.219e−05 0.027231019.7−8.995e−05−9.007e−05254.93− [1.272] e−06
0.052901019.7−0.002983−0.002985254.93−4.219e−05 0.027231019.7−8.995e−05−9.007e−05254.93−1.272e− [06] 0.011,4441019.7−4.273e−05−4.275e−05254.93−6.042e−07
0.052901019.7−0.002983−0.002985254.93−4.219e−05 0.027231019.7−8.995e−05−9.007e−05254.93−1.272e−06 [0.011] ,4441019.7−4.273e−05−4.275e−05254.93−6.042e−07
0.027231019.7−8.995e−05−9.007e−05254.93−1.272e−06 0.011, [4441019.7] −4.273e−05−4.275e−05254.93−6.042e−07
0.027231019.7−8.995e−05−9.007e−05254.93−1.272e−06 0.011,4441019.7− [4.273] e−05−4.275e−05254.93−6.042e−07
0.027231019.7−8.995e−05−9.007e−05254.93−1.272e−06 0.011,4441019.7−4.273e− [05] −4.275e−05254.93−6.042e−07
0.027231019.7−8.995e−05−9.007e−05254.93−1.272e−06 0.011,4441019.7−4.273e−05− [4.275] e−05254.93−6.042e−07
0.027231019.7−8.995e−05−9.007e−05254.93−1.272e−06 0.011,4441019.7−4.273e−05−4.275e− [05254.93] −6.042e−07
0.027231019.7−8.995e−05−9.007e−05254.93−1.272e−06 0.011,4441019.7−4.273e−05−4.275e−05254.93− [6.042] e−07
0.027231019.7−8.995e−05−9.007e−05254.93−1.272e−06 0.011,4441019.7−4.273e−05−4.275e−05254.93−6.042e− [07] Three
close to [70.7] in
of Beat [3] ,
to \(v_{ [0] y}\Delta
\(v_{0y}\Delta t/ [2] \).
as Beat [5] showed,
of Eq. [9] .
\(\Delta t^ [2] \)
\(\Delta t= [0.1] \)
s and [0.05] s
of about [2.7] ,
and between [0.05] s
s and [0.02] s
by about [33] ,
factor \(\varphi( [1] -\varphi)\)
of Beat [5] is
of Beat [5] ,
factor \(\varphi( [1] -\varphi)\)
of Beat [5] ),
of Beat [7] ,
of Beat [2] is
error \(\approx+v_{ [0] y}\Delta
\(\approx+v_{0y}\Delta t/ [2] \).
of Beat [5] ,
lies between [0] and
\(g\Delta t^ [2] u_0/(8v_{0y})\).
\(g\Delta t^2u_0/( [8] v_{0y})\).
\(g\Delta t^2u_0/(8v_{ [0] y})\).
beat: Experiment [6] in
""",
    "beat7": r"""
Beat [7] ·
main.py --max_steps [145] python
main.py --max_steps [146] python
main.py --speed [1000] Look
ceiling. --max_steps [145] prints
and --max_steps [146] runs
flight needs [145] steps,
which is [146] stored
points (Beat [0] ),
0), so [146] is
launch at [1000] m/s
range of [1.0197e+05] m,
m, about [102] km,
height of [25493] m
time of [144.21] s,
s, from [1] ,444
from 1, [444] trajectory
vertical. Over [102] km
about \(R^ [2] /(2R_\oplus)\approx816\)
about \(R^2/( [2] R_\oplus)\approx816\)
with \(R_\oplus= [6.371] \times10^6\)
with \(R_\oplus=6.371\times10^ [6] \)
by about [0.9] °.
main.py --speed [1e308] --dt
1e308 --dt [1e-3] .
height of [0] ,
of Beat [2] possible,
beat: Experiment [3] in
""",
    "equations": r"""
code Eq. [1] ODEHorizontal
physics_cannon.py Eq. [2] ODEVertical
physics_cannon.py Eq. [3] DEFINITIONLaunch
driver_cannon.py Eq. [4] DERIVEDAnalytic
check Eq. [5] DERIVEDRange
check Eq. [6] ALGORITHMForward
physics_cannon.py Eq. [7] ALGORITHMHeun
physics_cannon.py Eq. [8] ALGORITHMHeun
physics_cannon.py Eq. [9] ALGORITHMLanding
driver_cannon.py Eq. [10] ALGORITHMFlight
driver_cannon.py Eq. [11] ALGORITHMMaximum
height (Beat [2] ,
lie (Beat [3] ,
update (Beat [4] ,
approximation (Beat [5] ,
""",
    "algorithm": r"""
steps, Eqs. [6] to
6 to [8] ,
in Beats [3] and
3 and [4] ,
loop, Eqs. [9] to
9 to [11] ,
in Beat [5] .
lie between [0] °
0° and [90] °,
at least [2] ,
with Eq. [3] and
with Eq. [6] or
with Eqs. [7] and
7 and [8] ,
with Eqs. [9] ,
Eqs. 9, [11] and
11 and [10] ,
point \(x=h= [0] \).
with \(h< [0] \).
at \(h= [0] \).
exactly \(h= [0] \),
""",
    "modules": r"""
Defines \(g= [9.80665] \;\text{m
\(g=9.80665\;\text{m s}^{- [2] }\),
derivatives \([u,v, [0] ,-g]\),
""",
    "quickstart": r"""
Requirements Python [3.10] or
main.py --angle_deg [30] python
""",
    "output": r"""
""",
    "summary": r"""
run is [146] .
ground crossing): [1019.7] m
(parabolic interpolation): [254.93] m
ground crossing): [14.421] s
points, Eq. [9] .
neighbours, Eq. [11] .
crossing, Eq. [10] .
figures. Experiments [1] –3,
Experiments 1– [3] ,
Experiments 1–3, [5] ,
in Experiment [6] can
summary (Beat [7] ).
""",
    "experiments": r"""
code. Experiment [1] ·
with Beat [1] .
speed of [100] m/s
timestep of [0.1] s,
main.py --angle_deg [30] .
angles from [1] °
1° to [89] °
89° in [1] °
occurs at [45] °
such as [30] °
30° and [60] °,
effect; Experiment [5] shows
the defaults, [30] °
range of [883.1] m
m and [60] °
60° prints [883.09] m
m (Beat [1] );
(Beat 1); [40] °
40° and [50] °
both print [1004.2] m;
m; and [45] °
45° prints [1019.7] m.
m. Experiment [2] ·
with Beat [3] .
the same [100] m/s,
100 m/s, [45] °
main.py --dt [0.5] ,
0.5, then [0.2] ,
then 0.2, [0.1] ,
0.2, 0.1, [0.05] ,
0.05, and [0.02] s.
in Experiments [5] and
5 and [6] .
points at [0.5] ,
at 0.5, [0.2] ,
0.5, 0.2, [0.1] ,
0.2, 0.1, [0.05] and
0.05 and [0.02] s
s are [31] ,
are 31, [75] ,
31, 75, [147] ,
75, 147, [291] and
291 and [724] with
euler, and [30] ,
and 30, [74] ,
30, 74, [146] ,
74, 146, [290] and
290 and [723] with
later (Beat [3] ).
3). Experiment [3] ·
with Beat [2] .
angle at [45] °
main.py --speed [50] through
through --speed [500] .
to \(v_0^ [2] \).
against \(v_0^ [2] \)
with \( [1] /g\).
from Experiment [5] for
defaults, --speed [50] prints
range of [254.92] m
and --speed [200] prints
200 prints [4078.9] m,
m, against [1019.7] m
m at [100] m/s
m/s (Beat [2] ).
2). --speed [1000] prints
1000 prints [1.0197e+05] m;
guide (Beat [7] ).
7). Experiment [4] ·
with Beat [1] .
angles of [15] °,
of 15°, [30] °,
15°, 30°, [45] °,
30°, 45°, [60] °,
60°, and [75] °
speed = [100] m/s.
in ( [15] ,
in (15, [30] ,
(15, 30, [45] ,
30, 45, [60] ,
45, 60, [75] ):
= run_cannon_trajectory(speed= [100] ,
angle_deg=angle, dt= [0.1] )
times. The [15] °/75°
The 15°/ [75] °
and the [30] °/60°
the 30°/ [60] °
in Experiment [5] rather
timestep of [0.1] s,
of Experiment [5] are
5 are [509.83] m
m at [15] °
15° and [509.86] m
m at [75] °,
75°, and [883.1] m
m at [30] °
30° and [883.09] m
m at [60] °.
heights are [34.154] ,
are 34.154, [475.7] ,
34.154, 475.7, [127.46] and
127.46 and [382.39] m,
times are [5.2781] ,
are 5.2781, [19.699] ,
5.2781, 19.699, [10.197] and
10.197 and [17.662] s.
s. Experiment [5] ·
with Beat [5] .
which \(h< [0] \),
are \(h_{n- [1] }\geq0\)
and \(h_n< [0] \),
= x_{n- [1] }
+ (x_n-x_{n- [1] })
(x_n-x_{n-1}) \frac{h_{n- [1] }}{h_{n-1}-h_n}.
(x_n-x_{n-1}) \frac{h_{n-1}}{h_{n- [1] }-h_n}.
If \(h_{n- [1] }=0\)
If \(h_{n-1}= [0] \)
main.py --dt [1] and
inspects xs[- [1] ];
with --dt [1] the
range is [1018.5] m,
which is [1.18] m
the exact [1019.72] m,
while xs[- [1] ]
xs[-1] is [1060.66] m,
which is [40.9] m
long (Beat [5] ).
method, xs[- [1] ]
xs[-1] is [1025.3] m
both --dt [0.5] and
and --dt [0.1] ,
is at [14.5] s.
s. Experiment [6] ·
with Beat [6] .
value \(R=v_0^ [2] \sin(2\theta)/g\).
value \(R=v_0^2\sin( [2] \theta)/g\).
in Beat [6] ,
them. Experiment [7] ·
with Beat [4] .
is state[ [1] ]
* ds2[ [1] ],
Use Experiment [5] 's
\(\Delta t= [0.1] \)
and Experiment [5] 's
values are [1012.6] m,
1012.6 m, [251.41] m
m and [14.321] s,
s, against [1019.72] m,
1019.72 m, [254.93] m
m and [14.4210] s
exact and [1026.8] m,
1026.8 m, [258.48] m
m and [14.521] s
Euler (Beat [3] ).
""",
}

LEDGER_DIRECTIONS = {
    "overview": "first Euler more improved Euler Heun more later below larger does_not height range height time",
    "beats": "below first time speed range height time later ODE does_not DEFINITION DERIVED ALGORITHM never",
    "beat0": "above range height time range height time landing last below last after later time first below range height time speed height never ODE ODE after speed",
    "beat1": "speed speed speed higher longer below after first range height time height time range speed exactly range speed DEFINITION last speed exactly",
    "beat2": "speed speed exactly range height time speed range height time range height time twice half range range last range height speed time speed below height DERIVED time height DERIVED range time DERIVED speed range range does_not more",
    "beat3": "forward Euler time after forward Euler range height time Euler speed halves Euler ALGORITHM Euler exactly after height Euler DERIVED Euler exactly speed longer exactly higher range time height height exactly range time smaller landing more range Euler range time speed",
    "beat4": "improved Euler Euler Euler Heun first Euler improved first height improved falls Euler range time improved Euler range improved Euler Heun ALGORITHM Heun ALGORITHM height Heun height DERIVED exactly improved less Heun height",
    "beat5": "peak last first below range height time first last last range time last below range range above below range time time height height last ALGORITHM time ALGORITHM range time time last above exactly landing height height peak height ALGORITHM falls first last peak more half larger height range time peak below range time never landing falls exactly landing falls exactly range range time DERIVED range DERIVED never more range more smaller larger above earlier range last falls",
    "beat6": "last range height time improved last first Euler range Euler range height Euler range speed height more Euler range improved height range time smaller landing roughly range falls smaller Euler roughly improved does_not does_not Euler range height Euler height range above",
    "beat7": "first before larger range range height time first larger speed range height always height",
    "equations": "after Euler time height time height DERIVED Euler DERIVED Heun height DERIVED range DERIVED after first below",
    "algorithm": "after speed improved speed last height last height last first below range height time before first last height below height height first below first above last below range time last first exactly landing height",
    "modules": "Heun after first does_not above does_not first",
    "quickstart": "later",
    "output": "first below first exactly",
    "summary": "first height time last below height time time",
    "experiments": "below roughly more range height time speed range after range range range range range height below improved landing more improved Euler later speed range height first range first speed range more range landing later larger range speed speed range landing first below landing after first last exactly landing time range range range improved last improved landing range range after forward Euler first Heun less height after height before landing does_not Euler speed range time forward Euler",
}


if __name__ == "__main__":
    unittest.main(verbosity=2)
