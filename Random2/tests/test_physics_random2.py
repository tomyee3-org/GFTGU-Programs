"""Regression tests for the Random2 program.

The discovery logic deliberately supports both the repository layout
(``Random2/tests/test_physics_random2.py``) and a flattened upload in which this
test file sits beside the four program modules.
"""

from __future__ import annotations

import ast
import hashlib
import math
import random
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CORE_MODULE_FILES = (
    "random2_physics.py",
    "random2_driver.py",
    "main.py",
    "random2_plot.py",
)
HELP_FILENAME = "Random2.html"


def find_module_dir(start: Path) -> Path:
    """Find the nearest ancestor containing all four core program modules."""
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        if all((directory / name).is_file() for name in CORE_MODULE_FILES):
            return directory

    names = ", ".join(CORE_MODULE_FILES)
    raise FileNotFoundError(
        f"Could not find a directory containing the Random2 core files: {names}"
    )


MODULE_DIR = find_module_dir(Path(__file__))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import random2_driver as driver
import random2_physics as physics
import random2_plot as plot


def recompute_build_id() -> str:
    digest = hashlib.sha256()
    for name in CORE_MODULE_FILES:
        with (MODULE_DIR / name).open(
            "r", encoding="utf-8", newline=None
        ) as source:
            content = source.read().encode("utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()[:12]


class TestPortableDiscovery(unittest.TestCase):
    def test_finds_module_from_tests_directory(self):
        self.assertEqual(find_module_dir(Path(__file__).parent), MODULE_DIR)

    def test_finds_module_from_module_file(self):
        self.assertEqual(
            find_module_dir(MODULE_DIR / "random2_physics.py"), MODULE_DIR
        )

    def test_finds_nearest_matching_ancestor(self):
        with tempfile.TemporaryDirectory(dir=MODULE_DIR) as temporary:
            nested = Path(temporary) / "one" / "two"
            nested.mkdir(parents=True)
            self.assertEqual(find_module_dir(nested), MODULE_DIR)

    def test_missing_modules_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "Random2 core files"):
                find_module_dir(Path(temporary))


class TestReleaseMetadataAndCompatibility(unittest.TestCase):
    def test_all_core_files_exist(self):
        self.assertTrue(all((MODULE_DIR / name).is_file() for name in CORE_MODULE_FILES))

    def test_model_version_has_semantic_version_form(self):
        self.assertRegex(physics.MODEL_VERSION, r"^\d+\.\d+\.\d+$")

    def test_build_id_covers_exactly_the_four_core_files(self):
        self.assertTupleEqual(physics.BUILD_ID_COVERS, CORE_MODULE_FILES)

    def test_build_id_matches_core_file_contents(self):
        self.assertNotEqual(physics.BUILD_ID, "unknown")
        self.assertEqual(physics.BUILD_ID, recompute_build_id())

    def test_help_version_and_build_match_program(self):
        help_text = (MODULE_DIR / HELP_FILENAME).read_text(encoding="utf-8")
        match = re.search(
            r'id="version_build"[^>]*>\s*Version\s+([^&<\s]+)'
            r'&nbsp;(?:&nbsp;)+Build\s+([0-9a-f]{12})',
            help_text,
        )
        self.assertIsNotNone(match, "Help file lacks a parseable version/build line")
        self.assertEqual(match.group(1), physics.MODEL_VERSION)
        self.assertEqual(match.group(2), physics.BUILD_ID)

    def test_help_describes_current_modes_and_defaults(self):
        help_text = (MODULE_DIR / HELP_FILENAME).read_text(encoding="utf-8")
        for required in (
            'scaled_distance',
            'walk2d',
            'maxSteps</code></td><td><code>4096',
            'nTrials</code></td><td><code>100',
            'reference_steps</code></td><td><code>2000',
            'n_walks</code></td><td><code>4',
            'mean_free_path</code></td><td><code>1.0',
            'radius_factor</code></td><td><code>2.0',
            'ray_length_factor</code></td><td><code>0.6',
            'step_cap</code></td><td><code>200000',
            'upper right',
        ):
            self.assertIn(required, help_text)

    def test_student_help_does_not_embed_java_listing(self):
        help_text = (MODULE_DIR / HELP_FILENAME).read_text(encoding="utf-8")
        self.assertNotIn("Listing of the Java code", help_text)
        self.assertNotIn("Math.random()", help_text)

    def test_all_program_files_parse_with_python_310_grammar(self):
        for name in CORE_MODULE_FILES:
            source = (MODULE_DIR / name).read_text(encoding="utf-8")
            ast.parse(source, filename=name, feature_version=(3, 10))

    def test_version_command_runs_from_module_directory(self):
        completed = subprocess.run(
            [sys.executable, str(MODULE_DIR / "main.py"), "--version"],
            cwd=MODULE_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        expected = f"Random2 {physics.MODEL_VERSION} (build {physics.BUILD_ID})"
        self.assertEqual(completed.stdout.strip(), expected)


class TestPhysicsValidation(unittest.TestCase):
    def test_component_uniform_uses_all_three_components(self):
        with mock.patch.object(physics.random, "random", side_effect=[0.0, 0.5, 1.0]):
            self.assertEqual(physics.generate_component_step("uniform"), (-1.0, 0.0, 1.0))

    def test_component_gaussian_uses_three_standard_normals(self):
        with mock.patch.object(physics.random, "gauss", side_effect=[-2.0, 0.0, 3.5]) as gauss:
            self.assertEqual(physics.generate_component_step("gaussian"), (-2.0, 0.0, 3.5))
            self.assertEqual(gauss.call_args_list, [mock.call(0.0, 1.0)] * 3)

    def test_invalid_component_distribution_is_rejected(self):
        for value in ("normal", "Uniform", "", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "uniform.*gaussian"):
                physics.generate_component_step(value)

    def test_isotropic_step_has_requested_length(self):
        random.seed(8101)
        for _ in range(100):
            self.assertAlmostEqual(math.hypot(*physics.generate_isotropic_step(2.75)), 2.75, places=12)

    def test_isotropic_step_angles(self):
        with mock.patch.object(physics.random, "random", return_value=0.0):
            self.assertEqual(physics.generate_isotropic_step(3.0), (3.0, 0.0))
        with mock.patch.object(physics.random, "random", return_value=0.25):
            x, y = physics.generate_isotropic_step(2.0)
            self.assertAlmostEqual(x, 0.0, places=14)
            self.assertAlmostEqual(y, 2.0, places=14)

    def test_isotropic_step_rejects_invalid_lengths(self):
        for value in (0, -1, math.inf, -math.inf, math.nan, True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                physics.generate_isotropic_step(value)

    def test_default_radius_formula(self):
        self.assertAlmostEqual(physics.default_radius(2000, 1.5, 2.0), 3.0 * math.sqrt(2000))

    def test_default_radius_rejects_invalid_parameters(self):
        for value in (0, -1, 1.5, True, "2"):
            with self.subTest(reference_steps=value), self.assertRaises(ValueError):
                physics.default_radius(value)
        for parameter in ("mean_free_path", "radius_factor"):
            for value in (0, -1, math.inf, math.nan, True, "1"):
                kwargs = {parameter: value}
                with self.subTest(parameter=parameter, value=value), self.assertRaises(ValueError):
                    physics.default_radius(4, **kwargs)

    def test_circle_crossing_fraction_for_radial_segment(self):
        self.assertAlmostEqual(physics.circle_crossing_fraction((0, 0), (2, 0), 1), 0.5)

    def test_circle_crossing_fraction_for_oblique_segment(self):
        t = physics.circle_crossing_fraction((0.0, 0.0), (2.0, 2.0), math.sqrt(2.0))
        self.assertAlmostEqual(t, 0.5)

    def test_circle_tangent_and_no_crossing_cases(self):
        self.assertAlmostEqual(physics.circle_crossing_fraction((-2, 1), (2, 1), 1), 0.5)
        self.assertIsNone(physics.circle_crossing_fraction((0, 0), (0.5, 0), 1))
        self.assertIsNone(physics.circle_crossing_fraction((0, 0), (0, 0), 1))

    def test_circle_crossing_rejects_bad_points_and_radius(self):
        bad_points = ((1,), (1, 2, 3), (math.nan, 0), (math.inf, 0), (True, 0), ("x", 0))
        for value in bad_points:
            with self.subTest(p0=value), self.assertRaises(ValueError):
                physics.circle_crossing_fraction(value, (2, 0), 1)
            with self.subTest(p1=value), self.assertRaises(ValueError):
                physics.circle_crossing_fraction((0, 0), value, 1)
        for radius in (0, -1, math.nan, math.inf, True, "1"):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                physics.circle_crossing_fraction((0, 0), (2, 0), radius)

    def test_point_at_interpolates_and_extrapolates(self):
        self.assertEqual(physics.point_at((1, 2), (5, 10), 0.25), (2.0, 4.0))
        self.assertEqual(physics.point_at((1, 2), (5, 10), 1.5), (7.0, 14.0))

    def test_point_at_rejects_invalid_input(self):
        for t in (math.nan, math.inf, True, "0.5"):
            with self.subTest(t=t), self.assertRaises(ValueError):
                physics.point_at((0, 0), (1, 1), t)
        with self.assertRaises(ValueError):
            physics.point_at((0,), (1, 1), 0.5)


class TestDriver(unittest.TestCase):
    def test_single_walk_3d_computes_distance_and_mean_step(self):
        steps = [(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)]
        with mock.patch.object(driver, "generate_component_step", side_effect=steps):
            distance, mean_step = driver._perform_single_walk_3d(2)
        self.assertAlmostEqual(distance, math.sqrt(5.0))
        self.assertAlmostEqual(mean_step, 1.5)

    def test_single_walk_rejects_invalid_step_count(self):
        for value in (0, -1, 1.5, True, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                driver._perform_single_walk_3d(value)

    def test_trials_average_scaled_distance(self):
        results = [(6.0, 2.0), (9.0, 3.0), (4.0, 1.0)]
        with mock.patch.object(driver, "_perform_single_walk_3d", side_effect=results):
            self.assertAlmostEqual(driver._perform_trials_3d(4, 3), 10.0 / 3.0)

    def test_trials_reject_nonpositive_or_nonfinite_generated_mean(self):
        for mean_step in (0.0, -1.0, math.nan, math.inf):
            with mock.patch.object(driver, "_perform_single_walk_3d", return_value=(1.0, mean_step)):
                with self.subTest(mean_step=mean_step), self.assertRaises(RuntimeError):
                    driver._perform_trials_3d(2, 1)

    def test_scaled_experiment_uses_integer_halving_and_returns_ascending_counts(self):
        with mock.patch.object(driver, "_perform_trials_3d", side_effect=lambda n, *_: n + 0.5):
            lengths, averages = driver.run_scaled_distance_experiment(10, 3)
        self.assertEqual(lengths, [2.0, 5.0, 10.0])
        self.assertEqual(averages, [2.5, 5.5, 10.5])

    def test_scaled_experiment_validates_inputs(self):
        invalid_cases = (
            (1, 10, "uniform"),
            (0, 10, "uniform"),
            (8, 0, "uniform"),
            (8, True, "uniform"),
            (8, 10, "bad"),
        )
        for args in invalid_cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                driver.run_scaled_distance_experiment(*args)

    def test_walk2d_clips_crossing_and_builds_ray(self):
        with mock.patch.object(driver, "generate_isotropic_step", return_value=(1.0, 0.0)):
            result = driver.run_walk2d(
                reference_steps=4,
                n_walks=1,
                radius=2.5,
                mean_free_path=1.0,
                ray_length_factor=0.6,
                step_cap=10,
            )
        walk = result.walks[0]
        self.assertTrue(walk.escaped)
        self.assertEqual(walk.steps_taken, 3)
        self.assertEqual(walk.points[-1], (2.5, 0.0))
        self.assertEqual(walk.ray, ((2.5, 0.0), (4.0, 0.0)))
        self.assertEqual(result.model_version, physics.MODEL_VERSION)
        self.assertEqual(result.build_id, physics.BUILD_ID)

    def test_walk2d_can_suppress_outgoing_ray(self):
        with mock.patch.object(driver, "generate_isotropic_step", return_value=(1.0, 0.0)):
            result = driver.run_walk2d(n_walks=1, radius=0.5, ray_length_factor=0, step_cap=2)
        self.assertTrue(result.walks[0].escaped)
        self.assertIsNone(result.walks[0].ray)

    def test_walk2d_records_step_cap_without_false_escape(self):
        alternating = [(1.0, 0.0), (-1.0, 0.0)] * 2
        with mock.patch.object(driver, "generate_isotropic_step", side_effect=alternating):
            result = driver.run_walk2d(n_walks=1, radius=5, step_cap=4)
        walk = result.walks[0]
        self.assertFalse(walk.escaped)
        self.assertEqual(walk.steps_taken, 4)
        self.assertIsNone(walk.ray)
        self.assertEqual(len(walk.points), 5)

    def test_walk2d_crossing_invariant_failure_is_explicit(self):
        with (
            mock.patch.object(driver, "generate_isotropic_step", return_value=(2.0, 0.0)),
            mock.patch.object(driver, "circle_crossing_fraction", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "crossing.*no circle"):
                driver.run_walk2d(n_walks=1, radius=1, step_cap=1)

    def test_walk2d_validates_all_parameters(self):
        cases = {
            "reference_steps": (0, -1, 1.5, True),
            "n_walks": (0, -1, 1.5, True),
            "step_cap": (0, -1, 1.5, True),
            "mean_free_path": (0, -1, math.nan, math.inf, True, "1"),
            "radius_factor": (0, -1, math.nan, math.inf, True, "1"),
            "radius": (0, -1, math.nan, math.inf, True, "1"),
            "ray_length_factor": (-1, math.nan, math.inf, True, "1"),
        }
        for name, values in cases.items():
            for value in values:
                kwargs = {"n_walks": 1, "step_cap": 1}
                kwargs[name] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    driver.run_walk2d(**kwargs)

    def test_zero_ray_length_is_valid(self):
        result = driver.run_walk2d(n_walks=1, radius=100, ray_length_factor=0, step_cap=1)
        self.assertEqual(result.walks[0].steps_taken, 1)


class TestStatisticalProperties(unittest.TestCase):
    def test_uniform_component_statistics_and_mean_step_length(self):
        random.seed(20817)
        sample = [physics.generate_component_step("uniform") for _ in range(50_000)]
        for coordinate in range(3):
            values = [step[coordinate] for step in sample]
            self.assertGreaterEqual(min(values), -1.0)
            self.assertLessEqual(max(values), 1.0)
            self.assertAlmostEqual(sum(values) / len(values), 0.0, delta=0.012)
        mean_length = sum(math.sqrt(sum(x * x for x in step)) for step in sample) / len(sample)
        self.assertAlmostEqual(mean_length, 0.9605919565, delta=0.006)

    def test_isotropic_step_has_no_preferred_mean_direction(self):
        random.seed(314159)
        sample = [physics.generate_isotropic_step() for _ in range(30_000)]
        mean_x = sum(x for x, _ in sample) / len(sample)
        mean_y = sum(y for _, y in sample) / len(sample)
        self.assertAlmostEqual(mean_x, 0.0, delta=0.015)
        self.assertAlmostEqual(mean_y, 0.0, delta=0.015)

    def test_scaled_distance_exponent_is_close_to_one_half(self):
        random.seed(8675309)
        lengths, averages = driver.run_scaled_distance_experiment(512, 180, "uniform")
        log_x = [math.log(x) for x in lengths]
        log_y = [math.log(y) for y in averages]
        mean_x = sum(log_x) / len(log_x)
        mean_y = sum(log_y) / len(log_y)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_x, log_y)) / sum(
            (x - mean_x) ** 2 for x in log_x
        )
        self.assertAlmostEqual(slope, 0.5, delta=0.08)

    def test_escape_steps_follow_quadratic_radius_scaling(self):
        def mean_escape_steps(radius: float) -> float:
            result = driver.run_walk2d(
                reference_steps=1,
                n_walks=500,
                radius=radius,
                mean_free_path=1.0,
                ray_length_factor=0,
                step_cap=20_000,
            )
            self.assertTrue(all(walk.escaped for walk in result.walks))
            return sum(walk.steps_taken for walk in result.walks) / len(result.walks)

        random.seed(271828)
        small = mean_escape_steps(6.0)
        large = mean_escape_steps(12.0)
        self.assertAlmostEqual(large / small, 4.0, delta=0.75)


class TestPlotting(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_scaled_plot_accepts_valid_data(self):
        with mock.patch.object(plot.plt, "show") as show:
            plot.plot_scaled_distance([2.0, 4.0], [1.5, 2.1])
        show.assert_called_once_with()

    def test_scaled_plot_rejects_invalid_data(self):
        bad_pairs = (
            ([], []),
            ([1], [1, 2]),
            ([0], [1]),
            ([1], [-1]),
            ([math.nan], [1]),
            ([1], [math.inf]),
            ([True], [1]),
            (["1"], [1]),
        )
        for lengths, averages in bad_pairs:
            with self.subTest(lengths=lengths, averages=averages), self.assertRaises(ValueError):
                plot.plot_scaled_distance(lengths, averages)

    def test_walk_plot_marks_capped_path(self):
        result = driver.Walk2DResult(
            model_version=physics.MODEL_VERSION,
            build_id=physics.BUILD_ID,
            radius=2.0,
            mean_free_path=1.0,
            reference_steps=4,
            step_cap=1,
            walks=[driver.WalkPath(points=[(0, 0), (1, 0)], escaped=False, steps_taken=1)],
        )
        with mock.patch.object(plot.plt, "show"):
            plot.plot_walk2d(result)
        markers = [line.get_marker() for line in plt.gca().lines]
        self.assertIn("x", markers)
        annotation_text = "\n".join(text.get_text() for text in plt.gca().texts)
        self.assertIn("x = step cap reached", annotation_text)

    def test_walk_plot_rejects_invalid_corner(self):
        result = driver.run_walk2d(n_walks=1, radius=100, step_cap=1)
        with self.assertRaisesRegex(ValueError, "corner"):
            plot.plot_walk2d(result, "center")


if __name__ == "__main__":
    unittest.main(verbosity=2)
