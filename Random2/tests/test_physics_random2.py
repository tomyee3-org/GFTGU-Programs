"""Regression tests for the Random2 program.

The discovery logic deliberately supports both the repository layout
(``Random2/tests/test_physics_random2.py``) and a flattened upload in which this
test file sits beside the four program modules.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import html as html_module
import io
import math
import os
import random
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from html.parser import HTMLParser
from pathlib import Path
from typing import NamedTuple
from unittest import mock


CORE_MODULE_FILES = (
    "random2_physics.py",
    "random2_driver.py",
    "main.py",
    "random2_plot.py",
)
# The Beats Help is accepted under either of its names; the Reference Guide
# version, Random2-original.html, is optional and never required.
HELP_FILENAMES = ("Random2-claude.html", "Random2.html")


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
import numpy as np

import random2_driver as driver
import random2_physics as physics
import random2_plot as plot
import main as cli


class _HelpSemanticParser(HTMLParser):
    """Collect visible Help text, version text, and table cells."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.visible_text = []
        self.version_text = []
        self.table_rows = []
        self._in_version = False
        self._in_cell = False
        self._cell_text = []
        self._row = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id") == "version_build":
            self._in_version = True
        if tag in ("td", "th"):
            self._in_cell = True
            self._cell_text = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._in_cell:
            self._row.append(" ".join("".join(self._cell_text).split()))
            self._in_cell = False
        elif tag == "tr":
            if self._row:
                self.table_rows.append(tuple(self._row))
            self._row = []
        elif tag == "p" and self._in_version:
            self._in_version = False

    def handle_data(self, data):
        self.visible_text.append(data)
        if self._in_version:
            self.version_text.append(data)
        if self._in_cell:
            self._cell_text.append(data)


def find_help_file(module_dir: Path) -> Path:
    """Find the Beats Help beside the modules or in GFTGU-Documentation/Random2/."""
    program_name = "Random2"
    candidates = []
    for help_filename in HELP_FILENAMES:
        candidates.append(module_dir / help_filename)
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
        f"Could not find {' or '.join(HELP_FILENAMES)} beside the modules or in "
        "GFTGU-Documentation/Random2/."
    )


HELP_FILE = find_help_file(MODULE_DIR)
HELP_HTML = HELP_FILE.read_text(encoding="utf-8")


def parse_help_file() -> _HelpSemanticParser:
    parser = _HelpSemanticParser()
    parser.feed(HELP_HTML)
    parser.close()
    return parser


@contextmanager
def isolated_rng(seed: int):
    """Route program randomness through a private RNG without global mutation."""
    rng = random.Random(seed)
    with mock.patch.object(physics, "random", rng):
        yield rng


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
        help_data = parse_help_file()
        version_text = " ".join("".join(help_data.version_text).split())
        match = re.search(
            r"^Version\s+(\S+)\s+Build\s+([0-9a-f]{12})$",
            version_text,
        )
        self.assertIsNotNone(match, "Help file lacks a parseable version/build line")
        self.assertEqual(match.group(1), physics.MODEL_VERSION)
        self.assertEqual(match.group(2), physics.BUILD_ID)

    def test_help_describes_current_modes_and_defaults(self):
        help_data = parse_help_file()
        visible_text = " ".join("".join(help_data.visible_text).split())
        self.assertIn("scaled_distance", visible_text)
        self.assertIn("walk2d", visible_text)

        defaults = {
            row[0]: row[1]
            for row in help_data.table_rows
            if len(row) >= 2 and row[0] != "Parameter"
        }
        self.assertEqual(
            {name: defaults.get(name) for name in (
                "--display",
                "--max_steps",
                "--n_trials",
                "--step_distribution",
                "--reference_steps",
                "--n_walks",
                "--mean_free_path",
                "--radius_factor",
                "--radius",
                "--ray_length_factor",
                "--step_cap",
                "--corner",
                "--seed",
            )},
            {
                "--display": "scaled_distance",
                "--max_steps": "4096",
                "--n_trials": "100",
                "--step_distribution": "uniform",
                "--reference_steps": "2000",
                "--n_walks": "4",
                "--mean_free_path": "1.0",
                "--radius_factor": "2.0",
                "--radius": "omitted",
                "--ray_length_factor": "0.6",
                "--step_cap": "200000",
                "--corner": "upper_right",
                "--seed": "omitted",
            },
        )

    def test_student_help_does_not_embed_java_listing(self):
        visible_text = " ".join("".join(parse_help_file().visible_text).split())
        self.assertNotIn("Listing of the Java code", visible_text)
        self.assertNotIn("Math.random()", visible_text)

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


class TestCommandLine(unittest.TestCase):
    def test_defaults_cover_driver_and_plot_inputs(self):
        args = cli.parse_args([])
        self.assertEqual(args.display, "scaled_distance")
        self.assertEqual(args.max_steps, 4096)
        self.assertEqual(args.n_trials, 100)
        self.assertEqual(args.step_distribution, "uniform")
        self.assertEqual(args.reference_steps, 2000)
        self.assertEqual(args.n_walks, 4)
        self.assertEqual(args.mean_free_path, 1.0)
        self.assertEqual(args.radius_factor, 2.0)
        self.assertIsNone(args.radius)
        self.assertEqual(args.ray_length_factor, 0.6)
        self.assertEqual(args.step_cap, 200_000)
        self.assertEqual(args.corner, "upper_right")
        self.assertIsNone(args.seed)

    def test_selector_choices_and_numeric_values_parse(self):
        args = cli.parse_args([
            "--display", "walk2d",
            "--step_distribution", "gaussian",
            "--reference_steps", "64",
            "--n_walks", "6",
            "--mean_free_path", "0.5",
            "--radius_factor", "3",
            "--radius", "20",
            "--ray_length_factor", "0",
            "--step_cap", "900",
            "--corner", "lower_left",
        ])
        self.assertEqual(args.display, "walk2d")
        self.assertEqual(args.step_distribution, "gaussian")
        self.assertEqual(args.reference_steps, 64)
        self.assertEqual(args.n_walks, 6)
        self.assertEqual(args.mean_free_path, 0.5)
        self.assertEqual(args.radius_factor, 3.0)
        self.assertEqual(args.radius, 20.0)
        self.assertEqual(args.ray_length_factor, 0.0)
        self.assertEqual(args.step_cap, 900)
        self.assertEqual(args.corner, "lower_left")

    def test_scaled_distance_arguments_are_forwarded(self):
        result = driver.ScaledDistanceResult(
            physics.MODEL_VERSION, physics.BUILD_ID, "gaussian", 7,
            (2.0,), (1.0,), (0.1,),
        )
        with (
            mock.patch.object(
                cli,
                "run_scaled_distance_statistics",
                return_value=result,
            ) as run,
            mock.patch.object(cli, "plot_scaled_distance") as draw,
            mock.patch.object(cli, "seed_generator") as seed,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            cli.main([
                "--display", "scaled_distance",
                "--max_steps", "32",
                "--n_trials", "7",
                "--step_distribution", "gaussian",
                "--seed", "5",
            ])
        seed.assert_called_once_with(5)
        run.assert_called_once_with(32, 7, step_distribution="gaussian")
        draw.assert_called_once_with((2.0,), (1.0,))

    def test_walk_arguments_and_corner_are_forwarded(self):
        result = object()
        with (
            mock.patch.object(cli, "run_walk2d", return_value=result) as run,
            mock.patch.object(cli, "walk2d_summary", return_value=["summary"]),
            mock.patch.object(cli, "plot_walk2d") as draw,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            cli.main([
                "--display", "walk2d",
                "--reference_steps", "20",
                "--n_walks", "3",
                "--mean_free_path", "2",
                "--radius_factor", "4",
                "--radius", "12",
                "--ray_length_factor", "0.25",
                "--step_cap", "99",
                "--corner", "lower_right",
            ])
        run.assert_called_once_with(
            reference_steps=20,
            n_walks=3,
            radius=12.0,
            mean_free_path=2.0,
            radius_factor=4.0,
            ray_length_factor=0.25,
            step_cap=99,
        )
        draw.assert_called_once_with(result, corner="lower right")

    def test_invalid_command_line_values_are_rejected(self):
        for arguments in (
            ["--max_steps", "0"],
            ["--radius", "nan"],
            ["--ray_length_factor", "-1"],
            ["--step_distribution", "normal"],
            ["--corner", "upper right"],
            ["--max_steps", "1"],
            ["--seed", "-1"],
            ["--seed", "1.5"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                cli.parse_args(arguments)


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
        with isolated_rng(8101):
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

    def test_default_radius_rejects_computed_underflow_and_overflow(self):
        with self.assertRaisesRegex(ValueError, "computed.*positive and finite"):
            physics.default_radius(1, 5e-324, 0.5)
        with self.assertRaisesRegex(ValueError, "computed.*positive and finite"):
            physics.default_radius(sys.maxsize, 1e308, 1e308)

    def test_numpy_real_scalars_are_supported(self):
        with isolated_rng(123):
            x, y = physics.generate_isotropic_step(np.float64(2.0))
        self.assertAlmostEqual(x * x + y * y, 4.0)
        self.assertAlmostEqual(
            physics.default_radius(4, np.float32(1.5), np.int32(2)), 6.0
        )

    def test_circle_crossing_fraction_for_radial_segment(self):
        self.assertAlmostEqual(physics.circle_crossing_fraction((0, 0), (2, 0), 1), 0.5)

    def test_circle_crossing_fraction_for_oblique_segment(self):
        t = physics.circle_crossing_fraction((0.0, 0.0), (2.0, 2.0), math.sqrt(2.0))
        self.assertAlmostEqual(t, 0.5)

    def test_circle_crossing_is_stable_at_extreme_scales(self):
        tiny_t = physics.circle_crossing_fraction(
            (0.0, 0.0), (2.0e-200, 0.0), 1.0e-200
        )
        self.assertAlmostEqual(tiny_t, 0.5, places=14)

        radius = 1.0e16
        start = radius - math.ulp(radius)
        end = 2.0 * radius
        expected = (radius - start) / (end - start)
        large_t = physics.circle_crossing_fraction(
            (start, 0.0), (end, 0.0), radius
        )
        self.assertAlmostEqual(large_t / expected, 1.0, places=14)

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
        with mock.patch.object(
            driver, "_perform_trials_3d_with_error", side_effect=lambda n, *_: (n + 0.5, n / 10)
        ) as trials:
            lengths, averages = driver.run_scaled_distance_experiment(10, 3)
        self.assertEqual([call.args[0] for call in trials.call_args_list], [10, 5, 2])
        self.assertEqual(lengths, [2.0, 5.0, 10.0])
        self.assertEqual(averages, [2.5, 5.5, 10.5])
        with mock.patch.object(
            driver, "_perform_trials_3d_with_error", side_effect=lambda n, *_: (n + 0.5, n / 10)
        ):
            result = driver.run_scaled_distance_statistics(10, 3)
        self.assertEqual(result.lengths, (2.0, 5.0, 10.0))
        self.assertEqual(result.standard_errors, (0.2, 0.5, 1.0))

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
    def test_isolated_rng_does_not_mutate_global_state(self):
        state_before = random.getstate()
        with isolated_rng(12345):
            physics.generate_component_step("uniform")
            physics.generate_component_step("gaussian")
        self.assertEqual(random.getstate(), state_before)

    def test_uniform_component_statistics_and_mean_step_length(self):
        with isolated_rng(20817):
            sample = [physics.generate_component_step("uniform") for _ in range(50_000)]
        for coordinate in range(3):
            values = [step[coordinate] for step in sample]
            self.assertGreaterEqual(min(values), -1.0)
            self.assertLessEqual(max(values), 1.0)
            self.assertAlmostEqual(sum(values) / len(values), 0.0, delta=0.012)
        mean_length = sum(math.sqrt(sum(x * x for x in step)) for step in sample) / len(sample)
        self.assertAlmostEqual(mean_length, 0.9605919565, delta=0.006)

    def test_isotropic_step_has_no_preferred_mean_direction(self):
        with isolated_rng(314159):
            sample = [physics.generate_isotropic_step() for _ in range(30_000)]
        mean_x = sum(x for x, _ in sample) / len(sample)
        mean_y = sum(y for _, y in sample) / len(sample)
        self.assertAlmostEqual(mean_x, 0.0, delta=0.015)
        self.assertAlmostEqual(mean_y, 0.0, delta=0.015)

    def _assert_scaled_distance_exponent(self, distribution, seed):
        with isolated_rng(seed):
            lengths, averages = driver.run_scaled_distance_experiment(
                512, 180, distribution
            )
        log_x = [math.log(x) for x in lengths]
        log_y = [math.log(y) for y in averages]
        mean_x = sum(log_x) / len(log_x)
        mean_y = sum(log_y) / len(log_y)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_x, log_y)) / sum(
            (x - mean_x) ** 2 for x in log_x
        )
        self.assertAlmostEqual(slope, 0.5, delta=0.08)

    def test_uniform_scaled_distance_exponent_is_close_to_one_half(self):
        self._assert_scaled_distance_exponent("uniform", 8675309)

    def test_gaussian_scaled_distance_exponent_is_close_to_one_half(self):
        self._assert_scaled_distance_exponent("gaussian", 90210)

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

        with isolated_rng(271828):
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
        ax = plt.gca()
        aspect = ax.get_aspect()
        self.assertTrue(
            aspect == "equal" or aspect == 1.0,
            f"expected equal log-axis aspect, got {aspect!r}",
        )
        self.assertEqual(ax.get_adjustable(), "box")
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_yscale(), "log")

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

    def test_walk_plot_rejects_malformed_constructed_results(self):
        valid_walk = driver.WalkPath(
            points=[(0.0, 0.0)], escaped=False, steps_taken=0
        )
        invalid_results = (
            driver.Walk2DResult("1", "b", 0.0, 1.0, 1, 1, [valid_walk]),
            driver.Walk2DResult("1", "b", 1.0, math.nan, 1, 1, [valid_walk]),
            driver.Walk2DResult("1", "b", 1.0, 1.0, 1, 1, []),
            driver.Walk2DResult(
                "1", "b", 1.0, 1.0, 1, 1,
                [driver.WalkPath([(math.inf, 0.0)], False, 0)],
            ),
            driver.Walk2DResult(
                "1", "b", 1.0, 1.0, 1, 1,
                [driver.WalkPath([(0.0, 0.0)], True, 1,
                                 ((0.0, 0.0), (math.nan, 1.0)))],
            ),
        )
        for result in invalid_results:
            with self.subTest(result=result), self.assertRaises(ValueError):
                plot.plot_walk2d(result)

    def test_walk_plot_accepts_microscopic_finite_geometry(self):
        result = driver.Walk2DResult(
            model_version=physics.MODEL_VERSION,
            build_id=physics.BUILD_ID,
            radius=1.0e-200,
            mean_free_path=1.0e-201,
            reference_steps=1,
            step_cap=1,
            walks=[driver.WalkPath([(0.0, 0.0)], False, 0)],
        )
        with mock.patch.object(plot.plt, "show"):
            plot.plot_walk2d(result)


BEAT_NUMBERS = tuple(range(0, 9))
VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
# Commands that the Help documents as being rejected by the program.
REJECTED_COMMANDS = {"python main.py --max_steps 1"}
SEED = ("--seed", "12")


def html_text(fragment: str) -> str:
    """Visible text of an HTML fragment, with entities decoded and spaces collapsed."""
    return " ".join(
        html_module.unescape(re.sub(r"</?[A-Za-z!][^>]*>", "", fragment)).split()
    )


def section_html(html: str, section_id: str) -> str:
    match = re.search(
        rf'<section id="{re.escape(section_id)}">(.*?)</section>', html, re.DOTALL
    )
    if match is None:
        raise AssertionError(f"section {section_id!r} not found in the Help file")
    return match.group(1)


def documented_commands(fragment: str) -> list:
    """Every ``python main.py ...`` command shown in a <pre> block."""
    commands = []
    for block in re.findall(r"<pre[^>]*>(.*?)</pre>", fragment, re.DOTALL):
        text = html_module.unescape(re.sub(r"<[^>]+>", "", block)).replace("\\\n", " ")
        for line in text.splitlines():
            line = " ".join(line.split())
            if line.startswith("python main.py"):
                commands.append(line)
    return commands


def command_arguments(command: str) -> tuple:
    return tuple(shlex.split(command)[2:])


def is_seeded(arguments) -> bool:
    return "--seed" in arguments


class CliRun(NamedTuple):
    stdout: str
    stderr: str
    exit_code: object
    result: object


_CLI_CACHE = {}


def run_cli(arguments=()) -> CliRun:
    """Run ``main.main()`` in-process with the plots suppressed.

    Seeded commands are cached, because they always print the same thing.
    """
    key = tuple(arguments)
    if key in _CLI_CACHE:
        return _CLI_CACHE[key]
    captured = {}
    real_scaled = cli.run_scaled_distance_statistics
    real_walk = cli.run_walk2d

    def capture_scaled(*args, **kwargs):
        captured["result"] = real_scaled(*args, **kwargs)
        return captured["result"]

    def capture_walk(*args, **kwargs):
        captured["result"] = real_walk(*args, **kwargs)
        return captured["result"]

    out, err = io.StringIO(), io.StringIO()
    exit_code = None
    state = random.getstate()
    try:
        with (
            mock.patch.object(cli, "run_scaled_distance_statistics", new=capture_scaled),
            mock.patch.object(cli, "run_walk2d", new=capture_walk),
            mock.patch.object(cli, "plot_scaled_distance"),
            mock.patch.object(cli, "plot_walk2d"),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            try:
                cli.main(list(key))
            except SystemExit as exc:
                exit_code = exc.code
    finally:
        random.setstate(state)
    run = CliRun(out.getvalue(), err.getvalue(), exit_code, captured.get("result"))
    if is_seeded(key):
        _CLI_CACHE[key] = run
    return run


class HelpStructure(HTMLParser):
    """Ids, links and tag balance of a Help page."""

    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.hrefs = []
        self.sidebar_hrefs = []
        self.section_ids = []
        self.scripts = []
        self.errors = []
        self._stack = []
        self._in_nav = False
        self.feed(html)
        self.close()
        if self._stack:
            self.errors.append(f"unclosed tags: {self._stack}")

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.append(attributes["id"] or "")
        if tag == "section" and attributes.get("id"):
            self.section_ids.append(attributes["id"])
        if tag == "script" and attributes.get("src"):
            self.scripts.append(attributes["src"])
        if tag == "nav":
            self._in_nav = True
        href = attributes.get("href")
        if tag == "a" and href:
            self.hrefs.append(href)
            if self._in_nav and href.startswith("#"):
                self.sidebar_hrefs.append(href[1:])
        if tag not in VOID_TAGS:
            self._stack.append(tag)

    def handle_endtag(self, tag):
        if tag == "nav":
            self._in_nav = False
        if tag in VOID_TAGS:
            return
        if not self._stack or self._stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> with open {self._stack[-3:]}")
            if tag in self._stack:
                while self._stack and self._stack.pop() != tag:
                    pass
            return
        self._stack.pop()


STRUCTURE = HelpStructure(HELP_HTML)


class HelpStructureTests(unittest.TestCase):
    def test_help_file_is_html5_utf8_with_one_version_build(self):
        self.assertIn("<!DOCTYPE html>", HELP_HTML[:100])
        self.assertRegex(HELP_HTML[:500], r'<meta charset="utf-8"\s*/?>')
        self.assertEqual(STRUCTURE.ids.count("version_build"), 1)

    def test_help_file_is_the_beats_version_and_not_the_original(self):
        self.assertIn(HELP_FILE.name, HELP_FILENAMES)
        self.assertIn("Beat 0", HELP_HTML)

    def test_tags_are_balanced_and_ids_are_unique(self):
        self.assertEqual(STRUCTURE.errors, [])
        duplicates = sorted({i for i in STRUCTURE.ids if STRUCTURE.ids.count(i) > 1})
        self.assertEqual(duplicates, [])

    def test_every_internal_link_and_sidebar_entry_resolves(self):
        targets = set(STRUCTURE.ids)
        internal = [h[1:] for h in STRUCTURE.hrefs if h.startswith("#")]
        self.assertGreater(len(internal), 20)
        for target in internal:
            with self.subTest(target=target):
                self.assertIn(target, targets)
        for target in STRUCTURE.sidebar_hrefs:
            with self.subTest(sidebar=target):
                self.assertIn(target, STRUCTURE.section_ids)

    def test_page_has_the_beats_layout_in_order(self):
        expected = (
            ["overview", "beats"]
            + [f"beat{n}" for n in BEAT_NUMBERS]
            + ["equations", "algorithm", "modules", "quickstart", "parameters",
               "output", "summary", "experiments", "related", "license"]
        )
        self.assertEqual(STRUCTURE.section_ids, expected)
        self.assertEqual(STRUCTURE.sidebar_hrefs, expected)

    def test_mathjax_is_the_only_external_script(self):
        self.assertEqual(len(STRUCTURE.scripts), 1)
        self.assertRegex(STRUCTURE.scripts[0], r"^https://cdn\.jsdelivr\.net/npm/mathjax@3/")
        self.assertIn("needs no internet access to run", html_text(section_html(HELP_HTML, "overview")))

    def test_mathjax_source_has_no_text_mode_underscores(self):
        for block in re.findall(r"\\\[(.*?)\\\]", HELP_HTML, re.DOTALL):
            with self.subTest(block=block[:40]):
                self.assertNotIn("\\texttt", block)

    def test_student_content_contains_no_ai_or_review_history(self):
        student_text = HELP_HTML.split('<section id="license">', 1)[0]
        for term in ("Claude", "Copilot", "Gemini", "ChatGPT", "Anthropic", "Codex", "Grok",
                     "AI-generated", "audit", "Kickoff", "previous version", "reviewer"):
            with self.subTest(term=term):
                self.assertNotIn(term, student_text)

    def test_java_provenance_is_confined_to_license(self):
        before_license, license_and_after = HELP_HTML.split('<section id="license">', 1)
        self.assertNotIn("Java", before_license)
        self.assertIn("Java", license_and_after)

    def test_relative_links_resolve_when_the_documentation_tree_is_present(self):
        docs_root = HELP_FILE.parent.parent
        if docs_root.name != "GFTGU-Documentation" or not (docs_root / "Star").is_dir():
            self.skipTest("the sibling documentation folders are not present in this layout")
        for href in STRUCTURE.hrefs:
            if href.startswith(("#", "http://", "https://", "mailto:")):
                continue
            with self.subTest(href=href):
                self.assertTrue((HELP_FILE.parent / href).is_file(), href)


class HelpBeatTests(unittest.TestCase):
    TAGS = {"MODEL": "law", "DEFINITION": "def", "DERIVED": "der",
            "ALGORITHM": "alg", "PRESCRIPTION": "pre"}

    def equations(self):
        found = {}
        for number in BEAT_NUMBERS:
            body = section_html(HELP_HTML, f"beat{number}")
            for match in re.finditer(
                r'<span class="eq-label">\((\d+)\)</span>\s*<div class="eq-kind">.*?'
                r'<span class="kind kind-(\w+)">(\w+)</span>',
                body, re.DOTALL,
            ):
                found[int(match.group(1))] = (match.group(3), number, match.group(2))
        return found

    def test_every_beat_follows_the_same_pattern(self):
        for number in BEAT_NUMBERS:
            body = section_html(HELP_HTML, f"beat{number}")
            with self.subTest(beat=number):
                self.assertIn(f"<h2>Beat {number} · ", body)
                self.assertGreaterEqual(body.count("<pre>"), 1)
                self.assertRegex(body, r"<p>Look[ ,]")
                self.assertIn("Three tasks, in order.", body)
                self.assertIn("<p><em>Then</em>", body)
                self.assertIn("<p><em>Experiments that go with this beat:</em>", body)
                self.assertLess(body.index("<pre>"), body.index("Three tasks, in order."))
                self.assertLess(body.index("Three tasks, in order."), body.index("<em>Then</em>"))
                self.assertLess(body.index("<em>Then</em>"), body.index("Experiments that go with"))
                self.assertGreaterEqual(body.count('class="eq"'), 1)

    def test_sidebar_titles_match_the_beat_headings(self):
        for number in BEAT_NUMBERS:
            heading = re.search(
                rf"<h2>Beat {number} · (.*?)</h2>", section_html(HELP_HTML, f"beat{number}")
            ).group(1)
            link = re.search(rf'<a href="#beat{number}">(.*?)</a>', HELP_HTML).group(1)
            with self.subTest(beat=number):
                self.assertEqual(html_text(link), f"{number} · {html_text(heading)}")

    def test_equations_are_numbered_in_order_and_tagged_consistently(self):
        found = self.equations()
        self.assertEqual(sorted(found), list(range(1, 21)))
        beats = [found[n][1] for n in sorted(found)]
        self.assertEqual(beats, sorted(beats))
        for number, (kind, _, css) in found.items():
            with self.subTest(equation=number):
                self.assertEqual(self.TAGS[kind], css)
        self.assertIn("Twenty equations are numbered", html_text(section_html(HELP_HTML, "beats")))

    def test_tag_counts_match_the_reading_note(self):
        found = self.equations()
        by_kind = {}
        for number, (kind, _, _) in found.items():
            by_kind.setdefault(kind, []).append(number)
        self.assertEqual(sorted(by_kind["MODEL"]), [1, 12])
        self.assertEqual(sorted(by_kind["ALGORITHM"]), [8, 15, 19])
        self.assertEqual(sorted(by_kind["PRESCRIPTION"]), [14])
        note = html_text(section_html(HELP_HTML, "beats"))
        self.assertIn("Eqs. (1) and (12) are the only ones", note)
        self.assertIn("Eqs. (8), (15) and (19)", note)
        self.assertIn("Eq. (14)", note)

    def test_equation_citations_refer_to_numbered_equations(self):
        for section in STRUCTURE.section_ids:
            if section in ("related", "license"):
                continue
            text = html_text(section_html(HELP_HTML, section))
            for group in re.findall(r"Eqs?\.\s*\(([\d\s,and()to]+?)\)(?=[\s.,;:]|$)", text):
                for number in re.findall(r"\d+", group):
                    with self.subTest(section=section, equation=number):
                        self.assertIn(int(number), range(1, 21))

    def test_equation_index_lists_every_numbered_equation_once_with_its_kind_and_beat(self):
        found = self.equations()
        rows = re.findall(
            r"<tr><td>\((\d+)\)</td><td><span class=\"kind kind-\w+\">(\w+)</span></td>"
            r"<td>.*?</td><td>(\d+)</td>",
            section_html(HELP_HTML, "equations"),
        )
        self.assertEqual([int(r[0]) for r in rows], list(range(1, 21)))
        for number, kind, beat in rows:
            with self.subTest(equation=number):
                self.assertEqual((kind, int(beat)), found[int(number)][:2])

    def test_every_experiment_is_cited_by_a_beat_and_every_citation_exists(self):
        experiments = re.findall(r'<h3 id="exp(\d+)">', section_html(HELP_HTML, "experiments"))
        self.assertEqual(experiments, [str(n) for n in range(1, 12)])
        cited = set()
        for number in BEAT_NUMBERS:
            body = section_html(HELP_HTML, f"beat{number}")
            tail = body[body.index("Experiments that go with this beat:"):]
            links = re.findall(r'href="#exp(\d+)"', tail)
            with self.subTest(beat=number):
                self.assertTrue(links)
            cited.update(links)
        self.assertEqual(cited, set(experiments))

    def test_experiments_are_ranked_from_introductory_to_advanced(self):
        rank = {"Introductory": 0, "Introductory to Intermediate": 1, "Intermediate": 2,
                "Intermediate Programming": 2, "Intermediate to Advanced": 3,
                "Advanced": 4, "Advanced Programming": 4}
        levels = [
            rank[html_text(title).rsplit("— ", 1)[1]]
            for title in re.findall(r'<h3 id="exp\d+">(.*?)</h3>', section_html(HELP_HTML, "experiments"))
        ]
        self.assertEqual(levels, sorted(levels))

    def test_python_snippet_in_the_experiments_runs_and_matches_beats_0_and_4(self):
        blocks = [
            html_module.unescape(re.sub(r"<[^>]+>", "", b))
            for b in re.findall(r"<pre>(.*?)</pre>", section_html(HELP_HTML, "experiments"), re.DOTALL)
            if "random2_driver import" in b
        ]
        self.assertEqual(len(blocks), 1)
        completed = subprocess.run(
            [sys.executable, "-c", blocks[0]], cwd=MODULE_DIR,
            text=True, capture_output=True, timeout=120,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        first, second = completed.stdout.splitlines()
        length, average, error = (float(v) for v in first.split())
        self.assertEqual(length, 4096.0)
        beat0 = run_cli(SEED).stdout
        self.assertIn(f"{average:.4f}", beat0)
        self.assertIn(f"{error:.4f}", beat0)
        escaped, mean = second.split()
        beat4 = run_cli(("--display", "walk2d", "--radius", "30", "--n_walks", "200", *SEED)).stdout
        self.assertEqual(int(escaped), 200)
        self.assertIn(f"mean {float(mean):.1f}", beat4)


class HelpCommandTests(unittest.TestCase):
    def test_the_help_documents_a_meaningful_number_of_commands(self):
        self.assertGreaterEqual(len(dict.fromkeys(documented_commands(HELP_HTML))), 25)

    def test_every_beat_command_except_the_seed_demonstrations_is_seeded(self):
        for number in BEAT_NUMBERS:
            for command in documented_commands(section_html(HELP_HTML, f"beat{number}")):
                if number == 8 and command in ("python main.py", *REJECTED_COMMANDS):
                    continue
                with self.subTest(command=command):
                    self.assertTrue(is_seeded(command_arguments(command)))

    def test_every_documented_command_parses_and_runs_to_a_summary(self):
        for command in dict.fromkeys(documented_commands(HELP_HTML)):
            arguments = command_arguments(command)
            if "--help" in arguments:
                continue
            with self.subTest(command=command):
                run = run_cli(arguments)
                if command in REJECTED_COMMANDS:
                    self.assertEqual(run.exit_code, 2)
                    self.assertIn("max_steps must be at least 2", run.stderr)
                    self.assertEqual(run.stdout, "")
                else:
                    self.assertIn(run.exit_code, (None, 0), run.stderr)
                    lines = run.stdout.splitlines()
                    self.assertEqual(lines[0], f"Random2 {physics.MODEL_VERSION} (build {physics.BUILD_ID})")
                    self.assertTrue(lines[1].startswith("display: "))
                    self.assertTrue(lines[-1].startswith(("fitted log-log slope", "mean escape steps", "escape steps")))

    def test_documented_options_are_real_options(self):
        parser = cli.build_parser()
        real = {s for action in parser._actions for s in action.option_strings}
        used = set(re.findall(r"(?<![\w-])(--[A-Za-z_]+)", "\n".join(documented_commands(HELP_HTML))))
        self.assertTrue(used)
        self.assertLessEqual(used, real)


class HelpReferenceTests(unittest.TestCase):
    def test_parameter_table_matches_the_parser_options_and_defaults(self):
        rows = {
            row[0]: row[1]
            for row in parse_help_file().table_rows
            if len(row) == 3 and row[0].startswith("--")
        }
        options = {
            action.option_strings[0]: action
            for action in cli.build_parser()._actions
            if action.option_strings and action.option_strings[0] not in ("-h", "--version")
        }
        self.assertEqual(set(rows), set(options))
        for name, action in options.items():
            expected = "omitted" if action.default is None else str(action.default)
            with self.subTest(option=name):
                self.assertEqual(rows[name], expected)

    def test_printed_summary_blocks_are_exactly_what_beats_0_and_3_print(self):
        blocks = re.findall(r"<pre>(Random2 .*?)</pre>", section_html(HELP_HTML, "summary"), re.DOTALL)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(html_module.unescape(blocks[0]).strip(), run_cli(SEED).stdout.strip())
        self.assertEqual(
            html_module.unescape(blocks[1]).strip(),
            run_cli(("--display", "walk2d", *SEED)).stdout.strip(),
        )

    def test_every_summary_label_is_described_in_the_help(self):
        text = html_text(section_html(HELP_HTML, "summary"))
        scaled = run_cli(SEED).stdout
        walk = run_cli(("--display", "walk2d", "--radius", "30", "--n_walks", "200",
                        "--step_cap", "900", *SEED)).stdout
        labels = set()
        for output in (scaled, walk):
            for line in output.splitlines()[1:]:
                if ":" in line:
                    labels.add(line.split(":", 1)[0])
        labels = {label for label in labels if not label.startswith("walk ")}
        for label in sorted(labels):
            with self.subTest(label=label):
                self.assertIn(label, text)
        for column in ("mean scaled distance", "standard error", "ratio to sqrt(N)"):
            self.assertIn(column, scaled)
            self.assertIn(column, text)

    def test_constant_table_matches_the_program(self):
        rows = {
            row[0]: row[1]
            for row in parse_help_file().table_rows
            if len(row) == 3 and not row[0].startswith("--") and row[0] != "Name"
        }
        self.assertEqual(rows["UNIFORM_MEAN_STEP_LENGTH"], f"{physics.UNIFORM_MEAN_STEP_LENGTH:.6f}")
        self.assertEqual(rows["MAX_LISTED_WALKS"], str(cli.MAX_LISTED_WALKS))

    def test_code_identifiers_named_in_the_help_exist_in_the_program(self):
        modules = (physics, driver, plot, cli)
        names = set()
        for body in re.findall(r"<code>([^<]+)</code>", HELP_HTML):
            body = html_module.unescape(body)
            match = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*)\((?:\"\w+\")?\)", body)
            if match:
                names.add(match.group(1))
            elif re.fullmatch(r"[A-Z][A-Z_0-9]{3,}", body):
                names.add(body)
        for class_name in re.findall(r"<code>([A-Z][a-z]\w+)</code>", HELP_HTML):
            names.add(class_name)
        self.assertGreater(len(names), 15)
        for name in sorted(names):
            with self.subTest(name=name):
                self.assertTrue(any(hasattr(module, name) for module in modules), name)


class HelpQuotedNumberTests(unittest.TestCase):
    """Every number the Help quotes from a seeded run must be the number printed."""

    FOUR_DECIMALS = re.compile(r"(?<![\w.])\d+\.\d{4}(?![\w%])")
    ONE_DECIMAL = re.compile(r"(?<![\w.])\d+\.\d(?![\w%])")
    # Numbers that a student computes by hand from printed values or that are
    # constants of the equations; each is checked in HelpQuantitativeClaimTests.
    DERIVED_OK = {"0.9213", "1.5958", "0.5", "1.0", "2.0", "0.6", "16.4"}
    # The runs of Beats 0, 3 and 4, which later beats compare with.
    REFERENCE_RUNS = (
        SEED,
        ("--display", "walk2d", *SEED),
        ("--display", "walk2d", "--radius", "30", "--n_walks", "200", *SEED),
    )

    def printed_for(self, fragment):
        chunks = [run_cli(arguments).stdout for arguments in self.REFERENCE_RUNS]
        for command in documented_commands(fragment):
            arguments = command_arguments(command)
            if command not in REJECTED_COMMANDS and is_seeded(arguments):
                chunks.append(run_cli(arguments).stdout)
        return "\n".join(chunks)

    def tokens(self, fragment):
        text = html_text(re.sub(r"\\\[.*?\\\]", "", fragment, flags=re.DOTALL))
        found = set(self.FOUR_DECIMALS.findall(text)) | set(self.ONE_DECIMAL.findall(text))
        return found - self.DERIVED_OK

    def test_numbers_quoted_in_each_beat_are_printed_by_that_beat_s_commands(self):
        for number in BEAT_NUMBERS:
            body = section_html(HELP_HTML, f"beat{number}")
            printed = self.printed_for(body)
            tokens = self.tokens(body)
            with self.subTest(beat=number):
                self.assertTrue(tokens)
                self.assertEqual(sorted(t for t in tokens if t not in printed), [])

    # Whole numbers in Beats 3 to 7 that are inputs or bounds rather than output.
    WHOLE_NUMBERS_NOT_PRINTED = {"2000", "961", "256", "200000"}

    def test_whole_numbers_quoted_in_beats_3_to_7_are_printed(self):
        for number in range(3, 8):
            body = section_html(HELP_HTML, f"beat{number}")
            text = html_text(re.sub(r"\\\[.*?\\\]", "", body, flags=re.DOTALL))
            printed = self.printed_for(body)
            counts = set(re.findall(r"(?<![\w.])\d{3,6}(?![\w.])", text)) - self.WHOLE_NUMBERS_NOT_PRINTED
            with self.subTest(beat=number):
                self.assertTrue(counts)
                for count in sorted(counts):
                    self.assertRegex(printed, rf"(?<![\d.]){count}(?![\d])", count)

    def test_numbers_quoted_in_the_reference_sections_are_printed_by_some_command(self):
        every_output = "\n".join(
            run_cli(command_arguments(c)).stdout
            for c in dict.fromkeys(documented_commands(HELP_HTML))
            if c not in REJECTED_COMMANDS and is_seeded(command_arguments(c))
        )
        for section in ("overview", "beats", "algorithm", "modules", "quickstart",
                        "parameters", "output", "summary"):
            with self.subTest(section=section):
                tokens = set(self.FOUR_DECIMALS.findall(html_text(section_html(HELP_HTML, section))))
                tokens -= {"0.9213", "1.5958"}
                self.assertEqual(sorted(t for t in tokens if t not in every_output), [])


class HelpQuantitativeClaimTests(unittest.TestCase):
    """Independent checks of statements in the beats that go beyond the printed digits."""

    @staticmethod
    def scaled(*arguments):
        return run_cli((*arguments, *SEED)).result

    @staticmethod
    def walk(*arguments):
        return run_cli(("--display", "walk2d", *arguments, *SEED)).result

    def test_uniform_mean_step_length_closed_form_matches_numerical_integration(self):
        cells = 120
        centres = (np.arange(cells) + 0.5) / cells  # the octant [0, 1]^3 is enough by symmetry
        x, y, z = np.meshgrid(centres, centres, centres, indexing="ij")
        numeric = float(np.sqrt(x * x + y * y + z * z).mean())
        self.assertAlmostEqual(numeric, physics.UNIFORM_MEAN_STEP_LENGTH, delta=2e-5)
        self.assertAlmostEqual(physics.UNIFORM_MEAN_STEP_LENGTH, 0.960592, places=6)
        self.assertAlmostEqual(math.sqrt(8 / (3 * math.pi)), 0.92132, places=5)
        self.assertAlmostEqual(physics.large_n_scaled_distance_ratio("uniform"), 0.9591, places=4)
        self.assertEqual(physics.large_n_scaled_distance_ratio("gaussian"), 1.0)
        self.assertAlmostEqual(math.sqrt(8 / math.pi), 1.5958, places=4)

    def test_large_n_prediction_agrees_with_a_vectorised_simulation(self):
        rng = np.random.default_rng(2026)
        for distribution, draw in (
            ("uniform", lambda size: rng.uniform(-1.0, 1.0, size)),
            ("gaussian", lambda size: rng.standard_normal(size)),
        ):
            steps = draw((4000, 256, 3))
            net = np.linalg.norm(steps.sum(axis=1), axis=1)
            mean_step = np.linalg.norm(steps, axis=2).mean(axis=1)
            ratio = float((net / mean_step).mean()) / 16.0
            with self.subTest(distribution=distribution):
                self.assertAlmostEqual(
                    ratio, physics.large_n_scaled_distance_ratio(distribution), delta=0.03
                )

    def test_relative_spread_of_the_scaled_distance_is_0_42(self):
        self.assertAlmostEqual(math.sqrt(3 * math.pi / 8 - 1), 0.42, places=2)
        result = self.scaled("--n_trials", "1000")
        spread = result.standard_errors[-1] * math.sqrt(1000) / result.averages[-1]
        self.assertAlmostEqual(spread, 0.42, delta=0.03)

    def test_beat0_ratios_and_factors(self):
        result = self.scaled()
        ratios = [a / math.sqrt(n) for n, a in zip(result.lengths, result.averages)]
        self.assertAlmostEqual(min(ratios), 0.9074, places=4)
        self.assertAlmostEqual(max(ratios), 1.0237, places=4)
        self.assertAlmostEqual(result.averages[-1] / result.averages[-3], 2.05, places=2)
        self.assertAlmostEqual(result.averages[-3] / result.averages[-5], 2.05, places=2)

    def test_beat1_standard_errors_ranges_and_slope_uncertainty(self):
        few, default, many = self.scaled("--n_trials", "25"), self.scaled(), self.scaled("--n_trials", "1000")
        self.assertAlmostEqual(few.standard_errors[-1] / default.standard_errors[-1], 1.82, places=2)
        self.assertAlmostEqual(default.standard_errors[-1] / many.standard_errors[-1], 2.85, places=2)
        self.assertAlmostEqual(many.standard_errors[-1] / many.standard_errors[3], 16.4, places=1)
        prediction = physics.large_n_scaled_distance_ratio("uniform")
        late = [a / math.sqrt(n) for n, a in zip(many.lengths, many.averages) if n >= 16]
        self.assertLessEqual(max(abs(r - prediction) for r in late), 0.008)
        few_ratios = [a / math.sqrt(n) for n, a in zip(few.lengths, few.averages)]
        self.assertAlmostEqual(min(few_ratios), 0.8211, places=4)
        self.assertAlmostEqual(max(few_ratios), 1.0326, places=4)
        spacing = math.log(2.0)
        spread = sum((k * spacing - 5.5 * spacing) ** 2 for k in range(12))
        for trials, expected in ((25, 0.01), (1000, 0.002)):
            self.assertAlmostEqual(0.42 / math.sqrt(trials) / math.sqrt(spread), expected, delta=0.0006)

    def test_beat1_short_walk_bias_lowers_the_expected_slope(self):
        rng = np.random.default_rng(7)
        means = []
        lengths = [2 ** k for k in range(1, 13)]
        for n_steps in lengths:
            trials = max(400, 400_000 // n_steps)
            steps = rng.uniform(-1.0, 1.0, (trials, n_steps, 3))
            net = np.linalg.norm(steps.sum(axis=1), axis=1)
            means.append(float((net / np.linalg.norm(steps, axis=2).mean(axis=1)).mean()))
        self.assertGreater(means[0] / math.sqrt(2), 0.97)
        self.assertAlmostEqual(physics.fitted_loglog_slope(lengths, means), 0.498, delta=0.003)

    def test_beat2_gaussian_ranges_and_comparison(self):
        gaussian, uniform = self.scaled("--step_distribution", "gaussian"), self.scaled()
        late = [a / math.sqrt(n) for n, a in zip(gaussian.lengths, gaussian.averages) if n >= 64]
        self.assertAlmostEqual(min(late), 0.9408, places=4)
        self.assertAlmostEqual(max(late), 1.0502, places=4)
        difference = gaussian.averages[-1] - uniform.averages[-1]
        combined = math.hypot(gaussian.standard_errors[-1], uniform.standard_errors[-1])
        self.assertAlmostEqual(difference, 2.16, places=2)
        self.assertAlmostEqual(difference / combined, 0.54, places=2)
        self.assertAlmostEqual(1 / physics.large_n_scaled_distance_ratio("uniform") - 1, 0.04, places=2)

    def test_beat3_default_star(self):
        result = self.walk()
        self.assertAlmostEqual(result.radius, 2 * math.sqrt(2000), places=12)
        steps = [w.steps_taken for w in result.walks]
        self.assertEqual(steps, [11564, 7519, 8246, 2479])
        self.assertAlmostEqual(max(steps) / min(steps), 4.66, places=2)
        stats = driver.escape_statistics(result)
        self.assertAlmostEqual(stats.standard_error / stats.mean, 0.25, places=2)
        self.assertEqual(math.ceil(result.radius), 90)

    def test_beat4_mean_escape_bound_and_spread(self):
        stats = driver.escape_statistics(self.walk("--radius", "30", "--n_walks", "200"))
        self.assertEqual((stats.n_escaped, stats.minimum, stats.maximum), (200, 132, 3797))
        self.assertLessEqual(900 - 3 * stats.standard_error, stats.mean)
        self.assertLessEqual(stats.mean, 961 + 3 * stats.standard_error)
        self.assertGreater(stats.mean, stats.median)

    def test_eq16_bound_holds_for_a_large_sample(self):
        with isolated_rng(4242):
            result = driver.run_walk2d(n_walks=4000, radius=6.0, ray_length_factor=0)
        stats = driver.escape_statistics(result)
        self.assertEqual(stats.n_escaped, 4000)
        self.assertLessEqual(36.0 - 3 * stats.standard_error, stats.mean)
        self.assertLessEqual(stats.mean, 49.0 + 3 * stats.standard_error)

    def test_beat5_ratio_and_its_uncertainty(self):
        small = driver.escape_statistics(self.walk("--radius", "15", "--n_walks", "200"))
        large = driver.escape_statistics(self.walk("--radius", "30", "--n_walks", "200"))
        ratio = large.mean / small.mean
        self.assertAlmostEqual(ratio, 3.96, places=2)
        self.assertAlmostEqual(large.median / small.median, 3.99, places=2)
        relative = math.hypot(large.standard_error / large.mean, small.standard_error / small.mean)
        self.assertAlmostEqual(relative, 0.068, places=3)
        self.assertAlmostEqual(ratio * relative, 0.27, places=2)
        self.assertAlmostEqual(900 / 256, 3.52, places=2)
        self.assertAlmostEqual(961 / 225, 4.27, places=2)
        self.assertTrue(900 / 256 <= ratio <= 961 / 225)

    def test_beat6_step_counts_are_identical_for_factors_of_two(self):
        reference = [w.steps_taken for w in self.walk().walks]
        for length in ("0.5", "2"):
            result = self.walk("--mean_free_path", length)
            with self.subTest(mean_free_path=length):
                self.assertEqual([w.steps_taken for w in result.walks], reference)
                self.assertAlmostEqual((result.radius / result.mean_free_path) ** 2, 8000.0, places=6)

    def test_beat7_step_cap_bias(self):
        capped = self.walk("--radius", "30", "--n_walks", "200", "--step_cap", "900")
        loose = self.walk("--radius", "30", "--n_walks", "200", "--step_cap", "3000")
        capped_stats, loose_stats = driver.escape_statistics(capped), driver.escape_statistics(loose)
        self.assertEqual((capped_stats.n_escaped, capped_stats.maximum), (130, 897))
        self.assertEqual(loose_stats.n_escaped, 196)
        self.assertTrue(all(w.steps_taken == 900 for w in capped.walks if not w.escaped))
        self.assertLess(capped_stats.mean, loose_stats.mean)

    def test_beat8_seed_comparison(self):
        a, b = self.scaled(), run_cli(("--seed", "28")).result
        difference = abs(a.averages[-1] - b.averages[-1])
        combined = math.hypot(a.standard_errors[-1], b.standard_errors[-1])
        self.assertAlmostEqual(difference, 0.79, places=2)
        self.assertAlmostEqual(combined, 3.47, places=2)
        self.assertAlmostEqual(difference / combined, 0.23, places=2)
        single = run_cli(("--max_steps", "2", *SEED)).stdout
        self.assertIn("fitted log-log slope: n/a (only one step count)", single)


class NewFeatureTests(unittest.TestCase):
    """The seed option, printed summaries, frozen results and shared helpers."""

    def test_same_seed_repeats_and_different_seeds_differ(self):
        first = run_cli(("--max_steps", "64", "--seed", "3")).stdout
        _CLI_CACHE.clear()
        second = run_cli(("--max_steps", "64", "--seed", "3")).stdout
        other = run_cli(("--max_steps", "64", "--seed", "4")).stdout
        self.assertEqual(first, second)
        self.assertNotEqual(first.splitlines()[5:], other.splitlines()[5:])

    def test_unseeded_runs_say_so(self):
        output = run_cli(("--max_steps", "8")).stdout
        self.assertIn("seed: none (results differ from run to run)", output)

    def test_seed_generator_validates_and_seeds(self):
        for bad in (-1, 1.5, True, "3"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                physics.seed_generator(bad)
        state = random.getstate()
        try:
            physics.seed_generator(9)
            first = random.random()
            physics.seed_generator(9)
            self.assertEqual(random.random(), first)
            physics.seed_generator(None)
            random.random()
        finally:
            random.setstate(state)

    def test_fitted_slope_is_exact_for_a_power_law_and_validates(self):
        xs = [2.0, 4.0, 8.0, 16.0]
        self.assertAlmostEqual(physics.fitted_loglog_slope(xs, [3 * x ** 0.5 for x in xs]), 0.5, places=12)
        for bad in (([1.0], [1.0]), ([1.0, 1.0], [1.0, 2.0]), ([1.0, 2.0], [1.0]), ([0.0, 1.0], [1.0, 1.0])):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                physics.fitted_loglog_slope(*bad)

    def test_standard_errors_match_the_sample_definition(self):
        values = [(6.0, 2.0), (9.0, 3.0), (4.0, 1.0), (5.0, 2.0)]
        with mock.patch.object(driver, "_perform_single_walk_3d", side_effect=values):
            mean, error = driver._perform_trials_3d_with_error(4, 4)
        scaled = [a / b for a, b in values]
        self.assertAlmostEqual(mean, statistics.fmean(scaled))
        self.assertAlmostEqual(error, statistics.stdev(scaled) / 2.0)
        with mock.patch.object(driver, "_perform_single_walk_3d", return_value=(3.0, 1.0)):
            self.assertEqual(driver._perform_trials_3d_with_error(4, 1), (3.0, None))

    def test_single_trial_prints_n_a_standard_error(self):
        output = run_cli(("--max_steps", "4", "--n_trials", "1", *SEED)).stdout
        self.assertRegex(output, r"\n +2 +\d+\.\d{4} +n/a +\d+\.\d{4}\n")

    def test_results_are_frozen_tuples(self):
        result = driver.run_walk2d(n_walks=2, radius=3.0)
        self.assertIsInstance(result.walks, tuple)
        self.assertIsInstance(result.walks[0].points, tuple)
        with self.assertRaises(FrozenInstanceError):
            result.radius = 1.0
        with self.assertRaises(FrozenInstanceError):
            result.walks[0].steps_taken = 0
        table = driver.run_scaled_distance_statistics(8, 3)
        self.assertIsInstance(table.lengths, tuple)
        with self.assertRaises(FrozenInstanceError):
            table.n_trials = 1

    def test_escape_statistics(self):
        walks = (
            driver.WalkPath(((0.0, 0.0),), True, 10),
            driver.WalkPath(((0.0, 0.0),), True, 20),
            driver.WalkPath(((0.0, 0.0),), False, 50),
        )
        result = driver.Walk2DResult("v", "b", 1.0, 1.0, 1, 50, walks)
        stats = driver.escape_statistics(result)
        self.assertEqual((stats.n_walks, stats.n_escaped, stats.minimum, stats.maximum), (3, 2, 10, 20))
        self.assertEqual((stats.mean, stats.median), (15.0, 15.0))
        self.assertAlmostEqual(stats.standard_error, statistics.stdev([10, 20]) / math.sqrt(2))
        none = driver.escape_statistics(driver.Walk2DResult("v", "b", 1.0, 1.0, 1, 50, walks[2:]))
        self.assertEqual((none.n_escaped, none.mean, none.standard_error), (0, None, None))

    def test_walk_summary_lists_capped_walks_and_omits_lists_above_twelve(self):
        capped = run_cli(("--display", "walk2d", "--radius", "30", "--n_walks", "3",
                          "--step_cap", "5", *SEED)).stdout
        self.assertIn("walk 1: stopped at the step cap after 5 steps, not escaped", capped)
        self.assertIn("escape steps: n/a (no walk escaped)", capped)
        many = run_cli(("--display", "walk2d", "--radius", "3", "--n_walks", "13", *SEED)).stdout
        self.assertNotIn("walk 1:", many)
        self.assertIn("escaped walks: 13 of 13", many)

    def test_driver_and_plot_use_the_shared_physics_validators(self):
        self.assertIs(driver.require_positive_int, physics.require_positive_int)
        self.assertIs(plot.require_positive_finite_number, physics.require_positive_finite_number)
        for name in ("_require_positive_finite_number", "_require_nonnegative_finite_number",
                     "_require_positive_int"):
            self.assertFalse(hasattr(driver, name), name)

    def test_scaled_plot_shows_the_fitted_slope(self):
        with mock.patch.object(plot.plt, "show"):
            plot.plot_scaled_distance((2.0, 4.0, 8.0), (1.0, 2.0 ** 0.5, 2.0))
        text = "\n".join(t.get_text() for t in plt.gca().texts)
        plt.close("all")
        self.assertIn("fitted log-log slope = 0.5000", text)
        with mock.patch.object(plot.plt, "show"):
            plot.plot_scaled_distance((2.0,), (1.0,))
        self.assertEqual(len(plt.gca().texts), 0)
        plt.close("all")

    def test_walk_plot_annotation_median_matches_the_summary(self):
        with isolated_rng(12):
            result = driver.run_walk2d(n_walks=4, radius=10.0)
        with mock.patch.object(plot.plt, "show"):
            plot.plot_walk2d(result)
        text = "\n".join(t.get_text() for t in plt.gca().texts)
        plt.close("all")
        median = driver.escape_statistics(result).median
        self.assertIn(f"median escape steps = {median:.1f}", text)
        self.assertIn(f"median {median:.1f}", "\n".join(cli.walk2d_summary(result)))

    def test_walk_plot_leaves_room_for_long_rays(self):
        with isolated_rng(5):
            result = driver.run_walk2d(n_walks=3, radius=5.0, ray_length_factor=2.0)
        with mock.patch.object(plot.plt, "show"):
            plot.plot_walk2d(result)
        low, high = plt.gca().get_xlim()
        plt.close("all")
        for walk in result.walks:
            (_, _), (rx, ry) = walk.ray
            self.assertLess(abs(rx), high)
            self.assertLess(abs(ry), high)
            self.assertGreater(high, 1.8 * result.radius)


class EndToEndSubprocessTests(unittest.TestCase):
    """Both displays run as a student would run them, with a non-interactive backend."""

    def run_main(self, *arguments):
        environment = {**os.environ, "MPLBACKEND": "Agg"}
        return subprocess.run(
            [sys.executable, "main.py", *arguments], cwd=MODULE_DIR, env=environment,
            capture_output=True, text=True, timeout=120,
        )

    def test_scaled_distance_display_runs(self):
        completed = self.run_main("--max_steps", "64", "--n_trials", "20", "--seed", "1")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("display: scaled_distance", completed.stdout)
        self.assertIn("fitted log-log slope:", completed.stdout)
        in_process = run_cli(("--max_steps", "64", "--n_trials", "20", "--seed", "1")).stdout
        self.assertEqual(completed.stdout, in_process)

    def test_walk2d_display_runs(self):
        completed = self.run_main("--display", "walk2d", "--radius", "10", "--seed", "1")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("escaped walks: 4 of 4", completed.stdout)

    def test_rejected_input_exits_with_a_message(self):
        completed = self.run_main("--max_steps", "1")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("max_steps must be at least 2", completed.stderr)


class Audit21Tests(unittest.TestCase):
    """Experiment 9 bound, asymptotic wording, scale limits, annotation provenance."""

    def test_experiment9_bound_uses_r_over_lambda(self):
        card = html_text(section_html(HELP_HTML, "experiments"))
        exp9 = card[card.index("9 · Verify Escape Steps"):card.index("10 · ")]
        self.assertIn(r"(1+\lambda/R)^2", exp9)
        self.assertNotIn(r"(1+1/R)^2", exp9)
        self.assertIn("mean escape steps / (R / mean free path)^2", exp9)
        run = run_cli(("--display", "walk2d", "--radius", "30", "--mean_free_path", "2",
                       "--n_walks", "200", *SEED))
        stats = driver.escape_statistics(run.result)
        scale = physics.diffusion_step_scale(30.0, 2.0)
        ratio, error = stats.mean / scale, stats.standard_error / scale
        self.assertLessEqual(1.0 - 3 * error, ratio)
        self.assertLessEqual(ratio, (1 + 2 / 30) ** 2 + 3 * error)
        self.assertLess(stats.mean / 30 ** 2, 0.3)  # dividing by R^2 alone would be wrong

    def test_beat0_marks_eq5_as_asymptotic(self):
        body = section_html(HELP_HTML, "beat0")
        eq5 = body[body.index('<span class="eq-label">(5)</span>'):]
        eq5 = eq5[:eq5.index("\\]")]
        self.assertEqual(eq5.count(r"\approx"), 2)
        text = html_text(body)
        self.assertIn("Eq. (4) is exact for every", text)
        self.assertIn("approaches a constant", text)

    def test_beat2_constant_is_not_claimed_to_depend_only_weakly(self):
        text = html_text(section_html(HELP_HTML, "beat2"))
        self.assertNotIn("only weakly", text)
        self.assertIn("about 4%", text)

    def test_seed_wording_matches_the_commands(self):
        text = html_text(section_html(HELP_HTML, "beats"))
        self.assertNotIn("Every command on this page", text)
        self.assertIn("The commands of Beats 0 to 7 include --seed 12", text)
        for number in range(0, 8):
            for command in documented_commands(section_html(HELP_HTML, f"beat{number}")):
                self.assertIn("--seed 12", command)

    def test_extreme_scale_ratios_are_rejected_with_a_message(self):
        for radius in (1.0e155, 1.0e-160):
            with self.subTest(radius=radius):
                with self.assertRaisesRegex(ValueError, "between 1e-150 and 1e150"):
                    driver.run_walk2d(n_walks=1, radius=radius, step_cap=1)
                run = run_cli(("--display", "walk2d", "--radius", repr(radius), "--mean_free_path", "1",
                               "--step_cap", "1", "--n_walks", "1", *SEED))
                self.assertIn("Random2 input/model error", str(run.exit_code))
                self.assertEqual(run.stdout.count("\n"), 1)

    def test_scales_at_the_limits_are_accepted_and_printed_finitely(self):
        for radius in (1.0e150, 1.0e-150):
            with self.subTest(radius=radius):
                run = run_cli(("--display", "walk2d", "--radius", repr(radius), "--step_cap", "1",
                               "--n_walks", "1", *SEED))
                self.assertIn(run.exit_code, (None, 0))
                self.assertNotIn("inf", run.stdout)
                self.assertNotRegex(run.stdout, r": 0\.0+\n")

    def test_lengths_never_print_as_zero(self):
        self.assertEqual(physics.length_text(89.44271909999159), "89.4427")
        self.assertEqual(physics.length_text(30.0), "30.0000")
        self.assertEqual(physics.length_text(0.004), "0.004")
        self.assertEqual(physics.length_text(2.5e7), "2.5e+07")
        output = run_cli(("--display", "walk2d", "--radius", "0.004", "--mean_free_path", "0.001",
                          *SEED)).stdout
        self.assertIn("star radius R: 0.004\n", output)
        self.assertIn("(R / mean free path)^2: 16.0", output)

    def test_annotation_says_when_the_radius_was_given(self):
        for arguments, given in (({"radius": 0.004, "mean_free_path": 0.001}, True), ({}, False)):
            with isolated_rng(3):
                result = driver.run_walk2d(n_walks=2, **arguments)
            self.assertIs(result.radius_given, given)
            with mock.patch.object(plot.plt, "show"):
                plot.plot_walk2d(result)
            text = "\n".join(t.get_text() for t in plt.gca().texts)
            plt.close("all")
            with self.subTest(given=given):
                self.assertNotIn("radius = 0.00\n", text)
                if given:
                    self.assertIn("radius = 0.004  (given)", text)
                    self.assertNotIn("reference steps", text)
                else:
                    self.assertIn("reference steps = 2000", text)
                    self.assertNotIn("(given)", text)

    def test_plot_rejects_a_non_boolean_radius_given(self):
        result = driver.Walk2DResult("v", "b", 1.0, 1.0, 1, 1,
                                     (driver.WalkPath(((0.0, 0.0),), False, 0),), "yes")
        with self.assertRaisesRegex(ValueError, "radius_given"):
            plot.plot_walk2d(result)

    def test_original_help_lists_every_current_option(self):
        original = HELP_FILE.with_name("Random2-original.html")
        if not original.is_file():
            self.skipTest("Random2-original.html is not present in this layout")
        parser = _HelpSemanticParser()
        parser.feed(original.read_text(encoding="utf-8"))
        listed = {r[0] for r in parser.table_rows if r and r[0].startswith("--")}
        options = {
            action.option_strings[0] for action in cli.build_parser()._actions
            if action.option_strings and action.option_strings[0] not in ("-h", "--version")
        }
        self.assertEqual(listed, options)
        text = " ".join("".join(parser.visible_text).split())
        self.assertIn("the program prints a table", text)
        self.assertIn("at least 2", text)


def math_blocks(html: str) -> list:
    """TeX source of every \\( \\) and \\[ \\] block outside <pre> and <code>."""
    body = re.sub(r"<(pre|code|script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL)
    return [a or b for a, b in re.findall(r"\\\[(.*?)\\\]|\\\((.*?)\\\)", body, re.DOTALL)]


class Audit22Tests(unittest.TestCase):
    """Rendered meaning of the TeX, the seed sentence, and the original's validation note."""

    def test_no_math_block_contains_a_tex_line_break(self):
        documents = [("claude", HELP_HTML)]
        original = HELP_FILE.with_name("Random2-original.html")
        if original.is_file():
            documents.append(("original", original.read_text(encoding="utf-8")))
        for name, html in documents:
            blocks = math_blocks(html)
            self.assertGreater(len(blocks), 50)
            for block in blocks:
                with self.subTest(document=name, block=block[:40]):
                    self.assertNotIn("\\\\", block)

    def test_eq5_uses_the_logarithm_operator(self):
        body = section_html(HELP_HTML, "beat0")
        eq5 = body[body.index('<span class="eq-label">(5)</span>'):]
        eq5 = eq5[:eq5.index("\\]")]
        self.assertIn("\\log\\langle d\\rangle\\approx\\tfrac12\\log N+\\log C", eq5)
        self.assertNotRegex(eq5, r"\\\\[A-Za-z]")

    def test_seed_sentence_does_not_promise_agreement_within_one_standard_error(self):
        text = html_text(section_html(HELP_HTML, "beats"))
        self.assertNotIn("agree with these within their standard errors", text)
        self.assertIn("combined standard error of Beat 8", text)
        self.assertIn("more than two combined standard errors is expected", text)
        twelve = run_cli(SEED).result
        two = run_cli(("--seed", "2")).result
        difference = abs(twelve.averages[-1] - two.averages[-1])
        combined = math.hypot(twelve.standard_errors[-1], two.standard_errors[-1])
        self.assertAlmostEqual(difference / combined, 2.09, places=2)

    def test_original_distinguishes_parser_checks_from_derived_checks(self):
        original = HELP_FILE.with_name("Random2-original.html")
        if not original.is_file():
            self.skipTest("Random2-original.html is not present in this layout")
        parser = _HelpSemanticParser()
        parser.feed(original.read_text(encoding="utf-8"))
        text = " ".join("".join(parser.visible_text).split())
        self.assertIn("are rejected by the command-line parser", text)
        self.assertIn("checked by the program after parsing but before any walk starts", text)
        self.assertNotIn("out-of-range physical parameters", text)
        run = run_cli(("--display", "walk2d", "--radius", "1e155", "--mean_free_path", "1",
                       "--n_walks", "1", "--step_cap", "1", *SEED))
        self.assertTrue(run.stdout.startswith("Random2 "))
        self.assertEqual(run.stderr, "")
        self.assertIn("Random2 input/model error", str(run.exit_code))


class HelpOriginalCompatibilityTests(unittest.TestCase):
    """The Reference Guide version is optional; these tests never require it."""

    @classmethod
    def setUpClass(cls):
        cls.original = HELP_FILE.with_name("Random2-original.html")
        if not cls.original.is_file():
            raise unittest.SkipTest("Random2-original.html is not present in this layout")
        cls.text = cls.original.read_text(encoding="utf-8")

    def test_original_stamp_matches_the_program(self):
        match = re.search(
            r'<p id="version_build"[^>]*>\s*Version\s+([^&<\s]+)(?:&nbsp;)+Build\s+([0-9a-f]{12})',
            self.text,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.groups(), (physics.MODEL_VERSION, physics.BUILD_ID))

    def test_original_parameter_defaults_still_match_the_parser(self):
        parser = _HelpSemanticParser()
        parser.feed(self.text)
        rows = {r[0]: r[1] for r in parser.table_rows if len(r) >= 2 and r[0].startswith("--")}
        for action in cli.build_parser()._actions:
            name = action.option_strings[0] if action.option_strings else None
            if name in rows:
                expected = "omitted" if action.default is None else str(action.default)
                self.assertEqual(rows[name], expected)

    def test_original_commands_still_run(self):
        commands = []
        for line in html_module.unescape(re.sub(r"<[^>]+>", "\n", self.text)).splitlines():
            line = " ".join(line.replace("\\", " ").split())
            if line.startswith("python main.py"):
                commands.append(line)
        self.assertGreaterEqual(len(commands), 10)
        for command in dict.fromkeys(commands):
            arguments = command_arguments(command)
            if "--n_walks" in arguments and int(arguments[arguments.index("--n_walks") + 1]) > 10:
                arguments = (*arguments, "--radius", "10")  # keep the check quick
            with self.subTest(command=command):
                self.assertIn(run_cli(arguments).exit_code, (None, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
