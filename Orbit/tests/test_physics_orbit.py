"""Regression tests for the Orbit tutorial program.

The discovery code deliberately supports both repository layouts used by the
project: this file may live in ``tests/`` or may be flattened beside the four
program modules during an AI-review upload.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import html as html_module
from html.parser import HTMLParser
import importlib.util
import io
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import NamedTuple
import unittest
from unittest import mock

import numpy as np


CORE_MODULE_FILES = (
    "physics_orbit.py",
    "driver_orbit.py",
    "main.py",
    "plot_orbit.py",
)


def find_module_dir(start: Path | str) -> Path:
    """Return the nearest ancestor containing all four Orbit modules."""
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        if all((directory / name).is_file() for name in CORE_MODULE_FILES):
            return directory

    required = ", ".join(CORE_MODULE_FILES)
    raise FileNotFoundError(
        f"Could not find an Orbit module directory containing: {required}"
    )


MODULE_DIR = find_module_dir(Path(__file__).resolve().parent)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import driver_orbit as driver  # noqa: E402
import main as orbit_main  # noqa: E402
import physics_orbit as physics  # noqa: E402
import plot_orbit as plotting  # noqa: E402


def find_help_file(module_dir: Path) -> Path:
    """Find Help in a flattened upload or the GFTGU-Documentation tree.

    Documentation folders no longer use chapter-number prefixes, and the
    Help files live under the sibling ``GFTGU-Documentation`` repository
    rather than beside the program modules inside ``GFTGU-Programs``.
    """
    # The Beats Help is named Orbit-claude.html until it is adopted as the
    # live Help, when it is renamed Orbit.html; either name is accepted.  The
    # Reference Guide version, Orbit-original.html, is never used here.
    help_filenames = ("Orbit-claude.html", "Orbit.html")
    program_name = "Orbit"
    candidates = [module_dir / name for name in help_filenames]
    for ancestor in (module_dir, *module_dir.parents):
        for name in help_filenames:
            candidates.append(ancestor / "GFTGU-Documentation" / program_name / name)
            if ancestor.name != program_name:
                candidates.append(ancestor / program_name / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find Orbit-claude.html (or Orbit.html) beside the program or in "
        "GFTGU-Documentation/Orbit/."
    )


HELP_PATH = find_help_file(MODULE_DIR)
DOCUMENTATION_DIR = HELP_PATH.parent
ORIGINAL_HELP_PATH = DOCUMENTATION_DIR / "Orbit-original.html"
RELEASE_NOTES_PATH = DOCUMENTATION_DIR / "Orbit-ReleaseNotes.html"
SAMPLE_OUTPUTS_PATH = (
    DOCUMENTATION_DIR / "SampleOutputs" / "Orbit-SampleOutputs_Guide.html"
)


def expected_build_id(directory: Path = MODULE_DIR) -> str:
    """Independently reproduce the documented core-only build hash."""
    digest = hashlib.sha256()
    for name in CORE_MODULE_FILES:
        with (directory / name).open("r", encoding="utf-8", newline=None) as source:
            content = source.read().encode("utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()[:12]


def circular_result(**overrides: float) -> driver.OrbitResult:
    values = dict(
        xInit=1.0,
        yInit=0.0,
        vxInit=0.0,
        vyInit=1.0,
        k=1.0,
        dt0=0.1,
        maxSteps=5_000,
        eps1=0.05,
        eps2=1.0e-4,
        maxOrbits=1.0,
    )
    values.update(overrides)
    return driver.run_orbit(**values)


class IdTextParser(HTMLParser):
    """Collect text belonging to elements with an id attribute."""

    def __init__(self) -> None:
        super().__init__()
        self._stack: list[str | None] = []
        self.text_by_id: dict[str, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element_id = dict(attrs).get("id")
        self._stack.append(element_id)
        if element_id is not None:
            self.text_by_id.setdefault(element_id, [])

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element_id = dict(attrs).get("id")
        if element_id is not None:
            self.text_by_id.setdefault(element_id, [])

    def handle_endtag(self, tag: str) -> None:
        if self._stack:
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        for element_id in reversed(self._stack):
            if element_id is not None:
                self.text_by_id[element_id].append(data)
                break


class DiscoveryAndCompatibilityTests(unittest.TestCase):
    def test_find_module_dir_from_module_and_nested_directory(self) -> None:
        self.assertEqual(find_module_dir(MODULE_DIR / "main.py"), MODULE_DIR)
        self.assertEqual(find_module_dir(Path(__file__).parent), MODULE_DIR)

    def test_find_module_dir_prefers_nearest_complete_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            outer = root / "outer"
            inner = outer / "inner"
            nested = inner / "tests" / "deeper"
            nested.mkdir(parents=True)
            for directory in (outer, inner):
                for name in CORE_MODULE_FILES:
                    (directory / name).touch()
            self.assertEqual(find_module_dir(nested), inner)

    def test_find_module_dir_failure_names_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            with self.assertRaisesRegex(FileNotFoundError, "physics_orbit.py"):
                find_module_dir(temp_name)

    def test_all_core_modules_parse_with_python_310_grammar(self) -> None:
        for name in CORE_MODULE_FILES:
            source = (MODULE_DIR / name).read_text(encoding="utf-8")
            ast.parse(source, filename=name, feature_version=(3, 10))

    def test_version_command_works_from_module_directory(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(MODULE_DIR / "main.py"), "--version"],
            cwd=MODULE_DIR,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(
            completed.stdout.strip(),
            f"Orbit {physics.MODEL_VERSION} (build {physics.BUILD_ID})",
        )


class BuildMetadataTests(unittest.TestCase):
    def test_build_manifest_is_exactly_the_four_executable_modules(self) -> None:
        self.assertEqual(physics.BUILD_ID_COVERS, CORE_MODULE_FILES)

    def test_build_id_matches_independent_calculation(self) -> None:
        self.assertEqual(physics.BUILD_ID, expected_build_id())
        self.assertRegex(physics.BUILD_ID, r"^[0-9a-f]{12}$")

    def test_build_id_ignores_line_ending_only_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            copied = Path(temp_name)
            for name in CORE_MODULE_FILES:
                text = (MODULE_DIR / name).read_text(encoding="utf-8")
                with (copied / name).open("w", encoding="utf-8", newline="") as target:
                    target.write(text.replace("\n", "\r\n"))
            self.assertEqual(expected_build_id(copied), physics.BUILD_ID)

    def test_any_core_source_change_changes_build_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            copied = Path(temp_name)
            for name in CORE_MODULE_FILES:
                shutil.copy2(MODULE_DIR / name, copied / name)
            with (copied / "driver_orbit.py").open("a", encoding="utf-8") as target:
                target.write("\n# build-id regression probe\n")
            self.assertNotEqual(expected_build_id(copied), physics.BUILD_ID)

    def test_help_and_test_changes_do_not_change_build_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            copied = Path(temp_name)
            for name in CORE_MODULE_FILES:
                shutil.copy2(MODULE_DIR / name, copied / name)
            (copied / "Orbit-claude.html").write_text("changed help", encoding="utf-8")
            (copied / "test_physics_orbit.py").write_text("changed tests", encoding="utf-8")
            self.assertEqual(expected_build_id(copied), physics.BUILD_ID)

    def test_build_id_falls_back_to_unknown_when_core_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            copied_physics = Path(temp_name) / "physics_orbit.py"
            shutil.copy2(MODULE_DIR / "physics_orbit.py", copied_physics)
            spec = importlib.util.spec_from_file_location("physics_orbit_missing_core", copied_physics)
            self.assertIsNotNone(spec)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            self.assertEqual(module.BUILD_ID, "unknown")


class PhysicsFunctionTests(unittest.TestCase):
    def test_acceleration_known_three_four_five_geometry(self) -> None:
        ax, ay = physics.compute_acceleration(3.0, 4.0, 25.0)
        self.assertAlmostEqual(ax, -0.6)
        self.assertAlmostEqual(ay, -0.8)

    def test_acceleration_is_inward_and_has_inverse_square_magnitude(self) -> None:
        ax1, ay1 = physics.compute_acceleration(2.0, -1.0, 7.0)
        ax2, ay2 = physics.compute_acceleration(4.0, -2.0, 7.0)
        self.assertLess(2.0 * ax1 + (-1.0) * ay1, 0.0)
        self.assertAlmostEqual(math.hypot(ax2, ay2), math.hypot(ax1, ay1) / 4.0)

    def test_acceleration_scales_linearly_with_mu(self) -> None:
        a1 = physics.compute_acceleration(3.0, 4.0, 5.0)
        a2 = physics.compute_acceleration(3.0, 4.0, 10.0)
        np.testing.assert_allclose(a2, np.multiply(a1, 2.0), rtol=1.0e-15)

    def test_acceleration_avoids_premature_intermediate_overflow(self) -> None:
        ax, ay = physics.compute_acceleration(1.0e154, 0.0, 1.0e308)
        self.assertEqual(ax, -1.0)
        self.assertEqual(ay, -0.0)

    def test_acceleration_rejects_singular_invalid_and_unrepresentable_inputs(self) -> None:
        invalid_calls = (
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, -1.0),
            (math.nan, 0.0, 1.0),
            (1.0, math.inf, 1.0),
            (True, 0.0, 1.0),
            ("1", 0.0, 1.0),
            (1.0e-200, 0.0, 1.0),
        )
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                physics.compute_acceleration(*arguments)

    def test_specific_energy_known_value_and_rotation_invariance(self) -> None:
        expected = 0.5 * 3.0**2 - 4.0 / 2.0
        self.assertAlmostEqual(physics.specific_energy(2.0, 0.0, 0.0, 3.0, 4.0), expected)
        self.assertAlmostEqual(physics.specific_energy(0.0, 2.0, -3.0, 0.0, 4.0), expected)

    def test_specific_energy_rejects_bad_domain_or_overflow(self) -> None:
        cases = (
            (0.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, math.inf, 0.0, 1.0),
            (1.0, 0.0, 1.0e308, 0.0, 1.0),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                physics.specific_energy(*arguments)

    def test_specific_energy_rejects_every_invalid_type_in_each_position(self) -> None:
        valid = [1.0, 0.0, 0.0, 1.0, 1.0]
        for index in range(len(valid)):
            for bad in (None, "1", math.nan, math.inf, True):
                arguments = valid.copy()
                arguments[index] = bad
                with self.subTest(index=index, bad=bad), self.assertRaises(ValueError):
                    physics.specific_energy(*arguments)

    def test_specific_angular_momentum_sign_and_zero_radial_case(self) -> None:
        self.assertEqual(physics.specific_angular_momentum(2.0, 0.0, 0.0, 3.0), 6.0)
        self.assertEqual(physics.specific_angular_momentum(2.0, 0.0, 0.0, -3.0), -6.0)
        self.assertEqual(physics.specific_angular_momentum(2.0, 0.0, -3.0, 0.0), 0.0)

    def test_specific_angular_momentum_rejects_invalid_or_overflow(self) -> None:
        for arguments in ((1.0, 0.0, False, 1.0), (1.0e308, 0.0, 0.0, 1.0e308)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                physics.specific_angular_momentum(*arguments)

    def test_specific_angular_momentum_rejects_every_invalid_type_in_each_position(self) -> None:
        valid = [1.0, 0.0, 0.0, 1.0]
        for index in range(len(valid)):
            for bad in (None, "1", math.nan, math.inf, True):
                arguments = valid.copy()
                arguments[index] = bad
                with self.subTest(index=index, bad=bad), self.assertRaises(ValueError):
                    physics.specific_angular_momentum(*arguments)

    def test_keplerian_elements_for_circular_elliptic_parabolic_and_hyperbolic_states(self) -> None:
        circular = physics.keplerian_elements(1.0, 0.0, 0.0, 1.0, 1.0)
        self.assertEqual(circular.classification, "elliptic")
        self.assertAlmostEqual(circular.eccentricity, 0.0, places=14)
        self.assertAlmostEqual(circular.semimajor_axis, 1.0)
        self.assertAlmostEqual(circular.periapsis_radius, 1.0)
        self.assertAlmostEqual(circular.apoapsis_radius, 1.0)
        self.assertAlmostEqual(circular.orbital_period, 2.0 * math.pi)
        self.assertIsNone(circular.periapsis_longitude_degrees)
        self.assertIsNone(circular.initial_true_anomaly_degrees)

        elliptic = physics.keplerian_elements(1.0, 0.0, 0.0, 0.8, 1.0)
        self.assertEqual(elliptic.classification, "elliptic")
        self.assertAlmostEqual(elliptic.eccentricity, 0.36)
        self.assertAlmostEqual(elliptic.apoapsis_radius, 1.0)
        self.assertAlmostEqual(elliptic.periapsis_longitude_degrees, 180.0)
        self.assertAlmostEqual(elliptic.initial_true_anomaly_degrees, 180.0)

        parabolic = physics.keplerian_elements(
            1.0, 0.0, 0.0, math.sqrt(2.0), 1.0
        )
        self.assertEqual(parabolic.classification, "parabolic")
        self.assertAlmostEqual(parabolic.eccentricity, 1.0)
        self.assertIsNone(parabolic.semimajor_axis)
        self.assertIsNone(parabolic.apoapsis_radius)
        self.assertIsNone(parabolic.orbital_period)

        hyperbolic = physics.keplerian_elements(1.0, 0.0, 0.0, 2.0, 1.0)
        self.assertEqual(hyperbolic.classification, "hyperbolic")
        self.assertGreater(hyperbolic.eccentricity, 1.0)
        self.assertLess(hyperbolic.semimajor_axis, 0.0)
        self.assertIsNone(hyperbolic.apoapsis_radius)

    def test_keplerian_orientation_rotates_with_initial_state(self) -> None:
        elements = physics.keplerian_elements(0.0, 1.0, -1.2, 0.0, 1.0)
        self.assertEqual(elements.classification, "elliptic")
        self.assertAlmostEqual(elements.periapsis_longitude_degrees, 90.0)
        self.assertAlmostEqual(elements.initial_true_anomaly_degrees, 0.0)


class DriverValidationTests(unittest.TestCase):
    BASE = dict(
        xInit=1.0,
        yInit=0.0,
        vxInit=0.0,
        vyInit=1.0,
        k=1.0,
        dt0=0.1,
        maxSteps=2,
        eps1=0.05,
        eps2=1.0e-4,
        maxOrbits=1.0,
    )

    def assert_invalid(self, **replacement: object) -> None:
        values = self.BASE.copy()
        values.update(replacement)
        with self.assertRaises(ValueError):
            driver.run_orbit(**values)

    def test_rejects_origin_nonpositive_and_noninteger_controls(self) -> None:
        for replacement in (
            {"xInit": 0.0, "yInit": 0.0},
            {"k": 0.0},
            {"k": -1.0},
            {"dt0": 0.0},
            {"maxSteps": 0},
            {"maxSteps": 1.5},
            {"maxSteps": True},
            {"eps1": 0.0},
            {"eps2": -1.0},
            {"maxOrbits": 0.0},
        ):
            with self.subTest(replacement=replacement):
                self.assert_invalid(**replacement)

    def test_rejects_nonfinite_nonnumeric_and_boolean_scalar_values(self) -> None:
        for name in ("xInit", "yInit", "vxInit", "vyInit", "k", "dt0", "eps1", "eps2", "maxOrbits"):
            for value in (math.nan, math.inf, "1.0", True):
                with self.subTest(name=name, value=value):
                    self.assert_invalid(**{name: value})

    def test_accepts_independent_tolerances_larger_than_one(self) -> None:
        result = driver.run_orbit(**(self.BASE | {"eps1": 10.0, "eps2": 10.0}))
        self.assertEqual(result.termination_reason, "max_steps")

    def test_accepts_numpy_real_and_integer_scalars(self) -> None:
        values = self.BASE | {"dt0": np.float64(0.1), "maxSteps": np.int64(2)}
        result = driver.run_orbit(**values)
        self.assertEqual(result.accepted_steps, 2)

    def test_rejects_nonrepresentable_derived_initial_norms(self) -> None:
        self.assert_invalid(xInit=1.3e308, yInit=1.3e308)
        self.assert_invalid(vxInit=1.3e308, vyInit=1.3e308)


class DriverHelperAndFailureTests(unittest.TestCase):
    def test_minimum_segment_radius_is_overflow_safe(self) -> None:
        radius = driver._minimum_segment_radius(1.0e308, 0.0, 1.0e308, 1.0e308)
        self.assertEqual(radius, 1.0e308)
        self.assertEqual(driver._minimum_segment_radius(-2.0, 0.0, 2.0, 0.0), 0.0)

    def test_checked_time_advance_rejects_loss_of_progress_and_overflow(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "can no longer advance"):
            driver._checked_time_advance(1.0e20, 1.0)
        with self.assertRaisesRegex(RuntimeError, "can no longer advance"):
            driver._checked_time_advance(sys.float_info.max, sys.float_info.max)

    def test_forced_corrector_nonconvergence_exhausts_retries(self) -> None:
        with mock.patch.object(driver, "_relative_increment_change", return_value=math.inf):
            with self.assertRaisesRegex(RuntimeError, "80 retries"):
                circular_result(maxSteps=1)

    def test_main_presents_value_and_runtime_errors_without_tracebacks(self) -> None:
        for exception in (ValueError("bad input"), RuntimeError("no convergence")):
            with self.subTest(exception=type(exception).__name__):
                with (
                    mock.patch.object(orbit_main, "run_orbit", side_effect=exception),
                    mock.patch.object(sys, "argv", ["main.py"]),
                ):
                    with self.assertRaisesRegex(SystemExit, f"Orbit: {exception}"):
                        orbit_main.main()

    def test_main_displays_safeguard_and_event_refinement_diagnostics(self) -> None:
        result = circular_result(maxOrbits=0.1)
        output = io.StringIO()
        with (
            mock.patch.object(orbit_main, "run_orbit", return_value=result),
            mock.patch.object(orbit_main, "plot_orbit"),
            mock.patch.object(sys, "argv", ["main.py"]),
            contextlib.redirect_stdout(output),
        ):
            orbit_main.main()
        summary = output.getvalue()
        self.assertIn(
            f"angular-step rejections : {result.angular_step_rejections}",
            summary,
        )
        self.assertIn(
            f"endpoint refinement trials: {result.event_refinement_trials}",
            summary,
        )


class CommandLineAndSummaryTests(unittest.TestCase):
    def test_command_line_defaults_match_main_and_driver_configuration(self) -> None:
        args = orbit_main.parse_args([])
        self.assertEqual(args.xInit, 4.6e10)
        self.assertEqual(args.yInit, 0.0)
        self.assertEqual(args.vxInit, 0.0)
        self.assertEqual(args.vyInit, 58_980.0)
        self.assertEqual(args.k, physics.GM_SUN)
        self.assertEqual(args.dt0, 1.0e4)
        self.assertEqual(args.maxSteps, 20_000)
        self.assertEqual(args.eps1, 0.05)
        self.assertEqual(args.eps2, 1.0e-4)
        self.assertEqual(args.maxOrbits, 1.0)
        self.assertEqual(args.output, "orbit")

    def test_all_command_line_values_are_forwarded(self) -> None:
        result = circular_result(maxOrbits=0.1)
        argv = [
            "main.py", "--xInit", "2", "--yInit", "3",
            "--vxInit", "4", "--vyInit", "5", "--k", "6",
            "--dt0", "0.2", "--maxSteps", "7", "--eps1", "0.3",
            "--eps2", "0.004", "--maxOrbits", "0.5",
            "--output", "velocity",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(orbit_main, "run_orbit", return_value=result) as run,
            mock.patch.object(orbit_main, "plot_orbit") as plot,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            orbit_main.main()
        run.assert_called_once_with(
            xInit=2.0, yInit=3.0, vxInit=4.0, vyInit=5.0, k=6.0,
            dt0=0.2, maxSteps=7, eps1=0.3, eps2=0.004, maxOrbits=0.5,
        )
        plot.assert_called_once_with(result, output="velocity")

    def test_help_names_every_input_and_describes_output_choices(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(MODULE_DIR / "main.py"), "--help"],
            cwd=MODULE_DIR,
            check=True,
            text=True,
            capture_output=True,
        )
        for option in (
            "--xInit", "--yInit", "--vxInit", "--vyInit", "--k",
            "--dt0", "--maxSteps", "--eps1", "--eps2", "--maxOrbits",
            "--output",
        ):
            with self.subTest(option=option):
                self.assertIn(option, completed.stdout)
        for choice in orbit_main.OUTPUT_CHOICES:
            with self.subTest(choice=choice):
                self.assertIn(choice, completed.stdout)

    def test_five_significant_digit_formatter(self) -> None:
        expected = {
            0.0: "0.0000",
            1.0: "1.0000",
            12.3456: "12.346",
            0.000123456: "0.00012346",
            12345.6: "12346",
            123456.0: "1.2346e+05",
        }
        for value, text in expected.items():
            with self.subTest(value=value):
                self.assertEqual(orbit_main._five_significant(value), text)

    def test_default_summary_contains_requested_diagnostics_and_elements(self) -> None:
        result = driver.run_orbit(
            xInit=4.6e10, yInit=0.0, vxInit=0.0, vyInit=58_980.0,
            k=physics.GM_SUN, dt0=1.0e4, maxSteps=20_000,
            eps1=0.05, eps2=1.0e-4, maxOrbits=1.0,
        )
        summary = "\n".join(orbit_main._summary_lines(result))
        for text in (
            "termination", "accepted steps", "elapsed simulated time",
            "azimuthal revolutions", "angular-step rejections",
            "endpoint refinement trials", "max fractional energy drift",
            "max absolute specific-energy drift",
            "max fractional angular-momentum drift",
            "max absolute specific-angular-momentum drift",
            "closure radius residual", "closure velocity residual",
            "conic classification       : elliptic", "eccentricity",
            "semimajor axis", "semilatus rectum", "periapsis radius",
            "apoapsis radius", "Keplerian period", "periapsis longitude",
            "initial true anomaly",
        ):
            with self.subTest(text=text):
                self.assertIn(text, summary)

    def test_hyperbolic_step_limit_summary_reports_final_radius_and_speed(self) -> None:
        result = driver.run_orbit(
            xInit=4.6e10, yInit=0.0, vxInit=0.0, vyInit=85_000.0,
            k=physics.GM_SUN, dt0=1.0e4, maxSteps=30,
            eps1=0.05, eps2=1.0e-4, maxOrbits=1.0,
        )
        summary = "\n".join(orbit_main._summary_lines(result))
        self.assertEqual(result.termination_reason, "max_steps")
        self.assertIn("conic classification       : hyperbolic", summary)
        self.assertIn("apoapsis radius            : n/a (unbound)", summary)
        self.assertIn("final radius", summary)
        self.assertIn("final speed", summary)

    def test_hyperbolic_revolution_limit_summary_reports_final_radius_and_speed(self) -> None:
        result = driver.run_orbit(
            xInit=4.6e10, yInit=0.0, vxInit=0.0, vyInit=85_000.0,
            k=physics.GM_SUN, dt0=1.0e4, maxSteps=600,
            eps1=0.05, eps2=1.0e-4, maxOrbits=0.05,
        )
        summary = "\n".join(orbit_main._summary_lines(result))
        self.assertEqual(result.termination_reason, "max_orbits")
        self.assertIn("final radius", summary)
        self.assertIn("final speed", summary)


class OrbitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.circular = circular_result()

    def test_result_metadata_arrays_and_initial_state(self) -> None:
        result = self.circular
        self.assertEqual(result.model_version, physics.MODEL_VERSION)
        self.assertEqual(result.build_id, physics.BUILD_ID)
        arrays = (result.xs, result.ys, result.vxs, result.vys, result.ts, result.PEs, result.KEs, result.Hs)
        self.assertTrue(all(array.dtype == np.dtype(float) for array in arrays))
        self.assertTrue(all(len(array) == result.accepted_steps + 1 for array in arrays))
        self.assertTrue(all(np.isfinite(array).all() for array in arrays))
        self.assertEqual((result.xs[0], result.ys[0], result.vxs[0], result.vys[0], result.ts[0]), (1.0, 0.0, 0.0, 1.0, 0.0))
        self.assertTrue(np.all(np.diff(result.ts) > 0.0))

    def test_unit_circular_orbit_period_closure_and_conservation(self) -> None:
        result = self.circular
        self.assertEqual(result.termination_reason, "max_orbits")
        self.assertAlmostEqual(result.revolutions_completed, 1.0, places=12)
        self.assertLess(abs(result.final_time - 2.0 * math.pi) / (2.0 * math.pi), 2.0e-4)
        self.assertLess(result.closure_radius_residual, 1.0e-7)
        self.assertLess(result.closure_velocity_residual, 2.0e-4)
        self.assertLess(result.max_fractional_energy_drift, 3.0e-4)
        self.assertLess(result.max_fractional_angular_momentum_drift, 2.0e-4)

    def test_default_mercury_case_matches_analytic_kepler_period(self) -> None:
        radius = 4.6e10
        speed = 58_980.0
        energy = 0.5 * speed**2 - physics.GM_SUN / radius
        semimajor_axis = -physics.GM_SUN / (2.0 * energy)
        analytic_period = 2.0 * math.pi * math.sqrt(semimajor_axis**3 / physics.GM_SUN)
        result = driver.run_orbit(
            radius, 0.0, 0.0, speed, physics.GM_SUN,
            1.0e4, 20_000, 0.05, 1.0e-4, 1.0,
        )
        self.assertEqual(result.termination_reason, "max_orbits")
        self.assertLess(abs(result.final_time - analytic_period) / analytic_period, 1.0e-3)
        self.assertLess(result.max_fractional_energy_drift, 1.0e-4)
        self.assertLess(result.closure_velocity_residual, 1.0e-4)

    def test_clockwise_and_rotated_cases_are_equivalent(self) -> None:
        clockwise = circular_result(vyInit=-1.0)
        rotated = circular_result(xInit=0.0, yInit=1.0, vxInit=-1.0, vyInit=0.0)
        self.assertAlmostEqual(clockwise.final_time, self.circular.final_time, places=12)
        self.assertAlmostEqual(rotated.final_time, self.circular.final_time, places=12)
        self.assertAlmostEqual(clockwise.revolutions_completed, 1.0, places=12)
        self.assertAlmostEqual(rotated.revolutions_completed, 1.0, places=12)

    def test_fractional_orbit_has_integrated_target_angle_but_no_closure_diagnostics(self) -> None:
        result = circular_result(maxOrbits=0.5)
        self.assertEqual(result.termination_reason, "max_orbits")
        self.assertAlmostEqual(result.revolutions_completed, 0.5, places=12)
        self.assertAlmostEqual(abs(math.atan2(result.ys[-1], result.xs[-1])), math.pi, places=11)
        angles = np.arctan2(result.ys, result.xs)
        integrated_revolutions = abs(sum(
            driver._unwrap_delta(float(new), float(old))
            for old, new in zip(angles[:-1], angles[1:])
        )) / (2.0 * math.pi)
        self.assertEqual(result.revolutions_completed, integrated_revolutions)
        self.assertGreater(result.event_refinement_trials, 1)
        self.assertIsNone(result.closure_radius_residual)
        self.assertIsNone(result.closure_velocity_residual)

    def test_two_revolutions_have_closure_diagnostics(self) -> None:
        result = circular_result(maxSteps=10_000, maxOrbits=2.0)
        self.assertEqual(result.termination_reason, "max_orbits")
        self.assertAlmostEqual(result.revolutions_completed, 2.0, places=12)
        self.assertIsNotNone(result.closure_radius_residual)
        self.assertIsNotNone(result.closure_velocity_residual)

    def test_unbound_case_stops_at_max_steps(self) -> None:
        result = driver.run_orbit(1.0, 0.0, 0.0, 2.0, 1.0, 0.01, 100, 0.05, 1.0e-4, 1.0)
        self.assertEqual(result.termination_reason, "max_steps")
        self.assertEqual(result.accepted_steps, 100)
        self.assertIsNone(result.closure_radius_residual)

    def test_radial_infall_stops_at_singularity_guard_without_nonfinite_state(self) -> None:
        result = driver.run_orbit(1.0, 0.0, -0.1, 0.0, 1.0, 0.01, 10_000, 0.05, 1.0e-4, 1.0)
        self.assertEqual(result.termination_reason, "central_singularity")
        self.assertEqual(result.revolutions_completed, 0.0)
        self.assertIsNone(result.max_fractional_angular_momentum_drift)
        self.assertTrue(np.isfinite(result.xs).all())
        self.assertTrue(np.isfinite(result.vxs).all())

    def test_oversized_outward_trial_does_not_false_trigger_singularity(self) -> None:
        result = driver.run_orbit(1.0, 0.0, 2.0, 0.0, 1.0, 6.0, 1, 10.0, 10.0, 1.0)
        self.assertEqual(result.termination_reason, "max_steps")
        self.assertGreater(result.xs[-1], result.xs[0])

    def test_scale_relative_guard_accepts_microscopic_starting_radii(self) -> None:
        for radius in (1.0e-8, 5.0e-7, 1.0e-6, 2.0e-6):
            with self.subTest(radius=radius):
                result = driver.run_orbit(
                    radius, 0.0, 0.0, radius, radius**3,
                    0.01, 1, 0.05, 1.0e-4, 1.0,
                )
                self.assertEqual(result.termination_reason, "max_steps")
                self.assertEqual(result.accepted_steps, 1)

    def test_angular_step_limit_prevents_endpoint_unwrap_aliasing(self) -> None:
        result = driver.run_orbit(1.0, 0.0, 0.0, 1.0, 1.0, 10.0, 5_000, 10.0, 10.0, 1.0)
        angles = np.arctan2(result.ys, result.xs)
        deltas = [
            abs(driver._unwrap_delta(float(new), float(old)))
            for old, new in zip(angles[:-1], angles[1:])
        ]
        self.assertEqual(result.termination_reason, "max_orbits")
        self.assertAlmostEqual(result.revolutions_completed, 1.0, places=12)
        self.assertLess(max(deltas), 0.5 * math.pi)
        self.assertGreater(result.angular_step_rejections, 0)

    def test_angular_safeguard_survives_eccentric_periapsis_with_loose_tolerances(self) -> None:
        result = driver.run_orbit(
            1.0, 0.0, 0.0, 0.5, 1.0,
            10.0, 20_000, 10.0, 10.0, 1.0,
        )
        angles = np.arctan2(result.ys, result.xs)
        deltas = [
            abs(driver._unwrap_delta(float(new), float(old)))
            for old, new in zip(angles[:-1], angles[1:])
        ]
        self.assertEqual(result.termination_reason, "max_orbits")
        self.assertGreater(result.angular_step_rejections, 0)
        self.assertLess(max(deltas), math.pi)

    def test_parabolic_case_uses_absolute_not_fractional_energy_drift(self) -> None:
        result = driver.run_orbit(1.0, 0.0, 0.0, math.sqrt(2.0), 1.0, 0.01, 100, 0.05, 1.0e-4, 1.0)
        self.assertIsNone(result.max_fractional_energy_drift)
        self.assertGreater(result.max_absolute_specific_energy_drift, 0.0)
        self.assertTrue(math.isfinite(result.max_absolute_specific_energy_drift))

    def test_tighter_eps1_alone_improves_circular_orbit_trend(self) -> None:
        loose = circular_result(eps1=0.1, eps2=1.0e-4)
        tight = circular_result(eps1=0.01, eps2=1.0e-4)
        self.assertGreater(tight.accepted_steps, loose.accepted_steps)
        self.assertLess(tight.max_fractional_energy_drift, loose.max_fractional_energy_drift)
        self.assertLess(abs(tight.final_time - 2.0 * math.pi), abs(loose.final_time - 2.0 * math.pi))

    def test_tighter_eps2_alone_improves_corrector_conservation(self) -> None:
        loose = circular_result(eps1=0.05, eps2=1.0e-2)
        tight = circular_result(eps1=0.05, eps2=1.0e-8)
        self.assertLess(tight.max_fractional_energy_drift, loose.max_fractional_energy_drift)
        self.assertLess(tight.closure_velocity_residual, loose.closure_velocity_residual)


class PlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = circular_result(maxOrbits=0.1)

    def tearDown(self) -> None:
        plotting.plt.close("all")

    def test_all_five_documented_output_modes_render(self) -> None:
        for mode in ("orbit", "velocity", "position_time", "velocity_time", "energy"):
            with self.subTest(mode=mode), mock.patch.object(plotting.plt, "show") as show:
                plotting.plot_orbit(self.result, output=mode)
                show.assert_called_once_with()
                axis = plotting.plt.gcf().axes[0]
                self.assertTrue(axis.get_title().startswith("Orbit"))
                self.assertTrue(axis.get_xlabel())
                self.assertTrue(axis.get_ylabel())
                plotting.plt.close("all")

    def test_unknown_output_mode_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown output mode"):
            plotting.plot_orbit(self.result, output="not-a-mode")

    def test_plotted_data_and_equal_aspect_match_result(self) -> None:
        with mock.patch.object(plotting.plt, "show"):
            plotting.plot_orbit(self.result, output="orbit")
            axis = plotting.plt.gcf().axes[0]
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), self.result.xs)
            np.testing.assert_array_equal(axis.lines[0].get_ydata(), self.result.ys)
            self.assertEqual(axis.get_aspect(), 1.0)
            plotting.plt.close("all")

            plotting.plot_orbit(self.result, output="velocity")
            axis = plotting.plt.gcf().axes[0]
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), self.result.vxs)
            np.testing.assert_array_equal(axis.lines[0].get_ydata(), self.result.vys)
            self.assertEqual(axis.get_aspect(), 1.0)
            plotting.plt.close("all")

            plotting.plot_orbit(self.result, output="energy")
            axis = plotting.plt.gcf().axes[0]
            np.testing.assert_allclose(
                axis.lines[2].get_ydata(),
                self.result.KEs + self.result.PEs,
                rtol=0.0,
                atol=0.0,
            )


class TerminationReasonTests(unittest.TestCase):
    def test_members_are_plain_strings_with_stable_values(self) -> None:
        self.assertEqual(
            {member.name: member.value for member in driver.TerminationReason},
            {
                "MAX_ORBITS": "max_orbits",
                "MAX_STEPS": "max_steps",
                "CENTRAL_SINGULARITY": "central_singularity",
            },
        )
        for member in driver.TerminationReason:
            with self.subTest(member=member.name):
                self.assertIsInstance(member, str)
                self.assertEqual(member, member.value)
                self.assertEqual(str(member), member.value)
                self.assertEqual(f"{member}", member.value)
                self.assertEqual(hash(member), hash(member.value))
                self.assertEqual({member.value: 1}[member], 1)

    def test_each_kind_of_run_reports_its_member(self) -> None:
        cases = (
            (circular_result(), driver.TerminationReason.MAX_ORBITS),
            (circular_result(maxSteps=3), driver.TerminationReason.MAX_STEPS),
            (
                driver.run_orbit(
                    xInit=1.0, yInit=0.0, vxInit=-0.1, vyInit=0.0, k=1.0,
                    dt0=0.1, maxSteps=20_000, eps1=0.05, eps2=1.0e-4,
                    maxOrbits=1.0,
                ),
                driver.TerminationReason.CENTRAL_SINGULARITY,
            ),
        )
        for result, expected in cases:
            with self.subTest(expected=expected.name):
                self.assertIs(result.termination_reason, expected)

    def test_every_member_has_summary_text(self) -> None:
        self.assertEqual(
            set(orbit_main.TERMINATION_TEXT), set(driver.TerminationReason)
        )
        for member, text in orbit_main.TERMINATION_TEXT.items():
            with self.subTest(member=member.name):
                self.assertTrue(text)


class NumericalConstantTests(unittest.TestCase):
    EXPECTED = {
        "MAX_CORRECTOR_ITERATIONS": 10,
        "MAX_RETRIES_PER_STEP": 80,
        "MAX_EVENT_REFINEMENT_TRIALS": 80,
        "MAX_ANGULAR_STEP": 0.5 * math.pi,
        "TIMESTEP_SHRINK_FACTOR": 0.5,
        "TIMESTEP_GROWTH_FACTOR": 1.1,
        "SINGULARITY_GUARD_RELATIVE": 1.0e-12,
        "SINGULARITY_GUARD_ULPS": 32.0,
        "SINGULARITY_STOP_FACTOR": 1024.0,
        "EVENT_ANGLE_RELATIVE_TOLERANCE": 1.0e-12,
        "EVENT_ANGLE_FLOOR": 1.0e-14,
        "MIN_MAX_ORBITS": 1.0e-9,
    }

    def test_constants_exist_with_the_documented_values(self) -> None:
        for name, value in self.EXPECTED.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(driver, name), value)

    def test_no_numerical_limit_is_left_as_a_local_literal(self) -> None:
        source = (MODULE_DIR / "driver_orbit.py").read_text(encoding="utf-8")
        run_orbit_source = source.split("def run_orbit(", 1)[1]
        for forbidden in (
            "max_corrector_iterations", "max_retries_per_step",
            "max_angular_step", "max_event_refinement_trials",
            "dt_work *= 0.5", "dt_work * 1.1", "1024.0", "32.0 *",
        ):
            with self.subTest(literal=forbidden):
                self.assertNotIn(forbidden, run_orbit_source)

    def test_retry_limit_constant_controls_the_failure_message(self) -> None:
        with (
            mock.patch.object(driver, "MAX_RETRIES_PER_STEP", 3),
            mock.patch.object(driver, "_relative_increment_change", return_value=math.inf),
        ):
            with self.assertRaisesRegex(RuntimeError, "after 3 retries"):
                circular_result(maxSteps=1)

    def test_event_refinement_limit_constant_controls_the_failure_message(self) -> None:
        with mock.patch.object(driver, "MAX_EVENT_REFINEMENT_TRIALS", 1):
            with self.assertRaisesRegex(RuntimeError, "after 1 trials"):
                circular_result(maxOrbits=0.3)

    def test_corrector_iteration_limit_constant_forces_a_shorter_step(self) -> None:
        baseline = circular_result(eps2=1.0e-12)
        with mock.patch.object(driver, "MAX_CORRECTOR_ITERATIONS", 1):
            limited = circular_result(eps2=1.0e-12)
        self.assertGreater(limited.accepted_steps, baseline.accepted_steps)

    def test_growth_factor_constant_controls_recovery_after_a_shrunken_step(self) -> None:
        kwargs = dict(vyInit=0.3, dt0=0.5, maxSteps=20_000)
        baseline = circular_result(**kwargs)
        with mock.patch.object(driver, "TIMESTEP_GROWTH_FACTOR", 1.0):
            no_growth = circular_result(**kwargs)
        self.assertGreater(no_growth.accepted_steps, baseline.accepted_steps)

    def test_shrink_factor_constant_controls_rejection(self) -> None:
        kwargs = dict(vyInit=0.3, dt0=0.5, maxSteps=20_000)
        baseline = circular_result(**kwargs)
        with mock.patch.object(driver, "TIMESTEP_SHRINK_FACTOR", 0.9):
            gentle = circular_result(**kwargs)
        self.assertNotEqual(gentle.accepted_steps, baseline.accepted_steps)

    def test_angular_step_constant_controls_the_safeguard(self) -> None:
        baseline = circular_result(dt0=0.5)
        self.assertEqual(baseline.angular_step_rejections, 0)
        # 0.001 rad (not the value that a student could reach with ordinary settings) makes the
        # guard act on this small circular orbit, so its effect can be observed.
        with mock.patch.object(driver, "MAX_ANGULAR_STEP", 0.001):
            limited = circular_result(dt0=0.5)
        self.assertGreater(limited.angular_step_rejections, 0)

    def test_singularity_stop_factor_constant_moves_the_stop_radius(self) -> None:
        radial = dict(xInit=1.0, yInit=0.0, vxInit=-0.1, vyInit=0.0, k=1.0,
                      dt0=0.1, maxSteps=20_000, eps1=0.05, eps2=1.0e-4)
        default = driver.run_orbit(**radial)
        with mock.patch.object(driver, "SINGULARITY_STOP_FACTOR", 1.0e9):
            early = driver.run_orbit(**radial)
        self.assertLess(
            math.hypot(default.xs[-1], default.ys[-1]),
            math.hypot(early.xs[-1], early.ys[-1]),
        )
        self.assertLessEqual(math.hypot(early.xs[-1], early.ys[-1]), 1.0e-3)


class ReportedDefectRegressionTests(unittest.TestCase):
    def test_five_significant_digits_survive_rounding_at_a_decade_boundary(self) -> None:
        expected = {
            0.99999996: "1.0000",
            -0.99999996: "-1.0000",
            9.99996: "10.000",
            99999.6: "1.0000e+05",
            0.000099999996: "0.00010000",
            0.99994: "0.99994",
            99999.4: "99999",
        }
        for value, text in expected.items():
            with self.subTest(value=value):
                self.assertEqual(orbit_main._five_significant(value), text)

    def test_five_significant_digit_count_is_exact_for_many_values(self) -> None:
        def significant_digits(text: str) -> int:
            mantissa = text.lower().split("e")[0].lstrip("-")
            return len(mantissa.replace(".", "").lstrip("0"))

        rng = np.random.default_rng(20260924)
        exponents = rng.uniform(-9.0, 12.0, 4000)
        values = np.concatenate((
            10.0 ** exponents,
            -(10.0 ** exponents[:500]),
            10.0 ** np.round(exponents[:500]) * (1.0 - 1.0e-9),
            10.0 ** np.round(exponents[:500]) * (1.0 - 4.0e-6),
        ))
        for value in values:
            text = orbit_main._five_significant(float(value))
            self.assertEqual(significant_digits(text), 5, (float(value), text))

    def test_invalid_k_is_reported_under_the_option_name_the_student_typed(self) -> None:
        for value, message in (
            (0.0, "k=GM must be positive"),
            (-1.0, "k=GM must be positive"),
            (math.nan, "k must be finite"),
            (math.inf, "k must be finite"),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, message):
                    circular_result(k=value)
        for text, message in (("0", "k=GM must be positive"), ("nan", "k must be finite")):
            with self.subTest(argument=text):
                with (
                    mock.patch.object(sys, "argv", ["main.py", "--k", text]),
                    self.assertRaisesRegex(SystemExit, f"Orbit: {message}"),
                ):
                    orbit_main.main()

    def test_parser_defaults_are_defined_in_one_place(self) -> None:
        parser = orbit_main.build_parser()
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
            if option.startswith("--")
        }
        self.assertTrue({"--xInit", "--k", "--output", "--version", "--help"} <= options)
        self.assertEqual(orbit_main.parse_args([]).k, physics.GM_SUN)


# --------------------------------------------------------------------------
# Help file (Beats layout): structure, commands, quoted numbers, and claims.
# --------------------------------------------------------------------------

HELP_HTML = HELP_PATH.read_text(encoding="utf-8")
BEAT_NUMBERS = tuple(range(9))
EXPERIMENT_NUMBERS = tuple(range(1, 16))
EQUATION_COUNT = 17
VOID_TAGS = {"meta", "br", "hr", "img", "link", "input"}


def html_text(fragment: str) -> str:
    """Visible text of an HTML fragment, with entities decoded and spaces collapsed."""
    # Only real tags are removed: a "<" followed by a digit or space is text (e.g. "e<1").
    return " ".join(html_module.unescape(re.sub(r"</?[A-Za-z!][^>]*>", "", fragment)).split())


def section_html(html: str, section_id: str) -> str:
    match = re.search(
        rf'<section id="{re.escape(section_id)}">(.*?)</section>', html, re.DOTALL
    )
    if match is None:
        raise AssertionError(f"section {section_id!r} not found in the Help file")
    return match.group(1)


def documented_commands(fragment: str) -> list[str]:
    """Every ``python main.py ...`` command shown in a <pre> block."""
    commands = []
    for block in re.findall(r"<pre[^>]*>(.*?)</pre>", fragment, re.DOTALL):
        text = html_module.unescape(re.sub(r"<[^>]+>", "", block)).replace("\\\n", " ")
        for line in text.splitlines():
            line = " ".join(line.split())
            if line.startswith("python main.py"):
                commands.append(line)
    return commands


def is_template(command: str) -> bool:
    """True for a command containing an ALL-CAPS placeholder the student must replace."""
    return re.search(r"\b[A-Z][A-Z_]{3,}\b", command) is not None


def command_arguments(command: str) -> tuple[str, ...]:
    return tuple(shlex.split(command)[2:])


class CliRun(NamedTuple):
    stdout: str
    result: driver.OrbitResult | None
    acceleration_calls: int


_CLI_CACHE: dict[tuple[str, ...], CliRun] = {}


def run_cli(arguments: tuple[str, ...] = ()) -> CliRun:
    """Run ``main.main()`` in-process with the plot suppressed, caching each command."""
    key = tuple(arguments)
    if key in _CLI_CACHE:
        return _CLI_CACHE[key]
    captured: dict[str, driver.OrbitResult] = {}
    calls = [0]
    real_run = orbit_main.run_orbit
    real_acceleration = driver.compute_acceleration

    def counting_acceleration(*args):
        calls[0] += 1
        return real_acceleration(*args)

    def capturing_run(**kwargs):
        captured["result"] = real_run(**kwargs)
        return captured["result"]

    output = io.StringIO()
    with (
        mock.patch.object(sys, "argv", ["main.py", *key]),
        mock.patch.object(orbit_main, "run_orbit", new=capturing_run),
        mock.patch.object(orbit_main, "plot_orbit"),
        mock.patch.object(driver, "compute_acceleration", new=counting_acceleration),
        contextlib.redirect_stdout(output),
    ):
        try:
            orbit_main.main()
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    run = CliRun(output.getvalue(), captured.get("result"), calls[0])
    _CLI_CACHE[key] = run
    return run


def run_result(**overrides: float) -> driver.OrbitResult:
    """Run the driver with the default Mercury-like settings, overridden by keywords."""
    values = dict(
        xInit=4.6e10, yInit=0.0, vxInit=0.0, vyInit=58_980.0,
        k=physics.GM_SUN, dt0=1.0e4, maxSteps=20_000,
        eps1=0.05, eps2=1.0e-4, maxOrbits=1.0,
    )
    values.update(overrides)
    return driver.run_orbit(**values)


def predicted_acceleration_changes(result: driver.OrbitResult, mu: float = physics.GM_SUN) -> np.ndarray:
    """The delta_a of Eq. (16) for every accepted step of a completed run."""
    changes = []
    for index in range(len(result.ts) - 1):
        dt = result.ts[index + 1] - result.ts[index]
        x, y = result.xs[index], result.ys[index]
        vx, vy = result.vxs[index], result.vys[index]
        ax, ay = physics.compute_acceleration(x, y, mu)
        vx_pred, vy_pred = vx + ax * dt, vy + ay * dt
        x_pred = x + 0.5 * (vx + vx_pred) * dt
        y_pred = y + 0.5 * (vy + vy_pred) * dt
        ax_pred, ay_pred = physics.compute_acceleration(x_pred, y_pred, mu)
        changes.append(
            math.hypot(ax_pred - ax, ay_pred - ay)
            / max(math.hypot(ax_pred, ay_pred), math.hypot(ax, ay))
        )
    return np.array(changes)


class HelpStructure(HTMLParser):
    """Ids, links and tag balance of a Help page."""

    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.hrefs: list[str] = []
        self.sidebar_hrefs: list[str] = []
        self.section_ids: list[str] = []
        self.errors: list[str] = []
        self._stack: list[str] = []
        self._in_nav = False
        self.feed(html)
        self.close()
        if self._stack:
            self.errors.append(f"unclosed tags: {self._stack}")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.append(attributes["id"] or "")
        if tag == "section" and attributes.get("id"):
            self.section_ids.append(attributes["id"])
        if tag == "nav":
            self._in_nav = True
        href = attributes.get("href")
        if tag == "a" and href and href.startswith("#"):
            self.hrefs.append(href[1:])
            if self._in_nav:
                self.sidebar_hrefs.append(href[1:])
        if tag not in VOID_TAGS:
            self._stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
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


class HelpStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.structure = HelpStructure(HELP_HTML)

    def test_help_file_has_unique_version_and_build_matching_the_program(self) -> None:
        self.assertEqual(HELP_HTML.count('id="version_build"'), 1)
        parser = IdTextParser()
        parser.feed(HELP_HTML)
        stamp = " ".join(" ".join(parser.text_by_id["version_build"]).split())
        self.assertEqual(
            stamp, f"Version {physics.MODEL_VERSION} Build {physics.BUILD_ID}"
        )

    def test_tags_are_balanced_and_ids_are_unique(self) -> None:
        self.assertEqual(self.structure.errors, [])
        duplicates = {name for name in self.structure.ids if self.structure.ids.count(name) > 1}
        self.assertEqual(duplicates, set())

    def test_every_internal_link_and_sidebar_entry_resolves(self) -> None:
        known = set(self.structure.ids)
        self.assertEqual(sorted(set(self.structure.hrefs) - known), [])
        sidebar_targets = set(self.structure.sidebar_hrefs)
        self.assertEqual(sorted(sidebar_targets - known), [])
        for section_id in self.structure.section_ids:
            if section_id == "restore-title":
                continue
            with self.subTest(section=section_id):
                self.assertIn(section_id, sidebar_targets | {"restore-title"})

    def test_page_has_the_beats_layout_in_order(self) -> None:
        expected = (
            ["overview", "beats"]
            + [f"beat{number}" for number in BEAT_NUMBERS]
            + ["equations", "algorithm", "modules", "quickstart", "parameters",
               "output", "summary", "experiments"]
        )
        listed = [name for name in self.structure.section_ids if name in expected]
        self.assertEqual(listed, expected)
        self.assertEqual(self.structure.section_ids[-2:], ["related", "license"])

    def test_mathjax_is_the_only_external_script_and_offline_note_is_static(self) -> None:
        sources = re.findall(r'<script[^>]*\ssrc="([^"]+)"', HELP_HTML)
        self.assertEqual(sources, ["https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"])
        self.assertIn("loaded from a public CDN", HELP_HTML)
        self.assertIn("internet connection is needed", HELP_HTML)
        self.assertNotIn("navigator.onLine", HELP_HTML)
        self.assertNotIn("install MathJax locally", HELP_HTML)

    def test_development_history_is_confined_to_provenance(self) -> None:
        instructional, provenance = HELP_HTML.split('<section id="license">', 1)
        for phrase in ("revised solver", "Python port", "original Java", "Triana workflow"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, instructional)
        self.assertIn("original Java", provenance)
        self.assertIn("Python port", provenance)

    def test_help_preserves_original_textbook_cross_references(self) -> None:
        text = html_text(HELP_HTML)
        for reference in ("Table 4.3", "Table 4.2", "Investigation 4.1", "Investigation 4.2", "Chapter 6"):
            with self.subTest(reference=reference):
                self.assertIn(reference, text)


class HelpBeatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.beats = {n: section_html(HELP_HTML, f"beat{n}") for n in BEAT_NUMBERS}

    def test_every_beat_follows_the_same_pattern(self) -> None:
        for number, body in self.beats.items():
            with self.subTest(beat=number):
                self.assertRegex(body, rf"<h2>Beat {number} · ")
                self.assertEqual(body.count("<pre>"), 1)
                self.assertIn("Three tasks, in order.", body)
                self.assertIn("<em>Then</em>", body)
                self.assertRegex(body, r"<em>Experiments that go with this beat:</em>")
                self.assertLess(body.index("<pre>"), body.index("Three tasks, in order."))
                self.assertLess(body.index("Three tasks, in order."), body.index("<em>Then</em>"))
                self.assertGreaterEqual(len(documented_commands(body)), 1)
                self.assertRegex(body, r"<div class=\"eq\">")

    def test_sidebar_titles_match_the_beat_headings(self) -> None:
        for number, body in self.beats.items():
            heading = html_text(re.search(r"<h2>(.*?)</h2>", body).group(1))
            with self.subTest(beat=number):
                self.assertRegex(
                    HELP_HTML,
                    rf'<a href="#beat{number}">{number} · '
                    + re.escape(heading.split(" · ", 1)[1].replace("&", "&amp;"))
                    + "</a>",
                )

    def test_equations_are_numbered_in_order_and_tagged(self) -> None:
        labels = [int(n) for n in re.findall(r'<span class="eq-label">\((\d+)\)</span>', HELP_HTML)]
        self.assertEqual(labels, list(range(1, EQUATION_COUNT + 1)))
        tags = re.findall(r'<span class="kind kind-(\w+)">(\w+)</span>', HELP_HTML)
        allowed = {"law": "LAW", "ode": "ODE", "def": "DEFINITION",
                   "der": "DERIVED", "alg": "ALGORITHM"}
        for css_class, word in tags:
            self.assertEqual(allowed[css_class], word)
        for css_class in allowed:
            self.assertIn(f".kind-{css_class}", HELP_HTML)
        for block in re.findall(r'<div class="eq">(.*?)</div>', HELP_HTML, re.DOTALL):
            self.assertRegex(block, r'class="kind kind-(law|ode|def|der|alg)"')

    def test_only_the_gravitational_law_is_tagged_law(self) -> None:
        law_blocks = re.findall(
            r'<div class="eq"><span class="eq-label">\((\d+)\)</span>\s*'
            r'<div class="eq-kind">[^<]*<span class="kind kind-law">',
            HELP_HTML,
        )
        self.assertEqual(law_blocks, ["1"])

    def test_equation_citations_refer_to_numbered_equations(self) -> None:
        text = html_text(HELP_HTML)
        cited = {int(n) for n in re.findall(r"Eqs?\. \((\d+)\)", text)}
        cited |= {int(n) for n in re.findall(r"Eqs\. \(\d+\) (?:and|to) \((\d+)\)", text)}
        self.assertTrue(cited)
        self.assertLessEqual(max(cited), EQUATION_COUNT)
        self.assertGreaterEqual(min(cited), 1)

    def test_equation_index_lists_every_numbered_equation_once(self) -> None:
        index = section_html(HELP_HTML, "equations")
        rows = re.findall(r"<tr><td>\((\d+)\)</td><td><span class=\"kind kind-(\w+)\">", index)
        self.assertEqual([int(number) for number, _ in rows], list(range(1, EQUATION_COUNT + 1)))
        in_beats = dict(re.findall(
            r'<span class="eq-label">\((\d+)\)</span>\s*<div class="eq-kind">[^<]*'
            r'<span class="kind kind-(\w+)">',
            HELP_HTML,
        ))
        # Titles in the beats may contain markup; compare the tag kinds only.
        for number, kind in rows:
            with self.subTest(equation=number):
                self.assertEqual(kind, in_beats.get(number, kind))

    def test_every_experiment_is_cited_by_a_beat_and_every_citation_exists(self) -> None:
        cited: set[int] = set()
        for number, body in self.beats.items():
            line = re.search(r"<em>Experiments that go with this beat:</em>(.*?)</p>", body, re.DOTALL).group(1)
            links = [int(n) for n in re.findall(r'href="#exp(\d+)"', line)]
            with self.subTest(beat=number):
                self.assertTrue(links)
            cited |= set(links)
        defined = {int(n) for n in re.findall(r'<h3 id="exp(\d+)">', HELP_HTML)}
        self.assertEqual(defined, set(EXPERIMENT_NUMBERS))
        self.assertEqual(cited, defined)

    def test_experiments_are_ranked_from_introductory_to_advanced(self) -> None:
        titles = re.findall(r'<h3 id="exp(\d+)">(\d+) · (.*?)</h3>', HELP_HTML)
        self.assertEqual([int(a) for a, _, _ in titles], list(EXPERIMENT_NUMBERS))
        self.assertEqual([int(a) for a, b, _ in titles], [int(b) for _, b, _ in titles])
        self.assertIn("Introductory", titles[0][2])
        self.assertIn("Advanced Programming Extension", titles[-1][2])
        self.assertIn("Compare Error Measures", titles[-1][2])

    def test_parameter_only_experiments_use_command_line_examples(self) -> None:
        experiments = section_html(HELP_HTML, "experiments")
        for number in EXPERIMENT_NUMBERS:
            if number in (11, 14, 15):
                continue
            start = experiments.index(f'<h3 id="exp{number}">')
            end = experiments.find('<h3 id="exp', start + 1)
            with self.subTest(experiment=number):
                self.assertIn("python main.py", experiments[start:end if end != -1 else None])

    def test_exp11_shows_how_to_access_returned_arrays(self) -> None:
        text = html_text(section_html(HELP_HTML, "experiments"))
        for name in ("result.xs", "result.ys", "result.ts"):
            self.assertIn(name, text)


class HelpCommandTests(unittest.TestCase):
    """Every command shown in the Help must run, and must run to a summary."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.commands = [c for c in documented_commands(HELP_HTML) if not is_template(c)]

    def test_the_help_documents_a_meaningful_number_of_commands(self) -> None:
        self.assertGreaterEqual(len(self.commands), 40)
        self.assertTrue(any(is_template(c) for c in documented_commands(HELP_HTML)))

    def test_every_documented_command_parses_and_runs_to_a_summary(self) -> None:
        for command in dict.fromkeys(self.commands):
            with self.subTest(command=command):
                arguments = command_arguments(command)
                run = run_cli(arguments)
                if "--help" in arguments or "--version" in arguments:
                    self.assertTrue(run.stdout)
                else:
                    self.assertTrue(run.stdout.startswith(f"Orbit {physics.MODEL_VERSION} (build {physics.BUILD_ID}) summary"))
                    self.assertIsNotNone(run.result)

    def test_documented_options_are_real_options(self) -> None:
        parser = orbit_main.build_parser()
        real = {o for a in parser._actions for o in a.option_strings if o.startswith("--")}
        used = set()
        for block in re.findall(r"<pre[^>]*>(.*?)</pre>", HELP_HTML, re.DOTALL):
            used |= set(re.findall(r"(?<![\w-])(--[A-Za-z][\w]*)", html_module.unescape(block)))
        self.assertLessEqual(used, real)

    def test_printed_summary_block_is_exactly_what_the_default_run_prints(self) -> None:
        section = section_html(HELP_HTML, "summary")
        block = html_module.unescape(re.search(r"<pre><code>(.*?)</code></pre>", section, re.DOTALL).group(1))
        printed = run_cli(()).stdout.split("\n", 1)[1]
        self.assertEqual(block.strip("\n"), printed.strip("\n"))


class HelpReferenceTests(unittest.TestCase):
    """The reference sections must agree with the program that they describe."""

    def test_parameter_table_matches_the_parser_options_and_defaults(self) -> None:
        parser = orbit_main.build_parser()
        defaults = {
            option: action.default
            for action in parser._actions
            for option in action.option_strings
            if option.startswith("--") and option not in ("--help", "--version")
        }
        table = section_html(HELP_HTML, "parameters")
        rows = re.findall(
            r"<tr><td><code>(--\w+)</code></td><td>([^<]*)</td>", table
        )
        self.assertEqual([name for name, _ in rows], list(defaults))
        for name, shown in rows:
            with self.subTest(option=name):
                if isinstance(defaults[name], str):
                    self.assertEqual(shown, defaults[name])
                else:
                    self.assertEqual(float(shown), float(defaults[name]))

    def test_output_modes_match_the_selector_choices(self) -> None:
        section = section_html(HELP_HTML, "output")
        listed = re.findall(r'<span class="tag">(\w+)</span>', section)
        self.assertEqual(tuple(listed), orbit_main.OUTPUT_CHOICES)
        self.assertEqual(orbit_main.parse_args([]).output, "orbit")

    def test_every_option_and_output_choice_appears_in_program_help(self) -> None:
        text = run_cli(("--help",)).stdout
        for name in re.findall(r"<code>(--\w+)</code>", section_html(HELP_HTML, "parameters")):
            self.assertIn(name, text)
        for choice in orbit_main.OUTPUT_CHOICES:
            self.assertIn(choice, text)

    def test_constant_table_matches_the_driver_constants(self) -> None:
        table = section_html(HELP_HTML, "algorithm")
        rows = dict(re.findall(r"<tr><td><code>([A-Z_]+)</code></td><td>(.*?)</td>", table))
        display = {
            "MAX_CORRECTOR_ITERATIONS": "10",
            "MAX_RETRIES_PER_STEP": "80",
            "MAX_EVENT_REFINEMENT_TRIALS": "80",
            "MAX_ANGULAR_STEP": "&pi;/2",
            "TIMESTEP_SHRINK_FACTOR": "0.5",
            "TIMESTEP_GROWTH_FACTOR": "1.1",
            "SINGULARITY_GUARD_RELATIVE": "10<sup>&minus;12</sup>",
            "SINGULARITY_GUARD_ULPS": "32",
            "SINGULARITY_STOP_FACTOR": "1024",
            "EVENT_ANGLE_RELATIVE_TOLERANCE": "10<sup>&minus;12</sup>",
            "EVENT_ANGLE_FLOOR": "10<sup>&minus;14</sup>",
            "MIN_MAX_ORBITS": "10<sup>&minus;9</sup>",
        }
        self.assertEqual(set(rows), set(NumericalConstantTests.EXPECTED))
        for name, shown in display.items():
            with self.subTest(constant=name):
                self.assertEqual(rows[name], shown)
                self.assertEqual(getattr(driver, name), NumericalConstantTests.EXPECTED[name])

    def test_prose_about_limits_and_factors_agrees_with_the_constants(self) -> None:
        text = html_text(section_html(HELP_HTML, "algorithm"))
        self.assertEqual(driver.MAX_CORRECTOR_ITERATIONS, 10)
        self.assertIn("If ten passes do not suffice, halve", text)
        self.assertEqual(driver.TIMESTEP_GROWTH_FACTOR, 1.1)
        self.assertIn("grow by 10%", text)
        self.assertEqual(driver.TIMESTEP_SHRINK_FACTOR, 0.5)
        self.assertIn("halve ", text)
        self.assertEqual(driver.MAX_ANGULAR_STEP, math.pi / 2)
        self.assertIn("exceeds π/2", text)
        beat6 = html_text(section_html(HELP_HTML, "beat6"))
        self.assertIn("halved", beat6)
        self.assertIn("grow by 10%", beat6)
        self.assertIn("exceeds \\(\\pi/2\\)", beat6)

    def test_reference_text_names_the_termination_values_and_edge_cases(self) -> None:
        text = html_text(HELP_HTML)
        for member in driver.TerminationReason:
            with self.subTest(member=member.value):
                self.assertIn(member.value, text)
        for phrase in (
            "not the central mass in kg",
            "gravitational parameter",
            "Specific energy",
            "integral number of revolutions" if "integral number of revolutions" in text else "whole number of revolutions",
            "zero or nearly zero",
            "not interpolated",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)
        for summary_line in ("angular-step rejections", "endpoint refinement trials"):
            self.assertIn(summary_line, text)
            self.assertIn(summary_line, run_cli(()).stdout)

    def test_every_summary_label_printed_by_main_is_described_in_the_help(self) -> None:
        text = html_text(section_html(HELP_HTML, "summary"))
        labels = re.findall(r"^\s*([a-z][a-z \-()A-Z]*?)\s*:", run_cli(()).stdout, re.MULTILINE)
        self.assertGreater(len(labels), 15)
        for label in ("termination", "accepted steps", "elapsed simulated time",
                      "azimuthal revolutions", "angular-step rejections",
                      "endpoint refinement trials", "Keplerian elements at initial state",
                      "acceleration evaluations", "shortest accepted step",
                      "longest accepted step", "specific energy"):
            self.assertIn(label, text)
        for label in ("final radius", "final speed", "n/a"):
            self.assertIn(label, text)


class HelpQuotedNumberTests(unittest.TestCase):
    """Every number the Help quotes from the console must be the number the program prints."""

    PRINTED = re.compile(r"(?<![\w.])(?:\d\.\d{4}e[+-]\d\d|0\.\d{4,})(?![\w%])")
    # Numbers that a student computes by hand from printed values (not printed themselves).
    DERIVED_OK = {
        "0.2057",          # 10064/48916 in Beat 4
        "0.0075",          # 570 s over the period, in Beat 0 (percent)
        "0.0128", "0.0497", "0.0499",  # largest delta_a, verified in HelpQuantitativeClaimTests
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.default_output = run_cli(()).stdout

    # Commands that a beat describes in its prose ("run the first command again with
    # --dt0 1e6") rather than showing in a <pre> block.
    PROSE_COMMANDS = {
        6: [("--dt0", "1e6")],
        8: [
            ("--vxInit", "41705.15", "--vyInit", "41705.15"),
            ("--vxInit", "41705.15", "--vyInit", "41705.15", "--dt0", "5000"),
            ("--vxInit", "41705.15", "--vyInit", "41705.15", "--eps1", "0.005"),
        ],
    }

    def outputs_for(self, fragment: str, beat: int | None = None) -> str:
        chunks = [self.default_output]
        for arguments in self.PROSE_COMMANDS.get(beat, []):
            chunks.append(run_cli(arguments).stdout)
        for command in documented_commands(fragment):
            if not is_template(command):
                arguments = command_arguments(command)
                if "--help" not in arguments and "--version" not in arguments:
                    chunks.append(run_cli(arguments).stdout)
        return "\n".join(chunks)

    def quoted_tokens(self, fragment: str) -> set[str]:
        text = html_text(fragment)
        return set(self.PRINTED.findall(text)) - self.DERIVED_OK

    def test_numbers_quoted_in_each_beat_are_printed_by_that_beat_s_commands(self) -> None:
        for number in BEAT_NUMBERS:
            body = section_html(HELP_HTML, f"beat{number}")
            printed = self.outputs_for(body, number)
            tokens = self.quoted_tokens(body)
            with self.subTest(beat=number):
                self.assertTrue(tokens, "a beat should quote printed numbers")
                self.assertEqual(sorted(t for t in tokens if t not in printed), [])

    def test_numbers_quoted_in_the_reference_sections_are_printed_by_some_command(self) -> None:
        every_output = "\n".join(
            [self.default_output]
            + [
                run_cli(command_arguments(c)).stdout
                for c in dict.fromkeys(documented_commands(HELP_HTML))
                if not is_template(c)
                and "--help" not in c and "--version" not in c
            ]
        )
        for section in ("overview", "beats", "equations", "algorithm", "modules",
                        "quickstart", "parameters", "output", "summary", "experiments"):
            with self.subTest(section=section):
                tokens = self.quoted_tokens(section_html(HELP_HTML, section))
                self.assertEqual(sorted(t for t in tokens if t not in every_output), [])


class HelpQuantitativeClaimTests(unittest.TestCase):
    """The derived and measured statements of the beats, checked against the program."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.default = run_cli(()).result
        cls.mu = physics.GM_SUN
        cls.r0 = 4.6e10

    # Beat 0 ---------------------------------------------------------------
    def test_beat0_shape_and_period_statements(self) -> None:
        elements = self.default.orbital_elements
        r_p, r_a = elements.periapsis_radius, elements.apoapsis_radius
        self.assertAlmostEqual(r_p, 4.6e10, delta=1.0)
        self.assertAlmostEqual(r_a / r_p, 1.52, places=2)
        self.assertAlmostEqual((r_a - r_p) / (r_a + r_p), elements.eccentricity, places=12)
        excess = (self.default.final_time - elements.orbital_period) / elements.orbital_period
        self.assertGreater(excess, 0.00007)
        self.assertLess(excess, 0.00008)
        self.assertAlmostEqual(self.default.final_time - elements.orbital_period, 570.0, delta=5.0)
        self.assertAlmostEqual(self.default.final_time / 86400.0, 88.0, delta=0.05)

    # Beat 1 ---------------------------------------------------------------
    def test_beat1_energies_escape_speed_and_fractional_drift_statement(self) -> None:
        energies = {v: physics.specific_energy(self.r0, 0.0, 0.0, v, self.mu) for v in (70000.0, 75961.0, 85000.0)}
        self.assertAlmostEqual(energies[70000.0] / 1e8, -4.3505, places=4)
        self.assertAlmostEqual(energies[75961.0] / 1e4, -1.6283, places=4)
        self.assertAlmostEqual(energies[85000.0] / 1e8, 7.2745, places=4)
        self.assertAlmostEqual(math.sqrt(2 * self.mu / self.r0), 75961.2, places=1)
        self.assertAlmostEqual(math.sqrt(self.mu / self.r0), 53712.7, places=1)
        middle = run_cli(("--vyInit", "75961", "--maxSteps", "600")).result
        ratio = middle.max_absolute_specific_energy_drift / abs(energies[75961.0])
        self.assertAlmostEqual(ratio, middle.max_fractional_energy_drift, places=6)
        kinetic = 0.5 * 75961.0**2
        self.assertAlmostEqual(abs(energies[75961.0]) / kinetic, 6e-6, delta=0.7e-6)
        self.assertEqual(middle.orbital_elements.classification, "elliptic")
        self.assertAlmostEqual(middle.orbital_elements.orbital_period / 3.156e7 / 1e6, 4.5, delta=0.1)
        exact = run_cli(("--vyInit", "75961.2143594", "--maxSteps", "600")).result
        self.assertEqual(exact.orbital_elements.classification, "parabolic")
        self.assertIsNone(exact.max_fractional_energy_drift)

    def test_beat1_energy_conservation_identity_of_eq4(self) -> None:
        r, v_r = 3.0e10, 12345.0
        x, y = r, 0.0
        vx, vy = v_r, 20000.0
        ax, ay = physics.compute_acceleration(x, y, self.mu)
        r_dot = (x * vx + y * vy) / r
        self.assertAlmostEqual(
            vx * ax + vy * ay + self.mu * r_dot / r**2, 0.0, delta=1e-9 * abs(vx * ax)
        )

    # Beat 2 ---------------------------------------------------------------
    def test_beat2_equal_speed_gives_equal_energy_size_and_period(self) -> None:
        rotated = run_cli(("--vxInit", "41705.15", "--vyInit", "41705.15")).result
        a, b = self.default.orbital_elements, rotated.orbital_elements
        self.assertAlmostEqual(math.sqrt(2) * 41705.15, 58980.0, delta=0.02)
        # Tolerance 2e-6 relative (not tighter) because the Help's rotated-orbit inputs,
        # 41705.15, are rounded to seven digits and differ from 58980/sqrt(2) by ~6e-7.
        self.assertAlmostEqual(a.specific_energy, b.specific_energy, delta=abs(a.specific_energy) * 2e-6)
        self.assertAlmostEqual(a.semimajor_axis, b.semimajor_axis, delta=a.semimajor_axis * 2e-6)
        self.assertAlmostEqual(a.orbital_period, b.orbital_period, delta=a.orbital_period * 2e-6)
        self.assertAlmostEqual(a.periapsis_radius / b.periapsis_radius, 2.9, delta=0.06)
        self.assertAlmostEqual(
            rotated.max_fractional_energy_drift / self.default.max_fractional_energy_drift, 36.0, delta=0.5
        )

    def test_beat2_angular_momentum_and_eq7(self) -> None:
        rotated = run_cli(("--vxInit", "41705.15", "--vyInit", "41705.15")).result
        for result, h_expected in ((self.default, 2.7131e15), (rotated, 1.9184e15)):
            elements = result.orbital_elements
            h = elements.specific_angular_momentum
            self.assertAlmostEqual(h / h_expected, 1.0, places=4)
            energy = elements.specific_energy
            self.assertAlmostEqual(-self.mu / (2 * energy), elements.semimajor_axis, delta=1.0)
            p = h * h / self.mu
            self.assertAlmostEqual(p, elements.semilatus_rectum, delta=p * 1e-12)
            e = math.sqrt(1 + 2 * energy * h * h / self.mu**2)
            self.assertAlmostEqual(e, elements.eccentricity, places=9)
            self.assertAlmostEqual(p / (1 + e), elements.periapsis_radius, delta=1.0)
            self.assertAlmostEqual(p / (1 - e), elements.apoapsis_radius, delta=1.0)
            self.assertLess(np.max(np.abs(result.Hs / result.Hs[0] - 1.0)), 0.001)
        self.assertAlmostEqual(self.default.orbital_elements.specific_energy / 1e9, -1.1457, places=4)
        circular = physics.keplerian_elements(self.r0, 0.0, 0.0, math.sqrt(self.mu / self.r0), self.mu)
        self.assertIsNone(circular.periapsis_longitude_degrees)
        self.assertLess(circular.eccentricity, 1e-12)

    # Beat 3 ---------------------------------------------------------------
    def test_beat3_apoapsis_start_reproduces_the_same_ellipse(self) -> None:
        h = self.r0 * 58980.0
        self.assertAlmostEqual(h / 6.9832e10, 38851.5, delta=0.1)
        far = run_cli(("--xInit", "6.9832e10", "--vyInit", "38851.5")).result
        near, far_elements = self.default.orbital_elements, far.orbital_elements
        for name in ("eccentricity", "semimajor_axis", "orbital_period", "periapsis_radius"):
            self.assertAlmostEqual(getattr(far_elements, name) / getattr(near, name), 1.0, places=4)  # 1.0e-5 differences: the Help's apoapsis speed has five digits
        self.assertAlmostEqual(far_elements.periapsis_longitude_degrees, 180.0, places=1)
        self.assertAlmostEqual(far_elements.initial_true_anomaly_degrees, 180.0, places=1)
        self.assertAlmostEqual(58980.0 / 38851.5, 1.52, places=2)

    def test_beat3_kepler_second_and_third_laws(self) -> None:
        r = self.default
        areas = 0.5 * (r.xs[:-1] * r.ys[1:] - r.xs[1:] * r.ys[:-1])
        rate = areas / np.diff(r.ts)
        self.assertAlmostEqual(0.5 * r.Hs[0] / 1.3565e15, 1.0, places=4)
        self.assertLess(np.max(np.abs(rate / (0.5 * r.Hs[0]) - 1.0)), 5e-5)
        moon = run_cli(("--xInit", "3.626e8", "--vyInit", "1082", "--k", "3.986e14", "--dt0", "1000")).result
        for result, mu in ((self.default, physics.GM_SUN), (moon, 3.986e14)):
            e = result.orbital_elements
            self.assertAlmostEqual(e.orbital_period**2 * mu / e.semimajor_axis**3, 4 * math.pi**2, delta=1e-3 * 39.48)
            self.assertAlmostEqual(result.final_time / e.orbital_period, 1.0, delta=1e-4)
        self.assertAlmostEqual(moon.orbital_elements.orbital_period / 86400.0, 27.8, delta=0.05)
        for period, a, expected in ((7.6019e6, 5.7916e10, 2.9747e-19), (2.4034e6, 3.8780e8, 9.9044e-14)):
            self.assertAlmostEqual(period**2 / a**3 / expected, 1.0, places=4)
        self.assertAlmostEqual(4 * math.pi**2, 39.48, places=2)

    # Beat 4 ---------------------------------------------------------------
    def test_beat4_hodograph_circle_and_closure_growth(self) -> None:
        r = self.default
        h = r.Hs[0]
        vy_span = r.vys.max() - r.vys.min()
        self.assertAlmostEqual(r.vys.max(), 58980.0, delta=1.0)
        self.assertAlmostEqual(r.vys.min(), -38852.0, delta=2.0)
        self.assertAlmostEqual(vy_span / 2, self.mu / h, delta=5.0)
        self.assertAlmostEqual(self.mu / h, 48916.0, delta=1.0)
        centre = 0.5 * (r.vys.max() + r.vys.min())
        self.assertAlmostEqual(centre, r.orbital_elements.eccentricity * self.mu / h, delta=5.0)
        self.assertAlmostEqual(centre, 10064.0, delta=1.0)
        # Every velocity lies on that circle (Eq. 10).
        distance = np.hypot(r.vxs, r.vys - centre)
        self.assertLess(np.max(np.abs(distance / (self.mu / h) - 1.0)), 2e-4)
        one = run_cli(("--output", "velocity")).result
        two = run_cli(("--maxOrbits", "2")).result
        five = run_cli(("--maxOrbits", "5")).result
        self.assertAlmostEqual(two.closure_velocity_residual / one.closure_velocity_residual, 2.00, delta=0.01)
        self.assertAlmostEqual(five.closure_velocity_residual / one.closure_velocity_residual, 5.00, delta=0.01)
        self.assertAlmostEqual(two.closure_radius_residual / one.closure_radius_residual, 3.99, delta=0.02)
        self.assertAlmostEqual(five.closure_radius_residual / one.closure_radius_residual, 24.9, delta=0.2)
        half = run_cli(("--maxOrbits", "1.5")).result
        self.assertIsNone(half.closure_radius_residual)
        self.assertIsNone(half.closure_velocity_residual)
        self.assertEqual(one.event_refinement_trials, 7)

    # Beat 5 ---------------------------------------------------------------
    def test_beat5_eps2_changes_cost_and_closure_but_not_step_count_or_drift(self) -> None:
        strict = run_cli(("--eps2", "1e-8"))
        loose = run_cli(("--eps2", "10"))
        default = run_cli(())
        self.assertEqual({strict.result.accepted_steps, loose.result.accepted_steps, default.result.accepted_steps}, {761})
        self.assertEqual(
            (loose.acceleration_calls, default.acceleration_calls, strict.acceleration_calls),
            (1534, 2301, 3062),
        )
        self.assertAlmostEqual(strict.result.max_fractional_energy_drift, default.result.max_fractional_energy_drift, delta=1e-9)
        self.assertAlmostEqual(loose.result.max_fractional_energy_drift / default.result.max_fractional_energy_drift, 0.985, delta=0.002)
        self.assertAlmostEqual(loose.result.closure_radius_residual / default.result.closure_radius_residual, 91.0, delta=1.0)
        self.assertAlmostEqual(default.acceleration_calls / loose.acceleration_calls, 1.5, delta=0.02)
        self.assertAlmostEqual(strict.acceleration_calls / default.acceleration_calls, 1.3, delta=0.04)
        per_step = [run.acceleration_calls / run.result.accepted_steps for run in (loose, default, strict)]
        for measured, expected in zip(per_step, (2, 3, 4)):
            self.assertAlmostEqual(measured, expected, delta=0.05)
        for run in (strict, loose, default):
            steps = np.diff(run.result.ts)
            self.assertEqual(int(np.sum(steps < 1.0e4 * (1 - 1e-9))), 1)

    # Beat 6 ---------------------------------------------------------------
    def test_beat6_default_run_never_rejects_a_step(self) -> None:
        steps = np.diff(self.default.ts)
        self.assertEqual(int(np.sum(steps == 1.0e4)), 760)
        self.assertAlmostEqual(steps[-1], 2.5e3, delta=100.0)
        self.assertAlmostEqual(float(predicted_acceleration_changes(self.default).max()), 0.0128, delta=0.00005)

    def test_beat6_step_control_statistics(self) -> None:
        controlled = run_cli(("--dt0", "1e5")).result
        free = run_cli(("--dt0", "1e5", "--eps1", "10")).result
        wild = run_cli(("--dt0", "1e6", "--eps1", "10")).result
        also = run_cli(("--dt0", "1e6")).result
        self.assertEqual((controlled.accepted_steps, free.accepted_steps, wild.accepted_steps, also.accepted_steps), (181, 77, 14, 182))
        self.assertAlmostEqual(np.diff(controlled.ts).max(), 88595.0, delta=1.0)
        self.assertLess(np.diff(controlled.ts).max(), 1.0e5)
        self.assertAlmostEqual(float(predicted_acceleration_changes(controlled).max()), 0.0497, delta=0.00005)
        self.assertAlmostEqual(float(predicted_acceleration_changes(free).max()), 0.128, delta=0.0005)
        self.assertAlmostEqual(float(predicted_acceleration_changes(wild).max()), 0.65, delta=0.005)
        self.assertLess(float(predicted_acceleration_changes(controlled).max()), 0.05)
        default = self.default
        self.assertAlmostEqual(controlled.max_fractional_energy_drift / default.max_fractional_energy_drift, 15.6, delta=0.1)
        self.assertAlmostEqual(free.max_fractional_energy_drift / controlled.max_fractional_energy_drift, 6.4, delta=0.1)
        period = default.orbital_elements.orbital_period
        self.assertAlmostEqual((controlled.final_time - period) / period, 0.0010, delta=0.00005)
        self.assertAlmostEqual((free.final_time - period) / period, 0.0076, delta=0.0001)
        self.assertAlmostEqual(76.1, default.accepted_steps / 10, places=1)
        self.assertAlmostEqual(wild.closure_velocity_residual, 0.25, delta=0.005)
        self.assertEqual(wild.angular_step_rejections, 1)
        for other in (controlled, free, also, default):
            self.assertEqual(other.angular_step_rejections, 0)

    # Beat 7 ---------------------------------------------------------------
    def test_beat7_second_order_convergence(self) -> None:
        runs = [self.default] + [run_cli(("--dt0", value)).result for value in ("5000", "2500", "1250")]
        drifts = [r.max_fractional_energy_drift for r in runs]
        radius = [r.closure_radius_residual for r in runs]
        velocity = [r.closure_velocity_residual for r in runs]
        for earlier, later in zip(drifts, drifts[1:]):
            self.assertAlmostEqual(earlier / later, 4.00, delta=0.01)
        for earlier, later in zip(velocity, velocity[1:]):
            self.assertAlmostEqual(earlier / later, 4.00, delta=0.01)
        for earlier, later in zip(radius, radius[1:]):
            self.assertAlmostEqual(earlier / later, 16.0, delta=0.1)
        self.assertAlmostEqual(drifts[0] / 64.0, 8.3504e-7, delta=2e-9)
        self.assertEqual([r.accepted_steps for r in runs], [761, 1521, 3041, 6082])
        period = self.default.orbital_elements.orbital_period
        excess = [r.final_time - period for r in runs]
        self.assertAlmostEqual(excess[0], 570.0, delta=5.0)
        self.assertLess(excess[2], 50.0)
        self.assertEqual(f"{period:.4e}", "7.6019e+06")

    # Beat 8 ---------------------------------------------------------------
    def test_beat8_eccentric_orbit_is_limited_by_eps1_not_dt0(self) -> None:
        base = run_cli(("--vyInit", "10000")).result
        short = run_cli(("--vyInit", "10000", "--dt0", "1000")).result
        tight = run_cli(("--vyInit", "10000", "--eps1", "0.005")).result
        self.assertAlmostEqual(base.orbital_elements.eccentricity, 0.96534, places=5)
        self.assertAlmostEqual(base.orbital_elements.periapsis_radius / 4.6e10, 0.018, delta=0.0005)
        self.assertEqual((base.accepted_steps, short.accepted_steps, tight.accepted_steps), (579, 2176, 5044))
        steps = np.diff(base.ts)
        self.assertEqual(int(np.sum(steps < 1.0e4 * (1 - 1e-9))), 431)
        self.assertAlmostEqual(steps.min(), 36.8, delta=0.05)
        self.assertAlmostEqual(np.diff(tight.ts).min(), 3.6, delta=0.05)
        self.assertAlmostEqual(float(predicted_acceleration_changes(base).max()), 0.0499, delta=0.00005)
        self.assertAlmostEqual(base.max_fractional_energy_drift / self.default.max_fractional_energy_drift, 209.0, delta=1.0)
        self.assertAlmostEqual(base.max_fractional_energy_drift / tight.max_fractional_energy_drift, 102.0, delta=1.0)
        self.assertAlmostEqual(tight.accepted_steps / base.accepted_steps, 8.7, delta=0.05)
        self.assertGreater(short.max_fractional_energy_drift, 0.9 * base.max_fractional_energy_drift)
        period = base.orbital_elements.orbital_period
        self.assertAlmostEqual((period - base.final_time) / period, 0.0027, delta=0.00005)

    def test_beat8_rotated_orbit_also_responds_to_eps1_and_not_dt0(self) -> None:
        arguments = ("--vxInit", "41705.15", "--vyInit", "41705.15")
        base = run_cli(arguments).result
        tight = run_cli(arguments + ("--eps1", "0.005")).result
        shorter = run_cli(arguments + ("--dt0", "5000")).result
        self.assertEqual(tight.accepted_steps, 2853)
        self.assertAlmostEqual(base.max_fractional_energy_drift / tight.max_fractional_energy_drift, 100.0, delta=1.0)
        self.assertGreater(shorter.max_fractional_energy_drift, 0.5 * base.max_fractional_energy_drift)

    def test_beat8_radial_infall_and_singularity_guard(self) -> None:
        radial = run_cli(("--vxInit", "-1000", "--vyInit", "0", "--maxSteps", "20000")).result
        self.assertIs(radial.termination_reason, driver.TerminationReason.CENTRAL_SINGULARITY)
        self.assertEqual(radial.accepted_steps, 1171)
        elements = radial.orbital_elements
        self.assertAlmostEqual(elements.eccentricity, 1.0, places=9)
        self.assertEqual(elements.periapsis_radius, 0.0)
        self.assertIsNone(radial.max_fractional_angular_momentum_drift)
        guard = max(1e-12 * self.r0, 32 * math.ulp(self.r0))
        self.assertAlmostEqual(1024 * guard, 47.0, delta=0.2)
        self.assertLessEqual(math.hypot(radial.xs[-1], radial.ys[-1]), 1024 * guard)
        self.assertAlmostEqual(np.diff(radial.ts).min(), 2e-10, delta=1e-10)
        self.assertAlmostEqual(radial.final_time / 86400.0, 10.83, delta=0.01)
        self.assertGreater(radial.max_fractional_energy_drift, 1.8e5)
        slow = run_cli(("--vxInit", "-1000", "--vyInit", "0", "--maxSteps", "20000", "--eps1", "0.001")).result
        self.assertIs(slow.termination_reason, driver.TerminationReason.MAX_STEPS)
        self.assertLess(slow.max_fractional_energy_drift, 1e-4)
        self.assertGreater(math.hypot(slow.xs[-1], slow.ys[-1]), 1e6)

    # Experiments ----------------------------------------------------------
    def test_experiment_check_numbers(self) -> None:
        earth = run_cli(("--xInit", "1.471e11", "--vyInit", "30290")).result
        self.assertEqual(f"{earth.orbital_elements.eccentricity:.5f}", "0.01695")
        self.assertAlmostEqual(earth.orbital_elements.orbital_period / 86400.0, 365.4, delta=0.05)
        hyperbolic = run_cli(("--vyInit", "85000", "--maxSteps", "600")).result
        h = 4.6e10 * 85000.0
        self.assertAlmostEqual(self.mu / h, 33942.0, delta=1.0)
        self.assertAlmostEqual(hyperbolic.orbital_elements.eccentricity * self.mu / h, 51059.0, delta=1.0)
        self.assertGreater(hyperbolic.orbital_elements.eccentricity * self.mu / h, self.mu / h)
        same = run_cli(("--dt0", "5000", "--eps1", "0.01", "--eps2", "1e-6")).result
        plain = run_cli(("--dt0", "5000")).result
        self.assertEqual(same.accepted_steps, plain.accepted_steps)
        self.assertEqual(same.max_fractional_energy_drift, plain.max_fractional_energy_drift)
        self.assertEqual(same.accepted_steps, 1521)
        self.assertAlmostEqual(self.default.orbital_elements.specific_energy / 1e9, -1.1457, places=4)

    def test_experiment11_snippet_runs_and_agrees_with_the_quoted_rate(self) -> None:
        section = section_html(HELP_HTML, "experiments")
        block = [b for b in re.findall(r"<pre><code>(.*?)</code></pre>", section, re.DOTALL) if "areas =" in b]
        self.assertEqual(len(block), 1)
        source = html_module.unescape(block[0])
        namespace = {"result": self.default}
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec(compile(source, "<experiment 11>", "exec"), namespace)  # noqa: S102
        low, high, expected = (float(value) for value in captured.getvalue().split())
        self.assertAlmostEqual(expected / 1.3565e15, 1.0, places=4)
        self.assertLess(max(abs(low / expected - 1.0), abs(high / expected - 1.0)), 5e-5)


class RunDiagnosticTests(unittest.TestCase):
    """The summary lines that let a student see the cost and the step sizes of a run."""

    def test_acceleration_evaluations_equal_the_calls_the_driver_makes(self) -> None:
        for arguments in ((), ("--eps2", "1e-8"), ("--dt0", "1e6", "--eps1", "10")):
            with self.subTest(arguments=arguments):
                run = run_cli(arguments)
                self.assertEqual(run.result.acceleration_evaluations, run.acceleration_calls)
                self.assertIn(
                    f"acceleration evaluations : {run.acceleration_calls}\n", run.stdout
                )

    def test_step_extremes_are_those_of_the_accepted_time_steps(self) -> None:
        run = run_cli(("--dt0", "1e5"))
        steps = np.diff(run.result.ts)
        self.assertEqual(run.result.shortest_accepted_step, steps.min())
        self.assertEqual(run.result.longest_accepted_step, steps.max())
        self.assertIn(f"longest accepted step    : {orbit_main._five_significant(steps.max())} s", run.stdout)
        self.assertIn(f"shortest accepted step   : {orbit_main._five_significant(steps.min())} s", run.stdout)
        self.assertLessEqual(run.result.longest_accepted_step, 1.0e5)

    def test_step_extremes_are_not_available_without_an_accepted_step(self) -> None:
        import dataclasses

        empty = dataclasses.replace(circular_result(), ts=np.array([0.0]), accepted_steps=0)
        self.assertIsNone(empty.shortest_accepted_step)
        self.assertIsNone(empty.longest_accepted_step)
        lines = "\n".join(orbit_main._summary_lines(empty))
        self.assertIn("shortest accepted step   : n/a (no step accepted)", lines)
        self.assertIn("longest accepted step    : n/a (no step accepted)", lines)
        self.assertIn("n/a (no step accepted)", html_text(section_html(HELP_HTML, "summary")))

    def test_specific_energy_of_the_starting_state_is_printed(self) -> None:
        expectations = {
            (): "-1.1457e+09",
            ("--vyInit", "70000"): "-4.3505e+08",
            ("--vyInit", "75961", "--maxSteps", "600"): "-16283",
            ("--vyInit", "85000", "--maxSteps", "600"): "7.2745e+08",
        }
        for arguments, printed in expectations.items():
            with self.subTest(arguments=arguments):
                run = run_cli(arguments)
                self.assertIn(f"    specific energy            : {printed} J/kg\n", run.stdout)
                self.assertEqual(
                    orbit_main._five_significant(run.result.orbital_elements.specific_energy),
                    printed,
                )

    def test_final_specific_energy_is_printed_and_matches_the_last_state(self) -> None:
        for arguments, printed in (
            ((), "-1.1457e+09"),
            (("--vyInit", "75961", "--maxSteps", "600"), "82138"),
            (("--vyInit", "85000", "--maxSteps", "600"), "7.2756e+08"),
        ):
            with self.subTest(arguments=arguments):
                run = run_cli(arguments)
                result = run.result
                energy = (
                    0.5 * (result.vxs[-1] ** 2 + result.vys[-1] ** 2)
                    - physics.GM_SUN / math.hypot(result.xs[-1], result.ys[-1])
                )
                # energy is a difference of two large terms, so allow a few parts in 1e12 of the larger one
                self.assertAlmostEqual(result.final_specific_energy, energy, delta=1e-6)
                self.assertIn(f"  final specific energy    : {printed} J/kg\n", run.stdout)
                # The change from the start can never exceed the reported maximum drift.
                change = abs(result.final_specific_energy - result.orbital_elements.specific_energy)
                self.assertLessEqual(change, result.max_absolute_specific_energy_drift * (1 + 1e-12))

    def test_the_hodograph_marks_the_origin_and_the_initial_velocity(self) -> None:
        result = circular_result(maxOrbits=0.1)
        with mock.patch.object(plotting.plt, "show"):
            plotting.plot_orbit(result, output="velocity")
            axis = plotting.plt.gcf().axes[0]
            offsets = [tuple(map(float, collection.get_offsets()[0])) for collection in axis.collections]
            plotting.plt.close("all")
        self.assertIn((0.0, 0.0), offsets)
        self.assertIn((float(result.vxs[0]), float(result.vys[0])), offsets)


class SmallRevolutionTargetTests(unittest.TestCase):
    """A tiny --maxOrbits must be met, or rejected; it may not silently overshoot."""

    def test_targets_at_and_above_the_minimum_are_met_to_a_small_fraction(self) -> None:
        for target in (1.0e-9, 3.0e-9, 1.0e-8, 1.0e-6, 1.0e-3, 0.01, 0.1, 0.5, 1.5):
            with self.subTest(target=target):
                result = run_result(maxOrbits=target)
                self.assertIs(result.termination_reason, driver.TerminationReason.MAX_ORBITS)
                self.assertAlmostEqual(result.revolutions_completed / target, 1.0, delta=1e-5)
                self.assertIsNone(result.closure_radius_residual)
                self.assertIsNone(result.closure_velocity_residual)

    def test_minimum_target_is_met_from_any_starting_direction(self) -> None:
        for angle in (0.3, 2.0, 3.1, -2.5):
            with self.subTest(angle=angle):
                c, s_ = math.cos(angle), math.sin(angle)
                result = run_result(
                    xInit=4.6e10 * c, yInit=4.6e10 * s_,
                    vxInit=-58980.0 * s_, vyInit=58980.0 * c,
                    maxOrbits=driver.MIN_MAX_ORBITS,
                )
                self.assertAlmostEqual(result.revolutions_completed / driver.MIN_MAX_ORBITS, 1.0, delta=1e-5)

    def test_targets_below_the_minimum_are_rejected_with_a_clear_message(self) -> None:
        for target in (1.0e-10, 1.0e-13, 1.0e-16, 1.0e-300):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "maxOrbits must be at least 1e-09"):
                    run_result(maxOrbits=target)
        with (
            mock.patch.object(sys, "argv", ["main.py", "--maxOrbits", "1e-13"]),
            self.assertRaisesRegex(SystemExit, "Orbit: maxOrbits must be at least 1e-09"),
        ):
            orbit_main.main()

    def test_closure_needs_a_positive_whole_number_of_revolutions(self) -> None:
        for target, expected in ((1.0, True), (2.0, True), (1.5, False), (0.5, False), (1.0e-9, False)):
            with self.subTest(target=target):
                result = run_result(maxOrbits=target)
                self.assertEqual(result.closure_radius_residual is not None, expected)
                self.assertEqual(result.closure_velocity_residual is not None, expected)


class AuditFixClaimTests(unittest.TestCase):
    """Help statements that were corrected after review, checked against the program."""

    mu = physics.GM_SUN

    def test_hodograph_radii_use_the_magnitude_of_h_for_either_orientation(self) -> None:
        for arguments, sign in ((("--output", "velocity"), 1.0), (("--vyInit", "-58980", "--output", "velocity"), -1.0)):
            with self.subTest(arguments=arguments):
                result = run_cli(arguments).result
                elements = result.orbital_elements
                h = result.Hs[0]
                self.assertEqual(math.copysign(1.0, h), sign)
                radius = self.mu / abs(h)
                offset = elements.eccentricity * self.mu / abs(h)
                self.assertAlmostEqual(radius, 48916.0, delta=1.0)
                self.assertAlmostEqual(offset, 10064.0, delta=1.0)
                # The centre sits on the +vy side for h > 0 and on the -vy side for h < 0.
                centre = (result.vys.max() + result.vys.min()) / 2.0
                self.assertAlmostEqual(centre, sign * offset, delta=5.0)
                self.assertAlmostEqual((result.vys.max() - result.vys.min()) / 2.0, radius, delta=5.0)
                distance = np.hypot(result.vxs, result.vys - centre)
                self.assertLess(np.max(np.abs(distance / radius - 1.0)), 2e-4)
                # The origin lies inside the circle for a bound orbit.
                self.assertLess(abs(centre), radius)

    def test_clockwise_run_has_the_same_closure_residuals_as_counter_clockwise(self) -> None:
        ccw, cw = run_cli(()).result, run_cli(("--vyInit", "-58980", "--output", "velocity")).result
        self.assertEqual(cw.accepted_steps, ccw.accepted_steps)
        self.assertAlmostEqual(cw.closure_radius_residual, ccw.closure_radius_residual, delta=1e-12)
        self.assertAlmostEqual(cw.closure_velocity_residual, ccw.closure_velocity_residual, delta=1e-9)
        self.assertAlmostEqual(cw.vys.max(), 38850.0, delta=5.0)
        self.assertEqual(cw.vys.min(), -58980.0)

    def test_near_escape_run_loses_the_sign_of_its_energy(self) -> None:
        run = run_cli(("--vyInit", "75961", "--maxSteps", "600"))
        result = run.result
        initial = result.orbital_elements.specific_energy
        last = 0.5 * (result.vxs[-1] ** 2 + result.vys[-1] ** 2) - self.mu / math.hypot(result.xs[-1], result.ys[-1])
        self.assertLess(initial, 0.0)
        self.assertAlmostEqual(initial, -16283.0, delta=0.5)
        self.assertGreater(last, 0.0)
        self.assertAlmostEqual(last, 82138.0, delta=1.0)
        self.assertGreater(result.max_absolute_specific_energy_drift, 5.0 * abs(initial))
        self.assertAlmostEqual(result.revolutions_completed, 0.35591, places=5)
        self.assertIs(result.termination_reason, driver.TerminationReason.MAX_STEPS)
        self.assertIn("0.35591", run.stdout)

    def test_tight_control_makes_the_near_escape_drift_small_compared_with_its_binding_energy(self) -> None:
        result = run_cli(("--vyInit", "75961", "--dt0", "100", "--eps1", "0.001")).result
        self.assertAlmostEqual(result.max_absolute_specific_energy_drift, 9.8799, places=3)
        self.assertLess(result.max_absolute_specific_energy_drift, 1e-3 * 16283.0)
        self.assertAlmostEqual(result.max_fractional_energy_drift, 0.00060676, places=8)
        self.assertAlmostEqual(result.revolutions_completed, 0.27184, places=5)
        self.assertEqual(result.accepted_steps, 20000)

    def test_apoapsis_formula_is_negative_for_an_unbound_orbit_and_the_program_says_n_a(self) -> None:
        run = run_cli(("--vyInit", "85000", "--maxSteps", "600"))
        elements = run.result.orbital_elements
        formula = elements.semilatus_rectum / (1.0 - elements.eccentricity)
        self.assertGreater(elements.eccentricity, 1.0)
        self.assertAlmostEqual(formula, -2.2844e11, delta=1e7)
        self.assertIsNone(elements.apoapsis_radius)
        self.assertIsNone(elements.orbital_period)
        self.assertIn("apoapsis radius            : n/a (unbound)", run.stdout)
        self.assertIn("Keplerian period           : n/a (unbound)", run.stdout)
        text = html_text(section_html(HELP_HTML, "beat2"))
        self.assertIn("r_a", section_html(HELP_HTML, "beat2"))
        self.assertIn("(e<1\\ \\text{only})", section_html(HELP_HTML, "beat2"))
        self.assertIn("An unbound orbit has no apoapsis", text)

    def test_eps2_can_change_the_accepted_steps_when_the_corrector_hits_its_pass_limit(self) -> None:
        default = run_cli(("--dt0", "1e6", "--eps1", "10")).result
        strict = run_cli(("--dt0", "1e6", "--eps1", "10", "--eps2", "1e-12")).result
        self.assertEqual((default.accepted_steps, strict.accepted_steps), (14, 24))
        with mock.patch.object(driver, "MAX_CORRECTOR_ITERATIONS", 1000):
            unlimited = run_result(dt0=1.0e6, eps1=10.0, eps2=1.0e-12)
        self.assertEqual(unlimited.accepted_steps, 14)

    def test_documented_commands_reach_the_corrector_pass_limit_only_where_the_help_says(self) -> None:
        """Beats 5 to 8: the ten-pass limit acts in exactly the two commands the Help names."""
        commands: list[str] = []
        for number in (5, 6, 7, 8):
            commands += [c for c in documented_commands(section_html(HELP_HTML, f"beat{number}")) if not is_template(c)]
        reached = []
        for command in dict.fromkeys(commands):
            arguments = command_arguments(command)
            baseline = run_cli(arguments).result
            with mock.patch.object(driver, "MAX_CORRECTOR_ITERATIONS", 1000):
                relaxed = run_cli_uncached(arguments)
            if (baseline.accepted_steps, baseline.acceleration_evaluations) != (
                relaxed.accepted_steps, relaxed.acceleration_evaluations
            ):
                reached.append(" ".join(arguments))
        self.assertEqual(
            sorted(reached),
            sorted(["--dt0 1e6 --eps1 10", "--dt0 1e6 --eps1 10 --eps2 1e-12"]),
        )
        beat6 = html_text(section_html(HELP_HTML, "beat6"))
        self.assertIn("ten passes", beat6)


def run_cli_uncached(arguments: tuple[str, ...]) -> driver.OrbitResult:
    """Run the command line once, bypassing (and not filling) the cache."""
    saved = dict(_CLI_CACHE)
    _CLI_CACHE.clear()
    try:
        return run_cli(arguments).result
    finally:
        _CLI_CACHE.clear()
        _CLI_CACHE.update(saved)


@unittest.skipUnless(
    ORIGINAL_HELP_PATH.is_file(),
    "Orbit-original.html (the Reference Guide Help) is not present; nothing else depends on it",
)
class ReferenceGuideHelpTests(unittest.TestCase):
    """While the Reference Guide Help is kept, it must still describe this program."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.html = ORIGINAL_HELP_PATH.read_text(encoding="utf-8")

    def test_version_build_stamp_matches_the_program(self) -> None:
        parser = IdTextParser()
        parser.feed(self.html)
        stamp = " ".join(" ".join(parser.text_by_id["version_build"]).split())
        self.assertEqual(stamp, f"Version {physics.MODEL_VERSION} Build {physics.BUILD_ID}")

    def test_every_option_and_output_choice_is_documented(self) -> None:
        text = html_text(self.html)
        parser = orbit_main.build_parser()
        for action in parser._actions:
            for option in action.option_strings:
                if option.startswith("--") and option not in ("--help", "--version"):
                    with self.subTest(option=option):
                        self.assertIn(option, text)
        for choice in orbit_main.OUTPUT_CHOICES:
            self.assertIn(f'"{choice}"', text)


# ``ReleaseNotes`` and ``SampleOutputs_Guide`` are revised only after the audit
# rounds are complete.  Until then each legitimately still describes the release
# named here.  Set this to ``None`` when they are updated; the synchronization
# checks then apply in full.  While it is set, the checks require the documents
# to still be at exactly that version, so the freeze cannot outlive the change
# it stands for.
COMPANION_DOCS_FROZEN_AT: str | None = None

RELEASE_NOTES_ACTIVE = re.compile(
    r"<b>Version:</b>\s*(?P<version>\d+\.\d+\.\d+)\s*<br>\s*<b>Build:</b>\s*(?P<build>[0-9a-f]{12})"
)
SAMPLE_OUTPUTS_ACTIVE = re.compile(
    r"<p>Version\s+(?P<version>\d+\.\d+\.\d+)(?:\s|&nbsp;)*\|(?:\s|&nbsp;)*"
    r"Build\s+(?P<build>[0-9a-f]{12})"
)


def active_version_and_build(pattern: re.Pattern[str], text: str) -> tuple[str, str]:
    """The FIRST match is the document's header, which names the active release.

    Later occurrences (release history, embedded examples) are deliberately ignored.
    """
    match = pattern.search(text)
    if match is None:
        raise AssertionError("could not find the active version/build header")
    return match["version"], match["build"]


def companion_status(
    active: tuple[str, str],
    current: tuple[str, str],
    frozen_at: str | None,
) -> str:
    """Return ``"current"``, ``"pending"`` or a message describing the failure."""
    if frozen_at is not None:
        if frozen_at == current[0]:
            return (
                f"COMPANION_DOCS_FROZEN_AT equals the current version {current[0]}; "
                "set it to None"
            )
        if active[0] != frozen_at:
            return (
                f"document is at {active[0]}, no longer the frozen {frozen_at}; "
                "set COMPANION_DOCS_FROZEN_AT = None and synchronise it"
            )
        return "pending"
    if active != current:
        return (
            f"document header says version {active[0]} build {active[1]}, "
            f"program is version {current[0]} build {current[1]}"
        )
    return "current"


class CompanionStatusTests(unittest.TestCase):
    CURRENT = ("1.6.0", "aaaaaaaaaaaa")

    def test_release_history_below_the_header_does_not_hide_a_stale_header(self) -> None:
        stale = (
            "<b>Version:</b> 1.4.0<br>\n<b>Build:</b> bbbbbbbbbbbb<br>\n"
            "<p><b>Version:</b> 1.6.0<br><b>Build:</b> aaaaaaaaaaaa</p>"
        )
        self.assertEqual(
            active_version_and_build(RELEASE_NOTES_ACTIVE, stale), ("1.4.0", "bbbbbbbbbbbb")
        )
        self.assertEqual(
            active_version_and_build(
                SAMPLE_OUTPUTS_ACTIVE,
                "<p>Version 1.4.0 &nbsp;|&nbsp; Build bbbbbbbbbbbb</p> ... Version 1.6.0 Build aaaaaaaaaaaa",
            ),
            ("1.4.0", "bbbbbbbbbbbb"),
        )

    def test_frozen_documents_are_pending_only_while_still_at_the_frozen_version(self) -> None:
        old = ("1.4.0", "bbbbbbbbbbbb")
        self.assertEqual(companion_status(old, self.CURRENT, "1.4.0"), "pending")
        self.assertIn("no longer the frozen", companion_status(self.CURRENT, self.CURRENT, "1.4.0"))
        self.assertIn("no longer the frozen", companion_status(("1.5.0", "cccccccccccc"), self.CURRENT, "1.4.0"))

    def test_stale_freeze_constant_is_reported(self) -> None:
        self.assertIn("set it to None", companion_status(self.CURRENT, self.CURRENT, "1.6.0"))

    def test_without_a_freeze_the_header_must_match_version_and_build(self) -> None:
        self.assertEqual(companion_status(self.CURRENT, self.CURRENT, None), "current")
        self.assertIn("program is version", companion_status(("1.6.0", "bbbbbbbbbbbb"), self.CURRENT, None))
        self.assertIn("program is version", companion_status(("1.4.0", "bbbbbbbbbbbb"), self.CURRENT, None))


@unittest.skipUnless(
    RELEASE_NOTES_PATH.is_file() and SAMPLE_OUTPUTS_PATH.is_file(),
    "the Release Notes and Sample Outputs Guide live in the documentation repository",
)
class DocumentationSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.release_notes = RELEASE_NOTES_PATH.read_text(encoding="utf-8")
        cls.samples = SAMPLE_OUTPUTS_PATH.read_text(encoding="utf-8")

    def require_synchronised_or_pending(self, pattern: re.Pattern[str], text: str) -> None:
        status = companion_status(
            active_version_and_build(pattern, text),
            (physics.MODEL_VERSION, physics.BUILD_ID),
            COMPANION_DOCS_FROZEN_AT,
        )
        if status == "pending":
            self.skipTest(
                f"companion document is intentionally still at {COMPANION_DOCS_FROZEN_AT}; "
                "it is updated after the audit rounds"
            )
        self.assertEqual(status, "current")

    def test_documentation_files_exist(self) -> None:
        self.assertTrue(RELEASE_NOTES_PATH.is_file())
        self.assertTrue(SAMPLE_OUTPUTS_PATH.is_file())

    # The two documents below follow Instructions_for_Release_Notes_and_Sample_Outputs_Guide.
    # These checks replace earlier wording checks ("five significant digits",
    # "Keplerian Elements", console-summary strings): the new specification asks for
    # a different structure and for no printed console tables, so the structure is
    # what is checked, together with the numbers the guide quotes.
    RELEASE_NOTES_HEADINGS = [
        "Release Status", "Open Bugs", "Major Improvements in This Release",
        "Test Suite Growth", "Known Limitations", "Known Minor Maintenance Items",
        "Version Identification",
    ]

    def test_release_notes_match_current_version_and_build(self) -> None:
        self.require_synchronised_or_pending(RELEASE_NOTES_ACTIVE, self.release_notes)
        self.assertIn(f"<title>Orbit {physics.MODEL_VERSION} Release Notes</title>", self.release_notes)
        self.assertRegex(self.release_notes, r"<b>Date:</b> \d{4}-\d{2}-\d{2}")

    def test_release_notes_have_the_required_sections_in_order_and_only_plain_html(self) -> None:
        headings = re.findall(r"<h2>(.*?)</h2>", self.release_notes)
        self.assertEqual(headings, self.RELEASE_NOTES_HEADINGS)
        for forbidden in ("<table", "<script", "<style", "style=", "<img"):
            self.assertNotIn(forbidden, self.release_notes)
        tags = set(re.findall(r"</?([a-zA-Z0-9]+)", self.release_notes))
        allowed = {"html", "head", "meta", "title", "body", "h1", "h2", "p", "ul", "ol", "li",
                   "b", "i", "code", "sub", "sup", "br"}
        self.assertLessEqual(tags, allowed)

    def test_open_bugs_are_identified_and_state_a_reproduction(self) -> None:
        section = re.search(r"<h2>Open Bugs</h2>(.*?)<h2>", self.release_notes, re.S).group(1)
        entries = re.findall(r"<li>(.*?)</li>", section, re.S)
        self.assertGreaterEqual(len(entries), 1)
        for entry in entries:
            with self.subTest(entry=entry[:40]):
                self.assertRegex(entry, r"<b>OB-\d+</b> \(P[123]; ")
                self.assertIn("Reproduction:", entry)
                self.assertIn("Observed:", entry)
                self.assertIn("Expected", entry)

    def test_open_bug_about_the_maxorbits_minimum_is_removed_once_the_program_states_it(self) -> None:
        help_text = run_cli(("--help",)).stdout
        stated_in_help = "1e-9" in help_text or "1e-09" in help_text
        listed_open = "<b>OB-1</b>" in self.release_notes and "lower limit of <code>--maxOrbits</code>" in self.release_notes
        self.assertNotEqual(stated_in_help, listed_open, "OB-1 and the program's --help disagree")

    def test_release_notes_test_growth_ends_at_the_current_suite_size(self) -> None:
        counts = [int(n) for n in re.findall(r"(?:\d\.\d\.\d: |, )(\d+)(?: tests)?", re.search(r"<li>1\.4\.0:.*?</li>", self.release_notes, re.S).group(0))]
        ran = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent)).countTestCases()
        self.assertEqual(counts[-1], ran)

    def test_sample_outputs_match_current_version_and_build(self) -> None:
        self.require_synchronised_or_pending(SAMPLE_OUTPUTS_ACTIVE, self.samples)
        self.assertRegex(self.samples, r"Build [0-9a-f]{12} &nbsp;\|&nbsp; Updated \d{1,2} \w+ \d{4}")
        provenance = re.search(r'<section id="provenance">(.*?)</section>', self.samples, re.S).group(1)
        self.assertIn(f"Version {physics.MODEL_VERSION}", provenance)
        self.assertIn(f"build {physics.BUILD_ID}", provenance)

    def sample_sections(self) -> dict[int, str]:
        return {
            int(number): body
            for number, body in re.findall(r'<section id="beat(\d)">(.*?)</section>', self.samples, re.S)
        }

    def test_sample_outputs_have_one_figure_section_per_beat_in_order(self) -> None:
        order = [int(n) for n in re.findall(r'<section id="beat(\d)">', self.samples)]
        self.assertEqual(order, list(BEAT_NUMBERS))
        self.assertNotIn("<table", self.samples)
        contents = re.findall(r'<li><a href="#(beat\d)">', self.samples)
        self.assertEqual(contents, [f"beat{n}" for n in BEAT_NUMBERS])
        for number, body in self.sample_sections().items():
            with self.subTest(beat=number):
                self.assertEqual(len(re.findall(r"<pre><code>", body)), 1)
                self.assertEqual(len(re.findall(r'<img src="data:image/png;base64,', body)), 1)
                self.assertEqual(len(re.findall(r'<p class="caption">', body)), 1)
                shows = re.search(r"<h3>What it shows</h3>\s*<ul>(.*?)</ul>", body, re.S).group(1)
                heads = re.search(r"<h3>Headline values</h3>\s*<ul>(.*?)</ul>", body, re.S).group(1)
                self.assertIn(shows.count("<li>"), (3, 4))
                self.assertIn(heads.count("<li>"), (3, 4, 5))

    def test_each_figure_command_is_a_command_of_its_beat_and_runs(self) -> None:
        def without_output(arguments: tuple[str, ...]) -> tuple[str, ...]:
            kept, skip = [], False
            for argument in arguments:
                if skip:
                    skip = False
                elif argument == "--output":
                    skip = True
                else:
                    kept.append(argument)
            return tuple(kept)

        for number, body in self.sample_sections().items():
            with self.subTest(beat=number):
                command = html_module.unescape(re.search(r"<pre><code>(.*?)</code></pre>", body, re.S).group(1))
                arguments = command_arguments(command)
                beat_commands = {
                    without_output(command_arguments(c))
                    for c in documented_commands(section_html(HELP_HTML, f"beat{number}"))
                }
                self.assertIn(without_output(arguments), beat_commands)
                self.assertTrue(run_cli(arguments).stdout.startswith("Orbit "))
        commands = [
            command_arguments(html_module.unescape(re.search(r"<pre><code>(.*?)</code></pre>", body, re.S).group(1)))
            for body in self.sample_sections().values()
        ]
        self.assertEqual(len(set(commands)), len(commands), "two Beats share a command")

    def test_numbers_quoted_in_the_sample_outputs_are_printed_by_the_program(self) -> None:
        printed = "\n".join(
            run_cli(command_arguments(c)).stdout
            for c in dict.fromkeys(documented_commands(HELP_HTML))
            if not is_template(c) and "--help" not in c and "--version" not in c
        )
        for number, body in self.sample_sections().items():
            body = re.sub(r"base64,[A-Za-z0-9+/=]+", "", body)
            text = html_text(re.sub(
                r"(\d\.\d{4})\u00d710<sup>(\u2212?)(\d+)</sup>",
                lambda m: f"{m.group(1)}e{'-' if m.group(2) else '+'}{int(m.group(3)):02d}",
                body,
            ))
            tokens = set(HelpQuotedNumberTests.PRINTED.findall(text))
            with self.subTest(beat=number):
                self.assertTrue(tokens)
                self.assertEqual(sorted(t for t in tokens if t not in printed), [])

    def test_sample_outputs_use_the_current_summary_wording(self) -> None:
        text = html_text(self.samples)
        printed = run_cli(("--vyInit", "85000", "--maxSteps", "600")).stdout
        for label in ("final specific energy", "n/a (unbound)", "specific energy", "final radius"):
            with self.subTest(label=label):
                self.assertIn(label, text)
                self.assertIn(label, printed)
        quoted = re.search(r"suite of (\d+) tests", text)
        self.assertIsNotNone(quoted)
        ran = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent)).countTestCases()
        self.assertEqual(int(quoted.group(1)), ran)


if __name__ == "__main__":
    unittest.main(verbosity=2)
