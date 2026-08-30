"""Regression tests for the Orbit tutorial program.

The discovery code deliberately supports both repository layouts used by the
project: this file may live in ``tests/`` or may be flattened beside the four
program modules during an AI-review upload.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
from html.parser import HTMLParser
import importlib.util
import io
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
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


HELP_PATH = MODULE_DIR / "Orbit.html"


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
            (copied / "Orbit.html").write_text("changed help", encoding="utf-8")
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

    def test_specific_angular_momentum_sign_and_zero_radial_case(self) -> None:
        self.assertEqual(physics.specific_angular_momentum(2.0, 0.0, 0.0, 3.0), 6.0)
        self.assertEqual(physics.specific_angular_momentum(2.0, 0.0, 0.0, -3.0), -6.0)
        self.assertEqual(physics.specific_angular_momentum(2.0, 0.0, -3.0, 0.0), 0.0)

    def test_specific_angular_momentum_rejects_invalid_or_overflow(self) -> None:
        for arguments in ((1.0, 0.0, False, 1.0), (1.0e308, 0.0, 0.0, 1.0e308)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                physics.specific_angular_momentum(*arguments)


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
            output = io.StringIO()
            with self.subTest(exception=type(exception).__name__):
                with (
                    mock.patch.object(orbit_main, "run_orbit", side_effect=exception),
                    mock.patch.object(sys, "argv", ["main.py"]),
                    contextlib.redirect_stdout(output),
                ):
                    orbit_main.main()
                self.assertIn("Orbit could not run:", output.getvalue())
                self.assertNotIn("Traceback", output.getvalue())
                output.seek(0)
                output.truncate(0)


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
        self.assertAlmostEqual(abs(math.atan2(result.ys[-1], result.xs[-1])), math.pi, places=12)
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


class HelpFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = HELP_PATH.read_text(encoding="utf-8")
        cls.parser = IdTextParser()
        cls.parser.feed(cls.html)

    def test_help_file_exists_and_has_unique_version_build_element(self) -> None:
        self.assertTrue(HELP_PATH.is_file())
        self.assertEqual(self.html.count('id="version_build"'), 1)
        version_text = " ".join(self.parser.text_by_id["version_build"]).split()
        self.assertIn(f"Version {physics.MODEL_VERSION}", " ".join(version_text))
        self.assertIn(f"Build {physics.BUILD_ID}", " ".join(version_text))

    def test_help_matches_python_parameters_and_outputs(self) -> None:
        for text in (
            "xInit",
            "vyInit",
            "58980.0",
            "1.3271244e20",
            "maxOrbits",
            '"orbit"',
            '"velocity"',
            '"position_time"',
            '"velocity_time"',
            '"energy"',
        ):
            with self.subTest(text=text):
                self.assertIn(text, self.html)

    def test_help_states_gm_specific_energy_and_fixed_center_limitations(self) -> None:
        self.assertIn("gravitational parameter", self.html)
        self.assertIn("not the central mass in kg", self.html)
        self.assertIn("Specific energy", self.html)
        self.assertIn("fixed central", self.html)

    def test_help_documents_termination_and_diagnostic_edge_cases(self) -> None:
        normalized_html = " ".join(self.html.split())
        for text in (
            "central_singularity",
            "integral number of revolutions",
            "initial energy is zero or nearly zero",
            "absolute specific-energy drift",
            "without linearly interpolating state components",
        ):
            with self.subTest(text=text):
                self.assertIn(text, normalized_html)

    def test_mathjax_offline_explanation_is_static_and_no_local_install_is_promised(self) -> None:
        self.assertIn("loaded from a public CDN", self.html)
        self.assertIn("internet connection is needed", self.html)
        self.assertNotIn("navigator.onLine", self.html)
        self.assertNotIn("install MathJax locally", self.html)

    def test_development_history_is_confined_to_provenance(self) -> None:
        instructional = self.html.split('<section id="license">', 1)[0]
        for phrase in ("revised solver", "Python port", "original Java", "Triana workflow"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, instructional)
        provenance = self.html.split('<section id="license">', 1)[1]
        self.assertIn("original Java", provenance)
        self.assertIn("Python port", provenance)

    def test_exercises_are_present_ranked_and_include_error_measure_comparison(self) -> None:
        self.assertEqual(self.html.count('class="exp-card"'), 15)
        self.assertIn("EXP-1 · Introductory", self.html)
        self.assertIn("EXP-15 · Advanced Programming Extension", self.html)
        self.assertIn("Compare Error Measures", self.html)

    def test_help_preserves_original_textbook_cross_references(self) -> None:
        for reference in ("Table 4.3", "Table 4.2", "Investigation 4.1", "Investigation 4.2", "Chapter 6"):
            with self.subTest(reference=reference):
                self.assertIn(reference, self.html)

    def test_exp11_tells_students_how_to_access_returned_arrays(self) -> None:
        self.assertIn("result.xs", self.html)
        self.assertIn("result.ys", self.html)
        self.assertIn("result.ts", self.html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
