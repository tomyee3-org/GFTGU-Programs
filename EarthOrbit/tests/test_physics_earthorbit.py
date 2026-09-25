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
import inspect
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
import main as entrypoint
import physics_earthorbit as physics
import plot_earthorbit as plotter


def find_help_file(module_dir):
    """Find Help in a flattened upload or the GFTGU-Documentation tree.

    Documentation folders no longer use chapter-number prefixes, and the
    Help files live under the sibling ``GFTGU-Documentation`` repository
    rather than beside the program modules inside ``GFTGU-Programs``.
    """
    # The Beats Help is named EarthOrbit-claude.html until it is adopted as
    # the live Help, when it is renamed EarthOrbit.html; either name is
    # accepted, and the first one found is used.  The Reference Guide version,
    # EarthOrbit-original.html, is optional: the tests written for its text
    # read it when it is present and skip otherwise.
    help_filenames = ("EarthOrbit-claude.html", "EarthOrbit.html")
    program_name = "EarthOrbit"
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
        "Could not find EarthOrbit-claude.html (or EarthOrbit.html) beside the "
        "program or in GFTGU-Documentation/EarthOrbit/."
    )


HELP_FILE = find_help_file(MODULE_DIR)
DOCUMENTATION_DIR = HELP_FILE.parent
ORIGINAL_HELP_FILE = DOCUMENTATION_DIR / "EarthOrbit-original.html"
RELEASE_NOTES_FILE = DOCUMENTATION_DIR / "EarthOrbit-ReleaseNotes.html"
SAMPLE_OUTPUTS_FILE = (
    DOCUMENTATION_DIR / "SampleOutputs" / "EarthOrbit-SampleOutputs_Guide.html"
)


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
        self.assertEqual(physics.MODEL_VERSION, "1.4.0")

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

    def test_coarse_step_that_skips_through_earth_is_rejected(self):
        """A single overlarge step can jump from outside Earth on one side
        to outside Earth on the other, with the connecting chord passing
        straight through the reference sphere. Both stored endpoints would
        satisfy the old ``r >= R_EARTH`` loop check, so without an explicit
        segment/sphere test this silently misreported an escape or ordinary
        orbit instead of the impact that actually occurred."""
        with self.assertRaisesRegex(FloatingPointError, "passed through"):
            driver.run_earth_orbit(
                h0=300_000.0, uInit=0.0, vInit=-20_000.0, dt=1000.0,
                maxSteps=2, force_law="inverse_square",
            )

    def test_ordinary_steps_do_not_false_positive_on_earth_crossing(self):
        """The segment/sphere check must not fire for legitimate runs,
        including a close-perigee orbit and the program's own default."""
        # Default near-surface impact scenario.
        driver.run_earth_orbit(maxSteps=2000)
        # A 300 km circular-ish orbit with a realistic dt.
        driver.run_earth_orbit(
            h0=300_000.0, uInit=7725.72, vInit=0.0, dt=1.0,
            maxSteps=7000, force_law="inverse_square",
        )
        # A genuinely non-impacting close-perigee orbit at a realistic dt.
        # uInit=7636.511278627493 gives an osculating (launch-state) perigee
        # altitude of +200.0 m at h0=300 km; verified directly against
        # driver._inverse_square_elements before use here. This differs from
        # an earlier version of this test that used h0=200, uInit=7900,
        # dt=0.1 -- that combination's own osculating perigee is actually
        # about -16,564 m (an impacting trajectory), so it did not exercise
        # a near-miss at all. See test_close_perigee_elements_match_target
        # below for the perigee-altitude check, and Experiment 6 in
        # EarthOrbit-claude.html for why the analytic (launch-state) perigee
        # and the numerically integrated outcome are not the same question.
        result = driver.run_earth_orbit(
            h0=300_000.0, uInit=7636.511278627493, vInit=0.0, dt=1.0,
            maxSteps=7000, force_law="inverse_square", return_diagnostics=True,
        )
        xs, ys, _, _, ts, us, vs = result
        summary = driver.analyze_earth_orbit(
            xs, ys, ts, us, vs, force_law="inverse_square", max_steps=7000
        )
        self.assertFalse(summary["impact"])
        self.assertTrue(summary["reached_orbit"])

    def test_close_perigee_elements_match_target(self):
        """The uInit used above really does target a +200 m osculating
        perigee altitude, independently of the integrated trajectory."""
        elements = driver._inverse_square_elements(
            0.0, physics.R_EARTH + 300_000.0, 7636.511278627493, 0.0
        )
        self.assertAlmostEqual(elements["perigee_altitude"], 200.0, places=3)

    def test_segment_dips_inside_earth_true_for_chord_grazing_under_surface(self):
        """Direct geometric check: a vertical chord whose closest approach
        to the origin is R_EARTH - 200 m must be reported as dipping inside
        Earth, independent of any full integration run."""
        x = physics.R_EARTH - 200.0
        self.assertTrue(
            driver._segment_dips_inside_earth(x, -1000.0, x, 1000.0)
        )

    def test_segment_dips_inside_earth_false_for_chord_clearing_surface(self):
        """Direct geometric check: a vertical chord whose closest approach
        to the origin is R_EARTH + 200 m must NOT be reported as dipping
        inside Earth -- the counterpart to the case immediately above, and
        the case Codex Audit22 #4 asked to see tested directly."""
        x = physics.R_EARTH + 200.0
        self.assertFalse(
            driver._segment_dips_inside_earth(x, -1000.0, x, 1000.0)
        )

    def test_segment_dips_inside_earth_handles_extreme_finite_inputs(self):
        """Squaring raw, wildly non-physical coordinates (far beyond any
        launch condition this program accepts) must not overflow to
        inf/nan and silently miss a crossing. Codex Audit22 #5."""
        self.assertTrue(
            driver._segment_dips_inside_earth(0.0, 1.0e155, 0.0, -1.0e155)
        )
        # A displacement of the same extreme magnitude that does NOT pass
        # near the origin must still read False, not nan-propagate to a
        # wrong answer in either direction.
        far = 1.0e155
        self.assertFalse(
            driver._segment_dips_inside_earth(far, far, far, far + 1.0e150)
        )


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

    _full_period_recovery_cache = None

    @classmethod
    def _full_period_recovery_errors(cls):
        """Run one full inverse-square orbit at three resolutions.

        Shared by the convergence-order test and the absolute-baseline
        test below, and cached on the class, so the (relatively
        expensive) triple integration runs only once per test session
        rather than once per concern.
        """
        if cls._full_period_recovery_cache is not None:
            return cls._full_period_recovery_cache

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
        cls._full_period_recovery_cache = (errors, initial_radius, initial_speed)
        return cls._full_period_recovery_cache

    def test_full_period_orbit_recovery_converges_toward_initial_state(self):
        """Errors shrink monotonically, and at roughly first order, as dt refines.

        This is a convergence-*rate* contract: it is agnostic to the
        absolute size of any one error. The fixed absolute ceilings
        carried over from the Version 1.1.1 baseline are checked
        separately, in test_full_period_orbit_recovery_meets_absolute_baseline.
        """
        errors, _, _ = self._full_period_recovery_errors()

        for metric_index in range(4):
            metric_errors = [row[metric_index] for row in errors]
            self.assertGreater(metric_errors[0], metric_errors[1])
            self.assertGreater(metric_errors[1], metric_errors[2])

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

    def test_full_period_orbit_recovery_meets_absolute_baseline(self):
        """The finest (2880-update) run stays within the Version 1.1.1 error baseline.

        Measured Version 1.1.1 errors at 2880 updates were approximately:
        0.0636 rad angular error, 1.37% radial error, 6.54% position
        error, and 6.37% velocity error. These ceilings are that measured
        baseline plus headroom for ordinary run-to-run floating-point
        variation on one machine, not a survey of multiple platforms --
        the suite has not been run cross-platform to confirm the margin
        holds elsewhere.
        """
        errors, initial_radius, initial_speed = self._full_period_recovery_errors()
        fine_angle, fine_radius, fine_position, fine_velocity = errors[-1]
        self.assertLess(fine_angle, 0.068)
        self.assertLess(fine_radius / initial_radius, 0.015)
        self.assertLess(fine_position / initial_radius, 0.068)
        self.assertLess(fine_velocity / initial_speed, 0.068)


class TrajectorySummaryTests(unittest.TestCase):
    """Interpolated event, revolution, and orbital-element reporting."""

    @staticmethod
    def run_and_analyze(**kwargs):
        result = driver.run_earth_orbit(return_diagnostics=True, **kwargs)
        xs, ys, _, _, ts, us, vs = result
        summary = driver.analyze_earth_orbit(
            xs, ys, ts, us, vs,
            force_law=kwargs.get("force_law", "simplified"),
            max_steps=kwargs.get("maxSteps", 15000),
        )
        return result, summary

    def test_five_significant_digit_formatter(self):
        expected = {
            1019.708: "1019.7",
            254.929: "254.93",
            14.42085: "14.421",
            300.0: "300.00",
            0.0: "0.0000",
            1_583_500.0: "1.5835e+06",
        }
        for value, formatted in expected.items():
            with self.subTest(value=value):
                self.assertEqual(driver._five_significant(value), formatted)

    def test_five_significant_handles_rounding_that_crosses_a_decade(self):
        """Rounding to five significant digits can carry the value up a
        power of ten (99999.9 -> 100000, 0.999999 -> 1.00000); the
        formatter must re-derive its exponent so the result always has
        exactly five significant digits, never six."""
        expected = {
            99999.9: "1.0000e+05",
            -99999.9: "-1.0000e+05",
            0.999999: "1.0000",
            -0.999999: "-1.0000",
            9.99996: "10.000",
            0.000099999: "9.9999e-05",
        }
        for value, formatted in expected.items():
            with self.subTest(value=value):
                self.assertEqual(driver._five_significant(value), formatted)

    def test_default_impact_summary_interpolates_to_surface(self):
        result, summary = self.run_and_analyze()
        xs, ys, _, _, ts, _, _ = result
        self.assertTrue(summary["impact"])
        self.assertFalse(summary["reached_orbit"])
        self.assertGreater(summary["impact_fraction"], 0.0)
        self.assertLess(summary["impact_fraction"], 1.0)
        self.assertLess(summary["total_time"], ts[-1])
        self.assertAlmostEqual(summary["maximum_altitude"], 300.0, places=6)
        self.assertGreater(summary["total_distance"], 0.0)
        self.assertIn("Surface impact before one revolution", summary["outcome"])
        self.assertTrue(any("interpolated to impact" in line for line in summary["lines"]))
        self.assertLess(math.hypot(xs[-1], ys[-1]), physics.R_EARTH)

    def test_completed_orbit_reports_each_revolution(self):
        h0 = 300_000.0
        speed = math.sqrt(physics.MU_EARTH / (physics.R_EARTH + h0))
        _, summary = self.run_and_analyze(
            h0=h0,
            uInit=speed,
            dt=1.0,
            maxSteps=7000,
            force_law="inverse_square",
        )
        self.assertTrue(summary["reached_orbit"])
        self.assertEqual(summary["completed_revolutions"], 1)
        self.assertEqual(len(summary["revolution_data"]), 1)
        revolution = summary["revolution_data"][0]
        self.assertGreater(revolution["time"], 5000.0)
        self.assertTrue(math.isfinite(revolution["minimum_altitude"]))
        self.assertTrue(math.isfinite(revolution["maximum_altitude"]))
        self.assertLessEqual(
            revolution["minimum_altitude"], revolution["maximum_altitude"]
        )
        self.assertTrue(any(line.startswith("Revolution 1:") for line in summary["lines"]))

    def test_exact_circular_initial_state_has_circular_osculating_elements(self):
        h0 = 300_000.0
        speed = math.sqrt(physics.MU_EARTH / (physics.R_EARTH + h0))
        _, summary = self.run_and_analyze(
            h0=h0,
            uInit=speed,
            dt=1.0,
            maxSteps=2,
            force_law="inverse_square",
        )
        elements = summary["elements"]
        self.assertAlmostEqual(elements["eccentricity"], 0.0, delta=2.0e-8)
        self.assertAlmostEqual(elements["perigee_altitude"], h0, delta=0.2)
        self.assertAlmostEqual(elements["apogee_altitude"], h0, delta=0.2)

    def test_escape_summary_prints_final_altitude_without_apogee(self):
        _, summary = self.run_and_analyze(
            h0=300_000.0,
            uInit=11_500.0,
            dt=2.0,
            maxSteps=3001,
            force_law="inverse_square",
        )
        self.assertTrue(summary["escape"])
        self.assertFalse(summary["reached_orbit"])
        self.assertIsNone(summary["elements"]["apogee_altitude"])
        self.assertTrue(
            any("Final/maximum altitude at maxSteps" in line for line in summary["lines"])
        )
        self.assertFalse(any("apogee" in line.lower() for line in summary["lines"]))


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


_HEADLESS_SHOW_WARNING = re.compile(
    r"^.*: UserWarning: \w+ is non-interactive, and thus cannot be shown$"
)


def without_headless_show_warning(stderr):
    """Remove Matplotlib's warning that a non-interactive figure cannot be shown.

    With MPLBACKEND=Agg, plt.show() is silent on Linux without a display but
    emits this warning, followed by the offending source line, on Windows,
    macOS and Linux desktops.  Only that two-line warning is removed; any
    other text is kept, so the empty-stderr checks stay strict.
    """
    lines = stderr.splitlines()
    kept = []
    index = 0
    while index < len(lines):
        if _HEADLESS_SHOW_WARNING.match(lines[index]):
            index += 1
            if index < len(lines) and lines[index].startswith((" ", "\t")):
                index += 1
            continue
        kept.append(lines[index])
        index += 1
    return "\n".join(kept)


class MainProgramTests(unittest.TestCase):
    """Command-line and documented console-interface contract tests."""

    def test_headless_show_warning_filter_removes_only_that_warning(self):
        warning = (
            "C:\\repo\\EarthOrbit\\plot_earthorbit.py:56: UserWarning: "
            "FigureCanvasAgg is non-interactive, and thus cannot be shown\n"
            "  plt.show()\n"
        )
        self.assertEqual(without_headless_show_warning(warning), "")
        self.assertEqual(without_headless_show_warning(""), "")
        other = "main.py:10: RuntimeWarning: overflow encountered\n  x = y * z\n"
        self.assertEqual(without_headless_show_warning(warning + other), other.rstrip("\n"))
        self.assertEqual(without_headless_show_warning("Traceback (most recent call last):"),
                         "Traceback (most recent call last):")

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
        for required in (
            "Surface impact before one revolution",
            "Fraction of a revolution:",
            "Total flight time:",
            "Total angular travel:",
            "Maximum altitude (interpolated):",
            "Total distance traveled (interpolated to impact):",
        ):
            with self.subTest(required=required):
                self.assertIn(required, result.stdout)
        # plt.show() warns under Agg except on Linux without a display.
        self.assertEqual(without_headless_show_warning(result.stderr), "")

    def test_command_line_defaults_match_driver_defaults(self):
        with mock.patch.object(sys, "argv", ["main.py"]):
            args = entrypoint.parse_args()
        signature = inspect.signature(driver.run_earth_orbit)
        for name in ("h0", "uInit", "vInit", "dt", "maxSteps", "force_law"):
            with self.subTest(name=name):
                self.assertEqual(getattr(args, name), signature.parameters[name].default)
        self.assertNotIn("return_diagnostics", vars(args))
        main_source = (MODULE_DIR / "main.py").read_text(encoding="utf-8")
        self.assertIn("return_diagnostics=True", main_source)

    def test_help_describes_all_cli_inputs_and_both_force_laws(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            cwd=MODULE_DIR,
            text=True,
            capture_output=True,
            timeout=20,
            check=True,
        )
        normalized = " ".join(result.stdout.split())
        for required in (
            "--h0", "--uInit", "--vInit", "--dt", "--maxSteps",
            "--force_law", "simplified keeps", "inverse_square uses",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized)

    def test_custom_escape_cli_prints_elements_and_final_altitude(self):
        environment = os.environ.copy()
        environment["MPLBACKEND"] = "Agg"
        result = subprocess.run(
            [
                sys.executable, "main.py", "--h0", "300000",
                "--uInit", "11500", "--vInit", "0", "--dt", "2",
                "--maxSteps", "3001", "--force_law", "inverse_square",
            ],
            cwd=MODULE_DIR,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        self.assertIn("Escape trajectory; maxSteps reached", result.stdout)
        self.assertIn("Osculating eccentricity:", result.stdout)
        self.assertIn("Osculating perigee altitude:", result.stdout)
        self.assertIn("Final/maximum altitude at maxSteps:", result.stdout)
        self.assertNotIn("Osculating apogee altitude:", result.stdout)


class HelpFileTests(unittest.TestCase):
    """Documentation-interface, scientific wording, and presentation contracts."""

    @classmethod
    def setUpClass(cls):
        cls.html = HELP_FILE.read_text(encoding="utf-8")

    def _original_html(self):
        """Text of the Reference Guide Help, for the tests written for it."""
        if not ORIGINAL_HELP_FILE.is_file():
            self.skipTest("EarthOrbit-original.html is not present; nothing else depends on it")
        return ORIGINAL_HELP_FILE.read_text(encoding="utf-8")

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
        html = self._original_html()
        required_text = (
            '<td class="pname">--h0</td>',
            '<td class="pdefault">300.0</td>',
            '<td class="pname">--uInit</td>',
            '<td class="pdefault">7900.0</td>',
            '<td class="pname">--vInit</td>',
            '<td class="pdefault">0.0</td>',
            '<td class="pname">--dt</td>',
            '<td class="pdefault">0.4</td>',
            '<td class="pname">--maxSteps</td>',
            '<td class="pdefault">15000</td>',
            '<td class="pname">--force_law</td>',
            '<td class="pdefault">simplified</td>',
            "return_diagnostics=True",
            "MU_EARTH",
            "3.986_004_355_07e14",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, html)
        self.assertIn("number of trajectory samples", html)

    def test_help_has_no_stale_inverse_square_energy_formula(self):
        self.assertNotIn("K_APPROX/r", self.html)

    def test_original_help_gives_the_inverse_square_energy_formula(self):
        html = self._original_html()
        self.assertNotIn("K_APPROX/r", html)
        self.assertIn("MU_EARTH/r", html)

    def test_help_contains_mathjax_offline_explanation(self):
        self.assertIn("MathJax", self.html)
        self.assertIn("loaded from a public CDN", self.html)
        self.assertIn("internet connection is needed", self.html)
        self.assertIn("program itself, which needs no internet access", self.html)

    def test_help_contains_exactly_ten_ranked_exercises(self):
        html = self._original_html()
        labels = re.findall(r'<div class="ec-num">EXP-(\d+) · ([^<]+)</div>', html)
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
        html = self._original_html()
        self.assertIn(r"v_c=\sqrt{gr}", html)
        self.assertIn("Requires <code>force_law=\"inverse_square\"</code>", html)
        self.assertIn("it has no finite escape speed", html)
        self.assertIn(r"T^2\propto r", html)
        self.assertIn(r"T^2/r^3", html)

    def test_help_documents_validation_and_impact_endpoint(self):
        html = self._original_html()
        self.assertIn("Accepted parameter values", html)
        self.assertIn("must be an integer of at least 2", html)
        self.assertIn("return_diagnostics", html)
        self.assertIn("final plotted point may lie slightly below", html)
        self.assertIn("Interpolate the Impact Point", html)

    def test_runnable_diagnostic_blocks_parse_and_execute(self):
        html = self._original_html()
        namespaces = {}
        for number in (4, 9, 10):
            with self.subTest(experiment=number):
                code = _exercise_code(html, number)
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
        self.assertTrue(np.all(np.isfinite(angle_travelled)))
        # EXP-9's accumulated angular travel is |theta - theta[0]| for a
        # circular orbit traversed in a fixed direction, so it should never
        # step backwards between stored samples.
        self.assertTrue(np.all(np.diff(angle_travelled) >= 0.0))

        exp10_r = namespaces[10]["r"]
        exp10_speed_squared = namespaces[10]["speed_squared"]
        self.assertEqual(len(exp10_r), len(exp10_speed_squared))
        self.assertEqual(len(exp10_r), len(namespaces[10]["ts"]))
        self.assertTrue(np.all(np.isfinite(exp10_r)))
        self.assertTrue(np.all(np.isfinite(exp10_speed_squared)))
        self.assertTrue(np.all(exp10_r > 0.0))
        self.assertTrue(np.all(exp10_speed_squared >= 0.0))

    def test_advanced_blocks_are_explicitly_runnable_starter_code(self):
        html = self._original_html()
        for number in (9, 10):
            with self.subTest(experiment=number):
                fragment = _exercise_fragment(html, number)
                self.assertIn("Runnable starter code", fragment)
                self.assertNotIn("run_earth_orbit(...", fragment)

        exp9_tree = ast.parse(_exercise_code(html, 9))
        top_level_targets = {
            target.id
            for node in exp9_tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertNotIn("force_law", top_level_targets)
        self.assertNotIn("G_SURFACE", _exercise_code(html, 10))

    def test_cannon_trajectory_is_listed_as_direct_predecessor(self):
        related = self.html.split('<section id="related">', 1)[1].split("</section>", 1)[0]
        text = " ".join(re.sub(r"<[^>]+>", " ", related).split())
        self.assertRegex(text, r"^Related Programs CannonTrajectory\b")
        self.assertIn("the simpler near-surface projectile calculation", text.lower())

    def test_related_programs_give_no_chapter_numbers_or_links(self):
        # Help file names will change, and the chapter order of a new edition is not known.
        related = self.html.split('<section id="related">', 1)[1].split("</section>", 1)[0]
        self.assertNotIn("<a ", related)
        self.assertNotRegex(re.sub(r"<[^>]+>", " ", related), r"\bChapter\b|\bCh\.|\bInvestigations?\b")

    def test_development_history_is_confined_to_license_provenance(self):
        pages = [("Beats Help", self.html)]
        if ORIGINAL_HELP_FILE.is_file():
            pages.append(("Reference Guide", ORIGINAL_HELP_FILE.read_text(encoding="utf-8")))
        for label, page in pages:
            student_content = page.split('<section id="license">', 1)[0]
            with self.subTest(help=label):
                self.assertNotIn("Triana", student_content)
                self.assertNotIn("original Java", student_content)
                # a whole word, so that "reporting" is not mistaken for it
                self.assertNotRegex(student_content.lower(), r"\bporting\b")

    def test_no_known_malformed_paragraph_nesting(self):
        self.assertIsNone(re.search(r"<p(?:\s[^>]*)?>\s*<p(?:\s[^>]*)?>", self.html))
        self.assertIsNone(re.search(r"</p>\s*</p>", self.html))

    def test_all_core_module_names_appear(self):
        for name in CORE_MODULE_FILENAMES:
            with self.subTest(module=name):
                self.assertIn(name, self.html)


class DocumentationSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.release_notes = RELEASE_NOTES_FILE.read_text(encoding="utf-8")
        cls.samples = SAMPLE_OUTPUTS_FILE.read_text(encoding="utf-8")

    def test_documentation_files_exist(self):
        self.assertTrue(RELEASE_NOTES_FILE.is_file())
        self.assertTrue(SAMPLE_OUTPUTS_FILE.is_file())

    def test_release_notes_match_current_version_and_build(self):
        self.assertIn(f"Version {physics.MODEL_VERSION}", self.release_notes)
        self.assertIn(f"<b>Build:</b> {physics.BUILD_ID}", self.release_notes)
        self.assertIn("command-line", self.release_notes)
        self.assertIn("five significant digits", self.release_notes)

    def test_sample_outputs_match_current_version_and_build(self):
        self.assertIn(f"Version {physics.MODEL_VERSION}", self.samples)
        self.assertIn(f"Build {physics.BUILD_ID}", self.samples)
        self.assertNotIn("Version 1.1.1", self.samples)
        self.assertNotIn("Build 77b768d3a136", self.samples)

    def test_sample_outputs_demonstrate_every_cli_parameter(self):
        for option in ("--h0", "--uInit", "--vInit", "--dt", "--maxSteps", "--force_law"):
            with self.subTest(option=option):
                self.assertIn(option, self.samples)

    def test_sample_outputs_cover_requested_result_classes(self):
        required_text = (
            "Surface impact before one revolution",
            "Fraction of a revolution",
            "Total distance traveled (interpolated to impact)",
            "Completed revolutions",
            "Revolution 1: time",
            "minimum altitude",
            "maximum altitude",
            "Osculating eccentricity",
            "Osculating perigee altitude",
            "Osculating apogee altitude",
            "Escape trajectory",
            "Final/maximum altitude at maxSteps",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, self.samples)


if __name__ == "__main__":
    unittest.main(verbosity=2)
