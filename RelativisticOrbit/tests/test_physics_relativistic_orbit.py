"""Regression tests for the complete RelativisticOrbit module.

The discovery logic deliberately supports both repository layout

    RelativisticOrbit/tests/test_physics_relativistic_orbit.py

and a flattened review/upload layout in which this test file is placed beside
the four program modules and RelativisticOrbit-claude.html.  The Beats Help,
RelativisticOrbit-claude.html, is required; the Reference Guide Help,
RelativisticOrbit-original.html, is optional and its tests skip without it.
"""

from __future__ import annotations

import contextlib
from dataclasses import fields
import ast
import hashlib
import html
from html.parser import HTMLParser
import importlib
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
from fractions import Fraction
from typing import NamedTuple
import unittest
from unittest import mock


CORE_MODULE_FILENAMES = (
    "physics_relativistic_orbit.py",
    "driver_relativistic_orbit.py",
    "main.py",
    "plot_relativistic_orbit.py",
)
# The Beats Help file; the Reference Guide version (-original) is never used here.
HELP_FILENAMES = ("RelativisticOrbit-claude.html", "RelativisticOrbit.html")
PROGRAM_NAME = "RelativisticOrbit"
MINIMUM_PYTHON_VERSION = (3, 10)


def find_module_dir(start: str | os.PathLike[str]) -> Path:
    """Find the nearest ancestor (including start) with all four core files."""
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        if all((directory / name).is_file() for name in CORE_MODULE_FILENAMES):
            return directory

    names = ", ".join(CORE_MODULE_FILENAMES)
    raise FileNotFoundError(
        f"Could not find a RelativisticOrbit module directory containing: {names}"
    )


def find_help_file(module_dir: Path) -> Path:
    """Find the Beats Help in a flattened upload or the GFTGU-Documentation tree.

    Documentation folders no longer use chapter-number prefixes, and the
    Help files live under the sibling ``GFTGU-Documentation`` repository
    rather than beside the program modules inside ``GFTGU-Programs``.
    """
    for filename in HELP_FILENAMES:
        candidates = [module_dir / filename]
        for ancestor in (module_dir, *module_dir.parents):
            candidates.append(ancestor / "GFTGU-Documentation" / PROGRAM_NAME / filename)
            if ancestor.name != PROGRAM_NAME:
                candidates.append(ancestor / PROGRAM_NAME / filename)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"Could not find {' or '.join(HELP_FILENAMES)} beside the program or in "
        f"GFTGU-Documentation/{PROGRAM_NAME}/."
    )


MODULE_DIR = find_module_dir(Path(__file__).resolve().parent)
HELP_FILE = find_help_file(MODULE_DIR)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import driver_relativistic_orbit as driver  # noqa: E402
import physics_relativistic_orbit as physics  # noqa: E402
from driver_relativistic_orbit import (  # noqa: E402
    RelativisticOrbitParams,
    integrate_relativistic_orbit,
)


class _HelpStructureParser(HTMLParser):
    """Collect IDs, fragment links, and table rows with the standard library."""

    def __init__(self):
        super().__init__()
        self.ids: list[str] = []
        self.fragments: list[str] = []
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.append(attributes["id"])
        href = attributes.get("href", "")
        if href.startswith("#"):
            self.fragments.append(href[1:])
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def params(**changes) -> RelativisticOrbitParams:
    """Return compact, valid defaults with selected fields replaced."""
    values = dict(
        x_init=15_000.0,
        u_init=1.2e8,
        dt=2.0e-6,
        max_steps=200,
        max_orbits=1,
        eps1=0.05,
        eps2=1.0e-4,
        model="schwarzschild",
    )
    values.update(changes)
    return RelativisticOrbitParams(**values)


class TestDiscoveryAndReleaseMetadata(unittest.TestCase):
    def test_module_directory_contains_all_core_files(self):
        self.assertTrue(all((MODULE_DIR / name).is_file() for name in CORE_MODULE_FILENAMES))

    def test_find_module_dir_from_module_directory(self):
        self.assertEqual(find_module_dir(MODULE_DIR), MODULE_DIR)

    def test_find_module_dir_from_tests_directory(self):
        self.assertEqual(find_module_dir(Path(__file__).parent), MODULE_DIR)

    def test_find_module_dir_from_test_file(self):
        self.assertEqual(find_module_dir(__file__), MODULE_DIR)

    def test_find_module_dir_failure_names_required_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            nested = Path(temp_dir) / "one" / "two"
            nested.mkdir(parents=True)
            with self.assertRaisesRegex(FileNotFoundError, "physics_relativistic_orbit.py"):
                find_module_dir(nested)

    def test_build_id_is_independently_reproducible(self):
        digest = hashlib.sha256()
        for name in CORE_MODULE_FILENAMES:
            with (MODULE_DIR / name).open("r", encoding="utf-8", newline=None) as source:
                content = source.read().encode("utf-8")
            digest.update(name.encode("utf-8"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        self.assertEqual(physics.BUILD_ID, digest.hexdigest()[:12])
        self.assertRegex(physics.BUILD_ID, r"^[0-9a-f]{12}$")

    def test_build_id_coverage_is_exact(self):
        self.assertEqual(tuple(physics.BUILD_ID_COVERS), CORE_MODULE_FILENAMES)

    def test_incomplete_core_is_explicitly_unpackaged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            copied = Path(temp_dir) / "physics_relativistic_orbit.py"
            shutil.copy2(MODULE_DIR / "physics_relativistic_orbit.py", copied)
            spec = importlib.util.spec_from_file_location("isolated_relativistic_physics", copied)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.BUILD_ID, "unpackaged")

    def test_help_version_and_build_match_program(self):
        text = HELP_FILE.read_text(encoding="utf-8")
        match = re.search(
            r'<p\s+id=["\']version_build["\'][^>]*>(.*?)</p>', text, re.DOTALL
        )
        self.assertIsNotNone(match, "Help file lacks <p id=\"version_build\">.")
        visible = html.unescape(re.sub(r"<[^>]+>", " ", match.group(1)))
        visible = " ".join(visible.split())
        self.assertIn(f"Version {physics.MODEL_VERSION}", visible)
        self.assertIn(f"Build {physics.BUILD_ID}", visible)

    def test_help_advertises_the_supported_minimum_python_version(self):
        text = HELP_FILE.read_text(encoding="utf-8")
        self.assertRegex(text, r"Python\s+3\.10\s+or\s+later")
        self.assertNotRegex(text, r"Python\s+3\.9\s+or\s+later")

    def test_release_sources_parse_with_python_310_grammar(self):
        """Guard the advertised syntax floor without assuming one fixed layout."""
        for name in (*CORE_MODULE_FILENAMES, Path(__file__).name):
            source_file = MODULE_DIR / name
            if not source_file.is_file():
                source_file = Path(__file__).resolve().parent / name
            with self.subTest(source=name):
                ast.parse(
                    source_file.read_text(encoding="utf-8"),
                    filename=str(source_file),
                    feature_version=MINIMUM_PYTHON_VERSION,
                )

    def test_help_has_no_hidden_control_characters(self):
        text = HELP_FILE.read_text(encoding="utf-8")
        bad = [char for char in text if ord(char) < 32 and char not in "\t\n\r"]
        self.assertEqual(bad, [])

    def test_help_revolution_formula_is_intact(self):
        text = HELP_FILE.read_text(encoding="utf-8")
        self.assertIn(r"N_{\rm rev}=\frac{|\Delta\phi_{\rm accumulated}|}{2\pi}", text)

    def test_help_avoids_deprecated_porting_history(self):
        text = HELP_FILE.read_text(encoding="utf-8").lower()
        for phrase in (
            "porting error",
            "previous python",
            "revised code",
            "no longer silently",
            "original notation defined",
            "program uses \\(h\\) directly",
        ):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, text)

    def test_help_has_unique_ids_and_resolved_fragment_links(self):
        parser = _HelpStructureParser()
        parser.feed(HELP_FILE.read_text(encoding="utf-8"))
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        self.assertEqual(set(parser.fragments) - set(parser.ids), set())

    def test_help_mathjax_delimiters_are_balanced(self):
        text = HELP_FILE.read_text(encoding="utf-8")
        self.assertEqual(text.count(r"\("), text.count(r"\)"))
        self.assertEqual(text.count(r"\["), text.count(r"\]"))

    def test_result_api_fields_are_stable(self):
        actual = {field.name for field in fields(driver.RelativisticOrbitResult)}
        expected = {
            "model_version", "build_id", "x", "y", "vx", "vy", "tau",
            "azimuth_unwrapped", "n_orbits", "final_step", "fell_into_hole",
            "termination_reason", "model", "periapsis_indices", "periapsis_tau",
            "periapsis_radius", "periapsis_azimuth", "mean_periapsis_advance",
            "max_fractional_h_drift", "max_fractional_energy_drift",
        }
        self.assertEqual(actual, expected)


class TestPhysicsConstantsAndEquations(unittest.TestCase):
    def test_nominal_constants(self):
        self.assertEqual(physics.GM_SUN, 1.3271244e20)
        self.assertEqual(physics.C, 299_792_458.0)

    def test_characteristic_radii(self):
        self.assertAlmostEqual(physics.HORIZON_RADIUS, 2.0 * physics.GM_SUN / physics.C2)
        self.assertAlmostEqual(physics.PHOTON_ORBIT_RADIUS, 1.5 * physics.HORIZON_RADIUS)
        self.assertAlmostEqual(physics.ISCO_RADIUS, 3.0 * physics.HORIZON_RADIUS)
        self.assertAlmostEqual(physics.HORIZON_RADIUS, 2953.2500761, places=5)

    def test_orbital_constants_match_schutz_notation(self):
        h, q = physics.orbital_constants(15_000.0, 1.2e8)
        self.assertEqual(h, 15_000.0 * 1.2e8)
        self.assertAlmostEqual(q, 3.0 * h * h / physics.C2)

    def test_orbital_constants_preserve_direction(self):
        h, q = physics.orbital_constants(10_000.0, -2.0e8)
        self.assertLess(h, 0.0)
        self.assertGreater(q, 0.0)

    def test_newtonian_acceleration(self):
        x, y = 3_000.0, 4_000.0
        ax, ay = physics.central_acceleration(x, y, 9.0e11, "newtonian")
        r3 = math.hypot(x, y) ** 3
        self.assertAlmostEqual(ax, -physics.GM_SUN * x / r3)
        self.assertAlmostEqual(ay, -physics.GM_SUN * y / r3)

    def test_schwarzschild_correction_factor(self):
        x, y, h = 12_000.0, 5_000.0, 1.7e12
        ax_n, ay_n = physics.central_acceleration(x, y, h, "newtonian")
        ax_s, ay_s = physics.central_acceleration(x, y, h, "schwarzschild")
        r2 = x * x + y * y
        correction = 1.0 + 3.0 * h * h / (physics.C2 * r2)
        self.assertAlmostEqual(ax_s / ax_n, correction)
        self.assertAlmostEqual(ay_s / ay_n, correction)

    def test_acceleration_is_central(self):
        ax, ay = physics.central_acceleration(12_000.0, -7_000.0, 1.0e12)
        self.assertAlmostEqual(12_000.0 * ay - (-7_000.0) * ax, 0.0, delta=1.0e-4)

    def test_model_name_is_case_insensitive(self):
        lower = physics.central_acceleration(10_000.0, 0.0, 1.0e12, "schwarzschild")
        upper = physics.central_acceleration(10_000.0, 0.0, 1.0e12, "SCHWARZSCHILD")
        self.assertEqual(lower, upper)

    def test_specific_angular_momentum(self):
        self.assertEqual(physics.specific_angular_momentum(2.0, 3.0, 5.0, 7.0), -1.0)

    def test_effective_energy_model_difference(self):
        state = (15_000.0, 2_000.0, -1.0e7, 1.1e8)
        h = physics.specific_angular_momentum(*state)
        e_n = physics.effective_specific_energy(*state, h, "newtonian")
        e_s = physics.effective_specific_energy(*state, h, "schwarzschild")
        r = math.hypot(state[0], state[1])
        expected = physics.GM_SUN * h * h / (physics.C2 * r**3)
        self.assertAlmostEqual(e_n - e_s, expected, delta=abs(expected) * 2e-15)

    def test_circular_speed_balances_radial_equation(self):
        radius = 10_000.0
        speed = physics.circular_proper_time_speed(radius)
        h = radius * speed
        ax, _ = physics.central_acceleration(radius, 0.0, h)
        # The two terms are about 1e12 m/s^2 and cancel to floating precision.
        self.assertAlmostEqual(speed * speed / radius + ax, 0.0, delta=2.0e-3)

    def test_circular_speed_requires_radius_above_photon_orbit(self):
        for radius in (physics.PHOTON_ORBIT_RADIUS, 0.99 * physics.PHOTON_ORBIT_RADIUS):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                physics.circular_proper_time_speed(radius)

    def test_circular_speed_at_nearest_representable_allowed_radius(self):
        radius = math.nextafter(physics.PHOTON_ORBIT_RADIUS, math.inf)
        speed = physics.circular_proper_time_speed(radius)
        self.assertTrue(math.isfinite(speed))
        self.assertGreater(speed, physics.C)

    def test_physics_functions_reject_invalid_values_cleanly(self):
        calls = (
            lambda: physics.orbital_constants("bad", 1.0),
            lambda: physics.orbital_constants(1.0, math.nan),
            lambda: physics.central_acceleration(0.0, 0.0, 1.0),
            lambda: physics.central_acceleration(1.0, 0.0, 1.0, None),
            lambda: physics.central_acceleration(1.0e-110, 0.0, 1.0),
            lambda: physics.specific_angular_momentum(1e308, 1e308, 1e308, -1e308),
            lambda: physics.effective_specific_energy(0.0, 0.0, 0.0, 0.0, 0.0),
            lambda: physics.circular_proper_time_speed(True),
            lambda: physics.circular_proper_time_speed(math.inf),
        )
        for call in calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()


class TestDriverUtilitiesAndValidation(unittest.TestCase):
    def test_relative_vector_change_is_rotation_invariant(self):
        self.assertAlmostEqual(driver._relative_vector_change(1, 0, 0, 1), math.sqrt(2))
        self.assertAlmostEqual(driver._relative_vector_change(0, 1, -1, 0), math.sqrt(2))

    def test_unwrap_delta_across_branch_cut(self):
        self.assertAlmostEqual(driver._unwrap_delta(-math.pi + 0.1, math.pi - 0.1), 0.2)
        self.assertAlmostEqual(driver._unwrap_delta(math.pi - 0.1, -math.pi + 0.1), -0.2)

    def test_segment_circle_intersection(self):
        fraction = driver._segment_circle_first_fraction(2.0, 0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(fraction, 0.5)
        self.assertIsNone(driver._segment_circle_first_fraction(2.0, 0.0, 2.0, 2.0, 1.0))

    def test_fractional_drift_zero_reference(self):
        self.assertEqual(driver._fractional_drift(0.0, 0.0), 0.0)
        self.assertTrue(math.isinf(driver._fractional_drift(1.0, 0.0)))

    def test_parameter_validation_matrix(self):
        invalid = (
            dict(x_init=0.0),
            dict(x_init=physics.HORIZON_RADIUS),
            dict(x_init=math.inf),
            dict(x_init="15000"),
            dict(u_init=math.nan),
            dict(u_init=False),
            dict(dt=0.0),
            dict(dt=True),
            dict(max_steps=0),
            dict(max_steps=True),
            dict(max_steps=2.5),
            dict(max_orbits=0),
            dict(max_orbits=False),
            dict(eps1=1.0),
            dict(eps2=0.0),
            dict(eps1=1e-4, eps2=1e-4),
            dict(eps1=1e-5, eps2=1e-4),
            dict(model="kerr"),
            dict(model=None),
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                integrate_relativistic_orbit(params(**changes))

    def test_newtonian_mode_allows_start_inside_schwarzschild_horizon(self):
        result = integrate_relativistic_orbit(
            params(x_init=1_000.0, u_init=1.0e8, dt=1e-10, max_steps=1, model="newtonian")
        )
        self.assertEqual(result.model, "newtonian")
        self.assertFalse(result.fell_into_hole)

    def test_predictor_gate_rejects_then_retries_without_accepting_state(self):
        values = [1.0, 0.0, 0.0]
        with mock.patch.object(driver, "_relative_vector_change", side_effect=values):
            result = integrate_relativistic_orbit(
                params(dt=1e-6, max_steps=1, max_orbits=10)
            )
        self.assertEqual(result.final_step, 1)
        self.assertEqual(len(result.tau), 2)
        self.assertAlmostEqual(result.tau[-1], 0.5e-6)

    def test_rejected_step_recovers_by_ten_percent_and_caps_at_user_dt(self):
        values = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        with mock.patch.object(driver, "_relative_vector_change", side_effect=values):
            result = integrate_relativistic_orbit(
                params(dt=1e-6, max_steps=3, max_orbits=10)
            )
        increments = [b - a for a, b in zip(result.tau, result.tau[1:])]
        expected = [0.5e-6, 0.55e-6, 0.605e-6]
        for actual, target in zip(increments, expected):
            self.assertAlmostEqual(actual, target, places=18)
        self.assertTrue(all(step <= 1e-6 for step in increments))

    def test_corrector_exhaustion_reaches_eighty_retry_safety_valve(self):
        call_number = 0

        def forced_change(*_args):
            nonlocal call_number
            value = 0.0 if call_number % 11 == 0 else 1.0
            call_number += 1
            return value

        with (
            mock.patch.object(driver, "_relative_vector_change", side_effect=forced_change),
            self.assertRaisesRegex(RuntimeError, "80 retries"),
        ):
            integrate_relativistic_orbit(params(dt=1e-6, max_steps=1))
        self.assertEqual(call_number, 80 * 11)

    def test_no_progress_timestep_safety_error(self):
        invalid = params(dt=0.0, max_steps=1)
        with (
            mock.patch.object(driver, "_validate_params"),
            self.assertRaisesRegex(RuntimeError, "cannot advance proper time"),
        ):
            integrate_relativistic_orbit(invalid)


class TestIntegratedOrbits(unittest.TestCase):
    def test_default_regression(self):
        result = integrate_relativistic_orbit(
            params(max_steps=6_000, max_orbits=10)
        )
        self.assertEqual(result.termination_reason, "max_steps")
        self.assertEqual(result.final_step, 6_000)
        self.assertAlmostEqual(result.n_orbits, 9.285146, places=5)
        self.assertEqual(len(result.periapsis_indices), 6)
        self.assertAlmostEqual(result.mean_periapsis_advance, 2.44005, places=4)
        self.assertLess(result.max_fractional_h_drift, 5e-5)
        self.assertLess(result.max_fractional_energy_drift, 2e-4)

    def test_default_rosette_has_regular_periapsis_radii(self):
        result = integrate_relativistic_orbit(params(max_steps=6_000, max_orbits=10))
        spread = max(result.periapsis_radius) - min(result.periapsis_radius)
        mean = sum(result.periapsis_radius) / len(result.periapsis_radius)
        self.assertLess(spread / mean, 2e-5)

    def test_newtonian_circular_orbit(self):
        radius = 10_000.0
        speed = math.sqrt(physics.GM_SUN / radius)
        result = integrate_relativistic_orbit(
            params(
                x_init=radius,
                u_init=speed,
                dt=2e-6,
                max_steps=5_000,
                max_orbits=1,
                eps1=0.02,
                eps2=1e-6,
                model="newtonian",
            )
        )
        radii = [math.hypot(x, y) for x, y in zip(result.x, result.y)]
        self.assertEqual(result.termination_reason, "max_orbits")
        self.assertLess(max(abs(value - radius) for value in radii), 2e-5)

    def test_schwarzschild_circular_orbit(self):
        radius = 10_000.0
        result = integrate_relativistic_orbit(
            params(
                x_init=radius,
                u_init=physics.circular_proper_time_speed(radius),
                dt=2e-6,
                max_steps=5_000,
                max_orbits=1,
                eps1=0.02,
                eps2=1e-6,
            )
        )
        radii = [math.hypot(x, y) for x, y in zip(result.x, result.y)]
        self.assertEqual(result.termination_reason, "max_orbits")
        self.assertLess(max(abs(value - radius) for value in radii), 2e-4)

    def test_clockwise_orbit_counting(self):
        radius = 10_000.0
        result = integrate_relativistic_orbit(
            params(
                x_init=radius,
                u_init=-physics.circular_proper_time_speed(radius),
                dt=2e-6,
                max_steps=5_000,
                max_orbits=1,
                eps1=0.02,
                eps2=1e-6,
            )
        )
        self.assertEqual(result.termination_reason, "max_orbits")
        self.assertGreaterEqual(result.n_orbits, 1.0)
        self.assertLess(result.azimuth_unwrapped[-1], 0.0)

    def test_radial_infall_stops_on_true_horizon(self):
        result = integrate_relativistic_orbit(
            params(x_init=10_000.0, u_init=0.0, dt=1e-5, max_steps=2_000)
        )
        self.assertEqual(result.termination_reason, "horizon")
        self.assertTrue(result.fell_into_hole)
        self.assertAlmostEqual(
            math.hypot(result.x[-1], result.y[-1]), physics.HORIZON_RADIUS, places=8
        )
        self.assertEqual(result.max_fractional_h_drift, 0.0)

    def test_nonradial_horizon_event_converges_and_is_excluded_from_drift(self):
        runs = []
        for dt in (2e-6, 1e-6, 0.5e-6):
            runs.append(
                integrate_relativistic_orbit(
                    params(
                        x_init=15_000.0,
                        u_init=5.0e7,
                        dt=dt,
                        max_steps=10_000,
                        max_orbits=10,
                        eps1=0.05,
                        eps2=1e-5,
                    )
                )
            )
        for result in runs:
            self.assertEqual(result.termination_reason, "horizon")
            self.assertAlmostEqual(math.hypot(result.x[-1], result.y[-1]), physics.HORIZON_RADIUS, places=8)
        coarse_change = abs(runs[1].tau[-1] - runs[0].tau[-1])
        fine_change = abs(runs[2].tau[-1] - runs[1].tau[-1])
        self.assertLess(fine_change, coarse_change)

        finest = runs[-1]
        h_initial = 15_000.0 * 5.0e7
        h_event = physics.specific_angular_momentum(
            finest.x[-1], finest.y[-1], finest.vx[-1], finest.vy[-1]
        )
        event_drift = abs(h_event - h_initial) / abs(h_initial)
        self.assertGreater(event_drift, finest.max_fractional_h_drift)

    def test_max_steps_termination(self):
        result = integrate_relativistic_orbit(params(max_steps=3, max_orbits=10))
        self.assertEqual(result.termination_reason, "max_steps")
        self.assertEqual(result.final_step, 3)

    def test_result_arrays_and_metadata_are_consistent(self):
        result = integrate_relativistic_orbit(params(max_steps=25, max_orbits=10))
        lengths = {
            len(result.x), len(result.y), len(result.vx), len(result.vy),
            len(result.tau), len(result.azimuth_unwrapped),
        }
        self.assertEqual(lengths, {result.final_step + 1})
        self.assertTrue(all(b > a for a, b in zip(result.tau, result.tau[1:])))
        self.assertEqual(result.model_version, physics.MODEL_VERSION)
        self.assertEqual(result.build_id, physics.BUILD_ID)

    def test_model_is_normalized_in_result(self):
        result = integrate_relativistic_orbit(
            params(model="NEWTONIAN", max_steps=1, max_orbits=10)
        )
        self.assertEqual(result.model, "newtonian")


class TestIndependentBenchmarksAndConvergence(unittest.TestCase):
    """Stored oracles generated independently with SciPy DOP853.

    Provenance: kickoff audit, 2026-08-27; Cartesian first-order formulation,
    rtol=2e-13, componentwise atol=(1e-8,1e-8,1e-5,1e-5), max_step=1e-6 s.
    SciPy is not required to run these regression tests.
    """

    DOP853_FINAL_AT_TAU_0012 = (
        -4600.295838439546,
        19494.73428803206,
        -83932611.77108411,
        -35596848.80564334,
    )
    DOP853_APSIDAL_ADVANCE = 2.440369610158

    @classmethod
    def setUpClass(cls):
        cls.default = integrate_relativistic_orbit(
            params(max_steps=6_000, max_orbits=10)
        )

    def test_default_endpoint_agrees_with_independent_oracle(self):
        reference = self.DOP853_FINAL_AT_TAU_0012
        position_error = math.hypot(
            self.default.x[-1] - reference[0], self.default.y[-1] - reference[1]
        )
        velocity_error = math.hypot(
            self.default.vx[-1] - reference[2], self.default.vy[-1] - reference[3]
        )
        self.assertLess(position_error, 250.0)
        self.assertLess(velocity_error, 1.1e6)

    def test_default_apsidal_advance_agrees_with_independent_oracle(self):
        self.assertLess(
            abs(self.default.mean_periapsis_advance - self.DOP853_APSIDAL_ADVANCE),
            5e-4,
        )

    def test_newtonian_circular_orbit_has_second_order_state_convergence(self):
        radius = 1.0e7
        speed = math.sqrt(physics.GM_SUN / radius)
        period = 2.0 * math.pi * radius / speed
        errors = []
        for steps in (50, 100, 200):
            result = integrate_relativistic_orbit(
                params(
                    x_init=radius,
                    u_init=speed,
                    dt=period / steps,
                    max_steps=steps,
                    max_orbits=100,
                    eps1=0.9,
                    eps2=1e-10,
                    model="newtonian",
                )
            )
            errors.append(math.hypot(result.x[-1] - radius, result.y[-1]))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        for coarse, fine in zip(errors, errors[1:]):
            self.assertGreater(coarse / fine, 3.5)
            self.assertLess(coarse / fine, 4.5)

    def test_selected_adversarial_matrix_remains_finite_and_ordered(self):
        cases = (
            params(x_init=physics.HORIZON_RADIUS * 1.001, u_init=0.0, dt=1e-8, max_steps=250),
            params(x_init=15_000.0, u_init=-2.5e8, dt=1e-7, max_steps=250),
            params(x_init=15_000.0, u_init=2.5e8, dt=1e-7, max_steps=250),
            params(x_init=2.0e5, u_init=1.0e7, dt=1e-5, max_steps=250),
            params(x_init=1_000.0, u_init=-1.5e8, dt=1e-9, max_steps=250, model="newtonian"),
            params(x_init=1.0e5, u_init=0.0, dt=1e-7, max_steps=250, model="newtonian"),
        )
        for case in cases:
            with self.subTest(case=case):
                result = integrate_relativistic_orbit(case)
                sequences = (result.x, result.y, result.vx, result.vy, result.tau)
                self.assertEqual(len({len(sequence) for sequence in sequences}), 1)
                self.assertTrue(all(math.isfinite(value) for sequence in sequences for value in sequence))
                self.assertTrue(all(b > a for a, b in zip(result.tau, result.tau[1:])))


class TestMainAndPlotIntegration(unittest.TestCase):
    def test_importing_main_has_no_simulation_side_effect(self):
        command = [sys.executable, "-c", "import main; print('import-ok')"]
        env = {**os.environ, "MPLBACKEND": "Agg"}
        completed = subprocess.run(
            command,
            cwd=MODULE_DIR,
            env=env,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        self.assertEqual(completed.stdout.strip(), "import-ok")
        self.assertEqual(completed.stderr, "")

    def test_version_command_matches_release_metadata(self):
        completed = subprocess.run(
            [sys.executable, "main.py", "--version"],
            cwd=MODULE_DIR,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        self.assertEqual(
            completed.stdout.strip(),
            f"RelativisticOrbit {physics.MODEL_VERSION} (build {physics.BUILD_ID})",
        )
        self.assertEqual(completed.stderr, "")

    def test_main_reports_summary_and_returns_result(self):
        main_module = importlib.import_module("main")
        short_params = params(max_steps=2, max_orbits=10)
        output = io.StringIO()
        with (
            mock.patch.object(main_module, "params", short_params),
            mock.patch.object(main_module, "plot_relativistic_orbit"),
            contextlib.redirect_stdout(output),
        ):
            result = main_module.main([])
        self.assertEqual(result.final_step, 2)
        self.assertIn(f"RelativisticOrbit {physics.MODEL_VERSION}", output.getvalue())
        self.assertIn("maximum accepted-step count reached", output.getvalue())
        self.assertIn(f"horizon radius    : {physics.HORIZON_RADIUS:.5g} m", output.getvalue())
        self.assertIn(f"ISCO radius       : {physics.ISCO_RADIUS:.5g} m", output.getvalue())

    def test_newtonian_summary_does_not_claim_horizon_or_isco(self):
        main_module = importlib.import_module("main")
        short_params = params(model="newtonian", max_steps=2, max_orbits=10)
        output = io.StringIO()
        with (
            mock.patch.object(main_module, "params", short_params),
            mock.patch.object(main_module, "plot_relativistic_orbit"),
            contextlib.redirect_stdout(output),
        ):
            main_module.main([])
        self.assertIn("model             : newtonian", output.getvalue())
        self.assertNotIn("horizon radius", output.getvalue())
        self.assertNotIn("ISCO radius", output.getvalue())

    def test_plot_draws_physical_reference_circles_only_for_schwarzschild(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from plot_relativistic_orbit import plot_relativistic_orbit

        schwarzschild = integrate_relativistic_orbit(params(max_steps=2, max_orbits=10))
        with mock.patch.object(plt, "show"):
            plot_relativistic_orbit(schwarzschild, show_isco=True)
        axes = plt.gcf().axes[0]
        radii = sorted(patch.radius for patch in axes.patches)
        self.assertEqual(radii, sorted([physics.HORIZON_RADIUS, physics.ISCO_RADIUS]))
        plt.close("all")

        newtonian = integrate_relativistic_orbit(
            params(model="newtonian", max_steps=2, max_orbits=10)
        )
        with mock.patch.object(plt, "show"):
            plot_relativistic_orbit(newtonian, show_isco=True)
        self.assertEqual(len(plt.gcf().axes[0].patches), 0)
        plt.close("all")


class TestCommandLineOptions(unittest.TestCase):
    def test_all_driver_inputs_and_plot_switches_have_defaults(self):
        main_module = importlib.import_module("main")
        options = main_module.parse_args([])
        for field in fields(RelativisticOrbitParams):
            self.assertEqual(getattr(options, field.name),
                             getattr(main_module.params, field.name))
        self.assertEqual(options.show_isco, main_module.show_isco)
        self.assertEqual(options.show_periapsides, main_module.show_periapsides)

    def test_selector_flags_signed_velocity_and_numerical_controls(self):
        main_module = importlib.import_module("main")
        options = main_module.parse_args([
            "--model", "newtonian", "--u_init", "-2.5e8",
            "--x_init", "10000", "--dt", "1e-7", "--max_steps", "99",
            "--max_orbits", "3", "--eps1", "0.01", "--eps2", "1e-6",
            "--no-show_isco", "--show_periapsides",
        ])
        self.assertEqual(options.u_init, -2.5e8)
        self.assertEqual(options.model, "newtonian")
        self.assertEqual(options.max_steps, 99)
        self.assertFalse(options.show_isco)
        self.assertTrue(options.show_periapsides)

    def test_invalid_cli_values_report_errors(self):
        main_module = importlib.import_module("main")
        for argv in (["--model", "unknown"], ["--x_init", "1000"],
                     ["--dt", "0"], ["--eps1", "0.01", "--eps2", "0.02"],
                     ["--max_orbits", "0"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main_module.parse_args(argv)

    def test_options_are_forwarded_to_driver_and_plot(self):
        main_module = importlib.import_module("main")
        real_solver = main_module.integrate_relativistic_orbit
        with mock.patch.object(main_module, "integrate_relativistic_orbit",
                               wraps=real_solver) as solve, \
             mock.patch.object(main_module, "plot_relativistic_orbit") as plot, \
             contextlib.redirect_stdout(io.StringIO()):
            result = main_module.main([
                "--model", "newtonian", "--u_init", "-1e8",
                "--max_steps", "2", "--no-show_isco", "--show_periapsides",
            ])
        self.assertEqual(result.final_step, 2)
        self.assertEqual(solve.call_args.args[0].u_init, -1e8)
        self.assertEqual(solve.call_args.args[0].model, "newtonian")
        plot.assert_called_once_with(result, show_isco=False,
                                     show_periapsides=True)

    def test_documented_inside_isco_example_returns_outward(self):
        main_module = importlib.import_module("main")
        options = main_module.parse_args([
            "--x_init", "20000", "--u_init", "8e7",
            "--max_steps", "8000", "--show_periapsides",
        ])
        example = RelativisticOrbitParams(**{
            field.name: getattr(options, field.name)
            for field in fields(RelativisticOrbitParams)
        })
        result = integrate_relativistic_orbit(example)
        self.assertNotEqual(result.termination_reason, "horizon")
        self.assertTrue(result.periapsis_radius)
        self.assertLess(min(result.periapsis_radius), physics.ISCO_RADIUS)
        self.assertGreater(min(result.periapsis_radius), physics.HORIZON_RADIUS)


# ---------------------------------------------------------------------------
# Exact orbit prediction
# ---------------------------------------------------------------------------

def _independent_advance(x_init, u_init):
    """Apsidal advance of Eq. (11) by quadrature, with roots from numpy.

    The cubic of Eq. (10) is solved in u = 1/r with numpy.roots rather than
    the scaled quadratic used by predict_orbit(), and the angle of a radial
    period is integrated numerically with u = u1 + (u2 - u1) sin^2(theta),
    which removes the square-root end-point singularities.
    """
    import numpy as np

    gm, c2 = physics.GM_SUN, physics.C2
    h = x_init * u_init
    energy = physics.effective_specific_energy(x_init, 0.0, 0.0, u_init, h)
    roots = np.roots([2.0 * gm / c2, -1.0, 2.0 * gm / h**2, 2.0 * energy / h**2])
    u1, u2, u3 = sorted(float(r.real) for r in roots)
    a3 = 2.0 * gm / c2
    cells = 4000
    total = 0.0
    for k in range(cells):
        theta = (k + 0.5) * (math.pi / 2.0) / cells
        u = u1 + (u2 - u1) * math.sin(theta) ** 2
        total += 2.0 / math.sqrt(a3 * (u3 - u))
    sweep = 2.0 * total * (math.pi / 2.0) / cells
    return sweep - 2.0 * math.pi, 1.0 / u2, 1.0 / u1


class TestOrbitPrediction(unittest.TestCase):
    """predict_orbit(): the outside check that the Help's beats rely on."""

    def test_default_advance_agrees_with_the_dop853_oracle(self):
        prediction = physics.predict_orbit(15_000.0, 1.2e8)
        self.assertEqual(prediction.kind, "bound")
        self.assertEqual(prediction.periapsis_radius, 15_000.0)
        self.assertLess(
            abs(prediction.apsidal_advance
                - TestIndependentBenchmarksAndConvergence.DOP853_APSIDAL_ADVANCE),
            1e-11,
        )

    def test_advance_and_turning_points_agree_with_quadrature(self):
        for x_init, u_init in ((15_000.0, 1.2e8), (15_000.0, 1.1e8), (20_000.0, 8e7),
                               (1e6, 1.153e7), (7383.12519025, 212197265.2804),
                               (15_000.0, 1.045e8), (1e5, 3.8e7)):
            prediction = physics.predict_orbit(x_init, u_init)
            advance, periapsis, apoapsis = _independent_advance(x_init, u_init)
            with self.subTest(x_init=x_init, u_init=u_init):
                self.assertEqual(prediction.kind, "bound")
                self.assertAlmostEqual(prediction.apsidal_advance / advance, 1.0, delta=1e-9)
                self.assertAlmostEqual(prediction.periapsis_radius / periapsis, 1.0, delta=1e-9)
                self.assertAlmostEqual(prediction.apoapsis_radius / apoapsis, 1.0, delta=1e-9)

    def test_turning_points_are_where_the_effective_potential_equals_the_energy(self):
        gm, c2 = physics.GM_SUN, physics.C2
        for x_init, u_init, model in ((15_000.0, 1.2e8, "schwarzschild"),
                                      (20_000.0, 8e7, "schwarzschild"),
                                      (15_000.0, 1.2e8, "newtonian")):
            prediction = physics.predict_orbit(x_init, u_init, model)
            h = x_init * u_init
            energy = physics.effective_specific_energy(x_init, 0.0, 0.0, u_init, h, model)
            cubic = gm * h * h / c2 if model == "schwarzschild" else 0.0
            for radius in (prediction.periapsis_radius, prediction.apoapsis_radius):
                potential = -gm / radius + h * h / (2 * radius**2) - cubic / radius**3
                with self.subTest(model=model, radius=radius):
                    self.assertAlmostEqual(potential / energy, 1.0, delta=1e-11)

    def test_kinds_of_orbit(self):
        m = physics.GM_SUN / physics.C2
        cases = (
            ((15_000.0, 1.044e8), "plunge"),
            ((15_000.0, 1.8e8), "escape"),
            ((15_000.0, 0.0), "radial"),
            ((8 * m, physics.circular_proper_time_speed(8 * m)), "circular (stable)"),
            ((5 * m, physics.circular_proper_time_speed(5 * m)), "circular (unstable)"),
            ((5000.0, 2e8), "plunge"),
            ((5000.0, 4.83e8), "escape"),
            ((20_000.0, 7.95e7), "plunge"),
            ((20_000.0, 7.99e7), "bound"),
        )
        for (x_init, u_init), kind in cases:
            with self.subTest(x_init=x_init, u_init=u_init):
                self.assertEqual(physics.predict_orbit(x_init, u_init).kind, kind)

    def test_newtonian_predictions(self):
        bound = physics.predict_orbit(15_000.0, 1.2e8, "newtonian")
        self.assertEqual((bound.kind, bound.apsidal_advance, bound.weak_field_advance),
                         ("bound", 0.0, 0.0))
        p = (15_000.0 * 1.2e8) ** 2 / physics.GM_SUN  # semi-latus rectum h^2/GM
        self.assertAlmostEqual(p, 2 * 15_000.0 * bound.apoapsis_radius
                               / (15_000.0 + bound.apoapsis_radius), delta=1e-6)
        escape_speed = math.sqrt(2 * physics.GM_SUN / 15_000.0)
        self.assertEqual(physics.predict_orbit(15_000.0, 1.0001 * escape_speed, "newtonian").kind,
                         "escape")
        circle = physics.circular_proper_time_speed(15_000.0, "newtonian")
        self.assertEqual(circle, math.sqrt(physics.GM_SUN / 15_000.0))
        self.assertEqual(physics.predict_orbit(15_000.0, circle, "newtonian").kind,
                         "circular (stable)")

    def test_start_is_a_periapsis_above_the_circular_value_and_an_apoapsis_below(self):
        circle = physics.circular_proper_time_speed(15_000.0)
        faster = physics.predict_orbit(15_000.0, 1.001 * circle)
        slower = physics.predict_orbit(15_000.0, 0.999 * circle)
        self.assertEqual(faster.periapsis_radius, 15_000.0)
        self.assertGreater(faster.apoapsis_radius, 15_000.0)
        self.assertEqual(slower.apoapsis_radius, 15_000.0)
        self.assertLess(slower.periapsis_radius, 15_000.0)

    def test_marginal_orbit_between_plunge_and_bound(self):
        low, high = 1.044e8, 1.045e8
        kinds = set()
        for _ in range(80):
            middle = 0.5 * (low + high)
            kind = physics.predict_orbit(15_000.0, middle).kind
            kinds.add(kind)
            if kind == "plunge":
                low = middle
            elif kind == "bound":
                high = middle
            else:
                break
        self.assertAlmostEqual(low, 1.04444e8, delta=1e3)
        self.assertLessEqual(kinds, {"plunge", "bound", "marginal"})
        for value in (low, high):
            prediction = physics.predict_orbit(15_000.0, value)
            with self.subTest(u_init=value):
                if prediction.apsidal_advance is not None:
                    self.assertTrue(math.isfinite(prediction.apsidal_advance))

    def test_circular_advance_formula(self):
        m = physics.GM_SUN / physics.C2
        stable = physics.predict_orbit(8 * m, physics.circular_proper_time_speed(8 * m))
        self.assertAlmostEqual(stable.apsidal_advance, 2 * math.pi, places=9)
        self.assertAlmostEqual(stable.weak_field_advance, 0.75 * math.pi, places=9)
        unstable = physics.predict_orbit(5 * m, physics.circular_proper_time_speed(5 * m))
        self.assertIsNone(unstable.apsidal_advance)
        self.assertIsNone(unstable.weak_field_advance)
        nearly = physics.predict_orbit(20 * m, 1.00001 * physics.circular_proper_time_speed(20 * m))
        self.assertAlmostEqual(nearly.apsidal_advance,
                               2 * math.pi / math.sqrt(1 - 6 / 20) - 2 * math.pi, delta=1e-4)

    def test_weak_field_limit_and_second_order_term(self):
        m = physics.GM_SUN / physics.C2
        for x_init in (1e6, 1e8):
            u_init = 1.02 * physics.circular_proper_time_speed(x_init)
            prediction = physics.predict_orbit(x_init, u_init)
            rp, ra = prediction.periapsis_radius, prediction.apoapsis_radius
            p = 2 * rp * ra / (rp + ra)
            e = (ra - rp) / (ra + rp)
            second = (18 + e * e) / 4 * m / p
            ratio = prediction.apsidal_advance / prediction.weak_field_advance
            with self.subTest(x_init=x_init):
                self.assertAlmostEqual(prediction.weak_field_advance, 6 * math.pi * m / p, delta=1e-15)
                self.assertAlmostEqual((ratio - 1) / second, 1.0, delta=0.02)
        far = physics.predict_orbit(1e250, 1e-117)
        self.assertEqual(far.kind, "bound")
        self.assertEqual(far.apsidal_advance, far.weak_field_advance)

    def test_mercury_advance_is_43_arcseconds_per_century(self):
        prediction = physics.predict_orbit(4.6e10, 58980.0)
        arcseconds = prediction.apsidal_advance * 415 * 206265
        self.assertAlmostEqual(arcseconds, 43.0, delta=0.1)
        self.assertAlmostEqual(prediction.apsidal_advance / prediction.weak_field_advance, 1.0,
                               delta=1e-6)

    def test_elliptic_integral(self):
        self.assertAlmostEqual(physics._complete_elliptic_k(0.0), math.pi / 2, places=15)
        self.assertAlmostEqual(physics._complete_elliptic_k(0.5), 1.8540746773013719, places=14)
        for m in (1e-12, 1e-4, 0.05, 0.099, 0.2, 0.9):
            with self.subTest(m=m):
                self.assertAlmostEqual(
                    physics._elliptic_k_excess(m),
                    2 * physics._complete_elliptic_k(m) / math.pi - 1,
                    delta=1e-15 + 1e-12 * m,
                )
        with self.assertRaises(ValueError):
            physics._complete_elliptic_k(1.0)

    def test_prediction_never_fails_uncleanly(self):
        import random

        generator = random.Random(2026)
        for _ in range(20_000):
            x_init = 10 ** generator.uniform(math.log10(2954.0), 308)
            u_init = generator.choice((-1, 1)) * 10 ** generator.uniform(-320, 300)
            if generator.random() < 0.3:
                # Starts close to, but resolvably off, a circular orbit.
                circle = physics.circular_proper_time_speed(max(x_init, 4430.0))
                u_init = circle * (1 + generator.choice((-1, 1)) * 10 ** generator.uniform(-16, -2))
            model = generator.choice(("schwarzschild", "newtonian"))
            try:
                prediction = physics.predict_orbit(x_init, u_init, model)
            except ValueError:
                continue
            for value in prediction[1:]:
                self.assertTrue(value is None or math.isfinite(value),
                                (x_init, u_init, model, prediction))
        for bad in ((0.0, 1.0), (1000.0, 1.0), (math.nan, 1.0), (1.0e4, math.inf), (True, 1.0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                physics.predict_orbit(*bad)

    def test_prediction_is_immutable(self):
        prediction = physics.predict_orbit(15_000.0, 1.2e8)
        with self.assertRaises(AttributeError):
            prediction.kind = "escape"
        self.assertEqual(prediction._fields, ("kind", "periapsis_radius", "apoapsis_radius",
                                              "apsidal_advance", "weak_field_advance"))


def _sampled_kind(x_init, u_init):
    """Kind of orbit found by sampling the sign of the scaled cubic exactly.

    Independent of predict_orbit()'s discriminant logic: the orbit plunges if
    (w - 1) q(w) stays positive all the way from the start to the horizon,
    escapes if it stays positive all the way out to w = 0, and is otherwise
    bound.
    """
    x = Fraction(x_init)
    gm = Fraction(physics.GM_SUN)
    a = 2 * gm / (Fraction(physics.C2) * x)
    b = 2 * gm / (x * Fraction(u_init) ** 2)

    def f(w):
        return (w - 1) * (a * w * w + (a - 1) * w + a + b - 1)

    cells = 4000
    if f(1 + Fraction(1, 10**9)) > 0:
        horizon = x / Fraction(physics.HORIZON_RADIUS)
        inside = (1 + (horizon - 1) * Fraction(k, cells) for k in range(1, cells + 1))
        return "bound" if any(f(w) < 0 for w in inside) else "plunge"
    outside = (Fraction(k, cells) for k in range(1, cells))
    return "bound" if any(f(w) < 0 for w in outside) else "escape"


class TestPredictionAtBoundaries(unittest.TestCase):
    """The ISCO, the other kind boundaries, and very large radii."""

    def test_exact_isco_circle_is_marginally_stable(self):
        radius = physics.ISCO_RADIUS
        prediction = physics.predict_orbit(radius, physics.circular_proper_time_speed(radius))
        self.assertEqual(prediction.kind, "circular (marginally stable)")
        self.assertEqual((prediction.periapsis_radius, prediction.apoapsis_radius), (radius, radius))
        self.assertIsNone(prediction.apsidal_advance)
        self.assertIsNone(prediction.weak_field_advance)

    def test_circles_near_the_isco_are_labelled_consistently(self):
        radius = physics.ISCO_RADIUS
        for step in range(-2000, 2001, 7):
            x_init = radius * (1 + step * 1e-12)
            kind = physics.predict_orbit(x_init, physics.circular_proper_time_speed(x_init)).kind
            with self.subTest(step=step):
                if step > 1:
                    self.assertEqual(kind, "circular (stable)")
                elif step < -1:
                    self.assertEqual(kind, "circular (unstable)")
                else:
                    self.assertTrue(kind.startswith("circular"), kind)
        for factor, kind in ((1 + 1e-6, "circular (stable)"), (1 - 1e-6, "circular (unstable)")):
            x_init = radius * factor
            with self.subTest(factor=factor):
                self.assertEqual(
                    physics.predict_orbit(x_init, physics.circular_proper_time_speed(x_init)).kind, kind)

    def test_resolvable_changes_of_speed_are_not_circles(self):
        radius = physics.ISCO_RADIUS
        circle = physics.circular_proper_time_speed(radius)
        kinds = [physics.predict_orbit(radius, circle * factor).kind
                 for factor in (1 - 1e-13, 1.0, 1 + 1e-13)]
        self.assertEqual(kinds, ["plunge", "circular (marginally stable)", "bound"])
        m = physics.GM_SUN / physics.C2
        for radius in (5 * m, 8 * m, 1e6, 1e12):
            circle = physics.circular_proper_time_speed(radius)
            for factor in (1 - 1e-13, 1 + 1e-13):
                with self.subTest(radius=radius, factor=factor):
                    self.assertFalse(
                        physics.predict_orbit(radius, circle * factor).kind.startswith("circular"))
        # The 14-significant-digit value of the 8GM/c^2 circular speed is a
        # resolvable change of speed, so it is a (nearly circular) bound orbit.
        self.assertEqual(physics.predict_orbit(11813.0003044, 134071263.04595).kind, "bound")
        self.assertEqual(physics.predict_orbit(11813.0003044, 134071263.04595919).kind,
                         "circular (stable)")

    def test_circles_just_off_the_isco_are_not_the_isco(self):
        radius = physics.ISCO_RADIUS
        for factor, kind in ((1 - 1e-14, "circular (unstable)"), (1 + 1e-14, "circular (stable)")):
            x_init = radius * factor
            with self.subTest(factor=factor):
                self.assertNotEqual(x_init, radius)
                self.assertEqual(
                    physics.predict_orbit(x_init, physics.circular_proper_time_speed(x_init)).kind, kind)

    def test_the_program_s_own_circular_speed_is_always_recognised(self):
        import random

        generator = random.Random(22)
        for _ in range(2000):
            x_init = 10 ** generator.uniform(math.log10(4430.0), 300)
            with self.subTest(x_init=x_init):
                self.assertTrue(physics.predict_orbit(
                    x_init, physics.circular_proper_time_speed(x_init)).kind.startswith("circular"))
            x_newton = 10 ** generator.uniform(-3, 300)
            with self.subTest(x_newton=x_newton):
                self.assertEqual(physics.predict_orbit(
                    x_newton, physics.circular_proper_time_speed(x_newton, "newtonian"),
                    "newtonian").kind, "circular (stable)")
        for offset in (1e-9, 1e-6, 1e-3, 1.0, 100.0):
            x_init = physics.PHOTON_ORBIT_RADIUS + offset
            with self.subTest(offset=offset):
                self.assertEqual(physics.predict_orbit(
                    x_init, physics.circular_proper_time_speed(x_init)).kind, "circular (unstable)")

    def test_circular_speed_is_accurate_close_to_the_photon_orbit(self):
        exact_photon = 3 * Fraction(physics.GM_SUN) / Fraction(physics.C2)
        for offset in (1e-9, 1e-6, 1e-3, 1.0):
            radius = physics.PHOTON_ORBIT_RADIUS + offset
            exact = math.sqrt(float(Fraction(physics.GM_SUN) / (Fraction(radius) - exact_photon)))
            with self.subTest(offset=offset):
                self.assertAlmostEqual(physics.circular_proper_time_speed(radius) / exact, 1.0,
                                       delta=4e-16)

    def test_isco_circle_through_the_command_line(self):
        output = run_cli(("--x_init", "8859.750228300749", "--u_init", "173085256.32731956",
                          "--max_steps", "1", "--dt", "1e-8")).stdout
        self.assertIn("circular dy/dtau: 1.73085e+08 m/s at x_init", output)
        self.assertIn("motion          : circular (marginally stable), periapsis 8859.75 m", output)
        self.assertNotIn("plunge", output)
        self.assertNotIn("apsidal advance", output)

    def test_kinds_near_the_isco_agree_with_an_independent_sign_sampling(self):
        for offset in (-0.2, -1e-2, -1e-3, 1e-3, 1e-2, 0.5):
            x_init = physics.ISCO_RADIUS * (1 + offset)
            circle = physics.circular_proper_time_speed(x_init)
            for push in (-1e-2, -1e-3, -1e-4, 1e-4, 1e-3, 1e-2, 0.05):
                u_init = circle * (1 + push)
                with self.subTest(offset=offset, push=push):
                    self.assertEqual(physics.predict_orbit(x_init, u_init).kind,
                                     _sampled_kind(x_init, u_init))

    def test_kind_changes_once_across_the_plunge_boundary(self):
        # Consecutive representable starting values near 1.04444e8 m/s: the
        # kind must switch from plunge to bound once, never back and forth.
        low, high = 1.044e8, 1.045e8
        for _ in range(80):
            middle = 0.5 * (low + high)
            if physics.predict_orbit(15_000.0, middle).kind == "plunge":
                low = middle
            else:
                high = middle
        value = low
        for _ in range(200):
            value = math.nextafter(value, 0.0)
        kinds = []
        for _ in range(400):
            kinds.append(physics.predict_orbit(15_000.0, value).kind)
            value = math.nextafter(value, math.inf)
        order = {"plunge": 0, "marginal": 1, "bound": 2}
        ranks = [order[kind] for kind in kinds]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual((kinds[0], kinds[-1]), ("plunge", "bound"))

    def test_weak_field_advance_at_very_large_radii(self):
        m = physics.GM_SUN / physics.C2
        for radius in (1e155, 1e300, 1e308):
            with self.subTest(radius=radius):
                self.assertAlmostEqual(physics.weak_field_advance(radius, radius) / (6 * math.pi * m / radius),
                                       1.0, delta=1e-14)
        self.assertAlmostEqual(physics.weak_field_advance(15_000.0, 23954.438830190225),
                               6 * math.pi * m / (2 * 15_000.0 * 23954.438830190225
                                                  / (15_000.0 + 23954.438830190225)), delta=1e-14)
        for bad in ((0.0, 1.0), (-1.0, 1.0), (1.0, math.inf)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                physics.weak_field_advance(*bad)

    def test_matched_speed_and_radius_scales_give_finite_positive_advances(self):
        import random

        generator = random.Random(2125)
        for _ in range(3000):
            x_init = 10 ** generator.uniform(math.log10(1e4), 307.5)
            u_init = math.sqrt(physics.GM_SUN / x_init) * generator.uniform(0.75, 1.3)
            try:
                prediction = physics.predict_orbit(x_init, u_init)
            except ValueError:
                continue
            if prediction.kind != "bound":
                continue
            with self.subTest(x_init=x_init, u_init=u_init):
                self.assertGreater(prediction.weak_field_advance, 0.0)
                self.assertGreater(prediction.apsidal_advance, 0.0)
                self.assertTrue(math.isfinite(prediction.apsidal_advance))
                if x_init > 1e10:
                    self.assertAlmostEqual(
                        prediction.apsidal_advance / prediction.weak_field_advance, 1.0, delta=1e-5)
        far = physics.predict_orbit(1e155, 1.1 * math.sqrt(physics.GM_SUN / 1e155))
        self.assertAlmostEqual(far.weak_field_advance / far.apsidal_advance, 1.0, delta=1e-12)
        self.assertGreater(far.weak_field_advance, 0.0)

    def test_unrepresentable_turning_point_is_reported_cleanly(self):
        with self.assertRaisesRegex(ValueError, "too large to represent"):
            physics.predict_orbit(1e308, 0.9 * math.sqrt(2 * physics.GM_SUN / 1e308))
        cli = importlib.import_module("main")
        run_params = RelativisticOrbitParams(1e308, 1.0, 1e-6, 1, 1, 0.05, 1e-4)
        with mock.patch.object(physics, "predict_orbit", side_effect=ValueError("too big")):
            lines = cli.prediction_lines(run_params)
        self.assertEqual(lines[-1], "    motion          : not predicted (too big)")


class TestSciPyOracleRegeneration(unittest.TestCase):
    """Maintenance: regenerate the stored DOP853 oracle when SciPy is present."""

    def test_stored_oracle_is_reproducible(self):
        try:
            oracle = generate_dop853_oracle()
        except ImportError:
            self.skipTest("SciPy is not installed; the stored oracle values are used as they are")
        final, advance = oracle
        stored = TestIndependentBenchmarksAndConvergence.DOP853_FINAL_AT_TAU_0012
        for value, reference, scale in zip(final, stored, (1e4, 1e4, 1e8, 1e8)):
            self.assertAlmostEqual(value, reference, delta=1e-6 * scale)
        self.assertAlmostEqual(
            advance, TestIndependentBenchmarksAndConvergence.DOP853_APSIDAL_ADVANCE, delta=1e-10
        )


def generate_dop853_oracle():
    """Regenerate the independent DOP853 reference values stored in the tests.

    This is the oracle-generation procedure itself, kept with the tests so that
    it cannot be lost: the Cartesian first-order form of the default orbit,
    integrated with SciPy's DOP853 at rtol=2e-13, componentwise
    atol=(1e-8, 1e-8, 1e-5, 1e-5) and max_step=1e-6 s to tau=0.012 s.  The
    unwrapped angle is integrated alongside, as a fifth variable that does not
    feed back on the orbit, and the periapsides are the zeros of r.v at which
    it increases.  Returns (final (x, y, vx, vy), mean apsidal advance).

    Run from the program folder with, for example,
        python -c "from tests.test_physics_relativistic_orbit import generate_dop853_oracle as g; print(g())"
    """
    from scipy.integrate import solve_ivp

    x0, u0 = 15_000.0, 1.2e8
    h = x0 * u0
    gm, c2 = physics.GM_SUN, physics.C2

    def rhs(_tau, state):
        x, y, vx, vy, _phi = state
        r2 = x * x + y * y
        factor = -gm * (1.0 + 3.0 * h * h / (c2 * r2)) / (r2 * math.sqrt(r2))
        return [vx, vy, factor * x, factor * y, (x * vy - y * vx) / r2]

    def periapsis(_tau, state):
        return state[0] * state[2] + state[1] * state[3]

    periapsis.direction = 1.0
    solution = solve_ivp(
        rhs, (0.0, 0.012), [x0, 0.0, 0.0, u0, 0.0], method="DOP853",
        rtol=2e-13, atol=[1e-8, 1e-8, 1e-5, 1e-5, 1e-12], max_step=1e-6,
        events=periapsis,
    )
    angles = [state[4] for state in solution.y_events[0]]
    advances = [b - a - 2.0 * math.pi for a, b in zip(angles, angles[1:])]
    final = tuple(float(value) for value in solution.y[:4, -1])
    return final, sum(advances) / len(advances)


# ---------------------------------------------------------------------------
# The Beats Help file
# ---------------------------------------------------------------------------

BEAT_NUMBERS = tuple(range(0, 9))
EQUATION_COUNT = 22
VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
# Commands that the Help documents as being rejected by the program.
REJECTED_COMMANDS = {"python main.py --x_init 2000": "--x_init must lie outside the Schwarzschild horizon"}
HELP_HTML = HELP_FILE.read_text(encoding="utf-8")


def html_text(fragment: str) -> str:
    """Visible text of an HTML fragment, with entities decoded and spaces collapsed."""
    return " ".join(
        html.unescape(re.sub(r"</?[A-Za-z!][^>]*>", "", fragment)).split()
    )


def section_html(page: str, section_id: str) -> str:
    match = re.search(
        rf'<section id="{re.escape(section_id)}">(.*?)</section>', page, re.DOTALL
    )
    if match is None:
        raise AssertionError(f"section {section_id!r} not found in the Help file")
    return match.group(1)


def documented_commands(fragment: str) -> list:
    """Every ``python main.py ...`` command shown in a <pre> block."""
    commands = []
    for block in re.findall(r"<pre[^>]*>(.*?)</pre>", fragment, re.DOTALL):
        text = html.unescape(re.sub(r"<[^>]+>", "", block))
        for line in text.splitlines():
            line = " ".join(line.split())
            if line.startswith("python main.py"):
                commands.append(line)
    return commands


def command_arguments(command: str) -> tuple:
    return tuple(shlex.split(command)[2:])


class CliRun(NamedTuple):
    stdout: str
    stderr: str
    exit_code: object
    result: object


_CLI_CACHE = {}


def run_cli(arguments=()) -> CliRun:
    """Run ``main.main()`` in-process with the plot suppressed.

    The program has no randomness, so every run is cached.
    """
    key = tuple(arguments)
    if key in _CLI_CACHE:
        return _CLI_CACHE[key]
    cli = importlib.import_module("main")
    captured = {}
    real = cli.integrate_relativistic_orbit

    def capture(*args, **kwargs):
        captured["result"] = real(*args, **kwargs)
        return captured["result"]

    out, err = io.StringIO(), io.StringIO()
    exit_code = None
    with (
        mock.patch.object(cli, "integrate_relativistic_orbit", new=capture),
        mock.patch.object(cli, "plot_relativistic_orbit"),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        try:
            cli.main(list(key))
        except SystemExit as exc:
            exit_code = exc.code
    run = CliRun(out.getvalue(), err.getvalue(), exit_code, captured.get("result"))
    _CLI_CACHE[key] = run
    return run


def beat_run(command: str) -> CliRun:
    return run_cli(command_arguments(command))


class HelpStructure(HTMLParser):
    """Ids, links, table rows and tag balance of a Help page."""

    def __init__(self, page: str) -> None:
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.hrefs = []
        self.sidebar_hrefs = []
        self.section_ids = []
        self.scripts = []
        self.errors = []
        self.table_rows = []
        self._row = None
        self._cell = None
        self._stack = []
        self._in_nav = False
        self.feed(page)
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
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        if tag not in VOID_TAGS:
            self._stack.append(tag)

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag == "nav":
            self._in_nav = False
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.table_rows.append(self._row)
            self._row = None
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


class TestHelpStructure(unittest.TestCase):
    def test_help_file_is_html5_utf8_with_one_version_build(self):
        self.assertIn("<!DOCTYPE html>", HELP_HTML[:100])
        self.assertRegex(HELP_HTML[:500], r'<meta charset="utf-8"\s*/?>')
        self.assertEqual(STRUCTURE.ids.count("version_build"), 1)

    def test_help_file_is_the_beats_version_and_not_the_original(self):
        self.assertIn(HELP_FILE.name, HELP_FILENAMES)
        self.assertNotEqual(HELP_FILE.name, "RelativisticOrbit-original.html")
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

    def test_display_math_has_no_line_breaks_or_text_mode_commands(self):
        blocks = re.findall(r"\\\[(.*?)\\\]", HELP_HTML, re.DOTALL)
        self.assertGreaterEqual(len(blocks), EQUATION_COUNT)
        for block in blocks:
            with self.subTest(block=block[:40]):
                self.assertNotIn("\\\\", block)
                self.assertNotIn("\\texttt", block)
                self.assertEqual(block.count("{"), block.count("}"))

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
        if docs_root.name != "GFTGU-Documentation" or not (docs_root / "Orbit").is_dir():
            self.skipTest("the sibling documentation folders are not present in this layout")
        relative = [h for h in STRUCTURE.hrefs
                    if not h.startswith(("#", "http://", "https://", "mailto:"))]
        self.assertEqual(len(relative), 3)
        for href in relative:
            with self.subTest(href=href):
                self.assertTrue((HELP_FILE.parent / href).is_file(), href)


class TestHelpBeats(unittest.TestCase):
    TAGS = {"LAW": "law", "ODE": "ode", "DEFINITION": "def", "DERIVED": "der",
            "ALGORITHM": "alg"}

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
                self.assertEqual(body.count("Three tasks, in order."), 1)
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
        self.assertEqual(sorted(found), list(range(1, EQUATION_COUNT + 1)))
        beats = [found[n][1] for n in sorted(found)]
        self.assertEqual(beats, sorted(beats))
        for number, (kind, _, css) in found.items():
            with self.subTest(equation=number):
                self.assertEqual(self.TAGS[kind], css)
        self.assertIn("Twenty-two equations are numbered", html_text(section_html(HELP_HTML, "beats")))

    def test_tag_counts_match_the_reading_note(self):
        by_kind = {}
        for number, (kind, _, _) in self.equations().items():
            by_kind.setdefault(kind, []).append(number)
        self.assertEqual(sorted(by_kind["LAW"]), [1])
        self.assertEqual(sorted(by_kind["ODE"]), [2, 6])
        self.assertEqual(sorted(by_kind["ALGORITHM"]), [5, 18, 19, 20])
        note = html_text(section_html(HELP_HTML, "beats"))
        self.assertIn("Eq. (1) is the only one", note)
        self.assertIn("Eqs. (2) and (6)", note)
        self.assertIn("Eqs. (5), (18), (19) and (20)", note)

    def test_equation_citations_refer_to_numbered_equations(self):
        for section in STRUCTURE.section_ids:
            if section in ("related", "license"):
                continue
            text = html_text(section_html(HELP_HTML, section))
            for group in re.findall(r"Eqs?\.\s*\(([\d\s,and()to]+?)\)(?=[\s.,;:]|$)", text):
                for number in re.findall(r"\d+", group):
                    with self.subTest(section=section, equation=number):
                        self.assertIn(int(number), range(1, EQUATION_COUNT + 1))

    def test_equation_index_lists_every_numbered_equation_once_with_its_kind_and_beat(self):
        found = self.equations()
        rows = re.findall(
            r"<tr><td>\((\d+)\)</td><td><span class=\"kind kind-\w+\">(\w+)</span></td>"
            r"<td>.*?</td><td>(\d+)</td>",
            section_html(HELP_HTML, "equations"),
        )
        self.assertEqual([int(r[0]) for r in rows], list(range(1, EQUATION_COUNT + 1)))
        for number, kind, beat in rows:
            with self.subTest(equation=number):
                self.assertEqual((kind, int(beat)), found[int(number)][:2])

    def test_every_experiment_is_cited_by_a_beat_and_every_citation_exists(self):
        experiments = re.findall(r'<h3 id="exp(\d+)">', section_html(HELP_HTML, "experiments"))
        self.assertEqual(experiments, [str(n) for n in range(1, 13)])
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

    def python_snippet(self, marker):
        blocks = [
            html.unescape(re.sub(r"<[^>]+>", "", block))
            for block in re.findall(r"<pre>(.*?)</pre>", section_html(HELP_HTML, "experiments"), re.DOTALL)
            if marker in block
        ]
        self.assertEqual(len(blocks), 1)
        completed = subprocess.run(
            [sys.executable, "-c", blocks[0]], cwd=MODULE_DIR,
            text=True, capture_output=True, timeout=120,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return completed.stdout.splitlines()

    def test_experiment_12_script_reproduces_beat_0(self):
        (line,) = self.python_snippet("integrate_relativistic_orbit(params)")
        count, measured, predicted = line.split()
        default = run_cli(()).stdout
        self.assertEqual(int(count), 6)
        self.assertIn(f"{float(measured):.6g} rad", default)
        self.assertIn(f"apsidal advance : {float(predicted):.6g} rad", default)

    def test_experiment_6_script_finds_the_stability_boundary(self):
        lines = self.python_snippet("circular_proper_time_speed(r)")
        kinds = [line.split(" ", 3)[3].rsplit(" ", 1)[0] for line in lines]
        self.assertEqual(kinds, ["circular (unstable)", "circular (unstable)",
                                 "circular (stable)", "circular (stable)", "circular (stable)"])
        # The printed radius and speed are complete: typed back in as
        # --x_init and --u_init they start the same circular orbit.
        for line, kind in zip(lines, kinds):
            _, radius, speed, _ = line.split(" ", 3)
            with self.subTest(radius=radius):
                output = run_cli(("--x_init", radius, "--u_init", speed,
                                  "--max_steps", "1", "--dt", "1e-9")).stdout
                self.assertIn(f"motion          : {kind},", output)

    def test_printed_circular_speed_is_documented_as_rounded(self):
        # The summary shows the circular value to six significant figures; the
        # Help says that typing that value back in is a different orbit, with
        # the ISCO as its example, and that the full value starts a circle.
        radius = "8859.750228300749"
        shown = run_cli(("--x_init", radius, "--u_init", "1.8e8", "--max_steps", "1",
                         "--dt", "1e-9")).stdout
        printed = re.search(r"circular dy/dtau: (\S+) m/s at x_init", shown).group(1)
        self.assertEqual(printed, "1.73085e+08")
        typed = run_cli(("--x_init", radius, "--u_init", printed, "--max_steps", "1",
                         "--dt", "1e-9")).stdout
        self.assertIn("motion          : plunge", typed)
        full = repr(physics.circular_proper_time_speed(float(radius)))
        exact = run_cli(("--x_init", radius, "--u_init", full, "--max_steps", "1",
                         "--dt", "1e-9")).stdout
        self.assertIn("motion          : circular (marginally stable)", exact)
        self.assertEqual(float(radius), physics.ISCO_RADIUS)
        algorithm = html_text(section_html(HELP_HTML, "algorithm"))
        self.assertIn("six significant figures only", algorithm)
        self.assertIn("--x_init 8859.750228300749 --u_init 1.73085e8 is predicted to plunge", algorithm)
        self.assertNotIn("recognises the value printed", algorithm)
        self.assertIn("six significant figures", html_text(section_html(HELP_HTML, "summary")))


class TestHelpCommands(unittest.TestCase):
    def test_the_help_documents_a_meaningful_number_of_commands(self):
        self.assertGreaterEqual(len(dict.fromkeys(documented_commands(HELP_HTML))), 30)

    def test_every_documented_command_parses_and_runs_to_a_summary(self):
        for command in dict.fromkeys(documented_commands(HELP_HTML)):
            arguments = command_arguments(command)
            if "--help" in arguments:
                continue
            with self.subTest(command=command):
                run = run_cli(arguments)
                if command in REJECTED_COMMANDS:
                    self.assertEqual(run.exit_code, 2)
                    self.assertIn(REJECTED_COMMANDS[command], run.stderr)
                    self.assertEqual(run.stdout, "")
                else:
                    self.assertIn(run.exit_code, (None, 0), run.stderr)
                    lines = run.stdout.splitlines()
                    self.assertEqual(
                        lines[0],
                        f"RelativisticOrbit {physics.MODEL_VERSION} (build {physics.BUILD_ID}) summary",
                    )
                    self.assertIn("  predicted from the constants of motion:", lines)

    def test_documented_options_are_real_options(self):
        parser = importlib.import_module("main").build_parser()
        real = {s for action in parser._actions for s in action.option_strings}
        used = set(re.findall(r"(?<![\w-])(--[A-Za-z0-9_-]+)", "\n".join(documented_commands(HELP_HTML))))
        self.assertTrue(used)
        self.assertLessEqual(used, real)


class TestHelpReference(unittest.TestCase):
    def test_parameter_table_matches_the_parser_options_and_defaults(self):
        rows = {
            row[0]: row[1]
            for row in STRUCTURE.table_rows
            if len(row) == 3 and row[0].startswith("--")
        }
        options = {
            action.option_strings[0]: action
            for action in importlib.import_module("main").build_parser()._actions
            if action.option_strings and action.option_strings[0] not in ("-h", "--version")
        }
        self.assertEqual(set(rows), set(options))
        for name, action in options.items():
            with self.subTest(option=name):
                if isinstance(action.default, bool):
                    self.assertEqual(rows[name], "on" if action.default else "off")
                elif isinstance(action.default, str):
                    self.assertEqual(rows[name], action.default)
                else:
                    self.assertEqual(float(rows[name]), float(action.default))

    def test_printed_summary_blocks_are_exactly_what_beats_0_and_6_print(self):
        blocks = re.findall(r"<pre>(RelativisticOrbit .*?)</pre>", section_html(HELP_HTML, "summary"),
                            re.DOTALL)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(html.unescape(blocks[0]).strip(), run_cli(()).stdout.strip())
        plunge = documented_commands(section_html(HELP_HTML, "beat6"))[0]
        self.assertEqual(html.unescape(blocks[1]).strip(), beat_run(plunge).stdout.strip())

    def test_every_summary_label_is_described_in_the_help(self):
        text = html_text(section_html(HELP_HTML, "summary"))
        labels = set()
        for command in dict.fromkeys(documented_commands(HELP_HTML)):
            if command in REJECTED_COMMANDS or "--help" in command:
                continue
            for line in beat_run(command).stdout.splitlines()[1:]:
                if ":" in line:
                    labels.add(line.split(":", 1)[0].strip())
        self.assertGreaterEqual(len(labels), 17)
        for label in sorted(labels):
            with self.subTest(label=label):
                self.assertIn(label, text)

    def test_every_motion_wording_the_program_can_print_is_described(self):
        text = html_text(section_html(HELP_HTML, "summary"))
        cli = importlib.import_module("main")
        m = physics.GM_SUN / physics.C2
        starts = ((15_000.0, 1.8e8), (15_000.0, 0.0), (15_000.0, 1.044e8),
                  (5 * m, physics.circular_proper_time_speed(5 * m)),
                  (8 * m, physics.circular_proper_time_speed(8 * m)),
                  (physics.ISCO_RADIUS, physics.circular_proper_time_speed(physics.ISCO_RADIUS)))
        for x_init, u_init in starts:
            run_params = RelativisticOrbitParams(x_init, u_init, 1e-6, 1, 1, 0.05, 1e-4)
            motion = cli.prediction_lines(run_params)[2].split(": ", 1)[1]
            with self.subTest(motion=motion):
                self.assertIn(motion.split(",")[0].split(" (no")[0].split(" (zero")[0], text)

    def test_constant_table_matches_the_program(self):
        rows = {
            row[0]: row[1]
            for row in STRUCTURE.table_rows
            if len(row) == 3 and not row[0].startswith("--") and row[0] != "Name"
            and not row[0].startswith("(")
        }
        for name in ("GM_SUN", "C", "HORIZON_RADIUS", "PHOTON_ORBIT_RADIUS", "ISCO_RADIUS",
                     "CIRCULAR_ULPS", "ISCO_ULPS"):
            with self.subTest(name=name):
                self.assertAlmostEqual(float(rows[name]) / getattr(physics, name), 1.0, delta=2e-6)

    def test_code_identifiers_named_in_the_help_exist_in_the_program(self):
        import plot_relativistic_orbit as plot
        cli = importlib.import_module("main")
        modules = (physics, driver, plot, cli)
        names = set()
        for body in re.findall(r"<code>([^<]+)</code>", HELP_HTML):
            body = html.unescape(body)
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

    def test_help_version_python_and_formulas(self):
        self.assertRegex(HELP_HTML, r"Python\s+3\.10\s+or\s+later")
        for item in (
            r"1+\frac{3h^2}{c^2r^2}",
            r"\sqrt{\frac{GM}{r-3GM/c^2}}",
            r"\frac{6\pi GM}{c^2p}",
            r"N_{\rm rev}=\frac{|\Delta\phi_{\rm accumulated}|}{2\pi}",
            "Schwarzschild horizon crossed",
            "maximum accepted-step count reached",
            "requested revolution count reached",
        ):
            with self.subTest(item=item):
                self.assertIn(item, HELP_HTML)


class TestHelpQuotedNumbers(unittest.TestCase):
    """Every number the beats quote from a run is the number the run prints."""

    NUMBER = re.compile(r"(?<![\w.])\d+\.\d+(?:e[+-]\d+)?(?![\w%]|\.\d)")
    WHOLE = re.compile(r"(?<![\w.+-])\d{4,}(?![\w%]|\.\d)")
    # Numbers a student works out from printed values, or constants of the
    # equations; each is checked in TestHelpQuantitativeClaims.
    DERIVED = {
        0: {"499.805", "1.388", "6.69"},
        1: set(),
        2: {"0.00032", "2.4403696101582"},
        3: {"0.0000318", "0.0067", "0.080", "1.36", "1.62"},
        4: {"11813.0", "7383.1", "4.89", "4429.9"},
        5: set(),
        6: set(),
        7: {"4.00", "3.98", "0.00032", "0.00072", "0.00023", "0.00004",
            "0.0032", "0.0016", "0.0008"},
        8: {"5906.5", "0.0128", "0.0032"},
    }

    @staticmethod
    def visible(fragment):
        fragment = re.sub(r"<pre[^>]*>.*?</pre>", " ", fragment, flags=re.DOTALL)
        fragment = re.sub(r"\\\[.*?\\\]", " ", fragment, flags=re.DOTALL)
        fragment = re.sub(r"\\\(.*?\\\)", " ", fragment, flags=re.DOTALL)
        return html_text(fragment)

    def printed_for(self, fragment):
        chunks = [run_cli(()).stdout]
        for command in documented_commands(fragment):
            chunks.append(command)
            if command not in REJECTED_COMMANDS:
                chunks.append(beat_run(command).stdout)
            else:
                chunks.append(beat_run(command).stderr)
        return "\n".join(chunks)

    def test_numbers_quoted_in_each_beat_are_printed_by_that_beat_s_commands(self):
        for number in BEAT_NUMBERS:
            body = section_html(HELP_HTML, f"beat{number}")
            printed = self.printed_for(body)
            text = self.visible(body)
            tokens = set(self.NUMBER.findall(text)) | set(self.WHOLE.findall(text))
            with self.subTest(beat=number):
                self.assertGreaterEqual(len(tokens), 5)
                missing = sorted(
                    t for t in tokens - self.DERIVED[number]
                    if not re.search(rf"(?<![\d.]){re.escape(t)}(?![\d])", printed)
                )
                self.assertEqual(missing, [])
                self.assertEqual(sorted(self.DERIVED[number] - tokens), [])

    def test_numbers_quoted_in_the_reference_sections_are_printed_or_checked(self):
        every_output = "\n".join(
            beat_run(c).stdout for c in dict.fromkeys(documented_commands(HELP_HTML))
            if c not in REJECTED_COMMANDS and "--help" not in c
        )
        allowed = {"2953.25", "4429.88", "8859.75", "299792458.0", "7564.95", "9.40306",
                   "206265", "1915", "3.10",
                   # ISCO_RADIUS in full, checked in test_printed_circular_speed_is_documented_as_rounded
                   "8859.750228300749"}
        for section in ("overview", "beats", "algorithm", "modules", "quickstart",
                        "parameters", "output", "experiments"):
            text = self.visible(section_html(HELP_HTML, section))
            tokens = set(self.NUMBER.findall(text)) | set(self.WHOLE.findall(text))
            with self.subTest(section=section):
                self.assertEqual(sorted(t for t in tokens - allowed if t not in every_output), [])


class TestHelpQuantitativeClaims(unittest.TestCase):
    """Independent checks of statements in the beats that go beyond the printed digits."""

    @staticmethod
    def result(command):
        return beat_run(command).result

    @staticmethod
    def commands(number):
        return documented_commands(section_html(HELP_HTML, f"beat{number}"))

    def test_beat0_rosette_arithmetic(self):
        default = self.result("python main.py")
        degrees = math.degrees(default.mean_periapsis_advance)
        self.assertEqual(f"{360 + degrees:.3f}", "499.805")
        per_period = (2 * math.pi + default.mean_periapsis_advance) / (2 * math.pi)
        self.assertEqual(f"{per_period:.3f}", "1.388")
        self.assertEqual(f"{default.n_orbits / per_period:.2f}", "6.69")
        self.assertAlmostEqual(15_000.0 / physics.HORIZON_RADIUS, 5.0, delta=0.1)
        self.assertEqual(len(default.periapsis_radius), 6)

    def test_beat1_extra_term_and_newtonian_eccentricity(self):
        self.assertEqual(f"{3 * 1.2e8**2 / physics.C2:.3f}", "0.481")
        newton = physics.predict_orbit(15_000.0, 1.2e8, "newtonian")
        e = (newton.apoapsis_radius - 15_000.0) / (newton.apoapsis_radius + 15_000.0)
        self.assertEqual(f"{e:.3f}", "0.628")
        run = self.result(self.commands(1)[0])
        self.assertEqual(len(run.periapsis_radius), 2)

    def test_beat2_prediction_and_start_direction(self):
        default = self.result("python main.py")
        exact = physics.predict_orbit(15_000.0, 1.2e8).apsidal_advance
        self.assertEqual(f"{exact:.13f}", "2.4403696101582")
        self.assertEqual(f"{exact - default.mean_periapsis_advance:.5f}", "0.00032")
        second = self.result(self.commands(2)[1])
        self.assertEqual(len(second.periapsis_radius), 6)
        self.assertLess(1.1e8, physics.circular_proper_time_speed(15_000.0))

    def test_beat3_weak_field_numbers(self):
        m = physics.GM_SUN / physics.C2
        prediction = physics.predict_orbit(1e6, 1.153e7)
        rp, ra = prediction.periapsis_radius, prediction.apoapsis_radius
        p = 2 * rp * ra / (rp + ra)
        self.assertAlmostEqual(p, 997272, delta=1)
        self.assertEqual(f"{m / p:.5f}", "0.00148")
        e = (ra - rp) / (ra + rp)
        self.assertEqual(f"{(18 + e * e) / 4 * m / p:.4f}", "0.0067")
        self.assertEqual(f"{prediction.apsidal_advance / prediction.weak_field_advance - 1:.4f}", "0.0067")
        self.assertEqual(f"{1 - prediction.weak_field_advance / prediction.apsidal_advance:.3f}", "0.007")
        run = self.result(self.commands(3)[0])
        self.assertEqual(f"{prediction.apsidal_advance - run.mean_periapsis_advance:.7f}", "0.0000318")
        control = self.result(self.commands(3)[1])
        self.assertAlmostEqual(control.mean_periapsis_advance / prediction.apsidal_advance, 0.01, delta=0.003)
        default = physics.predict_orbit(15_000.0, 1.2e8)
        p0 = 2 * 15_000.0 * default.apoapsis_radius / (15_000.0 + default.apoapsis_radius)
        e0 = (default.apoapsis_radius - 15_000.0) / (default.apoapsis_radius + 15_000.0)
        self.assertEqual(f"{m / p0:.3f}", "0.080")
        self.assertEqual(f"{1 + (18 + e0 * e0) / 4 * m / p0:.2f}", "1.36")
        self.assertEqual(f"{default.apsidal_advance / default.weak_field_advance:.2f}", "1.62")
        self.assertEqual(f"{default.weak_field_advance / default.apsidal_advance:.2f}", "0.62")
        self.assertAlmostEqual(1e6 / physics.HORIZON_RADIUS, 339, delta=0.5)

    def test_beat4_circular_numbers(self):
        m = physics.GM_SUN / physics.C2
        self.assertEqual(f"{8 * m:.1f}", "11813.0")
        self.assertEqual(f"{5 * m:.1f}", "7383.1")
        self.assertEqual(f"{3 * m:.1f}", "4429.9")
        self.assertAlmostEqual(212197265.2804 / physics.circular_proper_time_speed(7383.12519025),
                               1.001, delta=1e-6)
        self.assertEqual(134071263.04595919, physics.circular_proper_time_speed(11813.0003044))
        prediction = physics.predict_orbit(7383.12519025, 212197265.2804)
        self.assertEqual(f"{(prediction.apsidal_advance + 2 * math.pi) / (2 * math.pi):.2f}", "4.89")
        self.assertAlmostEqual(prediction.apoapsis_radius / 7383.12519025, 2.0, delta=0.05)

    def test_beat5_barrier_numbers(self):
        gm, c = physics.GM_SUN, physics.C
        h = 20_000.0 * 8e7
        root = math.sqrt(1 - 12 * (gm / (c * h)) ** 2)
        inner, outer = h * h / (2 * gm) * (1 - root), h * h / (2 * gm) * (1 + root)
        self.assertEqual(f"{inner:.1f}", "6893.0")
        self.assertEqual(f"{outer:.1f}", "12396.8")
        self.assertEqual(f"{math.sqrt(12) * gm / c:.5e}", "1.53349e+12")
        energy = physics.effective_specific_energy(20_000.0, 0.0, 0.0, 8e7, h)
        peak = -gm / inner + h * h / (2 * inner**2) - gm * h * h / (physics.C2 * inner**3)
        self.assertLess(energy, peak)
        coarse, fine = (self.result(c) for c in self.commands(5))
        prediction = physics.predict_orbit(20_000.0, 8e7)
        self.assertAlmostEqual(prediction.periapsis_radius - min(coarse.periapsis_radius), 89, delta=1)
        self.assertAlmostEqual(coarse.mean_periapsis_advance / prediction.apsidal_advance, 1.07, delta=0.005)
        default = self.result("python main.py")
        self.assertEqual(
            round(coarse.max_fractional_energy_drift / default.max_fractional_energy_drift), 32)
        self.assertLess(abs(fine.periapsis_radius[0] - prediction.periapsis_radius), 1.0)

    def test_beat6_plunge_boundary_and_barrier(self):
        gm, c = physics.GM_SUN, physics.C
        low, high = 1.044e8, 1.045e8
        for _ in range(60):
            middle = 0.5 * (low + high)
            if physics.predict_orbit(15_000.0, middle).kind == "plunge":
                low = middle
            else:
                high = middle
        self.assertEqual(f"{low:.5e}".replace("e+0", "e"), "1.04444e8")
        for u_init, above in ((1.044e8, True), (1.045e8, False)):
            h = 15_000.0 * u_init
            inner = h * h / (2 * gm) * (1 - math.sqrt(1 - 12 * (gm / (c * h)) ** 2))
            peak = -gm / inner + h * h / (2 * inner**2) - gm * h * h / (physics.C2 * inner**3)
            energy = physics.effective_specific_energy(15_000.0, 0.0, 0.0, u_init, h)
            with self.subTest(u_init=u_init):
                self.assertEqual(energy > peak, above)
        plunge, whirl, default = (self.result(c) for c in self.commands(6))
        self.assertEqual(plunge.termination_reason, "horizon")
        self.assertEqual(whirl.termination_reason, "max_steps")
        self.assertEqual(default.termination_reason, "horizon")
        self.assertEqual(physics.predict_orbit(15_000.0, 1.045e8).kind, "bound")
        self.assertGreater(physics.predict_orbit(15_000.0, 1.045e8).apsidal_advance, 4 * math.pi)

    def test_beat7_drift_ratios_errors_and_sampling_bounds(self):
        exact = physics.predict_orbit(15_000.0, 1.2e8).apsidal_advance
        runs = [self.result("python main.py")] + [self.result(c) for c in self.commands(7)]
        drifts = [r.max_fractional_energy_drift for r in runs]
        self.assertEqual(f"{drifts[0] / drifts[1]:.2f}", "4.00")
        self.assertEqual(f"{drifts[1] / drifts[2]:.2f}", "3.98")
        errors = [f"{r.mean_periapsis_advance - exact:+.5f}" for r in runs]
        self.assertEqual(errors, ["-0.00032", "+0.00072", "-0.00023", "-0.00004"])
        omega = 15_000.0 * 1.2e8 / 15_000.0**2
        self.assertEqual(omega, 8000.0)
        bounds = []
        for run, dt in zip(runs, (2e-6, 1e-6, 5e-7, 2e-7)):
            bound = omega * dt / (len(run.periapsis_radius) - 1)
            bounds.append(round(bound, 5))
            with self.subTest(dt=dt):
                self.assertLess(abs(run.mean_periapsis_advance - exact), bound)
                steps = [b - a for a, b in zip(run.tau, run.tau[1:])]
                self.assertLessEqual(max(steps), dt * (1 + 1e-9))
        self.assertEqual(bounds, [0.0032, 0.0016, 0.0008, 0.00032])

    def test_beat8_escape_and_mercury_numbers(self):
        m = physics.GM_SUN / physics.C2
        self.assertEqual(f"{4 * m:.1f}", "5906.5")
        u_escape = math.sqrt(2 * physics.GM_SUN / (15_000.0 * (1 - 2 * m / 15_000.0)))
        self.assertEqual(f"{u_escape:.5e}", "1.48435e+08")
        self.assertEqual(f"{math.sqrt(2 * physics.GM_SUN / 15_000.0):.5e}", "1.33023e+08")
        self.assertEqual(physics.predict_orbit(15_000.0, 1.0001 * u_escape).kind, "escape")
        self.assertEqual(physics.predict_orbit(15_000.0, 0.9999 * u_escape).kind, "bound")
        # Inside 4GM/c^2 the escape value is below the circular value.
        self.assertLess(math.sqrt(2 * physics.GM_SUN / (5000.0 * (1 - 2 * m / 5000.0))),
                        physics.circular_proper_time_speed(5000.0))
        omega = 58980.0 / 4.6e10
        self.assertEqual(f"{omega * 1e4:.4f}", "0.0128")
        _, relativistic, newtonian, _ = self.commands(8)
        gr, newton = self.result(relativistic), self.result(newtonian)
        self.assertEqual(len(gr.periapsis_radius), 5)
        self.assertEqual(f"{omega * 1e4 / 4:.4f}", "0.0032")
        signal = physics.predict_orbit(4.6e10, 58980.0).apsidal_advance
        for run in (gr, newton):
            self.assertAlmostEqual(run.mean_periapsis_advance / signal, 240, delta=5)
        self.assertGreater(abs(gr.mean_periapsis_advance - newton.mean_periapsis_advance), signal)
        step_needed = signal * 4 / omega
        self.assertLess(step_needed, 2.0)
        period = gr.tau[-1] / gr.n_orbits
        self.assertAlmostEqual(period / step_needed / 1e6, 5, delta=0.3)

        # The integration error alone, free of periapsis sampling: the turn of
        # the Laplace-Runge-Lenz vector of the Newtonian control per orbit.
        def lrl_angle(run, i):
            x, y, vx, vy = run.x[i], run.y[i], run.vx[i], run.vy[i]
            h = x * vy - y * vx
            r = math.hypot(x, y)
            return math.atan2(-vx * h / physics.GM_SUN - y / r, vy * h / physics.GM_SUN - x / r)

        drift = (lrl_angle(newton, -1) - lrl_angle(newton, 0)) / newton.n_orbits
        self.assertGreater(abs(drift), 100 * signal)


class TestNewFeatures(unittest.TestCase):
    """Prediction lines, frozen results and the extracted parser."""

    def test_results_are_frozen_tuples(self):
        from dataclasses import FrozenInstanceError

        result = integrate_relativistic_orbit(params(max_steps=20, max_orbits=10))
        for name in ("x", "y", "vx", "vy", "tau", "azimuth_unwrapped", "periapsis_indices",
                     "periapsis_tau", "periapsis_radius", "periapsis_azimuth"):
            with self.subTest(name=name):
                self.assertIsInstance(getattr(result, name), tuple)
        with self.assertRaises(FrozenInstanceError):
            result.n_orbits = 0.0

    def test_build_parser_defaults_follow_the_editable_params(self):
        cli = importlib.import_module("main")
        changed = params(x_init=12_345.0, max_steps=77)
        with mock.patch.object(cli, "params", changed):
            options = cli.build_parser().parse_args([])
        self.assertEqual((options.x_init, options.max_steps), (12_345.0, 77))

    def test_prediction_lines_for_each_kind(self):
        cli = importlib.import_module("main")
        m = physics.GM_SUN / physics.C2
        expectations = (
            ((15_000.0, 1.2e8, "schwarzschild"),
             ("bound, periapsis 15000 m, apoapsis 23954.4 m", "apsidal advance", "weak-field")),
            ((15_000.0, 1.2e8, "newtonian"), ("apsidal advance : 0 rad = 0 deg",)),
            ((15_000.0, 1.044e8, "schwarzschild"), ("plunge (no inner turning point",)),
            ((15_000.0, 1.8e8, "schwarzschild"), ("escape (no outer turning point)",)),
            ((15_000.0, 0.0, "schwarzschild"), ("radial (zero angular momentum)",)),
            ((4000.0, 1e8, "schwarzschild"), ("circular dy/dtau: none",)),
            ((5 * m, physics.circular_proper_time_speed(5 * m), "schwarzschild"),
             ("circular (unstable)",)),
        )
        for (x_init, u_init, model), fragments in expectations:
            run_params = RelativisticOrbitParams(x_init, u_init, 1e-6, 1, 1, 0.05, 1e-4, model)
            text = "\n".join(cli.prediction_lines(run_params))
            for fragment in fragments:
                with self.subTest(x_init=x_init, u_init=u_init, model=model, fragment=fragment):
                    self.assertIn(fragment, text)
            if "plunge" in text or "escape" in text or "unstable" in text:
                self.assertNotIn("apsidal advance", text)
                self.assertNotIn("weak-field", text)
            if model == "newtonian":
                self.assertNotIn("weak-field", text)

    def test_marginal_orbit_is_printed(self):
        cli = importlib.import_module("main")
        marginal = physics.OrbitPrediction("marginal", 7354.47, 15_000.0, None, None)
        run_params = RelativisticOrbitParams(15_000.0, 1.04443887e8, 1e-6, 1, 1, 0.05, 1e-4)
        with mock.patch.object(physics, "predict_orbit", return_value=marginal):
            text = "\n".join(cli.prediction_lines(run_params))
        self.assertIn("marginal (whirls toward the unstable circular orbit at 7354.47 m)", text)

    def test_summary_reports_periapsis_radius_range_only_when_found(self):
        with_periapsides = run_cli(()).stdout
        self.assertIn("  periapsis radius  : min 14999.8 m, max 15000 m", with_periapsides)
        none_found = run_cli(("--u_init", "1.8e8", "--max_steps", "200")).stdout
        self.assertIn("periapsides found : 0", none_found)
        self.assertNotIn("periapsis radius", none_found)


class TestEndToEndSubprocess(unittest.TestCase):
    """The program runs as a student would run it, with a non-interactive backend."""

    def run_main(self, *arguments):
        environment = {**os.environ, "MPLBACKEND": "Agg"}
        return subprocess.run(
            [sys.executable, "main.py", *arguments], cwd=MODULE_DIR, env=environment,
            capture_output=True, text=True, timeout=120,
        )

    def test_default_run_prints_the_same_summary_as_in_process(self):
        completed = self.run_main()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, run_cli(()).stdout)

    def test_newtonian_plunge_and_rejected_runs(self):
        newton = self.run_main("--model", "newtonian", "--max_steps", "50")
        self.assertEqual(newton.returncode, 0, newton.stderr)
        self.assertIn("apsidal advance : 0 rad", newton.stdout)
        plunge = self.run_main("--u_init", "1e8")
        self.assertEqual(plunge.returncode, 0, plunge.stderr)
        self.assertIn("Schwarzschild horizon crossed", plunge.stdout)
        rejected = self.run_main("--x_init", "2000")
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("--x_init must lie outside the Schwarzschild horizon", rejected.stderr)


class TestOriginalHelpCompatibility(unittest.TestCase):
    """The Reference Guide version is optional; these tests never require it."""

    @classmethod
    def setUpClass(cls):
        cls.original = HELP_FILE.with_name("RelativisticOrbit-original.html")
        if not cls.original.is_file():
            raise unittest.SkipTest("RelativisticOrbit-original.html is not present in this layout")
        cls.text = cls.original.read_text(encoding="utf-8")

    def test_original_stamp_matches_the_program(self):
        match = re.search(
            r'<p\s+id=["\']version_build["\'][^>]*>(.*?)</p>', self.text, re.DOTALL
        )
        self.assertIsNotNone(match)
        visible = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", match.group(1))).split())
        self.assertIn(f"Version {physics.MODEL_VERSION}", visible)
        self.assertIn(f"Build {physics.BUILD_ID}", visible)

    def test_original_parameter_defaults_still_match_main(self):
        parser = _HelpStructureParser()
        parser.feed(self.text)
        rows = {row[0]: row[1] for row in parser.rows if len(row) >= 2}
        cli = importlib.import_module("main")
        documented = {
            "x_init": float(rows["x_init"].split()[0]),
            "u_init": float(rows["u_init"].split()[0]),
            "dt": float(rows["dt"].split()[0]),
            "max_steps": int(rows["max_steps"]),
            "max_orbits": int(rows["max_orbits"]),
            "eps1": float(rows["eps1"]),
            "eps2": float(rows["eps2"]),
        }
        for name, value in documented.items():
            with self.subTest(name=name):
                self.assertEqual(value, getattr(cli.params, name))
        self.assertEqual(rows["model"].strip('"'), cli.params.model)

    def test_original_commands_still_run(self):
        commands = documented_commands(self.text)
        self.assertGreaterEqual(len(commands), 10)
        for command in dict.fromkeys(commands):
            with self.subTest(command=command):
                run = run_cli(command_arguments(command))
                self.assertIn(run.exit_code, (None, 0), run.stderr)

    def test_original_describes_the_new_summary_and_the_corrected_exercise(self):
        text = html_text(self.text)
        self.assertIn("predicted from the constants of motion", text)
        self.assertNotIn("approximately 7476 m", text)
        self.assertIn("7565 m", text)
        weak = [c for c in documented_commands(self.text) if "--x_init 1e6" in c]
        self.assertEqual(len(weak), 2)
        for command in weak:
            with self.subTest(command=command):
                self.assertIsNotNone(run_cli(command_arguments(command)).result.mean_periapsis_advance)


if __name__ == "__main__":
    unittest.main(verbosity=2)
