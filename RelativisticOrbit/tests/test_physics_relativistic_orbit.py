"""Regression tests for the complete RelativisticOrbit module.

The discovery logic deliberately supports both repository layout

    RelativisticOrbit/tests/test_physics_relativistic_orbit.py

and a flattened review/upload layout in which this test file is placed beside
the four program modules and RelativisticOrbit.html.
"""

from __future__ import annotations

import contextlib
import hashlib
import html
import importlib
import io
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


CORE_MODULE_FILENAMES = (
    "physics_relativistic_orbit.py",
    "driver_relativistic_orbit.py",
    "main.py",
    "plot_relativistic_orbit.py",
)
HELP_FILENAME = "RelativisticOrbit.html"


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


MODULE_DIR = find_module_dir(Path(__file__).resolve().parent)
HELP_FILE = MODULE_DIR / HELP_FILENAME
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import driver_relativistic_orbit as driver  # noqa: E402
import physics_relativistic_orbit as physics  # noqa: E402
from driver_relativistic_orbit import (  # noqa: E402
    RelativisticOrbitParams,
    integrate_relativistic_orbit,
)


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

    def test_help_has_no_hidden_control_characters(self):
        text = HELP_FILE.read_text(encoding="utf-8")
        bad = [char for char in text if ord(char) < 32 and char not in "\t\n\r"]
        self.assertEqual(bad, [])

    def test_help_revolution_formula_is_intact(self):
        text = HELP_FILE.read_text(encoding="utf-8")
        self.assertIn(r"N_{\rm rev}=\frac{|\Delta\phi_{\rm accumulated}|}{2\pi}", text)

    def test_help_avoids_deprecated_porting_history(self):
        text = HELP_FILE.read_text(encoding="utf-8").lower()
        for phrase in ("porting error", "previous python", "revised code", "no longer silently"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, text)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
