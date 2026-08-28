"""Regression tests for the complete Binary teaching module.

The discovery logic supports both the canonical tests subdirectory and a
review upload in which this file is flattened beside the four core modules.
"""

from __future__ import annotations

import ast
import hashlib
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
    "physics_binary.py",
    "driver_binary.py",
    "main.py",
    "plot_binary.py",
)


def find_module_dir(start: Path) -> Path:
    """Find the nearest ancestor containing every Binary core module."""
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if all((directory / name).is_file() for name in CORE_MODULE_FILENAMES):
            return directory
    names = ", ".join(CORE_MODULE_FILENAMES)
    raise FileNotFoundError(
        f"Could not find a directory containing all Binary core modules: {names}"
    )


MODULE_DIR = find_module_dir(Path(__file__))
HELP_FILE = MODULE_DIR / "Binary.html"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import driver_binary as driver  # noqa: E402
import physics_binary as physics  # noqa: E402
import plot_binary as plotting  # noqa: E402


DEFAULTS = {
    "MA": 2.0e30,
    "MB": 2.0e30,
    "xInitA": 4.6e10,
    "yInitA": 0.0,
    "vInitA": 0.0,
    "uInitA": 13000.0,
    "xInitB": -4.6e10,
    "yInitB": 0.0,
    "vInitB": 0.0,
    "uInitB": -13000.0,
    "dt": 2000.0,
    "max_steps": 10000,
    "eps1": 0.05,
    "eps2": 1.0e-4,
    "stop_after_one_orbit": True,
}


def integrate(**changes):
    parameters = dict(DEFAULTS)
    parameters.update(changes)
    return driver.integrate_binary(**parameters)


def energy_drift(result):
    initial = result.E[0]
    return max(abs(value - initial) for value in result.E) / abs(initial)


class TestModuleDiscovery(unittest.TestCase):
    def test_canonical_layout(self):
        self.assertEqual(find_module_dir(Path(__file__).parent), MODULE_DIR)

    def test_flattened_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in CORE_MODULE_FILENAMES:
                (root / name).touch()
            test_file = root / "test_physics_binary.py"
            test_file.touch()
            self.assertEqual(find_module_dir(test_file), root.resolve())

    def test_nearest_matching_ancestor_wins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "outer" / "inner"
            nested.mkdir(parents=True)
            for directory in (root, root / "outer"):
                for name in CORE_MODULE_FILENAMES:
                    (directory / name).touch()
            self.assertEqual(find_module_dir(nested), (root / "outer").resolve())

    def test_missing_set_raises(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "all Binary core modules"):
                find_module_dir(Path(temporary))


class TestVersionAndBuild(unittest.TestCase):
    def test_build_coverage_is_exact(self):
        self.assertEqual(physics.BUILD_ID_COVERS, CORE_MODULE_FILENAMES)

    def test_metadata_formats(self):
        self.assertRegex(physics.MODEL_VERSION, r"^\d+\.\d+\.\d+$")
        self.assertRegex(physics.BUILD_ID, r"^[0-9a-f]{12}$")

    def test_build_id_matches_core_sources(self):
        digest = hashlib.sha256()
        for name in CORE_MODULE_FILENAMES:
            with (MODULE_DIR / name).open(
                "r", encoding="utf-8", newline=None
            ) as source:
                content = source.read().encode("utf-8")
            digest.update(name.encode("utf-8"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        self.assertEqual(physics.BUILD_ID, digest.hexdigest()[:12])

    def test_result_carries_metadata(self):
        result = integrate(max_steps=1, stop_after_one_orbit=False)
        self.assertEqual(result.model_version, physics.MODEL_VERSION)
        self.assertEqual(result.build_id, physics.BUILD_ID)

    def test_command_line_version(self):
        completed = subprocess.run(
            [sys.executable, "main.py", "--version"],
            cwd=MODULE_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        expected = f"Binary {physics.MODEL_VERSION} (build {physics.BUILD_ID})"
        self.assertEqual(completed.stdout.strip(), expected)

    def test_python_310_syntax(self):
        for name in CORE_MODULE_FILENAMES:
            with self.subTest(name=name):
                source = (MODULE_DIR / name).read_text(encoding="utf-8")
                ast.parse(source, filename=name, feature_version=(3, 10))


class TestPhysicsFunctions(unittest.TestCase):
    def test_relative_displacement(self):
        self.assertEqual(
            physics.relative_displacement(4.0, 6.0, 1.0, 2.0),
            (3.0, 4.0, 5.0, 125.0),
        )

    def test_translation_invariance(self):
        first = physics.relative_displacement(4.0, 6.0, 1.0, 2.0)
        second = physics.relative_displacement(14.0, -1.0, 11.0, -5.0)
        self.assertEqual(first, second)

    def test_zero_separation(self):
        with self.assertRaisesRegex(ValueError, "zero separation"):
            physics.relative_displacement(1.0, 2.0, 1.0, 2.0)

    def test_bad_coordinates(self):
        for index in range(4):
            for bad in (math.nan, math.inf, -math.inf, True, "1.0"):
                values = [1.0, 2.0, -3.0, -4.0]
                values[index] = bad
                with self.subTest(index=index, bad=bad):
                    with self.assertRaises(ValueError):
                        physics.relative_displacement(*values)

    def test_unrepresentable_separations(self):
        cases = (
            (1.0e308, 0.0, -1.0e308, 0.0),
            (5.0e-324, 0.0, 0.0, 0.0),
            (1.0e150, 0.0, 0.0, 0.0),
        )
        for coordinates in cases:
            with self.subTest(coordinates=coordinates):
                with self.assertRaisesRegex(ValueError, "numerical range"):
                    physics.relative_displacement(*coordinates)

    def test_accelerations_and_signs(self):
        mass_a, mass_b, separation = 2.0e30, 3.0e30, 4.0e10
        ax_a, ay_a, ax_b, ay_b = physics.accelerations(
            mass_a, mass_b, separation / 2, 0.0, -separation / 2, 0.0
        )
        expected_a = -physics.G * mass_b / separation**2
        expected_b = physics.G * mass_a / separation**2
        self.assertAlmostEqual(ax_a, expected_a, delta=abs(expected_a) * 1e-15)
        self.assertAlmostEqual(ax_b, expected_b, delta=abs(expected_b) * 1e-15)
        self.assertEqual((ay_a, ay_b), (0.0, 0.0))

    def test_newtons_third_law(self):
        mass_a, mass_b = 2.3e30, 7.1e29
        ax_a, ay_a, ax_b, ay_b = physics.accelerations(
            mass_a, mass_b, 7e10, -2e10, -1e10, 3e10
        )
        scale = max(abs(mass_a * ax_a), abs(mass_a * ay_a))
        self.assertAlmostEqual(
            mass_a * ax_a + mass_b * ax_b, 0.0, delta=scale * 2e-15
        )
        self.assertAlmostEqual(
            mass_a * ay_a + mass_b * ay_b, 0.0, delta=scale * 2e-15
        )

    def test_inverse_square_scaling(self):
        near = physics.accelerations(2.0, 3.0, 1.0, 0.0, 0.0, 0.0)
        far = physics.accelerations(2.0, 3.0, 2.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(abs(far[0] / near[0]), 0.25, places=15)
        self.assertAlmostEqual(abs(far[2] / near[2]), 0.25, places=15)

    def test_invalid_masses(self):
        for bad in (0.0, -1.0, math.nan, math.inf, True, "2"):
            for name in ("MA", "MB"):
                masses = {"MA": 2.0, "MB": 3.0}
                masses[name] = bad
                with self.subTest(name=name, bad=bad):
                    with self.assertRaises(ValueError):
                        physics.accelerations(
                            masses["MA"], masses["MB"], 1.0, 0.0, 0.0, 0.0
                        )

    def test_energy_components(self):
        potential, kinetic, total = physics.energies(
            2.0, 3.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 4.0
        )
        self.assertAlmostEqual(potential, -physics.G * 6.0 / 5.0)
        self.assertAlmostEqual(kinetic, 49.0)
        self.assertAlmostEqual(total, kinetic + potential)

    def test_energy_symmetries(self):
        first = physics.energies(
            2.0, 3.0, 3.0, 4.0, 5.0, 6.0, -1.0, 2.0, -7.0, 8.0
        )
        swapped = physics.energies(
            3.0, 2.0, -1.0, 2.0, -7.0, 8.0, 3.0, 4.0, 5.0, 6.0
        )
        translated = physics.energies(
            2.0, 3.0, 103.0, -46.0, 5.0, 6.0, 99.0, -48.0, -7.0, 8.0
        )
        self.assertEqual(first, swapped)
        self.assertEqual(first, translated)

    def test_velocity_boost_changes_only_kinetic_terms(self):
        first = physics.energies(
            2.0, 3.0, 3.0, 4.0, 5.0, 6.0, -1.0, 2.0, -7.0, 8.0
        )
        boosted = physics.energies(
            2.0, 3.0, 3.0, 4.0, 15.0, -14.0, -1.0, 2.0, 3.0, -12.0
        )
        self.assertEqual(boosted[0], first[0])
        self.assertNotEqual(boosted[1:], first[1:])

    def test_bad_velocity_and_energy_overflow(self):
        for bad in (math.nan, math.inf, -math.inf, True, "3"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    physics.energies(
                        2.0, 3.0, 1.0, 0.0, bad, 0.0, 0.0, 0.0, 0.0, 0.0
                    )
        with self.assertRaisesRegex(ValueError, "calculated energy"):
            physics.energies(
                1e308, 1e308, 1.0, 0.0, 0.0, 0.0,
                -1.0, 0.0, 0.0, 0.0,
            )


class TestDriverValidation(unittest.TestCase):
    REAL_PARAMETERS = (
        "MA", "MB", "xInitA", "yInitA", "vInitA", "uInitA",
        "xInitB", "yInitB", "vInitB", "uInitB", "dt", "eps1", "eps2",
    )

    def test_all_real_inputs_reject_bad_values(self):
        for name in self.REAL_PARAMETERS:
            for bad in (math.nan, math.inf, -math.inf, True, "invalid"):
                with self.subTest(name=name, bad=bad):
                    with self.assertRaisesRegex(ValueError, "finite real number"):
                        integrate(**{name: bad})

    def test_positive_bounds(self):
        for name in ("MA", "MB"):
            for bad in (0.0, -1.0):
                with self.subTest(name=name, bad=bad):
                    with self.assertRaisesRegex(ValueError, "must both be positive"):
                        integrate(**{name: bad})
        for bad in (0.0, -1.0):
            with self.subTest(dt=bad):
                with self.assertRaisesRegex(ValueError, "dt must be positive"):
                    integrate(dt=bad)

    def test_max_steps_validation(self):
        for bad in (0, -1, 1.5, True, "10"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    integrate(max_steps=bad)

    def test_tolerance_validation(self):
        for name in ("eps1", "eps2"):
            for bad in (0.0, -0.1, 1.0, 1.1):
                with self.subTest(name=name, bad=bad):
                    with self.assertRaisesRegex(ValueError, r"0 < .* < 1"):
                        integrate(**{name: bad})

    def test_stop_flag_validation(self):
        for bad in (0, 1, 0.0, "True", None):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "True or False"):
                    integrate(stop_after_one_orbit=bad)

    def test_initial_collision(self):
        with self.assertRaisesRegex(ValueError, "zero separation"):
            integrate(xInitB=DEFAULTS["xInitA"], yInitB=DEFAULTS["yInitA"])

    def test_internal_numerical_guards(self):
        with self.assertRaisesRegex(RuntimeError, "numerical safety limit"):
            driver._halve_timestep(1.0, 0.75)
        with self.assertRaisesRegex(RuntimeError, "numerical safety limit"):
            driver._halve_timestep(5e-324, 0.0)
        with self.assertRaisesRegex(RuntimeError, "non-finite vector"):
            driver._vector_relative_change(0.0, 0.0, math.inf, 0.0)


class TestIntegrationRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default = integrate()

    def test_lengths_and_initial_record(self):
        result = self.default
        series = (
            result.times, result.xA, result.yA, result.vA, result.uA,
            result.xB, result.yB, result.vB, result.uB,
            result.U, result.K, result.E,
        )
        self.assertTrue(
            all(len(values) == result.accepted_steps + 1 for values in series)
        )
        self.assertEqual(result.times[0], 0.0)
        self.assertEqual(result.xA[0], DEFAULTS["xInitA"])
        self.assertEqual(result.uB[0], DEFAULTS["uInitB"])

    def test_outputs_are_finite_and_time_increases(self):
        result = self.default
        series = (
            result.times, result.xA, result.yA, result.vA, result.uA,
            result.xB, result.yB, result.vB, result.uB,
            result.U, result.K, result.E,
        )
        self.assertTrue(
            all(math.isfinite(value) for values in series for value in values)
        )
        self.assertTrue(all(b > a for a, b in zip(result.times, result.times[1:])))

    def test_default_orbit_and_energy(self):
        self.assertTrue(self.default.completed_orbit)
        self.assertLess(self.default.accepted_steps, DEFAULTS["max_steps"])
        self.assertTrue(4.4e6 < self.default.times[-1] < 4.8e6)
        self.assertLess(energy_drift(self.default), 2e-3)

    def test_center_of_mass_and_momentum(self):
        result = self.default
        total_mass = DEFAULTS["MA"] + DEFAULTS["MB"]
        com = [
            (DEFAULTS["MA"] * a + DEFAULTS["MB"] * b) / total_mass
            for arrays in ((result.xA, result.xB), (result.yA, result.yB))
            for a, b in zip(*arrays)
        ]
        momentum = [
            DEFAULTS["MA"] * a + DEFAULTS["MB"] * b
            for arrays in ((result.vA, result.vB), (result.uA, result.uB))
            for a, b in zip(*arrays)
        ]
        self.assertTrue(all(value == 0.0 for value in com + momentum))

    def test_angular_momentum_drift(self):
        result = self.default
        values = [
            DEFAULTS["MA"] * (xa * ua - ya * va)
            + DEFAULTS["MB"] * (xb * ub - yb * vb)
            for xa, ya, va, ua, xb, yb, vb, ub in zip(
                result.xA, result.yA, result.vA, result.uA,
                result.xB, result.yB, result.vB, result.uB,
            )
        ]
        initial = values[0]
        drift = max(abs(value - initial) for value in values) / abs(initial)
        self.assertLess(drift, 3e-4)

    def test_analytic_circular_case(self):
        radius = DEFAULTS["xInitA"]
        speed = math.sqrt(physics.G * DEFAULTS["MA"] / (4.0 * radius))
        result = integrate(uInitA=speed, uInitB=-speed)
        separations = [
            math.hypot(xa - xb, ya - yb)
            for xa, xb, ya, yb in zip(result.xA, result.xB, result.yA, result.yB)
        ]
        variation = (max(separations) - min(separations)) / separations[0]
        period = 2 * math.pi * math.sqrt(
            separations[0] ** 3
            / (physics.G * (DEFAULTS["MA"] + DEFAULTS["MB"]))
        )
        self.assertTrue(result.completed_orbit)
        self.assertLess(variation, 2e-5)
        self.assertLess(abs(result.times[-1] - period) / period, 5e-4)

    def test_timestep_convergence(self):
        finer = integrate(dt=1000.0)
        self.assertLess(energy_drift(finer), energy_drift(self.default))

    def test_translation_invariance_of_integrated_relative_motion(self):
        shifted = integrate(
            xInitA=DEFAULTS["xInitA"] + 1e10,
            xInitB=DEFAULTS["xInitB"] + 1e10,
            yInitA=-2e10,
            yInitB=-2e10,
        )
        self.assertEqual(shifted.accepted_steps, self.default.accepted_steps)
        for first_pair, second_pair in (
            ((self.default.xA, self.default.xB), (shifted.xA, shifted.xB)),
            ((self.default.yA, self.default.yB), (shifted.yA, shifted.yB)),
        ):
            for a, b, c, d in zip(*first_pair, *second_pair):
                self.assertAlmostEqual((c - d) - (a - b), 0.0, delta=1e-3)

    def test_galilean_invariance_of_integrated_relative_motion(self):
        boosted = integrate(
            vInitA=1000.0,
            vInitB=1000.0,
            uInitA=14000.0,
            uInitB=-12000.0,
        )
        self.assertEqual(boosted.accepted_steps, self.default.accepted_steps)
        scale = max(
            math.hypot(xa - xb, ya - yb)
            for xa, xb, ya, yb in zip(
                self.default.xA, self.default.xB,
                self.default.yA, self.default.yB,
            )
        )
        error = max(
            math.hypot(
                (xc - xd) - (xa - xb),
                (yc - yd) - (ya - yb),
            )
            for xa, xb, ya, yb, xc, xd, yc, yd in zip(
                self.default.xA, self.default.xB,
                self.default.yA, self.default.yB,
                boosted.xA, boosted.xB, boosted.yA, boosted.yB,
            )
        )
        self.assertLess(error / scale, 1e-5)

    def test_max_steps_and_unbound_motion(self):
        bound = integrate(max_steps=20, stop_after_one_orbit=False)
        unbound = integrate(
            uInitA=60000.0,
            uInitB=-60000.0,
            max_steps=100,
            stop_after_one_orbit=False,
        )
        self.assertEqual((bound.accepted_steps, bound.completed_orbit), (20, False))
        self.assertEqual(
            (unbound.accepted_steps, unbound.completed_orbit), (100, False)
        )
        self.assertGreater(unbound.E[0], 0.0)

    def test_head_on_case_fails_instead_of_freezing(self):
        with self.assertRaisesRegex(RuntimeError, "numerical safety limit"):
            integrate(uInitA=0.0, uInitB=0.0, max_steps=100000)

    def test_extreme_trajectory_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "numerical range"):
            integrate(xInitA=1e308, xInitB=-1e308)


class TestPlotting(unittest.TestCase):
    OUTPUT_TYPES = (
        "orbits",
        "velocity space",
        "position vs. time, body A",
        "position vs. time, body B",
        "velocity vs. time, body A",
        "velocity vs. time, body B",
        "energy vs time",
    )

    @classmethod
    def setUpClass(cls):
        cls.result = integrate(max_steps=2, stop_after_one_orbit=False)

    def tearDown(self):
        plotting.plt.close("all")

    def test_every_output(self):
        for output_type in self.OUTPUT_TYPES:
            with self.subTest(output_type=output_type):
                plotting.plt.close("all")
                with mock.patch.object(plotting.plt, "show") as show:
                    plotting.plot_binary(self.result, output_type)
                show.assert_called_once_with()
                self.assertEqual(len(plotting.plt.gcf().axes), 1)

    def test_equal_aspect_outputs(self):
        for output_type in ("orbits", "velocity space"):
            with self.subTest(output_type=output_type):
                with mock.patch.object(plotting.plt, "show"):
                    plotting.plot_binary(self.result, output_type)
                self.assertEqual(plotting.plt.gca().get_aspect(), 1.0)
                plotting.plt.close("all")

    def test_unknown_output(self):
        with self.assertRaisesRegex(ValueError, "Unknown output_type"):
            plotting.plot_binary(self.result, "not an output")


class TestHelpFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HELP_FILE.is_file():
            raise AssertionError(f"Required Help file not found: {HELP_FILE}")
        cls.html = HELP_FILE.read_text(encoding="utf-8")
        cls.prose = re.sub(r"\s+", " ", cls.html)

    def test_version_build_sync(self):
        match = re.search(
            r'<p\s+id="version_build"[^>]*>\s*'
            r"Version\s+([0-9]+\.[0-9]+\.[0-9]+)"
            r"(?:&nbsp;|\s)+Build\s+([0-9a-f]{12})",
            self.html,
            flags=re.IGNORECASE,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), physics.MODEL_VERSION)
        self.assertEqual(match.group(2), physics.BUILD_ID)

    def test_core_files_and_outputs_are_documented(self):
        for item in CORE_MODULE_FILENAMES + TestPlotting.OUTPUT_TYPES:
            with self.subTest(item=item):
                self.assertIn(item, self.html)

    def test_equation_numbers(self):
        labels = [
            int(value)
            for value in re.findall(
                r'class="eq-label">\((\d+)\)</span>', self.html
            )
        ]
        self.assertEqual(labels, list(range(1, 11)))

    def test_bounds_are_documented(self):
        for phrase in (
            "finite real numbers",
            "strictly between zero and one",
            "0 and 1 are not allowed",
            "double-precision arithmetic",
            "nonzero initial separation",
        ):
            self.assertIn(phrase, self.prose)

    def test_termination_and_energy_qualifications(self):
        for phrase in (
            "relative vector",
            "relative revolution",
            "does not by itself prove",
            "kinetic energy of the centre of mass",
            "subtract its translational kinetic energy",
            "problem-dependent",
            "There is no universal factor",
        ):
            self.assertIn(phrase, self.prose)

    def test_exercise_order_and_difficulty(self):
        headings = re.findall(r"<h3>(\d+) · ([^<]+)</h3>", self.html)
        self.assertEqual([int(number) for number, _ in headings], list(range(1, 9)))
        joined = " ".join(title for _, title in headings)
        for level in ("Introductory", "Intermediate", "Advanced"):
            self.assertIn(level, joined)

    def test_reflex_setup_is_reproducible(self):
        for phrase in (
            "Jupiter", "Saturn", "Earth",
            r"v_{\rm rel}", r"x_A=-aM_B",
            "precision Solar-System ephemeris",
        ):
            self.assertIn(phrase, self.html)

    def test_no_development_history(self):
        for pattern in (
            r"\bAI-generated\b", r"\bChatGPT\b", r"\bClaude\b",
            r"\bCopilot\b", r"\bGemini\b", r"\bported from\b",
            r"\bporting history\b", r"\bbug fix history\b",
        ):
            self.assertIsNone(re.search(pattern, self.html, re.IGNORECASE))

    def test_license_provenance(self):
        for phrase in (
            "Bernard Schutz",
            "Gravity from the Ground Up",
            "Cambridge University Press",
            "CC BY-NC-SA 4.0",
        ):
            self.assertIn(phrase, self.html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
