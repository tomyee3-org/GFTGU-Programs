"""Regression tests for the SphereGravity tutorial program.

The discovery logic deliberately supports both the repository layout
(`tests/test_physics_spheregravity.py`) and a flattened upload in which this
file is placed beside the four core modules.  The Beats Help,
SphereGravity-claude.html, is required; the Reference Guide Help,
SphereGravity-original.html, is optional and its tests skip without it.
"""

import ast
from contextlib import redirect_stdout
import hashlib
import html
from html.parser import HTMLParser
import inspect
import io
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from typing import NamedTuple
import unittest
from unittest import mock

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np


CORE_MODULE_FILENAMES = (
    "physics_spheregravity.py",
    "driver_spheregravity.py",
    "main.py",
    "plot_spheregravity.py",
)


def find_module_dir(start):
    """Find the nearest ancestor containing all four SphereGravity modules."""
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        if all((directory / name).is_file() for name in CORE_MODULE_FILENAMES):
            return directory

    names = ", ".join(CORE_MODULE_FILENAMES)
    raise FileNotFoundError(
        f"Could not find a directory containing the SphereGravity modules: {names}"
    )


MODULE_DIR = find_module_dir(Path(__file__).resolve().parent)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import driver_spheregravity as driver
import main as entry_point
import physics_spheregravity as physics
import plot_spheregravity as plotting



# The Beats Help file; the Reference Guide version (-original) is never used here.
HELP_FILENAMES = ("SphereGravity-claude.html", "SphereGravity.html")


def find_help_file(module_dir):
    """Find the Beats Help in a flattened upload or the GFTGU-Documentation tree.

    Documentation folders no longer use chapter-number prefixes, and the
    Help files live under the sibling ``GFTGU-Documentation`` repository
    rather than beside the program modules inside ``GFTGU-Programs``.
    """
    program_name = "SphereGravity"
    for help_filename in HELP_FILENAMES:
        candidates = [module_dir / help_filename]
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
        "Could not find SphereGravity-claude.html beside the program or in "
        "GFTGU-Documentation/SphereGravity/."
    )


HELP_FILE = find_help_file(MODULE_DIR)


class HelpHTMLParser(HTMLParser):
    """Collect structural information needed by Help-file regressions."""

    def __init__(self):
        super().__init__()
        self.ids = []
        self.local_targets = []
        self.script_sources = []
        self.module_card_count = 0

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.ids.append(element_id)
        href = attributes.get("href", "")
        if href.startswith("#"):
            self.local_targets.append(href[1:])
        if tag == "script" and attributes.get("src"):
            self.script_sources.append(attributes["src"])
        if "module-card" in attributes.get("class", "").split():
            self.module_card_count += 1


def reference_shell_mass(n_div, epsilon=0.001):
    """Direct transcription of the mass loop in Schutz's Java program."""
    d_phi = 2.0 * math.pi / n_div
    d_theta = 0.5 * d_phi
    theta = 0.5 * d_theta - 0.5 * math.pi
    mass = 0.0
    for _ in range(n_div):
        dm = d_theta * d_phi * math.cos(theta) * epsilon
        mass += dm * n_div
        theta += d_theta
    return mass


def reference_acceleration(n_div, radial_index, epsilon=0.001):
    """Direct transcription of the Java acceleration loop at one radius."""
    if radial_index == physics.SURFACE_INDEX:
        return 0.0

    r = radial_index * physics.RADIUS_STEP
    d_phi = 2.0 * math.pi / n_div
    d_theta = 0.5 * d_phi
    theta = 0.5 * d_theta - 0.5 * math.pi
    acceleration = 0.0
    for _ in range(n_div):
        sine = math.sin(theta)
        distance = math.sqrt(1.0 + r * r + 2.0 * r * sine)
        dm = d_theta * d_phi * math.cos(theta) * epsilon
        acceleration += dm * (r + sine) / distance**3 * n_div
        theta += d_theta
    return acceleration


class TestPortableDiscovery(unittest.TestCase):
    def test_actual_module_directory_is_found(self):
        self.assertEqual(MODULE_DIR, find_module_dir(Path(__file__)))
        for filename in CORE_MODULE_FILENAMES:
            self.assertTrue((MODULE_DIR / filename).is_file())

    def test_nearest_matching_ancestor_is_used(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            outer = root / "outer"
            inner = outer / "inner"
            nested = inner / "tests" / "nested"
            nested.mkdir(parents=True)
            for directory in (outer, inner):
                for filename in CORE_MODULE_FILENAMES:
                    (directory / filename).touch()
            self.assertEqual(inner, find_module_dir(nested))

    def test_missing_module_set_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "SphereGravity modules"):
                find_module_dir(temporary_directory)


class TestReleaseMetadata(unittest.TestCase):
    def test_model_version_is_semantic(self):
        self.assertRegex(physics.MODEL_VERSION, r"^\d+\.\d+\.\d+$")

    def test_build_id_covers_exactly_the_four_core_modules(self):
        self.assertEqual(tuple(physics.BUILD_ID_COVERS), CORE_MODULE_FILENAMES)

    def test_build_id_matches_independent_calculation(self):
        digest = hashlib.sha256()
        for filename in CORE_MODULE_FILENAMES:
            with (MODULE_DIR / filename).open(
                "r", encoding="utf-8", newline=None
            ) as source_file:
                content = source_file.read().encode("utf-8")
            digest.update(filename.encode("utf-8"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        self.assertEqual(digest.hexdigest()[:12], physics.BUILD_ID)
        self.assertRegex(physics.BUILD_ID, r"^[0-9a-f]{12}$")

    def test_build_id_controlled_fixture_and_newline_normalization(self):
        fixture = {
            "physics_spheregravity.py": "alpha\n",
            "driver_spheregravity.py": "beta\n",
            "main.py": "gamma\n",
            "plot_spheregravity.py": "delta\n",
        }
        observed = []
        for newline in ("\n", "\r\n"):
            with tempfile.TemporaryDirectory() as temporary_directory:
                directory = Path(temporary_directory)
                for filename, content in fixture.items():
                    (directory / filename).write_bytes(
                        content.replace("\n", newline).encode("utf-8")
                    )
                with mock.patch.object(
                    physics, "__file__", str(directory / "physics_spheregravity.py")
                ):
                    observed.append(physics._compute_build_id())
        self.assertEqual(observed, ["68df75f68f4f", "68df75f68f4f"])

    def test_build_id_fallback_for_file_errors(self):
        decode_error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")
        for error in (OSError("missing"), decode_error):
            with self.subTest(error=type(error).__name__):
                with mock.patch("builtins.open", side_effect=error):
                    self.assertEqual(physics._compute_build_id(), "unknown")

    def test_driver_reports_physics_metadata(self):
        self.assertEqual(
            driver.get_version_info(),
            {
                "model_version": physics.MODEL_VERSION,
                "build_id": physics.BUILD_ID,
            },
        )

    def test_help_version_and_build_match_program(self):
        help_text = HELP_FILE.read_text(encoding="utf-8")
        match = re.search(
            r'<p\s+id="version_build"[^>]*>\s*Version\s+([^&<\s]+)'
            r'(?:&nbsp;|\s)+Build\s+([0-9a-f]{12})\s*</p>',
            help_text,
            flags=re.IGNORECASE,
        )
        self.assertIsNotNone(match, "Help file lacks a parseable version_build element")
        self.assertEqual(match.group(1), physics.MODEL_VERSION)
        self.assertEqual(match.group(2), physics.BUILD_ID)

    def test_all_core_sources_parse_as_python_3_10(self):
        for filename in CORE_MODULE_FILENAMES:
            with self.subTest(filename=filename):
                source = (MODULE_DIR / filename).read_text(encoding="utf-8")
                ast.parse(source, filename=filename, feature_version=(3, 10))


class TestInputValidation(unittest.TestCase):
    def test_invalid_n_div_is_rejected_by_public_physics_functions(self):
        invalid_values = (0, -1, 1.5, True, np.bool_(True), "100", None)
        for value in invalid_values:
            with self.subTest(value=value, function="mass"):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    physics.compute_shell_mass(value)
            for implementation in (
                physics.compute_acceleration_profile_textbook,
                physics.compute_acceleration_profile_optimized,
                lambda n_div: physics.compute_acceleration_at_radii(n_div, [0.5]),
            ):
                with self.subTest(value=value, function=implementation.__name__):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        implementation(value)

    def test_numpy_integer_n_div_is_accepted(self):
        for value in (np.int32(8), np.int64(8), np.uint16(8)):
            with self.subTest(value=value):
                self.assertGreater(physics.compute_shell_mass(value), 0.0)

    def test_n_div_above_resource_limit_is_rejected(self):
        for value in (physics.MAX_NDIV + 1, np.uint64(physics.MAX_NDIV + 1)):
            for implementation in (
                physics.compute_acceleration_profile_textbook,
                physics.compute_acceleration_profile_optimized,
            ):
                with self.subTest(value=value, function=implementation.__name__):
                    with self.assertRaisesRegex(ValueError, "must not exceed"):
                        implementation(value)

    def test_maximum_n_div_is_accepted_for_mass_calculation(self):
        self.assertGreater(physics.compute_shell_mass(physics.MAX_NDIV), 0.0)

    def test_numpy_epsilon_scalars_are_accepted(self):
        for value in (np.float32(0.001), np.float64(0.001), np.int64(1)):
            with self.subTest(value=value):
                self.assertGreater(physics.compute_shell_mass(8, value), 0.0)

    def test_invalid_epsilon_is_rejected_by_public_physics_functions(self):
        invalid_values = (
            0,
            -0.001,
            float("nan"),
            float("inf"),
            True,
            np.bool_(False),
            1j,
            10**1000,
            "0.001",
            None,
        )
        for value in invalid_values:
            with self.subTest(value=value, function="mass"):
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    physics.compute_shell_mass(8, epsilon=value)
            for implementation in (
                physics.compute_acceleration_profile_textbook,
                physics.compute_acceleration_profile_optimized,
                lambda n_div, epsilon: physics.compute_acceleration_at_radii(
                    n_div, [0.5], epsilon
                ),
            ):
                with self.subTest(value=value, function=implementation.__name__):
                    with self.assertRaisesRegex(ValueError, "positive finite"):
                        implementation(8, epsilon=value)

    def test_arbitrary_radius_input_is_validated(self):
        invalid_values = (
            [],
            [[0.5]],
            [float("nan")],
            [float("inf")],
            [-0.5],
            [physics.SHELL_RADIUS],
            ["radius"],
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "radii|surface"):
                    physics.compute_acceleration_at_radii(8, value)

    def test_invalid_output_type_is_rejected(self):
        for value in ("relative_difference", "", None, 1):
            for implementation in (
                physics.compute_acceleration_profile_textbook,
                physics.compute_acceleration_profile_optimized,
            ):
                with self.subTest(value=value, function=implementation.__name__):
                    with self.assertRaisesRegex(ValueError, "outputType"):
                        implementation(8, outputType=value)


class TestShellMass(unittest.TestCase):
    def test_mass_loop_matches_java_reference(self):
        for n_div in (1, 2, 7, 20, 100):
            with self.subTest(n_div=n_div):
                self.assertAlmostEqual(
                    physics.compute_shell_mass(n_div),
                    reference_shell_mass(n_div),
                    places=14,
                )

    def test_mass_is_positive_and_finite(self):
        mass = physics.compute_shell_mass(100)
        self.assertGreater(mass, 0.0)
        self.assertTrue(math.isfinite(mass))

    def test_mass_scales_linearly_with_epsilon(self):
        mass_1 = physics.compute_shell_mass(100, epsilon=0.001)
        mass_2 = physics.compute_shell_mass(100, epsilon=0.007)
        self.assertAlmostEqual(mass_2 / mass_1, 7.0, places=12)

    def test_mass_converges_quadratically_to_continuum_value(self):
        exact_mass = physics.compute_continuum_shell_mass()
        errors = [
            abs(physics.compute_shell_mass(n_div) - exact_mass)
            for n_div in (10, 100, 1000)
        ]
        self.assertGreater(errors[0] / errors[1], 95.0)
        self.assertLess(errors[0] / errors[1], 105.0)
        self.assertGreater(errors[1] / errors[2], 95.0)
        self.assertLess(errors[1] / errors[2], 105.0)

    def test_continuum_mass_scales_with_epsilon(self):
        self.assertAlmostEqual(
            physics.compute_continuum_shell_mass(0.007),
            4.0 * math.pi * 0.007 * physics.SHELL_RADIUS**2,
            places=15,
        )


class TestAccelerationPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.radius_100, cls.raw_100 = physics.compute_acceleration_profile(
            100, "acceleration"
        )
        radius, cls.relative_100 = physics.compute_acceleration_profile(
            100, "relative difference"
        )
        np.testing.assert_array_equal(radius, cls.radius_100)
        cls.radius_1000, cls.raw_1000 = physics.compute_acceleration_profile(
            1000, "acceleration"
        )
        radius, cls.relative_1000 = physics.compute_acceleration_profile_optimized(
            1000, "relative difference"
        )
        np.testing.assert_array_equal(radius, cls.radius_1000)
        cls.radius_10000, cls.relative_10000 = physics.compute_acceleration_profile_optimized(
            10000, "relative difference"
        )

    def test_textbook_implementation_is_selected_by_default(self):
        self.assertIs(
            physics.compute_acceleration_profile,
            physics.compute_acceleration_profile_textbook,
        )

    def test_implementations_have_identical_signatures(self):
        self.assertEqual(
            inspect.signature(physics.compute_acceleration_profile_textbook),
            inspect.signature(physics.compute_acceleration_profile_optimized),
        )

    def test_textbook_and_optimized_implementations_agree(self):
        for output_type in ("acceleration", "relative difference"):
            with self.subTest(output_type=output_type):
                textbook = physics.compute_acceleration_profile_textbook(
                    24, output_type, epsilon=0.003
                )
                optimized = physics.compute_acceleration_profile_optimized(
                    24, output_type, epsilon=0.003
                )
                np.testing.assert_array_equal(textbook[0], optimized[0])
                np.testing.assert_allclose(
                    textbook[1], optimized[1], rtol=2e-13, atol=2e-15
                )

    def test_radial_grid_contract(self):
        self.assertEqual(self.radius_100.shape, (physics.NUM_RADII,))
        self.assertEqual(self.raw_100.shape, (physics.NUM_RADII,))
        self.assertEqual(self.radius_100[0], 0.0)
        self.assertEqual(self.radius_100[-1], 4.995)
        np.testing.assert_allclose(
            np.diff(self.radius_100), physics.RADIUS_STEP, rtol=0.0, atol=1e-15
        )

    def test_outputs_are_finite(self):
        for array in (
            self.raw_100,
            self.relative_100,
            self.raw_1000,
            self.relative_1000,
            self.relative_10000,
        ):
            self.assertTrue(np.all(np.isfinite(array)))

    def test_surface_sample_is_zero_placeholder(self):
        index = physics.SURFACE_INDEX
        self.assertEqual(self.radius_100[index], physics.SHELL_RADIUS)
        self.assertEqual(self.raw_100[index], 0.0)
        self.assertEqual(self.relative_100[index], 0.0)

    def test_selected_points_match_java_reference_loop(self):
        radius, acceleration = physics.compute_acceleration_profile(12, "acceleration")
        for radial_index in (0, 50, 180, 200, 201, 220, 400, 999):
            with self.subTest(radial_index=radial_index):
                self.assertEqual(radius[radial_index], radial_index * 0.005)
                self.assertAlmostEqual(
                    acceleration[radial_index],
                    reference_acceleration(12, radial_index),
                    places=14,
                )

    def test_requested_report_radii_match_independent_reference(self):
        requested_indices = (100, 199, 201, 400, 600, 800, 1000)
        requested_radii = [index * physics.RADIUS_STEP for index in requested_indices]
        radius, acceleration = physics.compute_acceleration_at_radii(
            12, requested_radii
        )
        np.testing.assert_array_equal(radius, requested_radii)
        for observed, radial_index in zip(acceleration, requested_indices):
            with self.subTest(radius=radial_index * physics.RADIUS_STEP):
                self.assertAlmostEqual(
                    observed,
                    reference_acceleration(12, radial_index),
                    places=14,
                )

    def test_center_acceleration_is_zero_to_roundoff(self):
        self.assertLess(abs(self.raw_100[0]), 1e-14)
        self.assertLess(abs(self.raw_1000[0]), 1e-14)

    def test_interior_residual_converges_toward_zero(self):
        index = 180  # r = 0.9
        self.assertLess(abs(self.relative_1000[index]), abs(self.relative_100[index]) / 90.0)

    def test_exterior_field_is_close_to_inverse_square_law(self):
        mass = physics.compute_shell_mass(1000)
        for radial_index in (220, 400, 800):
            r = self.radius_1000[radial_index]
            newton = mass / r**2
            with self.subTest(r=r):
                self.assertAlmostEqual(
                    self.raw_1000[radial_index] / newton, 1.0, delta=3e-5
                )

    def test_against_independent_continuum_shell_theorem(self):
        exact_mass = 4.0 * math.pi * physics.DEFAULT_EPSILON
        self.assertLess(abs(self.raw_1000[100]) / exact_mass, 3e-5)  # r = 0.5
        for radial_index in (400, 800):
            r = self.radius_1000[radial_index]
            exact_acceleration = exact_mass / r**2
            with self.subTest(r=r):
                self.assertAlmostEqual(
                    self.raw_1000[radial_index] / exact_acceleration,
                    1.0,
                    delta=3e-5,
                )

    def test_inverse_square_ratio_between_r_2_and_r_4(self):
        ratio = self.raw_1000[400] / self.raw_1000[800]
        self.assertAlmostEqual(ratio, 4.0, delta=2e-5)

    def test_documented_r_1_1_convergence_values(self):
        observed = np.array(
            [self.relative_100[220], self.relative_1000[220], self.relative_10000[220]]
        )
        np.testing.assert_allclose(
            observed, np.array([2.5e-3, 2.5e-5, 2.5e-7]), rtol=0.03, atol=0.0
        )
        ratios = observed[:-1] / observed[1:]
        self.assertTrue(np.all((ratios > 95.0) & (ratios < 105.0)))

    def test_relative_difference_transformation(self):
        mass = physics.compute_shell_mass(100)
        for radial_index in (0, 100, 180, 200):
            with self.subTest(region="inside", radial_index=radial_index):
                self.assertAlmostEqual(
                    self.relative_100[radial_index],
                    self.raw_100[radial_index] / mass,
                    places=14,
                )
        for radial_index in (201, 220, 400, 999):
            r = self.radius_100[radial_index]
            newton = mass / r**2
            expected = (self.raw_100[radial_index] - newton) / newton
            with self.subTest(region="outside", radial_index=radial_index):
                self.assertAlmostEqual(
                    self.relative_100[radial_index], expected, places=14
                )

    def test_epsilon_scales_raw_field_but_not_relative_difference(self):
        _, raw_1 = physics.compute_acceleration_profile(20, "acceleration", 0.001)
        _, raw_7 = physics.compute_acceleration_profile(20, "acceleration", 0.007)
        np.testing.assert_allclose(raw_7, raw_1 * 7.0, rtol=2e-14, atol=2e-16)

        _, relative_1 = physics.compute_acceleration_profile(
            20, "relative difference", 0.001
        )
        _, relative_7 = physics.compute_acceleration_profile(
            20, "relative difference", 0.007
        )
        np.testing.assert_allclose(relative_7, relative_1, rtol=2e-13, atol=2e-15)

    def test_each_call_returns_independent_arrays(self):
        radius_1, acceleration_1 = physics.compute_acceleration_profile(8)
        radius_2, acceleration_2 = physics.compute_acceleration_profile(8)
        self.assertIsNot(radius_1, radius_2)
        self.assertIsNot(acceleration_1, acceleration_2)
        radius_1[0] = 99.0
        acceleration_1[0] = 99.0
        self.assertEqual(radius_2[0], 0.0)
        self.assertNotEqual(acceleration_2[0], 99.0)


class TestDriverAndEntryPoint(unittest.TestCase):
    def test_driver_matches_physics_function(self):
        radius_driver, acceleration_driver = driver.run_spheregravity(
            16, "relative difference", epsilon=0.003
        )
        radius_physics, acceleration_physics = physics.compute_acceleration_profile(
            16, "relative difference", epsilon=0.003
        )
        np.testing.assert_array_equal(radius_driver, radius_physics)
        np.testing.assert_array_equal(acceleration_driver, acceleration_physics)

    def test_importing_main_has_no_execution_side_effect(self):
        completed = subprocess.run(
            [sys.executable, "-c", "import main; print('import-complete')"],
            cwd=str(MODULE_DIR),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "import-complete")

    def test_version_command_reports_current_metadata(self):
        completed = subprocess.run(
            [sys.executable, "main.py", "--version"],
            cwd=str(MODULE_DIR),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        expected = f"SphereGravity {physics.MODEL_VERSION} (build {physics.BUILD_ID})"
        self.assertEqual(completed.stdout.strip(), expected)

    def test_normal_execution_completes_headlessly(self):
        environment = os.environ.copy()
        environment["MPLBACKEND"] = "Agg"
        completed = subprocess.run(
            [sys.executable, "main.py"],
            cwd=str(MODULE_DIR),
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        expected = f"SphereGravity {physics.MODEL_VERSION} (build {physics.BUILD_ID})"
        self.assertTrue(completed.stdout.startswith(expected))
        self.assertIn("Shell mass comparison", completed.stdout)
        self.assertIn("Acceleration comparison", completed.stdout)
        for radius in entry_point.REPORT_RADII:
            self.assertIn(entry_point._format_value(radius), completed.stdout)

    def test_command_line_defaults_cover_all_driver_parameters(self):
        args = entry_point.parse_args([])
        signature = inspect.signature(driver.run_spheregravity)
        self.assertEqual(args.nDiv, signature.parameters["nDiv"].default)
        self.assertEqual(
            entry_point.CLI_OUTPUT_TYPES[args.outputType],
            "relative difference",
        )
        self.assertEqual(args.epsilon, signature.parameters["epsilon"].default)

    def test_command_line_accepts_all_custom_parameters(self):
        args = entry_point.parse_args(
            [
                "--nDiv",
                "250",
                "--outputType",
                "acceleration",
                "--epsilon",
                "0.004",
            ]
        )
        self.assertEqual(args.nDiv, 250)
        self.assertEqual(args.outputType, "acceleration")
        self.assertEqual(args.epsilon, 0.004)

    def test_command_line_rejects_legacy_spaced_selector(self):
        with self.assertRaises(SystemExit):
            entry_point.parse_args(["--outputType", "relative difference"])

    def test_comparison_prints_all_requested_quantities(self):
        output = io.StringIO()
        with redirect_stdout(output):
            entry_point.print_comparison(100, physics.DEFAULT_EPSILON)
        text = output.getvalue()
        self.assertIn("numerical midpoint mass", text)
        self.assertIn("continuum mass", text)
        self.assertIn("relative difference", text)
        self.assertIn("undefined; g/M=", text)
        for radius in entry_point.REPORT_RADII:
            self.assertIn(entry_point._format_value(radius), text)

    def test_main_uses_documented_user_settings(self):
        fake_radius = np.array([0.0, 0.5])
        fake_acceleration = np.array([0.0, 0.0])
        with mock.patch.object(
            entry_point,
            "run_spheregravity",
            return_value=(fake_radius, fake_acceleration),
        ) as run_mock, mock.patch.object(
            entry_point, "plot_spheregravity"
        ) as plot_mock, mock.patch.object(
            entry_point, "print_comparison"
        ) as comparison_mock, mock.patch("builtins.print"):
            entry_point.main(
                [
                    "--nDiv",
                    "16",
                    "--outputType",
                    "acceleration",
                    "--epsilon",
                    "0.003",
                ]
            )

        run_mock.assert_called_once_with(
            nDiv=16, outputType="acceleration", epsilon=0.003
        )
        plot_mock.assert_called_once_with(
            fake_radius, fake_acceleration, outputType="acceleration"
        )
        comparison_mock.assert_called_once_with(16, 0.003, list(entry_point.REPORT_RADII))


class TestPlotting(unittest.TestCase):
    def tearDown(self):
        plotting.plt.close("all")

    def test_valid_plot_has_expected_labels_and_surface_marker(self):
        radius = np.array([0.0, 0.5, 1.0, 2.0])
        acceleration = np.array([0.0, 0.0, 0.0, 0.25])
        with mock.patch.object(plotting.plt, "show") as show_mock:
            plotting.plot_spheregravity(radius, acceleration, "acceleration")
        show_mock.assert_called_once_with()
        axes = plotting.plt.gcf().axes[0]
        self.assertIn("Gravitational acceleration", axes.get_ylabel())
        self.assertEqual(axes.get_lines()[1].get_xdata()[0], physics.SHELL_RADIUS)
        np.testing.assert_array_equal(axes.get_lines()[0].get_xdata(), radius)
        plotted_y = axes.get_lines()[0].get_ydata()
        np.testing.assert_array_equal(plotted_y[[0, 1, 3]], acceleration[[0, 1, 3]])
        self.assertTrue(np.isnan(plotted_y[2]))
        self.assertEqual(acceleration[2], 0.0, "plotting must not mutate caller data")

    def test_each_plot_call_creates_a_new_figure(self):
        with mock.patch.object(plotting.plt, "show"):
            plotting.plot_spheregravity([0.0, 1.0], [0.0, 0.0])
            first_numbers = set(plotting.plt.get_fignums())
            plotting.plot_spheregravity([0.0, 1.0], [0.0, 0.0])
            second_numbers = set(plotting.plt.get_fignums())
        self.assertEqual(len(first_numbers), 1)
        self.assertEqual(len(second_numbers), 2)

    def test_relative_plot_uses_relative_labels(self):
        with mock.patch.object(plotting.plt, "show"):
            plotting.plot_spheregravity([0.0, 2.0], [0.0, 0.01], "relative difference")
        axes = plotting.plt.gcf().axes[0]
        self.assertIn("relative difference", axes.get_title().lower())
        self.assertIn("relative difference", axes.get_ylabel().lower())

    def test_invalid_output_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outputType"):
            plotting.plot_spheregravity([0.0], [0.0], "error")

    def test_non_one_dimensional_arrays_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            plotting.plot_spheregravity([[0.0]], [[0.0]])

    def test_empty_arrays_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            plotting.plot_spheregravity([], [])

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            plotting.plot_spheregravity([0.0, 1.0], [0.0])

    def test_non_numeric_arrays_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "real numeric"):
            plotting.plot_spheregravity(["zero"], ["zero"])

    def test_complex_arrays_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "real numeric"):
            plotting.plot_spheregravity([0.0 + 0.0j], [1.0 + 2.0j])

    def test_non_finite_arrays_are_rejected(self):
        for bad_value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=bad_value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    plotting.plot_spheregravity([0.0, 1.0], [0.0, bad_value])

    def test_negative_radius_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            plotting.plot_spheregravity([-1.0, 0.0], [0.1, 0.0])

    def test_nonincreasing_radius_is_rejected(self):
        for radius in ([0.0, 2.0, 1.0], [0.0, 1.0, 1.0]):
            with self.subTest(radius=radius):
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    plotting.plot_spheregravity(radius, [0.0, 0.1, 0.2])


# ---------------------------------------------------------------------------
# The midpoint-rule error estimate, the --radii option and the plot return
# ---------------------------------------------------------------------------


def _euler_maclaurin_reference(n_div, r):
    """Independent leading-order error, from numerically differentiated slopes.

    The ring sum is 2*pi*epsilon times h * sum f(theta_k) with
    f(theta) = cos(theta) (r + sin(theta)) / d^3.  The Euler-Maclaurin
    leading term is -(h^2/24) [f'(pi/2) - f'(-pi/2)]; here the slopes are
    taken by central differences instead of the closed form used in the
    program, and the result is normalized as in the printed table.
    """
    def f(theta):
        return math.cos(theta) * (r + math.sin(theta)) / (1.0 + r * r + 2.0 * r * math.sin(theta)) ** 1.5

    step = 1e-5
    top = (f(0.5 * math.pi + step) - f(0.5 * math.pi - step)) / (2 * step)
    bottom = (f(-0.5 * math.pi + step) - f(-0.5 * math.pi - step)) / (2 * step)
    h = math.pi / n_div
    error = -(h * h / 24.0) * (top - bottom)  # error of h*sum f, per 2*pi*epsilon
    if r > 1.0:
        return error / (2.0 / r**2)  # relative to the exact 2/r^2
    return error / 2.0  # g/M, with M = 4 pi epsilon = 2 * (2 pi epsilon)


class TestMidpointErrorEstimate(unittest.TestCase):
    def test_estimate_matches_an_independent_euler_maclaurin_calculation(self):
        for radius in (0.0, 0.25, 0.5, 0.9, 1.2, 2.0, 5.0, 40.0):
            estimate = physics.midpoint_error_estimate(100, [radius])[0]
            reference = _euler_maclaurin_reference(100, radius)
            with self.subTest(radius=radius):
                self.assertAlmostEqual(estimate, reference, delta=1e-6 * abs(reference) + 1e-15)

    def test_estimate_predicts_the_actual_error_of_the_sum(self):
        radii = (0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 5.0, 20.0)
        for n_div in (100, 1000):
            _, g = physics.compute_acceleration_at_radii(n_div, radii)
            mass = physics.compute_continuum_shell_mass()
            estimate = physics.midpoint_error_estimate(n_div, radii)
            for radius, value, predicted in zip(radii, g, estimate):
                actual = value / mass if radius < 1 else (value - mass / radius**2) / (mass / radius**2)
                with self.subTest(n_div=n_div, radius=radius):
                    # The next term is smaller by about h^2 relative to this one.
                    self.assertAlmostEqual(actual / predicted, 1.0, delta=0.03 if n_div == 100 else 1e-3)

    def test_estimate_limits(self):
        h = math.pi / 100
        self.assertEqual(physics.midpoint_error_estimate(100, [0.0])[0], 0.0)
        self.assertAlmostEqual(physics.midpoint_error_estimate(100, [1e6])[0], h * h / 24, delta=1e-12)
        self.assertAlmostEqual(
            physics.midpoint_error_estimate(1000, [2.0])[0] / physics.midpoint_error_estimate(100, [2.0])[0],
            0.01, places=12)
        for bad in ([1.0], [-0.5], [math.nan], []):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                physics.midpoint_error_estimate(100, bad)
        for bad in (0, 1.5, True):
            with self.subTest(bad_n=bad), self.assertRaises(ValueError):
                physics.midpoint_error_estimate(bad, [0.5])

    def test_midpoint_mass_closed_form(self):
        for n_div in (1, 2, 3, 10, 100, 1000):
            h = math.pi / n_div
            ratio = physics.compute_shell_mass(n_div) / physics.compute_continuum_shell_mass()
            with self.subTest(n_div=n_div):
                self.assertAlmostEqual(ratio, (h / 2) / math.sin(h / 2), delta=1e-13)


class TestNumericalRange(unittest.TestCase):
    """Inputs are limited so that no printed quantity underflows or overflows."""

    def test_epsilon_and_radius_limits_are_enforced_everywhere(self):
        for epsilon in (1e-101, 2e100, 1e308, 1e-310):
            with self.subTest(epsilon=epsilon):
                with self.assertRaisesRegex(ValueError, "positive finite number from 1e-100 to 1e\\+100"):
                    physics.compute_shell_mass(8, epsilon)
                with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), \
                        mock.patch("sys.stderr", io.StringIO()):
                    entry_point.parse_args(["--epsilon", repr(epsilon)])
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            physics.compute_acceleration_at_radii(8, [2e100])
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), \
                mock.patch("sys.stderr", io.StringIO()):
            entry_point.parse_args(["--radii", "1e200"])
        self.assertEqual(entry_point.parse_args(["--epsilon", "1e-100", "--radii", "1e100"]).radii, [1e100])

    def test_extreme_accepted_inputs_print_finite_consistent_values(self):
        for epsilon, radii in ((1e-100, (1e100, 2.0, 0.5)), (1e100, (0.995, 1.005, 1e100))):
            for n_div in (1, 100):
                output = io.StringIO()
                with redirect_stdout(output):
                    entry_point.print_comparison(n_div, epsilon, radii)
                text = output.getvalue()
                with self.subTest(epsilon=epsilon, n_div=n_div):
                    self.assertNotIn("nan", text)
                    self.assertNotIn("inf", text)
                    # The relative differences do not depend on epsilon (Eq. 18).
                    reference = io.StringIO()
                    with redirect_stdout(reference):
                        entry_point.print_comparison(n_div, 0.001, radii)
                    column = lambda t: [line[44:] for line in t.splitlines() if re.match(r"^\s+\d", line)]
                    self.assertEqual(column(text), column(reference.getvalue()))
        radius, accel = physics.compute_acceleration_profile_optimized(10, "acceleration", 1e100)
        self.assertTrue(np.all(np.isfinite(accel)))

    def test_radius_column_distinguishes_radii_close_to_the_shell(self):
        output = io.StringIO()
        with redirect_stdout(output):
            entry_point.print_comparison(100, 0.001, (1.00001, 0.99999))
        self.assertIn("  1.00001  ", output.getvalue())
        self.assertIn("  0.99999  ", output.getvalue())


class TestComparisonTableOptions(unittest.TestCase):
    def test_radii_option_parses_and_rejects(self):
        args = entry_point.parse_args(["--radii", "0", "1.1", "250"])
        self.assertEqual(args.radii, [0.0, 1.1, 250.0])
        self.assertEqual(entry_point.parse_args([]).radii, list(entry_point.REPORT_RADII))
        for bad in ("1", "1.0", "-0.5", "nan", "inf", "x"):
            with self.subTest(bad=bad), redirect_stdout(io.StringIO()), \
                    mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                entry_point.parse_args(["--radii", bad])

    def test_table_prints_step_estimate_and_marks(self):
        output = io.StringIO()
        with redirect_stdout(output):
            entry_point.print_comparison(100, physics.DEFAULT_EPSILON, (0.5, 0.995, 1.1, 2.0))
        text = output.getvalue()
        self.assertIn("  latitude step h = pi/nDiv: 0.031416\n", text)
        self.assertIn("h^2 estimate", text)
        rows = {line.split()[0]: line for line in text.splitlines() if re.match(r"^\s+[\d.]+\s", line)}
        self.assertTrue(rows["0.995"].endswith(" *"))
        self.assertFalse(rows["1.1"].endswith("*"))
        self.assertFalse(rows["2"].endswith("*"))
        self.assertIn("-7.3108e-05", rows["0.5"])
        self.assertIn("(* h is not small compared with |r - 1| or 1;", text)

    def test_mark_rule_uses_the_smaller_of_distance_and_radius(self):
        def marked(n_div, radius):
            output = io.StringIO()
            with redirect_stdout(output):
                entry_point.print_comparison(n_div, physics.DEFAULT_EPSILON, (radius,))
            return "*" in output.getvalue()

        self.assertTrue(marked(1256, 1.005))
        self.assertFalse(marked(1257, 1.005))
        self.assertTrue(marked(2, 1000.0))    # h = 1.57 is not small compared with 1
        self.assertFalse(marked(10, 1000.0))  # h = 0.314
        self.assertFalse(marked(10000, 0.999))

    def test_table_without_marks_prints_no_footnote(self):
        output = io.StringIO()
        with redirect_stdout(output):
            entry_point.print_comparison(100, physics.DEFAULT_EPSILON, (2.0, 5.0))
        self.assertNotIn("*", output.getvalue())
        self.assertNotIn("undefined", output.getvalue())  # no interior row, so no interior note

    def test_long_radius_labels_keep_the_columns_aligned(self):
        def layout(radii):
            output = io.StringIO()
            with redirect_stdout(output):
                entry_point.print_comparison(10, physics.DEFAULT_EPSILON, radii)
            lines = output.getvalue().splitlines()
            header = next(line for line in lines if line.lstrip().startswith("radius"))
            rows = [line for line in lines if re.match(r"^\s+\d", line)]
            header_end = header.index("predicted g") + len("predicted g")
            # offset of the end of the predicted-g value from the end of its heading
            offsets = {row.index(row.split()[1], len(row.split()[0]) + 2) + len(row.split()[1]) - header_end
                       for row in rows}
            return lines, rows, offsets

        _, _, standard = layout((0.5, 2.0))
        lines, rows, offsets = layout((0.5, 1.00001, 1e100))
        self.assertEqual([row.split()[0] for row in rows], ["0.5", "1.00001", "1e+100"])
        self.assertEqual(offsets, standard)
        self.assertEqual(len(standard), 1)
        self.assertIn("  (Relative difference is undefined inside because expected g is zero.)", lines)

    def test_plot_returns_its_figure_and_axes(self):
        with mock.patch.object(plotting.plt, "show"):
            figure, axes = plotting.plot_spheregravity([0.0, 1.0, 2.0], [0.0, 0.0, 0.25])
        self.assertIs(axes.figure, figure)
        self.assertIn("Gravitational acceleration", axes.get_ylabel())
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png")
        self.assertGreater(len(buffer.getvalue()), 1000)
        plotting.plt.close("all")


# ---------------------------------------------------------------------------
# The Beats Help file
# ---------------------------------------------------------------------------

BEAT_NUMBERS = tuple(range(0, 9))
EQUATION_COUNT = 19
VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
# Commands that the Help documents as being rejected by the program.
REJECTED_COMMANDS = {
    "python main.py --radii 1": "radius must not be the shell radius 1",
    "python main.py --nDiv 100001": "value must not exceed 100000",
}
HELP_HTML = HELP_FILE.read_text(encoding="utf-8")


def html_text(fragment):
    """Visible text of an HTML fragment, with entities decoded and spaces collapsed."""
    return " ".join(
        html.unescape(re.sub(r"</?[A-Za-z!][^>]*>", "", fragment)).split()
    )


def section_html(page, section_id):
    match = re.search(
        rf'<section id="{re.escape(section_id)}">(.*?)</section>', page, re.DOTALL
    )
    if match is None:
        raise AssertionError(f"section {section_id!r} not found in the Help file")
    return match.group(1)


def documented_commands(fragment):
    """Every ``python main.py ...`` command shown in a <pre> block."""
    commands = []
    for block in re.findall(r"<pre[^>]*>(.*?)</pre>", fragment, re.DOTALL):
        text = html.unescape(re.sub(r"<[^>]+>", "", block))
        for line in text.splitlines():
            line = " ".join(line.split())
            if line.startswith("python main.py"):
                commands.append(line)
    return commands


def command_arguments(command):
    return tuple(shlex.split(command)[2:])


class CliRun(NamedTuple):
    stdout: str
    stderr: str
    exit_code: object


_CLI_CACHE = {}


def run_cli(arguments=()):
    """Run ``main.main()`` in-process with the plot suppressed.

    The printed summary does not depend on the plotted profile, so the
    profile is computed with the optimized implementation, which gives the
    same values as the textbook loops (see
    test_textbook_and_optimized_implementations_agree) in a fraction of the
    time.  The program has no randomness, so every run is cached.
    """
    key = tuple(arguments)
    if key in _CLI_CACHE:
        return _CLI_CACHE[key]
    out, err = io.StringIO(), io.StringIO()
    exit_code = None
    with mock.patch.object(entry_point, "plot_spheregravity"), \
            mock.patch.object(driver, "compute_acceleration_profile",
                              physics.compute_acceleration_profile_optimized), \
            redirect_stdout(out), mock.patch("sys.stderr", err):
        try:
            entry_point.main(list(key))
        except SystemExit as exc:
            exit_code = exc.code
    run = CliRun(out.getvalue(), err.getvalue(), exit_code)
    _CLI_CACHE[key] = run
    return run


def beat_run(command):
    return run_cli(command_arguments(command))


class HelpStructure(HTMLParser):
    """Ids, links, table rows and tag balance of a Help page."""

    def __init__(self, page):
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


class TestBeatsHelpStructure(unittest.TestCase):
    def test_help_file_is_html5_utf8_with_one_version_build(self):
        self.assertIn("<!DOCTYPE html>", HELP_HTML[:100])
        self.assertRegex(HELP_HTML[:500], r'<meta charset="utf-8"\s*/?>')
        self.assertEqual(STRUCTURE.ids.count("version_build"), 1)

    def test_help_file_is_the_beats_version_and_not_the_original(self):
        self.assertIn(HELP_FILE.name, HELP_FILENAMES)
        self.assertNotEqual(HELP_FILE.name, "SphereGravity-original.html")
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

    def test_display_math_has_no_line_breaks_or_unbalanced_braces(self):
        blocks = re.findall(r"\\\[(.*?)\\\]", HELP_HTML, re.DOTALL)
        self.assertGreaterEqual(len(blocks), EQUATION_COUNT)
        for block in blocks:
            with self.subTest(block=block[:40]):
                self.assertNotIn("\\texttt", block)
                self.assertEqual(block.count("{"), block.count("}"))
                # \\[6pt] inside a cases environment is the only permitted break.
                self.assertEqual(len(re.findall(r"\\\\(?!\[6pt\])", block)), 0)

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
        relative = [h for h in STRUCTURE.hrefs
                    if not h.startswith(("#", "http://", "https://", "mailto:"))]
        self.assertEqual(len(relative), 4)
        for href in relative:
            with self.subTest(href=href):
                self.assertTrue((HELP_FILE.parent / href).is_file(), href)


class TestBeatsHelpBeats(unittest.TestCase):
    TAGS = {"LAW": "law", "DEFINITION": "def", "DERIVED": "der", "ALGORITHM": "alg"}

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
        self.assertIn("Nineteen equations are numbered", html_text(section_html(HELP_HTML, "beats")))

    def test_tag_counts_match_the_reading_note(self):
        by_kind = {}
        for number, (kind, _, _) in self.equations().items():
            by_kind.setdefault(kind, []).append(number)
        self.assertEqual(sorted(by_kind["LAW"]), [1])
        self.assertEqual(sorted(by_kind["ALGORITHM"]), [8, 14, 19])
        note = html_text(section_html(HELP_HTML, "beats"))
        self.assertIn("Eq. (1) is the only one", note)
        self.assertIn("Eqs. (8), (14) and (19)", note)

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

    def test_experiment_6_script_runs_and_agrees_with_the_table(self):
        blocks = [
            html.unescape(re.sub(r"<[^>]+>", "", block))
            for block in re.findall(r"<pre>(.*?)</pre>", section_html(HELP_HTML, "experiments"), re.DOTALL)
            if "run_spheregravity" in block
        ]
        self.assertEqual(len(blocks), 1)
        completed = subprocess.run(
            [sys.executable, "-c", blocks[0]], cwd=str(MODULE_DIR),
            text=True, capture_output=True, timeout=120,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        values = dict(tuple(map(float, line.split())) for line in completed.stdout.splitlines())
        self.assertEqual(sorted(values), [1.1, 2.0, 4.995])
        # Eq. (17): the plotted value is the table's relative difference
        # divided through by the tiles' own mass.
        table = beat_run("python main.py --radii 2 5 10 100 1000").stdout
        delta = float(re.search(r"^\s+2\s+\S+\s+\S+\s+(\S+)", table, re.M).group(1))
        mass = float(re.search(r"relative difference\s+: (\S+)", table).group(1))
        self.assertAlmostEqual(values[2.0], (1 + delta) / (1 + mass) - 1, delta=2e-9)
        self.assertEqual(f"{values[4.995]:.1e}", "5.3e-06")


class TestBeatsHelpCommands(unittest.TestCase):
    def test_the_help_documents_a_meaningful_number_of_commands(self):
        self.assertGreaterEqual(len(dict.fromkeys(documented_commands(HELP_HTML))), 25)

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
                    self.assertEqual(lines[0], f"SphereGravity {physics.MODEL_VERSION} (build {physics.BUILD_ID})")
                    radii = (float(v) for v in arguments[arguments.index("--radii") + 1:]) \
                        if "--radii" in arguments else entry_point.REPORT_RADII
                    note = "  (Relative difference is undefined inside because expected g is zero.)"
                    self.assertEqual(note in lines, any(r < 1 for r in radii))

    def test_documented_options_are_real_options(self):
        parser = entry_point.build_parser()
        real = {s for action in parser._actions for s in action.option_strings}
        used = set(re.findall(r"(?<![\w-])(--[A-Za-z0-9_-]+)", "\n".join(documented_commands(HELP_HTML))))
        self.assertTrue(used)
        self.assertLessEqual(used, real)


class TestBeatsHelpReference(unittest.TestCase):
    def test_parameter_table_matches_the_parser_options_and_defaults(self):
        rows = {
            row[0]: row[1]
            for row in STRUCTURE.table_rows
            if len(row) == 3 and row[0].startswith("--")
        }
        options = {
            action.option_strings[0]: action
            for action in entry_point.build_parser()._actions
            if action.option_strings and action.option_strings[0] not in ("-h", "--version")
        }
        self.assertEqual(set(rows), set(options))
        for name, action in options.items():
            with self.subTest(option=name):
                if isinstance(action.default, list):
                    self.assertEqual([float(v) for v in rows[name].split()], action.default)
                elif isinstance(action.default, str):
                    self.assertEqual(rows[name], action.default)
                else:
                    self.assertEqual(float(rows[name]), float(action.default))

    def test_printed_summary_blocks_are_exactly_what_the_runs_print(self):
        blocks = re.findall(r"<pre>(SphereGravity .*?)</pre>", section_html(HELP_HTML, "summary"), re.DOTALL)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(html.unescape(blocks[0]).strip(), run_cli(()).stdout.strip())
        fine = documented_commands(section_html(HELP_HTML, "beat5"))[1]
        self.assertEqual(html.unescape(blocks[1]).strip(), beat_run(fine).stdout.strip())
        self.assertNotIn("*", blocks[1])

    def test_every_summary_label_is_described_in_the_help(self):
        text = html_text(section_html(HELP_HTML, "summary"))
        output = run_cli(()).stdout
        labels = [line.split(":", 1)[0].strip() for line in output.splitlines()[1:]
                  if ":" in line and not line.startswith("  (")]
        labels += ["predicted g", "expected g", "relative difference", "h^2 estimate", "undefined; g/M="]
        for label in labels:
            with self.subTest(label=label):
                self.assertIn(label, text)

    def test_constant_table_matches_the_program(self):
        rows = {
            row[0]: row[1]
            for row in STRUCTURE.table_rows
            if len(row) == 3 and not row[0].startswith("--") and row[0] != "Name"
            and not row[0].startswith("(")
        }
        for name in ("SHELL_RADIUS", "DEFAULT_EPSILON", "NUM_RADII", "RADIUS_STEP",
                     "SURFACE_INDEX", "MAX_NDIV", "MIN_EPSILON", "MAX_EPSILON",
                     "MAX_SAMPLE_RADIUS"):
            with self.subTest(name=name):
                self.assertEqual(float(rows[name]), float(getattr(physics, name)))
        self.assertEqual([float(v) for v in rows["REPORT_RADII"].split()], list(entry_point.REPORT_RADII))

    def test_code_identifiers_named_in_the_help_exist_in_the_program(self):
        modules = (physics, driver, plotting, entry_point)
        names = set()
        for body in re.findall(r"<code>([^<]+)</code>", HELP_HTML):
            body = html.unescape(body)
            match = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*)\(\)", body)
            if match:
                names.add(match.group(1))
            elif re.fullmatch(r"[A-Z][A-Z_0-9]{3,}", body):
                names.add(body)
        self.assertGreater(len(names), 12)
        for name in sorted(names):
            with self.subTest(name=name):
                self.assertTrue(any(hasattr(module, name) for module in modules), name)

    def test_selection_lines_quoted_in_the_help_are_the_ones_in_the_code(self):
        source = (MODULE_DIR / "physics_spheregravity.py").read_text(encoding="utf-8")
        for line in ("compute_acceleration_profile = compute_acceleration_profile_textbook",
                     "# compute_acceleration_profile = compute_acceleration_profile_optimized"):
            with self.subTest(line=line):
                self.assertIn(line, HELP_HTML)
                self.assertIn(line, source)

    def test_help_mentions_python_version_and_core_formulas(self):
        self.assertRegex(HELP_HTML, r"Python\s+3\.10\s+or\s+later")
        for item in (r"\frac{h/2}{\sin(h/2)}", r"d(\theta,r)=\sqrt{1+r^{2}+2r\sin\theta}",
                     r"\frac{h^{2}}{48}", "4\\pi G\\varepsilon"):
            with self.subTest(item=item):
                self.assertIn(item, HELP_HTML)


class TestBeatsHelpQuotedNumbers(unittest.TestCase):
    """Every number the beats quote from a run is the number the run prints."""

    NUMBER = re.compile(r"(?<![\w.])-?\d+\.\d+(?:e[+-]\d+)?(?![\w%]|\.\d)")
    WHOLE = re.compile(r"(?<![\w.+-])\d{4,}(?![\w%]|\.\d)")
    # Numbers a student works out from printed values, or constants of the
    # equations; each is checked in TestBeatsHelpQuantitativeClaims.
    DERIVED = {
        0: {"4.000", "0.012", "3400"},
        1: {"1.0000411245"},
        2: set(),
        3: set(),
        4: set(),
        5: {"0.012444", "0.005", "1257", "0.15"},
        6: {"0.15", "-0.15", "5.3e-06", "1000", "4.995"},
        7: set(),
        8: {"1000", "4.995"},
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
            run = beat_run(command)
            chunks.append(run.stdout + run.stderr)
        return "\n".join(chunks)

    def test_numbers_quoted_in_each_beat_are_printed_by_that_beat_s_commands(self):
        for number in BEAT_NUMBERS:
            body = section_html(HELP_HTML, f"beat{number}")
            printed = self.printed_for(body)
            text = self.visible(body)
            tokens = set(self.NUMBER.findall(text)) | set(self.WHOLE.findall(text))
            with self.subTest(beat=number):
                self.assertGreaterEqual(len(tokens), 3)
                missing = sorted(
                    t for t in tokens - self.DERIVED[number]
                    if not re.search(rf"(?<![\d.]){re.escape(t)}(?![\d])", printed)
                )
                self.assertEqual(missing, [])
                self.assertEqual(sorted(self.DERIVED[number] - tokens), [])

    def test_numbers_quoted_in_the_reference_sections_are_printed_or_checked(self):
        every_output = "\n".join(
            c + "\n" + beat_run(c).stdout for c in dict.fromkeys(documented_commands(HELP_HTML))
            if c not in REJECTED_COMMANDS and "--help" not in c
        )
        allowed = {"3.10", "4.995", "1000", "0.005", "0.001", "100000"}
        for section in ("overview", "beats", "algorithm", "modules", "quickstart",
                        "parameters", "output", "experiments"):
            text = self.visible(section_html(HELP_HTML, section))
            tokens = set(self.NUMBER.findall(text)) | set(self.WHOLE.findall(text))
            with self.subTest(section=section):
                self.assertEqual(sorted(t for t in tokens - allowed if t not in every_output), [])


def _table(command):
    """Parse a printed comparison table into {radius: (g, expected, comparison, estimate, marked)}."""
    rows = {}
    for line in beat_run(command).stdout.splitlines():
        match = re.match(r"^\s+(\d[\d.e+-]*)\s+(\S+)\s+(\S+)\s+(undefined; g/M=)?(\S+)\s+(\S+)( \*)?$", line)
        if match and not line.strip().startswith("radius"):
            rows[float(match.group(1))] = (float(match.group(2)), float(match.group(3)),
                                           float(match.group(5)), float(match.group(6)),
                                           bool(match.group(7)))
    return rows


def _mass_difference(command):
    return float(re.search(r"relative difference\s+: (\S+)", beat_run(command).stdout).group(1))


class TestBeatsHelpQuantitativeClaims(unittest.TestCase):
    """Independent checks of statements in the beats that go beyond the printed digits."""

    def test_beat0_inverse_square_and_interior(self):
        _, g = physics.compute_acceleration_at_radii(100, (0.5, 2.0, 4.0))
        self.assertEqual(f"{g[1] / g[2]:.3f}", "4.000")
        self.assertGreater(g[1] / abs(g[0]), 3400)
        radius, accel = physics.compute_acceleration_profile_optimized(100, "acceleration")
        near = accel[(radius > 1.0) & (radius < 1.05)]
        self.assertGreater(near.max(), 0.012)
        self.assertGreater(near.max(), physics.compute_continuum_shell_mass() / 1.015**2)
        self.assertLess(accel[(radius > 0.9) & (radius < 0.995)].min(), 0.0)
        self.assertLess(np.abs(accel[radius < 0.9]).max(), 3e-5)

    def test_beat1_midpoint_mass_formula(self):
        self.assertEqual(f"{(math.pi / 20) / math.sin(math.pi / 20):.7f}", "1.0041242")
        self.assertEqual(f"{(math.pi / 200) / math.sin(math.pi / 200):.10f}", "1.0000411245")
        coarse, fine = _mass_difference("python main.py --nDiv 10"), _mass_difference("python main.py")
        self.assertAlmostEqual(coarse / fine, 100, delta=0.5)
        self.assertGreater(physics.compute_shell_mass(10), physics.compute_continuum_shell_mass())

    def test_beat2_far_field_tends_to_the_mass_error(self):
        rows = _table("python main.py --radii 2 5 10 100 1000")
        self.assertEqual(f"{rows[1000.0][2]:.5g}", f"{_mass_difference('python main.py'):.5g}")
        values = [rows[r][2] for r in (2.0, 5.0, 10.0, 100.0, 1000.0)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_beat3_centre_and_interior_signs(self):
        rows = _table("python main.py --radii 0 0.25 0.5 0.75 0.9")
        self.assertLess(abs(rows[0.0][0]), 1e-15)
        self.assertEqual(rows[0.0][3], 0.0)
        for radius in (0.25, 0.5, 0.75, 0.9):
            with self.subTest(radius=radius):
                self.assertLess(rows[radius][2], 0.0)
        ring = physics.compute_continuum_shell_mass() / 2 * (math.pi / 100)  # n dm at the equator
        self.assertGreater(ring, 1e-5)
        self.assertLess(ring, 1e-3)

    def test_beat4_hundredfold_convergence_and_agreement_with_the_estimate(self):
        runs = [_table(c) for c in ("python main.py", "python main.py --nDiv 1000", "python main.py --nDiv 10000")]
        for coarse, fine in zip(runs, runs[1:]):
            for radius in (0.5, 2.0):
                with self.subTest(radius=radius):
                    self.assertAlmostEqual(coarse[radius][2] / fine[radius][2], 100, delta=0.2)
        self.assertEqual(f"{runs[1][2.0][2]:.4g}", f"{runs[1][2.0][3]:.4g}")
        self.assertEqual(f"{runs[2][2.0][2]:.5g}", f"{runs[2][2.0][3]:.5g}")

    def test_beat5_surface(self):
        coarse = _table("python main.py --radii 0.995 1.005 1.01 1.05 1.1")
        fine = _table("python main.py --nDiv 10000 --radii 0.995 1.005 1.01 1.05 1.1")
        self.assertTrue(all(coarse[r][4] for r in (0.995, 1.005, 1.01, 1.05)))
        self.assertFalse(coarse[1.1][4])
        self.assertEqual(f"{coarse[0.995][2]:.2f}", "0.15")
        self.assertFalse(any(row[4] for row in fine.values()))
        for radius in (0.995, 1.005):
            with self.subTest(radius=radius):
                self.assertLess(coarse[radius][2] * coarse[radius][3], 0.0)  # opposite signs
        _, g = physics.compute_acceleration_at_radii(10000, (0.995, 1.005))
        self.assertEqual(f"{g[1] - g[0]:.5g}", "0.012444")
        self.assertAlmostEqual((math.pi / 100) / 0.005, 6.28, delta=0.01)
        threshold = _table("python main.py --nDiv 1257 --radii 1.005")[1.005]
        self.assertFalse(threshold[4])
        self.assertAlmostEqual(abs(threshold[3] / threshold[2] - 1), 0.08, delta=0.01)
        self.assertAlmostEqual(0.005 / (math.pi / 10000), 15.9, delta=0.05)

    def test_beat6_plot_scale_and_definitions(self):
        self.assertEqual((physics.NUM_RADII, (physics.NUM_RADII - 1) * physics.RADIUS_STEP), (1000, 4.995))
        radius, rel = physics.compute_acceleration_profile_optimized(100, "relative difference")
        self.assertAlmostEqual(rel.max(), 0.15, delta=0.01)
        self.assertAlmostEqual(rel.min(), -0.15, delta=0.01)
        far = (radius > 1.05)
        self.assertLess(np.abs(rel[far]).max(), 0.01)
        rows = _table("python main.py")
        mass = _mass_difference("python main.py")
        self.assertEqual(f"{(1 + rows[5.0][2]) / (1 + mass) - 1:.1e}", "5.3e-06")
        self.assertEqual(f"{rel[-1]:.1e}", "5.3e-06")

    def test_beat7_epsilon_scaling(self):
        base = _table("python main.py --outputType acceleration")
        double = _table("python main.py --epsilon 0.002 --outputType acceleration")
        for radius in base:
            with self.subTest(radius=radius):
                self.assertAlmostEqual(double[radius][0] / base[radius][0], 2.0, delta=5e-4)
                self.assertEqual(double[radius][2], base[radius][2])

    @staticmethod
    def _experiment(number):
        match = re.search(rf'<h3 id="exp{number}">(.*?)(?=<h3 |$)',
                          section_html(HELP_HTML, "experiments"), re.DOTALL)
        return match.group(1)

    def test_experiment_7_commands_straddle_the_marking_threshold(self):
        commands = documented_commands(self._experiment(7))
        self.assertEqual(len(commands), 2)
        marks = []
        for command in commands:
            rows = _table(command)
            self.assertEqual(list(rows), [1.005])
            marks.append(rows[1.005][4])
        self.assertEqual(marks, [True, False])
        divisions = [int(command_arguments(c)[command_arguments(c).index("--nDiv") + 1]) for c in commands]
        self.assertEqual(divisions[1], divisions[0] + 1)

    def test_experiment_8_jump_matches_the_shell_theorem(self):
        fragment = self._experiment(8)
        commands = documented_commands(fragment)
        self.assertEqual(len(commands), 2)
        prose = html_text(fragment)
        epsilons = []
        for command in commands:
            arguments = command_arguments(command)
            radii = [float(v) for v in arguments[arguments.index("--radii") + 1:]]
            epsilon = (float(arguments[arguments.index("--epsilon") + 1])
                       if "--epsilon" in arguments else physics.DEFAULT_EPSILON)
            epsilons.append(epsilon)
            inner, outer = radii
            self.assertIn(f"4\\pi\\varepsilon/{outer:g}^2", prose)
            rows = _table(command)
            jump = rows[outer][0] - rows[inner][0]
            expected = 4 * math.pi * epsilon / outer**2
            with self.subTest(command=command):
                self.assertLess(inner, 1.0)
                self.assertGreater(outer, 1.0)
                self.assertAlmostEqual(jump / expected, 1.0, delta=1e-4)
                # the error of the sum is much smaller than the fall-off correction
                self.assertLess(abs(rows[outer][3]), 0.1 * (1 - 1 / outer**2))
                self.assertFalse(rows[outer][4] or rows[inner][4])
        self.assertEqual(epsilons[1], 2 * epsilons[0])  # "check that it doubles"

    def test_beat8_single_ring(self):
        self.assertEqual(f"{math.pi / 2 - 1:.4f}", "0.5708")
        rows = _table("python main.py --nDiv 1")
        self.assertTrue(all(row[4] for row in rows.values()))
        self.assertGreater(rows[0.5][2], 0.0)
        self.assertLess(rows[5.0][2], math.pi / 2 - 1)
        self.assertGreater(rows[5.0][2], rows[2.0][2])


class TestEndToEndSubprocess(unittest.TestCase):
    """The program runs as a student would run it, with a non-interactive backend."""

    def run_main(self, *arguments):
        environment = {**os.environ, "MPLBACKEND": "Agg"}
        return subprocess.run(
            [sys.executable, "main.py", *arguments], cwd=str(MODULE_DIR), env=environment,
            capture_output=True, text=True, timeout=120,
        )

    def test_default_run_prints_the_same_summary_as_in_process(self):
        completed = self.run_main()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, run_cli(()).stdout)

    def test_rejected_inputs_exit_with_a_message(self):
        for arguments, message in ((("--radii", "1"), "radius must not be the shell radius 1"),
                                   (("--nDiv", "0"), "value must be greater than zero"),
                                   (("--epsilon", "-1"), "positive finite number")):
            with self.subTest(arguments=arguments):
                completed = self.run_main(*arguments)
                self.assertEqual(completed.returncode, 2)
                self.assertIn(message, completed.stderr)


class TestOriginalHelpCompatibility(unittest.TestCase):
    """The Reference Guide version is optional; these tests never require it."""

    @classmethod
    def setUpClass(cls):
        cls.original = HELP_FILE.with_name("SphereGravity-original.html")
        if not cls.original.is_file():
            raise unittest.SkipTest("SphereGravity-original.html is not present in this layout")
        cls.help_text = cls.original.read_text(encoding="utf-8")
        cls.parser = HelpHTMLParser()
        cls.parser.feed(cls.help_text)
        cls.parser.close()

    def test_original_stamp_matches_the_program(self):
        match = re.search(
            r'<p\s+id="version_build"[^>]*>\s*Version\s+([^&<\s]+)(?:&nbsp;|\s)+Build\s+([0-9a-f]{12})',
            self.help_text,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.groups(), (physics.MODEL_VERSION, physics.BUILD_ID))

    def test_help_ids_are_unique_and_local_navigation_targets_exist(self):
        self.assertEqual(len(self.parser.ids), len(set(self.parser.ids)))
        self.assertEqual(set(self.parser.local_targets) - set(self.parser.ids), set())

    def test_help_loads_mathjax_from_the_documented_public_cdn(self):
        self.assertTrue(any("mathjax" in source.lower() for source in self.parser.script_sources))

    def test_help_has_exactly_four_module_cards(self):
        self.assertEqual(self.parser.module_card_count, 4)

    def test_help_describes_actual_radial_grid_and_surface_placeholder(self):
        self.assertIn(r"r \in [0, 4.995]", self.help_text)
        self.assertIn("compatibility placeholder", self.help_text)
        self.assertIn("plot leaves a gap", self.help_text)

    def test_help_documents_both_output_modes(self):
        self.assertIn("--outputType acceleration", self.help_text)
        self.assertIn("--outputType relative_difference", self.help_text)

    def test_help_documents_driver_epsilon_parameter(self):
        self.assertRegex(
            self.help_text,
            r"run_spheregravity\(nDiv=100,\s*outputType='acceleration',\s*epsilon=0\.001\)",
        )

    def test_help_documents_every_command_line_parameter(self):
        for option in ("--nDiv", "--outputType", "--epsilon", "--radii", "--version"):
            with self.subTest(option=option):
                self.assertIn(option, self.help_text)

    def test_help_documents_printed_comparison_radii_and_estimate(self):
        for radius in ("0.5", "0.995", "1.005", "2", "3", "4", "5"):
            with self.subTest(radius=radius):
                self.assertIn(radius, self.help_text)
        self.assertIn("h^2 estimate", self.help_text)

    def test_help_documents_interchangeable_implementations(self):
        self.assertIn(
            "compute_acceleration_profile = compute_acceleration_profile_textbook",
            self.help_text,
        )
        self.assertIn(
            "# compute_acceleration_profile = compute_acceleration_profile_optimized",
            self.help_text,
        )
        self.assertIn("identical signatures and return values", self.help_text)

    def test_exercises_are_numbered_and_ranked(self):
        expected_headings = (
            "1 · Introductory",
            "2 · Introductory–Intermediate",
            "3 · Intermediate",
            "4 · Intermediate",
            "5 · Intermediate–Advanced",
            "6 · Advanced Programming Extension",
            "7 · Advanced — NumPy Vectorization",
        )
        positions = []
        for heading in expected_headings:
            self.assertEqual(self.help_text.count(heading), 1)
            positions.append(self.help_text.index(heading))
        self.assertEqual(positions, sorted(positions))

    def test_vectorization_exercise_covers_scientific_computing_tradeoffs(self):
        self.assertIn("Identify which textbook loops", self.help_text)
        self.assertIn("equivalent results in both output modes", self.help_text)
        self.assertIn("bounded chunks", self.help_text)
        self.assertIn("algorithmic transparency, execution speed, and memory", self.help_text)

    def test_license_retains_java_provenance(self):
        license_section = self.help_text.split('<section id="license">', 1)[1]
        self.assertIn("original Java programs", license_section)
        self.assertIn("Bernard Schutz", license_section)
        self.assertIn("Thomas Yee", license_section)

    def test_original_commands_run(self):
        commands = documented_commands(self.help_text)
        self.assertGreaterEqual(len(commands), 4)
        for command in dict.fromkeys(commands):
            arguments = command_arguments(command)
            if "--help" in arguments:
                continue
            with self.subTest(command=command):
                self.assertIn(run_cli(arguments).exit_code, (None, 0))

    def test_original_array_lookup_snippet_runs_inside_main(self):
        # Experiment 2 tells students to add the lookup right after the call in
        # main.py, where radius and accel already exist.
        snippet = re.search(r"<pre># r = 1\.1 sits on the 0\.005 grid.*?</pre>", self.help_text, re.DOTALL)
        self.assertIsNotNone(snippet)
        code = html.unescape(re.sub(r"<[^>]+>", "", snippet.group(0)))
        source = (MODULE_DIR / "main.py").read_text(encoding="utf-8")
        self.assertIn("    radius, accel = run_spheregravity(", source)
        radius, accel = driver.run_spheregravity(nDiv=20, outputType="relative difference")
        namespace = {"radius": radius, "accel": accel}
        output = io.StringIO()
        with redirect_stdout(output):
            exec(code, namespace)
        self.assertTrue(output.getvalue().startswith("1.1 "))

    def test_original_relative_links_resolve_when_the_documentation_tree_is_present(self):
        docs_root = self.original.parent.parent
        if docs_root.name != "GFTGU-Documentation" or not (docs_root / "Star").is_dir():
            self.skipTest("the sibling documentation folders are not present in this layout")
        links = re.findall(r'href="(\.\./[^"]+)"', self.help_text)
        self.assertEqual(len(links), 4)
        for href in links:
            with self.subTest(href=href):
                self.assertTrue((self.original.parent / href).is_file(), href)


if __name__ == "__main__":
    unittest.main()
