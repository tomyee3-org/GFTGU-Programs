"""Regression tests for the SphereGravity tutorial program.

The discovery logic deliberately supports both the repository layout
(`tests/test_physics_spheregravity.py`) and a flattened upload in which this
file is placed beside the four core modules.
"""

import ast
import hashlib
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


HELP_FILE = MODULE_DIR / "SphereGravity.html"


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
            ):
                with self.subTest(value=value, function=implementation.__name__):
                    with self.assertRaisesRegex(ValueError, "positive finite"):
                        implementation(8, epsilon=value)

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
        exact_mass = 4.0 * math.pi * physics.DEFAULT_EPSILON
        errors = [
            abs(physics.compute_shell_mass(n_div) - exact_mass)
            for n_div in (10, 100, 1000)
        ]
        self.assertGreater(errors[0] / errors[1], 95.0)
        self.assertLess(errors[0] / errors[1], 105.0)
        self.assertGreater(errors[1] / errors[2], 95.0)
        self.assertLess(errors[1] / errors[2], 105.0)


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
        self.assertEqual(completed.stdout.strip(), expected)

    def test_main_uses_documented_user_settings(self):
        fake_radius = np.array([0.0, 0.5])
        fake_acceleration = np.array([0.0, 0.0])
        with mock.patch.object(
            entry_point,
            "run_spheregravity",
            return_value=(fake_radius, fake_acceleration),
        ) as run_mock, mock.patch.object(
            entry_point, "plot_spheregravity"
        ) as plot_mock, mock.patch("builtins.print"):
            entry_point.main([])

        run_mock.assert_called_once_with(
            nDiv=entry_point.nDiv, outputType=entry_point.outputType
        )
        plot_mock.assert_called_once_with(
            fake_radius, fake_acceleration, outputType=entry_point.outputType
        )


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


class TestHelpContent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.help_text = HELP_FILE.read_text(encoding="utf-8")
        cls.parser = HelpHTMLParser()
        cls.parser.feed(cls.help_text)
        cls.parser.close()

    def test_help_ids_are_unique_and_local_navigation_targets_exist(self):
        self.assertEqual(len(self.parser.ids), len(set(self.parser.ids)))
        self.assertEqual(
            set(self.parser.local_targets) - set(self.parser.ids),
            set(),
        )

    def test_help_loads_mathjax_from_the_documented_public_cdn(self):
        self.assertTrue(
            any("mathjax" in source.lower() for source in self.parser.script_sources)
        )

    def test_help_has_exactly_four_module_cards(self):
        self.assertEqual(self.parser.module_card_count, 4)

    def test_help_describes_actual_radial_grid_and_surface_placeholder(self):
        self.assertIn(r"r \in [0, 4.995]", self.help_text)
        self.assertIn("compatibility placeholder", self.help_text)
        self.assertIn("plot leaves a gap", self.help_text)

    def test_help_documents_both_output_modes(self):
        self.assertIn("outputType = 'acceleration'", self.help_text)
        self.assertIn("outputType = 'relative difference'", self.help_text)

    def test_help_documents_driver_epsilon_parameter(self):
        self.assertRegex(
            self.help_text,
            r"run_spheregravity\(nDiv=100,\s*outputType='acceleration',\s*epsilon=0\.001\)",
        )

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


if __name__ == "__main__":
    unittest.main()
