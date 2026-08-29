"""Regression and adversarial tests for Planck2.

The locator deliberately supports both repository layouts used for review:
this file may live in ``tests/`` or may be flattened beside the four program
modules during upload.
"""

import ast
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest import mock


CORE_MODULE_FILENAMES = (
    "planck2_physics.py",
    "planck2_driver.py",
    "main.py",
    "planck2_plot.py",
)


def find_module_dir(start):
    """Find the nearest ancestor containing all four Planck2 core modules."""
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        if all((directory / name).is_file() for name in CORE_MODULE_FILENAMES):
            return directory

    names = ", ".join(CORE_MODULE_FILENAMES)
    raise FileNotFoundError(
        f"Could not find a directory containing all Planck2 modules: {names}"
    )


MODULE_DIR = find_module_dir(Path(__file__))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import main as planck2_main  # noqa: E402
import planck2_driver as driver  # noqa: E402
import planck2_physics as phys  # noqa: E402
import planck2_plot as plotter  # noqa: E402


HELP_FILE = MODULE_DIR / "Planck2.html"


def relative_error(actual, expected):
    return abs(actual - expected) / abs(expected)


def independent_build_id():
    digest = hashlib.sha256()
    for name in CORE_MODULE_FILENAMES:
        content = (MODULE_DIR / name).read_text(encoding="utf-8").encode("utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()[:12]


class TestFileLocationAndReleaseMetadata(unittest.TestCase):
    def test_find_module_dir_from_module_and_nested_directory(self):
        self.assertEqual(find_module_dir(MODULE_DIR), MODULE_DIR)
        self.assertEqual(find_module_dir(MODULE_DIR / "main.py"), MODULE_DIR)
        self.assertEqual(find_module_dir(Path(__file__).parent), MODULE_DIR)

    def test_find_module_dir_chooses_nearest_complete_directory(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            outer = root / "outer"
            inner = outer / "inner"
            inner.mkdir(parents=True)
            for directory in (outer, inner):
                for name in CORE_MODULE_FILENAMES:
                    (directory / name).touch()
            self.assertEqual(find_module_dir(inner), inner)

    def test_find_module_dir_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temp_name:
            with self.assertRaisesRegex(FileNotFoundError, "all Planck2 modules"):
                find_module_dir(temp_name)

    def test_core_source_files_exist(self):
        for name in CORE_MODULE_FILENAMES:
            self.assertTrue((MODULE_DIR / name).is_file(), name)

    def test_sources_parse_as_python_3_10(self):
        for name in CORE_MODULE_FILENAMES:
            source = (MODULE_DIR / name).read_text(encoding="utf-8")
            ast.parse(source, filename=name, feature_version=(3, 10))

    def test_build_id_cover_list_is_exact(self):
        self.assertEqual(phys.BUILD_ID_COVERS, CORE_MODULE_FILENAMES)

    def test_build_id_matches_independent_recalculation(self):
        self.assertRegex(phys.BUILD_ID, r"^[0-9a-f]{12}$")
        self.assertEqual(phys.BUILD_ID, independent_build_id())
        self.assertEqual(phys.BUILD_ID, phys._compute_build_id())

    def test_help_version_and_build_match_program(self):
        text = HELP_FILE.read_text(encoding="utf-8")
        match = re.search(
            r'<p id="version_build"[^>]*>\s*Version\s+([^&<\s]+)'
            r'(?:&nbsp;)+Build\s+([0-9a-f]{12})\s*</p>',
            text,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), phys.MODEL_VERSION)
        self.assertEqual(match.group(2), phys.BUILD_ID)

    def test_command_line_version_matches_metadata(self):
        completed = subprocess.run(
            [sys.executable, "main.py", "--version"],
            cwd=MODULE_DIR,
            text=True,
            capture_output=True,
            check=True,
        )
        expected = f"Planck2 {phys.MODEL_VERSION} (build {phys.BUILD_ID})"
        self.assertEqual(completed.stdout.strip(), expected)

    @unittest.skipIf(
        os.environ.get("PLANCK2_SKIP_FLAT_LAYOUT_TEST") == "1",
        "avoid recursion inside the flattened-layout subprocess",
    )
    def test_complete_suite_runs_when_flattened_beside_modules(self):
        with tempfile.TemporaryDirectory() as temp_name:
            flat = Path(temp_name)
            for name in (*CORE_MODULE_FILENAMES, HELP_FILE.name):
                shutil.copy2(MODULE_DIR / name, flat / name)
            copied_test = flat / Path(__file__).name
            shutil.copy2(Path(__file__), copied_test)
            env = os.environ.copy()
            env["MPLBACKEND"] = "Agg"
            env["PLANCK2_SKIP_FLAT_LAYOUT_TEST"] = "1"
            completed = subprocess.run(
                [sys.executable, copied_test.name, "-q"],
                cwd=flat,
                env=env,
                text=True,
                capture_output=True,
                timeout=120,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )


class TestPhysicalConstantsAndConversions(unittest.TestCase):
    def test_si_constants_are_exact_defining_values(self):
        self.assertEqual(phys.H_PLANCK, 6.62607015e-34)
        self.assertEqual(phys.C_LIGHT, 2.99792458e8)
        self.assertEqual(phys.K_BOLTZMANN, 1.380649e-23)

    def test_stefan_boltzmann_constant(self):
        accepted = 5.670374419e-8
        self.assertLess(relative_error(phys.SIGMA_SB, accepted), 1e-10)

    def test_wavelength_and_frequency_conversions_are_inverse(self):
        for temperature in (2.725, 300.0, 5900.0, 1.0e7):
            for x in (0.01, 1.0, 2.821439, 4.965114, 100.0):
                wavelength = phys.x_to_wavelength(x, temperature)
                frequency = phys.x_to_frequency(x, temperature)
                self.assertAlmostEqual(
                    wavelength * frequency / phys.C_LIGHT,
                    1.0,
                    places=14,
                )

    def test_low_temperature_conversion_avoids_premature_underflow(self):
        wavelength = phys.x_to_wavelength(0.01, 1.0e-200)
        frequency = phys.x_to_frequency(100.0, 1.0e-200)
        self.assertTrue(math.isfinite(wavelength))
        self.assertGreater(wavelength, 0.0)
        self.assertTrue(math.isfinite(frequency))
        self.assertGreater(frequency, 0.0)

    def test_coordinate_jacobians(self):
        temperature = 5900.0
        x = 3.0
        expected_wavelength = phys.WIEN_SCALE / (temperature * x * x)
        expected_frequency = phys.FREQUENCY_SCALE * temperature
        self.assertAlmostEqual(
            phys.coordinate_jacobian("wavelength", x, temperature),
            expected_wavelength,
            places=15,
        )
        self.assertAlmostEqual(
            phys.coordinate_jacobian("frequency", x, temperature),
            expected_frequency,
            places=15,
        )
        self.assertEqual(
            phys.coordinate_jacobian("frequency", x, temperature),
            phys.coordinate_jacobian("energy_density", x, temperature),
        )

    def test_prefactor_relations(self):
        temperature = 5900.0
        b_nu = phys.prefactor("frequency", temperature)
        u_nu = phys.prefactor("energy_density", temperature)
        self.assertAlmostEqual(
            u_nu / b_nu,
            4.0 * math.pi / phys.C_LIGHT,
            places=22,
        )

    def test_exact_bolometric_integrals(self):
        temperature = 5900.0
        radiance = phys.SIGMA_SB * temperature**4 / math.pi
        energy_density = 4.0 * phys.SIGMA_SB * temperature**4 / phys.C_LIGHT
        self.assertEqual(phys.exact_physical_integral("wavelength", temperature), radiance)
        self.assertEqual(phys.exact_physical_integral("frequency", temperature), radiance)
        self.assertEqual(
            phys.exact_physical_integral("energy_density", temperature),
            energy_density,
        )

    def test_units_labels(self):
        self.assertIn("Wavelength", phys.units_label("wavelength")[0])
        self.assertIn("Frequency", phys.units_label("frequency")[0])
        self.assertIn("radiance", phys.units_label("frequency")[1])
        self.assertIn("energy density", phys.units_label("energy_density")[1])
        self.assertEqual(phys.physical_integral_units("wavelength"), "W m^-2 sr^-1")
        self.assertEqual(phys.physical_integral_units("frequency"), "W m^-2 sr^-1")
        self.assertEqual(phys.physical_integral_units("energy_density"), "J m^-3")


class TestDomainAndInputValidation(unittest.TestCase):
    def test_default_domain_is_valid(self):
        phys.PlanckDomain().validate()

    def test_domain_rejects_non_numeric_boolean_and_nonfinite_values(self):
        invalid_values = ("0.01", True, math.nan, math.inf, -math.inf)
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite numbers"):
                    phys.PlanckDomain(x_min=value).validate()

    def test_domain_rejects_invalid_orderings(self):
        cases = (
            phys.PlanckDomain(x_min=0.0),
            phys.PlanckDomain(x_min=-1.0),
            phys.PlanckDomain(x_min=2.0, x_max=1.0),
            phys.PlanckDomain(x_low=0.001),
            phys.PlanckDomain(x_low=20.0, x_high=20.0),
            phys.PlanckDomain(x_high=101.0),
        )
        for domain in cases:
            with self.subTest(domain=domain):
                with self.assertRaises(ValueError):
                    domain.validate()

    def test_quantity_validation_accepts_only_three_strings(self):
        for quantity in phys.SHAPE_EXPONENT:
            phys.validate_quantity(quantity)
        for quantity in ("", "WAVELENGTH", "lambda", None, 3, [], True):
            with self.subTest(quantity=quantity):
                with self.assertRaisesRegex(ValueError, "quantity must be"):
                    phys.validate_quantity(quantity)

    def test_temperature_validation_across_public_functions(self):
        invalid = (0.0, -1.0, math.nan, math.inf, -math.inf, True, "5900")
        functions = (
            lambda value: phys.prefactor("wavelength", value),
            lambda value: phys.coordinate_jacobian("frequency", 1.0, value),
            lambda value: phys.exact_physical_integral("frequency", value),
            lambda value: phys.x_to_wavelength(1.0, value),
            lambda value: phys.x_to_frequency(1.0, value),
            lambda value: driver.run_planck2(value, "frequency", 10),
        )
        for function in functions:
            for value in invalid:
                with self.subTest(function=function, value=value):
                    with self.assertRaises(ValueError):
                        function(value)

    def test_x_validation_across_public_functions(self):
        invalid = (0.0, -1.0, math.nan, math.inf, -math.inf, True, "1")
        functions = (
            lambda value: phys.ln_shape_function(value, 3, phys.PlanckDomain()),
            lambda value: phys.coordinate_jacobian("frequency", value, 5900.0),
            lambda value: phys.x_to_wavelength(value, 5900.0),
            lambda value: phys.x_to_frequency(value, 5900.0),
        )
        for function in functions:
            for value in invalid:
                with self.subTest(function=function, value=value):
                    with self.assertRaises(ValueError):
                        function(value)

    def test_resolution_validation(self):
        invalid = (0, -1, 1.5, True, "2000", driver.MAX_STEPS + 1)
        for n_steps in invalid:
            with self.subTest(n_steps=n_steps):
                with self.assertRaisesRegex(ValueError, "integer from 1"):
                    driver.run_planck2(5900.0, "wavelength", n_steps)

    def test_unrepresentable_step_size_is_rejected(self):
        smallest = math.nextafter(0.0, 1.0)
        domain = phys.PlanckDomain(
            x_min=smallest,
            x_max=2.0 * smallest,
            x_low=smallest,
            x_high=2.0 * smallest,
        )
        with self.assertRaisesRegex(ValueError, "step size"):
            driver.run_planck2(5900.0, "frequency", 2, domain)


class TestShapeFunction(unittest.TestCase):
    def setUp(self):
        self.domain = phys.PlanckDomain()

    def test_intermediate_branch_matches_planck_shape(self):
        for p, quantity in ((5, "wavelength"), (3, "frequency"), (3, "energy_density")):
            for x in (0.05, 0.1, 1.0, 5.0, 20.0):
                expected = x**p / math.expm1(x)
                actual = phys.shape_function(x, quantity, self.domain)
                self.assertLess(relative_error(actual, expected), 2e-14)

    def test_small_x_branch_matches_declared_rayleigh_jeans_limit(self):
        x = 0.01
        for p, quantity in ((5, "wavelength"), (3, "frequency")):
            self.assertAlmostEqual(
                phys.shape_function(x, quantity, self.domain),
                x ** (p - 1),
                delta=x ** (p - 1) * 1e-14,
            )

    def test_large_x_branch_matches_declared_wien_limit(self):
        x = 30.0
        for p, quantity in ((5, "wavelength"), (3, "frequency")):
            expected = math.exp(p * math.log(x) - x)
            self.assertAlmostEqual(
                phys.shape_function(x, quantity, self.domain),
                expected,
                delta=expected * 2e-14,
            )

    def test_shape_exponents(self):
        self.assertEqual(phys.SHAPE_EXPONENT["wavelength"], 5)
        self.assertEqual(phys.SHAPE_EXPONENT["frequency"], 3)
        self.assertEqual(phys.SHAPE_EXPONENT["energy_density"], 3)


class TestDriverScientificResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {
            quantity: driver.run_planck2(5900.0, quantity, 2000)
            for quantity in phys.SHAPE_EXPONENT
        }

    def test_grid_has_exact_interval_and_endpoint_counts(self):
        result = driver.run_planck2(5900.0, "frequency", 10)
        self.assertEqual(len(result.x_values), 11)
        self.assertEqual(len(result.coord_values), 11)
        self.assertEqual(len(result.y_values), 11)
        self.assertEqual(result.x_values[0], phys.PlanckDomain().x_min)
        self.assertEqual(result.x_values[-1], phys.PlanckDomain().x_max)

    def test_result_carries_release_metadata(self):
        result = self.results["wavelength"]
        self.assertEqual(result.model_version, phys.MODEL_VERSION)
        self.assertEqual(result.build_id, phys.BUILD_ID)

    def test_dimensionless_area_alias(self):
        result = self.results["frequency"]
        self.assertEqual(result.area, result.dimensionless_area)

    def test_analytic_peak_positions_within_half_a_grid_step(self):
        dx = (phys.PlanckDomain().x_max - phys.PlanckDomain().x_min) / 2000
        expected = {
            "wavelength": 4.965114231744276,
            "frequency": 2.821439372122079,
            "energy_density": 2.821439372122079,
        }
        for quantity, x_exact in expected.items():
            with self.subTest(quantity=quantity):
                self.assertLessEqual(
                    abs(self.results[quantity].x_peak - x_exact),
                    dx / 2.0,
                )

    def test_default_solar_wavelength_peak(self):
        peak_nm = self.results["wavelength"].coord_peak * 1e9
        self.assertLess(abs(peak_nm - 491.15), 1.0)

    def test_cmb_frequency_peak(self):
        result = driver.run_planck2(2.725, "frequency", 20000)
        peak_ghz = result.coord_peak / 1e9
        self.assertLess(abs(peak_ghz - 160.22), 0.2)

    def test_dimensionless_areas_match_gamma_zeta_integrals(self):
        expected = {
            "wavelength": 8.0 * math.pi**6 / 63.0,
            "frequency": math.pi**4 / 15.0,
            "energy_density": math.pi**4 / 15.0,
        }
        for quantity, exact in expected.items():
            with self.subTest(quantity=quantity):
                self.assertLess(
                    relative_error(self.results[quantity].dimensionless_area, exact),
                    1e-6,
                )

    def test_physical_integrals_match_stefan_boltzmann_references(self):
        for quantity, result in self.results.items():
            with self.subTest(quantity=quantity):
                self.assertLess(
                    relative_error(result.physical_integral, result.exact_physical_integral),
                    2e-6,
                )

    def test_frequency_radiance_and_energy_density_have_same_shape(self):
        frequency = self.results["frequency"]
        energy = self.results["energy_density"]
        self.assertEqual(frequency.x_peak, energy.x_peak)
        self.assertEqual(frequency.coord_peak, energy.coord_peak)
        expected_ratio = 4.0 * math.pi / phys.C_LIGHT
        self.assertLess(
            relative_error(energy.y_peak / frequency.y_peak, expected_ratio),
            3e-15,
        )
        self.assertLess(
            relative_error(
                energy.physical_integral / frequency.physical_integral,
                expected_ratio,
            ),
            3e-15,
        )

    def test_wavelength_and_frequency_spectra_transform_consistently(self):
        temperature = 5900.0
        domain = phys.PlanckDomain(x_min=1.0, x_max=5.0, x_low=1.0, x_high=5.0)
        wavelength = driver.run_planck2(temperature, "wavelength", 40, domain)
        frequency = driver.run_planck2(temperature, "frequency", 40, domain)
        for b_lambda, b_nu, lambda_value in zip(
            wavelength.y_values,
            frequency.y_values,
            wavelength.coord_values,
        ):
            transformed = b_lambda * lambda_value**2 / phys.C_LIGHT
            self.assertLess(relative_error(b_nu, transformed), 3e-14)

    def test_temperature_scaling(self):
        low = driver.run_planck2(3000.0, "wavelength", 2000)
        high = driver.run_planck2(6000.0, "wavelength", 2000)
        self.assertAlmostEqual(high.coord_peak / low.coord_peak, 0.5, places=14)
        self.assertAlmostEqual(high.physical_integral / low.physical_integral, 16.0, places=12)

    def test_truncating_domain_reduces_integral(self):
        full = self.results["frequency"]
        short_domain = phys.PlanckDomain(x_min=0.01, x_max=2.0, x_low=0.05, x_high=1.5)
        truncated = driver.run_planck2(5900.0, "frequency", 2000, short_domain)
        self.assertLess(truncated.dimensionless_area, full.dimensionless_area)
        self.assertLess(truncated.physical_integral, full.physical_integral)


class TestPlotting(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_plot_sorts_physical_coordinates_left_to_right(self):
        result = driver.run_planck2(5900.0, "wavelength", 100)
        with mock.patch.object(plt, "show"):
            plotter.plot_planck2(result)
        x_data = list(plt.gca().lines[0].get_xdata())
        self.assertEqual(x_data, sorted(x_data))

    def test_all_annotation_corners_are_accepted(self):
        result = driver.run_planck2(5900.0, "frequency", 20)
        for corner in ("upper right", "upper left", "lower right", "lower left"):
            with self.subTest(corner=corner), mock.patch.object(plt, "show"):
                plotter.plot_planck2(result, corner=corner)
                plt.close("all")

    def test_invalid_annotation_corner_is_rejected(self):
        result = driver.run_planck2(5900.0, "frequency", 20)
        with self.assertRaisesRegex(ValueError, "corner must be"):
            plotter.plot_planck2(result, corner="center")

    def test_window_fraction_validation(self):
        result = driver.run_planck2(5900.0, "frequency", 20)
        for value in (-0.1, 1.1, math.nan, math.inf, True, "0.1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite number from 0 to 1"):
                    plotter.plot_planck2(result, y_frac_window=value)

    def test_unit_threshold_produces_nonzero_axis_interval_without_warning(self):
        result = driver.run_planck2(5900.0, "frequency", 20)
        with warnings.catch_warnings(record=True) as caught, mock.patch.object(plt, "show"):
            warnings.simplefilter("always")
            plotter.plot_planck2(result, y_frac_window=1.0)
        left, right = plt.gca().get_xlim()
        self.assertLess(left, right)
        messages = [str(item.message) for item in caught]
        self.assertFalse(any("identical low and high" in message for message in messages))

    def test_zero_window_fraction_keeps_full_coordinate_domain(self):
        result = driver.run_planck2(5900.0, "frequency", 20)
        with mock.patch.object(plt, "show"):
            plotter.plot_planck2(result, y_frac_window=0.0)
        left, right = plt.gca().get_xlim()
        self.assertLessEqual(left, min(result.coord_values))
        self.assertGreaterEqual(right, max(result.coord_values))


class TestHelpFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = HELP_FILE.read_text(encoding="utf-8")

    def test_help_file_exists_and_is_html5_utf8(self):
        self.assertTrue(HELP_FILE.is_file())
        self.assertIn("<!DOCTYPE html>", self.text[:100])
        self.assertRegex(self.text[:500], r'<meta charset="utf-8"\s*/?>')

    def test_help_lists_current_defaults(self):
        expected_fragments = (
            '<td class="pdefault">5900.0</td>',
            '<td class="pdefault">"wavelength"</td>',
            '<td class="pdefault">2000</td>',
            '<td class="pdefault">0.01</td>',
            '<td class="pdefault">100.0</td>',
            '<td class="pdefault">0.05</td>',
            '<td class="pdefault">20.0</td>',
            '<td class="pdefault">"upper right"</td>',
            '<td class="pdefault">0.003</td>',
        )
        for fragment in expected_fragments:
            self.assertIn(fragment, self.text)

    def test_help_describes_three_quantity_modes_and_units(self):
        for quantity in ('"wavelength"', '"frequency"', '"energy_density"'):
            self.assertIn(quantity, self.text)
        for units in ("W m⁻³ sr⁻¹", "W m⁻² sr⁻¹ Hz⁻¹", "J m⁻³ Hz⁻¹"):
            self.assertIn(units, self.text)

    def test_help_distinguishes_dimensionless_and_physical_integrals(self):
        self.assertIn("distinct from", self.text)
        self.assertIn("coordinate Jacobian", self.text)
        self.assertIn(r"\frac{\sigma T^4}{\pi}", self.text)
        self.assertIn(r"\frac{4\sigma}{c}T^4", self.text)

    def test_help_documents_validation_contract(self):
        self.assertIn("1–1,000,000", self.text)
        self.assertIn("representable positive step", self.text)
        self.assertIn("finite range 0–1", self.text)

    def test_exercises_are_sequential_and_progressive(self):
        numbers = re.findall(r'<div class="ec-num">EXP-(\d+)</div>', self.text)
        self.assertEqual(numbers, [str(number) for number in range(1, 9)])
        headings = re.findall(
            r'<div class="ec-num">EXP-\d+</div><h4>(.*?)</h4>',
            self.text,
        )
        self.assertEqual(headings[0], "Solar Black-Body Approximation")
        self.assertEqual(headings[-1], "Rayleigh–Jeans and Wien Limits")

    def test_student_content_contains_no_ai_or_review_history(self):
        student_text = self.text.split('<section id="license">', 1)[0]
        for term in ("Claude", "Copilot", "Gemini", "AI-generated", "audit round"):
            self.assertNotIn(term, student_text)

    def test_java_provenance_is_confined_to_license(self):
        before_license, license_and_after = self.text.split('<section id="license">', 1)
        self.assertNotIn("Java/Triana", before_license)
        self.assertIn("Java/Triana", license_and_after)

    def test_first_edition_investigations_are_retained(self):
        self.assertIn("Investigations 10.1, 10.2, and 10.3", self.text)


class TestMainModule(unittest.TestCase):
    def test_closed_form_dimensionless_areas(self):
        self.assertEqual(planck2_main._exact_dimensionless_area(3), math.pi**4 / 15.0)
        self.assertEqual(planck2_main._exact_dimensionless_area(5), 8.0 * math.pi**6 / 63.0)
        with self.assertRaisesRegex(ValueError, "No closed form"):
            planck2_main._exact_dimensionless_area(4)

    def test_default_program_runs_headlessly(self):
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        completed = subprocess.run(
            [sys.executable, "main.py"],
            cwd=MODULE_DIR,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn(f"Planck2 {phys.MODEL_VERSION} (build {phys.BUILD_ID})", completed.stdout)
        self.assertIn("Dimensionless area", completed.stdout)
        self.assertIn("Exact bolometric value", completed.stdout)


if __name__ == "__main__":
    unittest.main()
