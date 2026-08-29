"""Regression and adversarial tests for Planck2.

The locator deliberately supports both repository layouts used for review:
this file may live in ``tests/`` or may be flattened beside the four program
modules during upload.
"""

import ast
import hashlib
from html.parser import HTMLParser
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


def independent_build_id(module_dir=MODULE_DIR):
    """Reproduce the release framing from raw bytes, independently of open()."""
    digest = hashlib.sha256()
    for name in CORE_MODULE_FILENAMES:
        raw = (Path(module_dir) / name).read_bytes()
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        content = text.encode("utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()[:12]


class HelpStructureParser(HTMLParser):
    """Collect IDs, links, and table rows without third-party dependencies."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.hrefs = []
        self.rows = []
        self.section_id = None
        self.in_row = False
        self.in_cell = False
        self.current_row = []
        self.current_cell = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.ids.append(element_id)
        if tag == "a" and "href" in attributes:
            self.hrefs.append((self.section_id, attributes["href"]))
        if tag == "section":
            self.section_id = element_id
        elif tag == "tr":
            self.in_row = True
            self.current_row = []
        elif tag in ("td", "th") and self.in_row:
            self.in_cell = True
            self.current_cell = []

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.in_cell:
            text = " ".join("".join(self.current_cell).split())
            self.current_row.append(text)
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            self.rows.append((self.section_id, tuple(self.current_row)))
            self.in_row = False
        elif tag == "section":
            self.section_id = None


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

    def test_build_id_is_invariant_under_lf_crlf_and_classic_cr(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            trees = [root / name for name in ("lf", "crlf", "cr")]
            endings = (b"\n", b"\r\n", b"\r")
            for tree, ending in zip(trees, endings):
                tree.mkdir()
                for name in CORE_MODULE_FILENAMES:
                    (tree / name).write_bytes(ending.join((b"alpha", b"beta", b"")))
            build_ids = [phys._compute_build_id(tree) for tree in trees]
            self.assertEqual(build_ids, [build_ids[0]] * len(build_ids))
            self.assertEqual(independent_build_id(trees[0]), build_ids[0])

    def test_build_id_treats_bom_and_unicode_normalization_as_changes(self):
        base = {name: "value = 'plain'\n" for name in CORE_MODULE_FILENAMES}
        with_bom = dict(base)
        with_bom[CORE_MODULE_FILENAMES[0]] = "\ufeff" + with_bom[CORE_MODULE_FILENAMES[0]]
        self.assertNotEqual(
            phys._build_id_from_texts(base),
            phys._build_id_from_texts(with_bom),
        )

        composed = dict(base)
        decomposed = dict(base)
        composed[CORE_MODULE_FILENAMES[0]] = "label = 'é'\n"
        decomposed[CORE_MODULE_FILENAMES[0]] = "label = 'e\u0301'\n"
        self.assertNotEqual(
            phys._build_id_from_texts(composed),
            phys._build_id_from_texts(decomposed),
        )

    def test_build_id_returns_unknown_for_incomplete_tree(self):
        with tempfile.TemporaryDirectory() as temp_name:
            self.assertEqual(phys._compute_build_id(temp_name), "unknown")

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

    def test_flat_layout_import_and_driver_smoke(self):
        with tempfile.TemporaryDirectory() as temp_name:
            flat = Path(temp_name)
            for name in CORE_MODULE_FILENAMES:
                shutil.copy2(MODULE_DIR / name, flat / name)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from planck2_driver import run_planck2; "
                        "r=run_planck2(5900.0, 'frequency', 20); "
                        "assert len(r.x_values)==21 and r.physical_integral>0"
                    ),
                ],
                cwd=flat,
                text=True,
                capture_output=True,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

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

    def test_ln_shape_function_validates_exponent_and_domain_contract(self):
        for p in (-1, 0, 2, 4, 6, 3.0, True, "3", None):
            with self.subTest(p=p):
                with self.assertRaisesRegex(ValueError, "integer 3 or 5"):
                    phys.ln_shape_function(1.0, p, phys.PlanckDomain())
        for domain in (None, object(), {}, "domain"):
            with self.subTest(domain=domain):
                with self.assertRaisesRegex(ValueError, "PlanckDomain"):
                    phys.ln_shape_function(1.0, 3, domain)
        with self.assertRaises(ValueError):
            phys.ln_shape_function(
                1.0,
                3,
                phys.PlanckDomain(x_low=5.0, x_high=4.0),
            )
        with self.assertRaisesRegex(ValueError, "PlanckDomain"):
            driver.run_planck2(5900.0, "frequency", 10, domain="invalid")

    def test_extreme_finite_values_raise_explanatory_value_errors(self):
        calls = (
            lambda: phys.prefactor("wavelength", 1.0e308),
            lambda: phys.prefactor("wavelength", 1.0e-200),
            lambda: phys.exact_physical_integral("frequency", 1.0e308),
            lambda: phys.exact_physical_integral("frequency", 1.0e-200),
            lambda: phys.coordinate_jacobian("wavelength", 1.0e308, 1.0e308),
            lambda: phys.x_to_wavelength(1.0e-308, 1.0e-308),
            lambda: phys.x_to_frequency(1.0e308, 1.0e308),
            lambda: driver.run_planck2(1.0e308, "frequency", 10),
            lambda: driver.run_planck2(1.0e-200, "frequency", 10),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, "representable|range"):
                    call()

    def test_driver_rejects_nonfinite_derived_result(self):
        with mock.patch.object(driver, "prefactor", return_value=math.inf):
            with self.assertRaisesRegex(ValueError, "representable"):
                driver.run_planck2(5900.0, "frequency", 10)


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

    def test_x_low_boundary_uses_exact_branch_only_at_and_above_threshold(self):
        p = 3
        boundary = self.domain.x_low
        below = math.nextafter(boundary, -math.inf)
        above = math.nextafter(boundary, math.inf)
        self.assertAlmostEqual(
            phys.ln_shape_function(below, p, self.domain),
            (p - 1) * math.log(below),
            places=14,
        )
        for x in (boundary, above):
            self.assertEqual(
                phys.ln_shape_function(x, p, self.domain),
                p * math.log(x) - math.log(math.expm1(x)),
            )

    def test_x_high_boundary_uses_wien_branch_only_above_threshold(self):
        p = 5
        boundary = self.domain.x_high
        below = math.nextafter(boundary, -math.inf)
        above = math.nextafter(boundary, math.inf)
        for x in (below, boundary):
            self.assertEqual(
                phys.ln_shape_function(x, p, self.domain),
                p * math.log(x) - math.log(math.expm1(x)),
            )
        self.assertEqual(
            phys.ln_shape_function(above, p, self.domain),
            p * math.log(above) - above,
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

    def test_result_lists_and_peak_fields_are_consistent(self):
        for result in self.results.values():
            with self.subTest(quantity=result.quantity):
                self.assertGreater(len(result.x_values), 0)
                self.assertEqual(len(result.x_values), len(result.coord_values))
                self.assertEqual(len(result.x_values), len(result.y_values))
                peak_index = result.x_values.index(result.x_peak)
                self.assertEqual(result.coord_values[peak_index], result.coord_peak)
                self.assertEqual(result.y_values[peak_index], result.y_peak)

    def test_coordinates_are_strictly_monotonic_in_sample_order(self):
        wavelength = self.results["wavelength"].coord_values
        self.assertTrue(all(a > b for a, b in zip(wavelength, wavelength[1:])))
        for quantity in ("frequency", "energy_density"):
            coords = self.results[quantity].coord_values
            with self.subTest(quantity=quantity):
                self.assertTrue(all(a < b for a, b in zip(coords, coords[1:])))

    def test_domain_wholly_below_peak_selects_right_endpoint(self):
        domain = phys.PlanckDomain(x_min=0.1, x_max=1.0, x_low=0.1, x_high=1.0)
        result = driver.run_planck2(5900.0, "frequency", 40, domain)
        self.assertEqual(result.x_peak, domain.x_max)

    def test_domain_wholly_above_peak_selects_left_endpoint(self):
        domain = phys.PlanckDomain(x_min=4.0, x_max=8.0, x_low=4.0, x_high=8.0)
        result = driver.run_planck2(5900.0, "frequency", 40, domain)
        self.assertEqual(result.x_peak, domain.x_min)

    def test_narrow_domain_brackets_analytic_peak(self):
        exact_peak = 2.821439372122079
        domain = phys.PlanckDomain(x_min=2.8, x_max=2.84, x_low=2.8, x_high=2.84)
        result = driver.run_planck2(5900.0, "frequency", 40, domain)
        self.assertLessEqual(abs(result.x_peak - exact_peak), 0.0005)

    def test_one_step_domain_uses_both_endpoints(self):
        domain = phys.PlanckDomain(x_min=0.1, x_max=1.0, x_low=0.1, x_high=1.0)
        result = driver.run_planck2(5900.0, "frequency", 1, domain)
        self.assertEqual(result.x_values, [domain.x_min, domain.x_max])
        self.assertEqual(result.x_peak, domain.x_max)

    def test_equal_sampled_values_keep_first_sample_as_peak(self):
        domain = phys.PlanckDomain(x_min=1.0, x_max=2.0, x_low=1.0, x_high=2.0)
        with mock.patch.object(driver, "_ln_shape_function_unchecked", return_value=0.0):
            result = driver.run_planck2(5900.0, "frequency", 1, domain)
        self.assertEqual(result.y_values[0], result.y_values[1])
        self.assertEqual(result.x_peak, domain.x_min)

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

    def test_peak_marker_title_and_axis_labels_for_every_mode(self):
        for quantity in phys.SHAPE_EXPONENT:
            result = driver.run_planck2(5900.0, quantity, 100)
            with self.subTest(quantity=quantity), mock.patch.object(plt, "show"):
                plotter.plot_planck2(result)
                axes = plt.gca()
                self.assertGreaterEqual(len(axes.lines), 2)
                marker_x = list(axes.lines[1].get_xdata())
                self.assertEqual(marker_x, [result.coord_peak, result.coord_peak])
                self.assertEqual(axes.get_xlabel(), result.x_label)
                self.assertEqual(axes.get_ylabel(), result.y_label)
                self.assertIn(quantity.replace("_", " ").title(), axes.get_title())
                self.assertIn("T = 5900 K", axes.get_title())
                plt.close("all")

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
        cls.parser = HelpStructureParser()
        cls.parser.feed(cls.text)

    def test_help_file_exists_and_is_html5_utf8(self):
        self.assertTrue(HELP_FILE.is_file())
        self.assertIn("<!DOCTYPE html>", self.text[:100])
        self.assertRegex(self.text[:500], r'<meta charset="utf-8"\s*/?>')

    def test_exactly_one_version_build_element(self):
        self.assertEqual(self.parser.ids.count("version_build"), 1)

    def test_parameter_names_and_defaults_are_paired_in_rows(self):
        rows = {
            row[0]: row[1]
            for section, row in self.parser.rows
            if section == "parameters" and len(row) >= 2 and row[0] != "Parameter"
        }
        expected = {
            "T": "5900.0",
            "quantity": '"wavelength"',
            "n_steps": "2000",
            "x_min": "0.01",
            "x_max": "100.0",
            "x_low": "0.05",
            "x_high": "20.0",
            "corner": '"upper right"',
            "y_frac_window": "0.003",
        }
        self.assertEqual({name: rows[name] for name in expected}, expected)

    def test_quantity_mode_fields_are_paired_in_rows(self):
        rows = {
            row[0]: row
            for section, row in self.parser.rows
            if section == "quantities" and len(row) == 4 and row[0] != "Mode string"
        }
        expected = {
            '"wavelength"': ("Spectral radiance", "x^5", "Wavelength (m)"),
            '"frequency"': ("Spectral radiance", "x^3", "Frequency (Hz)"),
            '"energy_density"': ("Spectral energy density", "x^3", "Frequency (Hz)"),
        }
        expected_units = {
            '"wavelength"': "W m⁻³ sr⁻¹",
            '"frequency"': "W m⁻² sr⁻¹ Hz⁻¹",
            '"energy_density"': "J m⁻³ Hz⁻¹",
        }
        self.assertEqual(set(rows), set(expected))
        for mode, (quantity_name, shape, coordinate) in expected.items():
            with self.subTest(mode=mode):
                row = rows[mode]
                self.assertIn(quantity_name, row[1])
                self.assertIn(expected_units[mode], row[1])
                self.assertIn(shape, row[2])
                self.assertEqual(row[3], coordinate)

    def test_help_distinguishes_dimensionless_and_physical_integrals(self):
        self.assertIn("distinct from", self.text)
        self.assertIn("coordinate Jacobian", self.text)
        self.assertIn(r"\frac{\sigma T^4}{\pi}", self.text)
        self.assertIn(r"\frac{4\sigma}{c}T^4", self.text)
        normalized = " ".join(self.text.split())
        self.assertIn(r"x=\frac{hc}{\lambda kT}=\frac{h\nu}{kT}", normalized)
        self.assertIn(r"\int_0^\infty\frac{x^3}{e^x-1}\,dx=\frac{\pi^4}{15}", normalized)
        self.assertIn(r"\int_0^\infty B_\lambda\,d\lambda", normalized)

    def test_help_documents_validation_contract(self):
        self.assertIn("1–1,000,000", self.text)
        self.assertIn("representable positive step", self.text)
        self.assertIn("finite range 0–1", self.text)
        self.assertIn("representable floating-point range", self.text)

    def test_exercise_identifiers_are_unique_and_sequential(self):
        numbers = re.findall(r'<div class="ec-num">EXP-(\d+)</div>', self.text)
        self.assertEqual(numbers, [str(number) for number in range(1, 9)])
        self.assertEqual(len(numbers), len(set(numbers)))
        headings = re.findall(
            r'<div class="ec-num">EXP-\d+</div><h4>(.*?)</h4>',
            self.text,
        )
        self.assertEqual(headings[0], "Solar Black-Body Approximation")
        self.assertEqual(headings[-1], "Rayleigh–Jeans and Wien Limits")

    def test_internal_navigation_targets_exist(self):
        targets = set(self.parser.ids)
        internal_links = [href for _, href in self.parser.hrefs if href.startswith("#")]
        self.assertGreater(len(internal_links), 0)
        for href in internal_links:
            with self.subTest(href=href):
                self.assertIn(href[1:], targets)

    def test_related_program_links_are_module_relative_html_links(self):
        related = [href for section, href in self.parser.hrefs if section == "related"]
        self.assertEqual(related, ["../08-Star/Star.html", "../08-Random2/Random2.html"])

    def test_student_content_contains_no_ai_or_review_history(self):
        student_text = self.text.split('<section id="license">', 1)[0]
        suspicious_terms = (
            "Claude",
            "Copilot",
            "Gemini",
            "ChatGPT",
            "Anthropic",
            "AI-generated",
            "audit round",
            "previous version",
            "porting fix",
            "legacy implementation",
            "reviewer",
        )
        for term in suspicious_terms:
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

    def test_default_program_runs_with_noninteractive_backend(self):
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
