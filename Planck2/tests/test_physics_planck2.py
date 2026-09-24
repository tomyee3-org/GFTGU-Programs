"""Regression and adversarial tests for Planck2.

The locator deliberately supports both repository layouts used for review:
this file may live in ``tests/`` or may be flattened beside the four program
modules during upload.
"""

import ast
import contextlib
from dataclasses import FrozenInstanceError, replace
import hashlib
import html as html_module
from html.parser import HTMLParser
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
from typing import NamedTuple
import unittest
import warnings
from types import MappingProxyType
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


HELP_FILENAMES = ("Planck2-claude.html", "Planck2.html")


def find_help_file(module_dir: Path) -> Path:
    """Find the Beats Help in a flattened upload or the GFTGU-Documentation tree.

    The live Beats Help is accepted under either of its names,
    ``Planck2-claude.html`` or ``Planck2.html``.  The Reference Guide version,
    ``Planck2-original.html``, is never returned, and no test requires it.
    """
    program_name = "Planck2"
    candidates = []
    for help_filename in HELP_FILENAMES:
        candidates.append(module_dir / help_filename)
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
        f"Could not find {' or '.join(HELP_FILENAMES)} beside the program or in "
        "GFTGU-Documentation/Planck2/."
    )


HELP_FILE = find_help_file(MODULE_DIR)


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

    def test_build_id_fails_fast_for_incomplete_tree(self):
        with tempfile.TemporaryDirectory() as temp_name:
            with self.assertRaisesRegex(RuntimeError, "Cannot compute BUILD_ID"):
                phys._compute_build_id(temp_name)

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
        self.assertLess(
            relative_error(
                phys.exact_physical_integral("wavelength", temperature),
                radiance,
            ),
            3e-15,
        )
        self.assertLess(
            relative_error(
                phys.exact_physical_integral("frequency", temperature),
                radiance,
            ),
            3e-15,
        )
        self.assertLess(
            relative_error(
                phys.exact_physical_integral("energy_density", temperature),
                energy_density,
            ),
            3e-15,
        )

    def test_avoidable_intermediate_overflow_counterexamples(self):
        exact = phys.exact_physical_integral("frequency", 1.0e78)
        self.assertTrue(math.isfinite(exact))
        self.assertLess(relative_error(exact, 1.8049362359900739e304), 2e-14)

        wavelength_prefactor = phys.prefactor("wavelength", 1.0e62)
        self.assertTrue(math.isfinite(wavelength_prefactor))
        self.assertLess(
            relative_error(wavelength_prefactor, 1.931791196317044e303),
            2e-14,
        )

        frequency = phys.x_to_frequency(1.0e300, 1.0e-300)
        self.assertTrue(math.isfinite(frequency))
        self.assertLess(relative_error(frequency, phys.FREQUENCY_SCALE), 2e-14)

    def test_scaled_product_supports_mixed_integer_powers(self):
        result = phys._scaled_product(
            ((8.0, 2), (4.0, -1), (2.0, 3)),
            "mixed-power result",
        )
        self.assertEqual(result, 128.0)
        self.assertAlmostEqual(
            phys._scaled_product(
                ((1.0e300, 1), (1.0e-300, 1)),
                "cancelled result",
            ),
            1.0,
            places=14,
        )

    def test_scaled_product_validates_its_private_contract(self):
        invalid_factors = (
            None,
            (),
            ((1.0,),),
            ((1.0, 1, 2),),
            ((0.0, 1),),
            ((-1.0, 1),),
            ((math.nan, 1),),
            ((math.inf, 1),),
            ((True, 1),),
            (("2", 1),),
            ((2.0, 1.5),),
            ((2.0, True),),
            ((2.0, "1"),),
        )
        for factors in invalid_factors:
            with self.subTest(factors=factors):
                with self.assertRaises(ValueError):
                    phys._scaled_product(factors, "invalid result")
        for label in ("", "   ", None, 3):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "label"):
                    phys._scaled_product(((2.0, 1),), label)

    def test_scaled_product_nextafter_range_boundaries(self):
        # 2**1023 times the float immediately below 2 is exactly the
        # largest finite binary64 value; the adjacent input 2 overflows.
        upper_inside = math.nextafter(2.0, 0.0)
        upper_factors = ((2.0, 1),) * 1023
        upper_result = phys._scaled_product(
            upper_factors + ((upper_inside, 1),),
            "upper-boundary result",
        )
        self.assertTrue(math.isfinite(upper_result))
        self.assertLess(relative_error(upper_result, sys.float_info.max), 3e-15)
        self.assertEqual(math.nextafter(upper_inside, math.inf), 2.0)
        with self.assertRaisesRegex(ValueError, "representable"):
            phys._scaled_product(
                upper_factors + ((2.0, 1),),
                "upper-boundary result",
            )

        # Half the smallest subnormal rounds to zero.  Moving the final
        # factor one float above 0.5 makes the result round up instead.
        lower_factors = ((0.5, 1),) * 1074
        lower_inside = math.nextafter(0.5, math.inf)
        lower_result = phys._scaled_product(
            lower_factors + ((lower_inside, 1),),
            "lower-boundary result",
        )
        self.assertTrue(math.isfinite(lower_result))
        self.assertGreater(lower_result, 0.0)
        self.assertEqual(math.nextafter(lower_inside, 0.0), 0.5)
        with self.assertRaisesRegex(ValueError, "representable"):
            phys._scaled_product(
                lower_factors + ((0.5, 1),),
                "lower-boundary result",
            )

    def test_scaled_prefactor_true_range_boundaries(self):
        scale = phys.WAVELENGTH_PREFACTOR_SCALE
        maximum_T = math.exp(
            (math.log(sys.float_info.max) - math.log(scale)) / 5.0
        )
        self.assertTrue(math.isfinite(phys.prefactor("wavelength", 0.99 * maximum_T)))
        with self.assertRaisesRegex(ValueError, "representable"):
            phys.prefactor("wavelength", 1.01 * maximum_T)

        minimum_positive = float.fromhex("0x0.0000000000001p-1022")
        minimum_T = math.exp(
            (math.log(minimum_positive) - math.log(scale)) / 5.0
        )
        self.assertGreater(phys.prefactor("wavelength", 1.1 * minimum_T), 0.0)
        with self.assertRaisesRegex(ValueError, "representable"):
            phys.prefactor("wavelength", 0.8 * minimum_T)

    def test_scaled_integral_and_frequency_true_range_boundaries(self):
        scale = phys.SIGMA_SB / math.pi
        maximum_T = math.exp(
            (math.log(sys.float_info.max) - math.log(scale)) / 4.0
        )
        self.assertTrue(
            math.isfinite(
                phys.exact_physical_integral("frequency", 0.99 * maximum_T)
            )
        )
        with self.assertRaisesRegex(ValueError, "representable"):
            phys.exact_physical_integral("frequency", 1.01 * maximum_T)

        minimum_positive = float.fromhex("0x0.0000000000001p-1022")
        minimum_T = math.exp(
            (math.log(minimum_positive) - math.log(scale)) / 4.0
        )
        self.assertGreater(
            phys.exact_physical_integral("frequency", 1.1 * minimum_T),
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "representable"):
            phys.exact_physical_integral("frequency", 0.8 * minimum_T)

        maximum_x = sys.float_info.max / phys.FREQUENCY_SCALE
        self.assertTrue(math.isfinite(phys.x_to_frequency(0.99 * maximum_x, 1.0)))
        with self.assertRaisesRegex(ValueError, "representable"):
            phys.x_to_frequency(1.01 * maximum_x, 1.0)

        self.assertGreater(phys.x_to_frequency(minimum_positive, 1.0e-10), 0.0)
        with self.assertRaisesRegex(ValueError, "representable"):
            phys.x_to_frequency(minimum_positive, 1.0e-20)

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

    def test_driver_rejects_nonfinite_first_coordinate_and_jacobian(self):
        with mock.patch.object(driver, "x_to_frequency", return_value=math.inf):
            with self.assertRaisesRegex(ValueError, "first sampled point"):
                driver.run_planck2(5900.0, "frequency", 10)
        with mock.patch.object(driver, "coordinate_jacobian", return_value=math.nan):
            with self.assertRaisesRegex(ValueError, "first sampled point"):
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
        self.assertEqual(result.x_values, (domain.x_min, domain.x_max))
        self.assertEqual(result.x_peak, domain.x_max)

    def test_equal_sampled_values_keep_first_sample_as_peak(self):
        domain = phys.PlanckDomain(x_min=1.0, x_max=2.0, x_low=1.0, x_high=2.0)
        with mock.patch.object(driver, "_ln_shape_function_unchecked", return_value=0.0):
            result = driver.run_planck2(5900.0, "frequency", 1, domain)
        self.assertEqual(result.y_values[0], result.y_values[1])
        self.assertEqual(result.x_peak, domain.x_min)

    def test_driver_does_not_rescan_completed_sample_arrays(self):
        """Behavioural check: validation work is bounded per sample and grows linearly.

        The count of ``math.isfinite`` calls is used as a proxy for validation
        work, so the test does not depend on which private helpers the driver
        happens to call.  A full re-validation of the finished arrays would add
        several checks per sample, and a quadratic rescan would break linearity.
        """
        def finite_checks(n_steps):
            with mock.patch.object(math, "isfinite", wraps=math.isfinite) as spy:
                driver.run_planck2(5900.0, "frequency", n_steps)
            return spy.call_count

        small, large = finite_checks(100), finite_checks(400)
        self.assertLessEqual(small, 18 * 101 + 50)
        self.assertLessEqual(large, 18 * 401 + 50)
        self.assertLess(large / small, 5.0)

    def test_result_is_fully_immutable(self):
        result = self.results["frequency"]
        self.assertIsInstance(result.x_values, tuple)
        self.assertIsInstance(result.coord_values, tuple)
        self.assertIsInstance(result.y_values, tuple)
        with self.assertRaises(FrozenInstanceError):
            result.x_peak = 1.0
        with self.assertRaises(TypeError):
            result.x_values[0] = 1.0

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
                    1e-6,
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

    def test_malformed_result_arrays_are_rejected(self):
        result = driver.run_planck2(5900.0, "frequency", 20)
        malformed = (
            replace(result, coord_values=result.coord_values[:-1]),
            replace(result, x_values=(), coord_values=(), y_values=()),
            replace(
                result,
                coord_values=(math.inf,) + result.coord_values[1:],
            ),
            replace(result, y_values=(math.nan,) + result.y_values[1:]),
            replace(result, y_peak=math.inf),
            replace(result, coord_peak=math.nan),
        )
        for bad_result in malformed:
            with self.subTest(bad_result=bad_result):
                with self.assertRaises(ValueError):
                    plotter.plot_planck2(bad_result)

    def test_peak_fields_must_identify_the_same_sample(self):
        result = driver.run_planck2(5900.0, "frequency", 20)
        peak_index = result.y_values.index(result.y_peak)
        other_index = 0 if peak_index != 0 else 1
        malformed = (
            replace(result, x_peak=result.x_values[other_index]),
            replace(result, coord_peak=result.coord_values[other_index]),
            replace(
                result,
                x_peak=result.x_values[other_index],
                coord_peak=result.coord_values[other_index],
            ),
            replace(result, x_peak=math.nan),
            replace(result, x_peak=math.inf),
        )
        for bad_result in malformed:
            with self.subTest(bad_result=bad_result):
                with self.assertRaises(ValueError):
                    plotter.plot_planck2(bad_result)

    def test_equal_maxima_require_first_sample_as_peak(self):
        domain = phys.PlanckDomain(x_min=1.0, x_max=2.0, x_low=1.0, x_high=2.0)
        with mock.patch.object(driver, "_ln_shape_function_unchecked", return_value=0.0):
            result = driver.run_planck2(5900.0, "frequency", 1, domain)
        second_peak = replace(
            result,
            x_peak=result.x_values[1],
            coord_peak=result.coord_values[1],
        )
        with self.assertRaisesRegex(ValueError, "first sampled maximum"):
            plotter.plot_planck2(second_peak)

    def test_displayed_scalars_and_metadata_are_validated(self):
        result = driver.run_planck2(5900.0, "frequency", 20)
        malformed = (
            replace(result, quantity="invalid"),
            replace(result, T=0.0),
            replace(result, T=math.nan),
            replace(result, T=math.inf),
            replace(result, dimensionless_area=0.0),
            replace(result, dimensionless_area=math.nan),
            replace(result, physical_integral=0.0),
            replace(result, physical_integral=math.inf),
            replace(result, exact_physical_integral=-1.0),
            replace(result, exact_physical_integral=math.nan),
            replace(result, x_label=""),
            replace(result, y_label="incorrect"),
            replace(result, physical_integral_units=""),
        )
        for bad_result in malformed:
            with self.subTest(bad_result=bad_result):
                with self.assertRaises(ValueError):
                    plotter.plot_planck2(bad_result)

    def test_annotation_distinguishes_exact_infinite_domain_value(self):
        result = driver.run_planck2(5900.0, "frequency", 20)
        with mock.patch.object(plt, "show"):
            plotter.plot_planck2(result)
        annotation_text = "\n".join(text.get_text() for text in plt.gca().texts)
        self.assertIn("Exact 0..∞ physical", annotation_text)

    def test_non_result_object_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Planck2Result"):
            plotter.plot_planck2(object())

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


HELP_HTML = HELP_FILE.read_text(encoding="utf-8")
BEAT_NUMBERS = tuple(range(0, 9))
VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
# Commands that the Help documents as being rejected by the program.
REJECTED_COMMANDS = {"python main.py --x_max 10"}


def html_text(fragment: str) -> str:
    """Visible text of an HTML fragment, with entities decoded and spaces collapsed."""
    # Only real tags are removed: a "<" followed by a digit or space is text.
    return " ".join(
        html_module.unescape(re.sub(r"</?[A-Za-z!][^>]*>", "", fragment)).split()
    )


def section_html(html: str, section_id: str) -> str:
    match = re.search(
        rf'<section id="{re.escape(section_id)}">(.*?)</section>', html, re.DOTALL
    )
    if match is None:
        raise AssertionError(f"section {section_id!r} not found in the Help file")
    return match.group(1)


def documented_commands(fragment: str) -> list:
    """Every ``python main.py ...`` command shown in a <pre> block."""
    commands = []
    for block in re.findall(r"<pre[^>]*>(.*?)</pre>", fragment, re.DOTALL):
        text = html_module.unescape(re.sub(r"<[^>]+>", "", block)).replace("\\\n", " ")
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
    annotation: str


_CLI_CACHE = {}


def run_cli(arguments=()) -> CliRun:
    """Run ``main.main()`` in-process, plotting on the Agg backend, caching each command."""
    key = tuple(arguments)
    if key in _CLI_CACHE:
        return _CLI_CACHE[key]
    captured = {}
    real_run = planck2_main.run_planck2

    def capturing_run(*args, **kwargs):
        captured["result"] = real_run(*args, **kwargs)
        return captured["result"]

    out, err = io.StringIO(), io.StringIO()
    exit_code = None
    with (
        mock.patch.object(planck2_main, "run_planck2", new=capturing_run),
        mock.patch.object(plt, "show"),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        try:
            planck2_main.main(list(key))
        except SystemExit as exc:
            exit_code = exc.code
    annotation = ""
    if plt.get_fignums():
        annotation = "\n".join(text.get_text() for text in plt.gca().texts)
    plt.close("all")
    run = CliRun(out.getvalue(), err.getvalue(), exit_code, captured.get("result"), annotation)
    _CLI_CACHE[key] = run
    return run


def printed_by(arguments) -> str:
    """Everything a command shows a student: the console summary and the plot annotation."""
    run = run_cli(arguments)
    return run.stdout + "\n" + run.annotation


def trapezoid_simpson(function, lower, upper, intervals=20000):
    """Composite Simpson rule, used only to check the Help's hand-derived numbers."""
    if intervals % 2:
        intervals += 1
    step = (upper - lower) / intervals
    total = function(lower) + function(upper)
    for index in range(1, intervals):
        total += (4 if index % 2 else 2) * function(lower + index * step)
    return total * step / 3.0


def shape(x, p):
    return x**p / math.expm1(x)


def peak_root(p):
    low, high = 1.0, 10.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if mid - p * (1.0 - math.exp(-mid)) < 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


class HelpStructure(HTMLParser):
    """Ids, links and tag balance of a Help page."""

    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.hrefs = []
        self.sidebar_hrefs = []
        self.section_ids = []
        self.scripts = []
        self.errors = []
        self._stack = []
        self._in_nav = False
        self.feed(html)
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
        if tag not in VOID_TAGS:
            self._stack.append(tag)

    def handle_endtag(self, tag):
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


STRUCTURE = HelpStructure(HELP_HTML)


class HelpStructureTests(unittest.TestCase):
    def test_help_file_is_html5_utf8_with_one_version_build(self):
        self.assertIn("<!DOCTYPE html>", HELP_HTML[:100])
        self.assertRegex(HELP_HTML[:500], r'<meta charset="utf-8"\s*/?>')
        self.assertEqual(STRUCTURE.ids.count("version_build"), 1)

    def test_help_file_is_the_beats_version_and_not_the_original(self):
        self.assertIn(HELP_FILE.name, HELP_FILENAMES)
        self.assertIn("Beat 0", HELP_HTML)
        self.assertNotIn("Reference Guide version", HELP_HTML)

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
               "quantities", "summary", "experiments", "related", "license"]
        )
        self.assertEqual(STRUCTURE.section_ids, expected)
        self.assertEqual(STRUCTURE.sidebar_hrefs, expected)

    def test_mathjax_is_the_only_external_script_and_offline_note_is_static(self):
        self.assertEqual(len(STRUCTURE.scripts), 1)
        self.assertRegex(STRUCTURE.scripts[0], r"^https://cdn\.jsdelivr\.net/npm/mathjax@3/")
        self.assertIn("needs no internet access to run", html_text(section_html(HELP_HTML, "overview")))

    def test_student_content_contains_no_ai_or_review_history(self):
        student_text = HELP_HTML.split('<section id="license">', 1)[0]
        for term in ("Claude", "Copilot", "Gemini", "ChatGPT", "Anthropic",
                     "AI-generated", "audit round", "previous version",
                     "porting fix", "legacy implementation", "reviewer",
                     "Kickoff", "Grok", "Codex"):
            with self.subTest(term=term):
                self.assertNotIn(term, student_text)

    def test_java_provenance_is_confined_to_license(self):
        before_license, license_and_after = HELP_HTML.split('<section id="license">', 1)
        self.assertNotIn("Java/Triana", before_license)
        self.assertIn("Java/Triana", license_and_after)

    def test_first_edition_investigations_are_retained(self):
        self.assertIn("Investigations 10.1, 10.2, and 10.3", HELP_HTML)

    def test_related_program_links_are_module_relative_html_links(self):
        related = re.findall(r'<a href="([^"#]+)"', section_html(HELP_HTML, "related"))
        self.assertEqual(related, ["../Star/Star.html", "../Random2/Random2.html"])

    def test_relative_links_resolve_when_the_documentation_tree_is_present(self):
        docs_root = HELP_FILE.parent.parent
        if docs_root.name != "GFTGU-Documentation" or not (docs_root / "Star").is_dir():
            self.skipTest("the sibling documentation folders are not present in this layout")
        for href in STRUCTURE.hrefs:
            if href.startswith(("#", "http://", "https://", "mailto:")):
                continue
            with self.subTest(href=href):
                self.assertTrue((HELP_FILE.parent / href).is_file(), href)


class HelpBeatTests(unittest.TestCase):
    def test_every_beat_follows_the_same_pattern(self):
        for number in BEAT_NUMBERS:
            body = section_html(HELP_HTML, f"beat{number}")
            with self.subTest(beat=number):
                self.assertIn(f"<h2>Beat {number} · ", body)
                self.assertGreaterEqual(body.count("<pre>"), 1)
                self.assertRegex(body, r"<p>Look[ ,]")
                self.assertIn("Three tasks, in order.", body)
                self.assertIn("<p><em>Then</em>", body)
                self.assertIn("<p><em>Experiments that go with this beat:</em>", body)
                self.assertLess(body.index("<pre>"), body.index("Three tasks, in order."))
                self.assertLess(body.index("Three tasks, in order."), body.index("<em>Then</em>"))
                self.assertLess(body.index("<em>Then</em>"), body.index("Experiments that go with"))
                self.assertGreaterEqual(body.count('class="eq"'), 1)

    def test_sidebar_titles_match_the_beat_headings(self):
        for number in BEAT_NUMBERS:
            heading = re.search(
                rf'<h2>Beat {number} · (.*?)</h2>', section_html(HELP_HTML, f"beat{number}")
            ).group(1)
            link = re.search(rf'<a href="#beat{number}">(.*?)</a>', HELP_HTML).group(1)
            with self.subTest(beat=number):
                self.assertEqual(html_text(link), f"{number} · {html_text(heading)}")

    def equations(self):
        found = {}
        for number in BEAT_NUMBERS:
            body = section_html(HELP_HTML, f"beat{number}")
            for match in re.finditer(
                r'<span class="eq-label">\((\d+)\)</span>\s*<div class="eq-kind">.*?'
                r'<span class="kind kind-(\w+)">(\w+)</span>',
                body, re.DOTALL,
            ):
                found[int(match.group(1))] = (match.group(3), number)
        return found

    def test_equations_are_numbered_in_order_and_tagged(self):
        found = self.equations()
        self.assertEqual(sorted(found), list(range(1, 22)))
        beats = [found[n][1] for n in sorted(found)]
        self.assertEqual(beats, sorted(beats))
        self.assertIn("Twenty-one equations are numbered", html_text(section_html(HELP_HTML, "beats")))

    def test_only_planck_s_law_is_tagged_law_and_algorithms_are_the_numerical_rules(self):
        found = self.equations()
        self.assertEqual([n for n, (kind, _) in found.items() if kind == "LAW"], [1])
        self.assertEqual(
            sorted(n for n, (kind, _) in found.items() if kind == "ALGORITHM"),
            [15, 17, 18, 19, 20],
        )
        self.assertIn("Eqs. (15), (17) and (18) to (20)", html_text(section_html(HELP_HTML, "beats")))
        self.assertFalse([n for n, (kind, _) in found.items() if kind == "ODE"])

    def test_equation_citations_refer_to_numbered_equations(self):
        for section in ("overview", "beats", "equations", "algorithm", "summary", "experiments",
                        *[f"beat{n}" for n in BEAT_NUMBERS]):
            text = html_text(section_html(HELP_HTML, section))
            for group in re.findall(r"Eqs?\.\s*\(([\d\s,and()to]+?)\)(?=[\s.,;:]|$)", text):
                for number in re.findall(r"\d+", group):
                    with self.subTest(section=section, equation=number):
                        self.assertIn(int(number), range(1, 22))

    def test_equation_index_lists_every_numbered_equation_once_with_its_kind_and_beat(self):
        found = self.equations()
        rows = re.findall(
            r"<tr><td>\((\d+)\)</td><td><span class=\"kind kind-\w+\">(\w+)</span></td>"
            r"<td>.*?</td><td>(\d+)</td>",
            section_html(HELP_HTML, "equations"),
        )
        self.assertEqual([int(r[0]) for r in rows], list(range(1, 22)))
        for number, kind, beat in rows:
            with self.subTest(equation=number):
                self.assertEqual((kind, int(beat)), found[int(number)])

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
        rank = {"Introductory": 0, "Intermediate": 1, "Intermediate to Advanced": 2,
                "Advanced": 3, "Advanced Programming": 3}
        levels = [
            rank[html_text(title).rsplit("— ", 1)[1]]
            for title in re.findall(r'<h3 id="exp\d+">(.*?)</h3>', section_html(HELP_HTML, "experiments"))
        ]
        self.assertEqual(levels, sorted(levels))
        self.assertEqual(levels[0], 0)
        self.assertEqual(levels[-1], 3)

    def test_python_snippets_in_the_experiments_run_and_agree_with_the_beats(self):
        blocks = [
            html_module.unescape(re.sub(r"<[^>]+>", "", b))
            for b in re.findall(r"<pre>(.*?)</pre>", section_html(HELP_HTML, "experiments"), re.DOTALL)
            if "from planck2_driver import" in b
        ]
        self.assertEqual(len(blocks), 2)
        outputs = []
        for block in blocks:
            completed = subprocess.run(
                [sys.executable, "-c", block], cwd=MODULE_DIR,
                text=True, capture_output=True, timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            outputs.append(completed.stdout.split("\n"))
        wavelength, frequency = outputs[0][0].split(), outputs[0][1].split()
        self.assertAlmostEqual(float(wavelength[1]), 4.965114231744, places=9)
        self.assertAlmostEqual(float(frequency[1]), 2.821439372122, places=9)
        self.assertLess(abs(float(wavelength[2]) - float(wavelength[1])), 0.0005)
        self.assertLess(abs(float(frequency[2]) - float(frequency[1])), 0.0005)
        self.assertAlmostEqual(float(outputs[1][0]), 2.9010e-3, delta=1e-7)
        self.assertAlmostEqual(float(outputs[1][1]), 1.0, delta=1e-6)


class HelpCommandTests(unittest.TestCase):
    def test_the_help_documents_a_meaningful_number_of_commands(self):
        commands = documented_commands(HELP_HTML)
        self.assertGreaterEqual(len(dict.fromkeys(commands)), 30)

    def test_every_documented_command_parses_and_runs_to_a_summary(self):
        for command in dict.fromkeys(documented_commands(HELP_HTML)):
            arguments = command_arguments(command)
            if "--help" in arguments:
                continue
            with self.subTest(command=command):
                run = run_cli(arguments)
                if command in REJECTED_COMMANDS:
                    self.assertEqual(run.exit_code, 2)
                    self.assertIn("Require x_min <= x_low < x_high <= x_max", run.stderr)
                    self.assertIn("x_max=10", run.stderr)
                    self.assertEqual(run.stdout, "")
                else:
                    self.assertIn(run.exit_code, (None, 0), run.stderr)
                    lines = run.stdout.splitlines()
                    self.assertEqual(len(lines), 5)
                    self.assertTrue(lines[0].startswith(f"Planck2 {phys.MODEL_VERSION} "))
                    self.assertRegex(lines[1], r"^Peak (wavelength|frequency) = \d\.\d{6}e[+-]\d\d (m|Hz)$")
                    self.assertIn("Exact 0..infinity bolometric value", lines[4])
                    self.assertIn("Peak", run.annotation)

    def test_rejected_commands_are_documented_as_rejected(self):
        body = section_html(HELP_HTML, "beat8")
        for command in REJECTED_COMMANDS:
            self.assertIn(command, html_text(body))
        self.assertIn("The first command below is rejected", html_text(body))

    def test_documented_options_are_real_options(self):
        parser = planck2_main.build_parser()
        real = {s for action in parser._actions for s in action.option_strings}
        used = set(re.findall(r"(?<![\w-])(--[A-Za-z_]+)", "\n".join(documented_commands(HELP_HTML))))
        self.assertTrue(used)
        self.assertLessEqual(used, real)


class HelpReferenceTests(unittest.TestCase):
    SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

    def rows(self, section):
        parser = HelpStructureParser()
        parser.feed(HELP_HTML)
        return [row for section_id, row in parser.rows if section_id == section]

    def test_parameter_table_matches_the_parser_options_and_defaults(self):
        rows = {row[0]: row for row in self.rows("parameters") if row[0] != "Parameter"}
        parser = planck2_main.build_parser()
        options = {
            action.option_strings[0]: action
            for action in parser._actions
            if action.option_strings and action.option_strings[0] not in ("-h", "--version")
        }
        self.assertEqual(set(rows), set(options))
        for name, action in options.items():
            default = action.default
            expected = f'"{default}"' if isinstance(default, str) else str(default)
            with self.subTest(option=name):
                self.assertEqual(rows[name][1], expected)

    def test_quantity_table_matches_the_shared_quantity_descriptor(self):
        rows = {row[0]: row for row in self.rows("quantities") if row[0] != "Mode string"}
        self.assertEqual(set(rows), {f'"{name}"' for name in phys.QUANTITY_SPECS})
        for name, spec in phys.QUANTITY_SPECS.items():
            row = rows[f'"{name}"']
            units = re.search(r"\((.*)\)$", spec.y_label).group(1)
            units = re.sub(
                r"\$\^\{-(\d)\}\$",
                lambda m: "⁻" + m.group(1).translate(self.SUPERSCRIPT),
                units,
            )
            with self.subTest(quantity=name):
                self.assertIn(units, row[1])
                self.assertIn(f"x^{spec.shape_exponent}", row[2])
                self.assertEqual(row[3], spec.x_label)

    def test_constants_and_defaults_table_matches_the_program(self):
        rows = {row[0]: row[1] for row in self.rows("algorithm") if len(row) == 3 and row[0] != "Name"}
        for name, value in (
            ("H_PLANCK", f"{phys.H_PLANCK:.8e}"),
            ("C_LIGHT", f"{phys.C_LIGHT:.8e}"),
            ("K_BOLTZMANN", f"{phys.K_BOLTZMANN:.6e}"),
            ("SIGMA_SB", f"{phys.SIGMA_SB:.6e}"),
        ):
            mantissa, exponent = value.split("e")
            sign = "\u2212" if exponent.startswith("-") else ""
            rendered = f"{mantissa}\u00d710{sign}{int(exponent.lstrip('+-'))}"
            with self.subTest(constant=name):
                self.assertTrue(rows[name].startswith(rendered), (rows[name], rendered))
        self.assertEqual(rows["MAX_STEPS"], f"{driver.MAX_STEPS:,}")
        domain = phys.PlanckDomain()
        for field, value in (("x_min", domain.x_min), ("x_max", domain.x_max),
                             ("x_low", domain.x_low), ("x_high", domain.x_high)):
            with self.subTest(field=field):
                self.assertEqual(rows[f"PlanckDomain.{field}"], str(value))

    def test_printed_summary_block_is_exactly_what_the_default_run_prints(self):
        block = re.search(r"<pre>(Planck2 .*?)</pre>", section_html(HELP_HTML, "summary"), re.DOTALL)
        self.assertIsNotNone(block)
        self.assertEqual(
            html_module.unescape(block.group(1)).strip(),
            run_cli(()).stdout.strip(),
        )

    def test_every_summary_label_printed_by_main_is_described_in_the_help(self):
        bullets = re.findall(r"<li>(.*?)</li>", section_html(HELP_HTML, "summary"), re.DOTALL)
        described = set()
        for bullet in bullets:
            head = bullet.split(":", 1)[0]
            described.update(re.findall(r"<code>([^<]+)</code>", head))
        self.assertEqual(
            described,
            {"peak at x", "Peak wavelength", "Peak frequency", "Dimensionless area",
             "Physical integral", "Exact 0..infinity bolometric value"},
        )
        for arguments, labels in (
            ((), ("peak at x", "Peak wavelength", "Dimensionless area",
                  "Physical integral", "Exact 0..infinity bolometric value")),
            (("--quantity", "frequency"), ("Peak frequency",)),
        ):
            output = run_cli(arguments).stdout
            for label in labels:
                self.assertIn(label, output)

    def test_every_annotation_line_is_described_in_the_help(self):
        annotation = run_cli(()).annotation
        text = html_text(section_html(HELP_HTML, "summary"))
        for label in ("Peak at x", "Peak λ", "Peak value", "∫ f(x) dx",
                      "Physical integral", "Exact 0..∞ physical"):
            with self.subTest(label=label):
                self.assertIn(label, annotation)
                self.assertIn(label, text)

    def test_code_identifiers_named_in_the_help_exist_in_the_program(self):
        modules = (phys, driver, plotter, planck2_main)
        names = set()
        for body in re.findall(r"<code>([^<]+)</code>", HELP_HTML):
            body = html_module.unescape(body)
            match = re.fullmatch(r"(?:PlanckDomain\.)?([A-Za-z_][A-Za-z_0-9]*)\(\)", body)
            if match:
                names.add(match.group(1))
            elif re.fullmatch(r"[A-Z][A-Z_0-9]{3,}", body):
                names.add(body)
        self.assertGreater(len(names), 20)
        allowed_elsewhere = {"expm1"}  # math.expm1
        for name in sorted(names - allowed_elsewhere):
            with self.subTest(name=name):
                self.assertTrue(
                    any(hasattr(module, name) for module in modules)
                    or (name == "validate" and hasattr(phys.PlanckDomain, name)),
                    name,
                )
        self.assertTrue(hasattr(phys.PlanckDomain, "validate"))


class HelpQuotedNumberTests(unittest.TestCase):
    """Every number the Help quotes from the console or the plot must be the number printed."""

    PRINTED = re.compile(r"(?<![\w.])(?:\d\.\d{4,6}e[+-]\d\d|\d+\.\d{6})(?![\w%])")
    # Numbers that a student computes by hand or reads from Eq. (5) and Eq. (6); each is
    # checked independently in HelpQuantitativeClaimTests.
    DERIVED_OK = {
        "4.965114", "2.821439",     # roots of Eq. (5)
        "2.897772",                 # Wien's constant, Eq. (6)
        "5.670374",                 # Stefan-Boltzmann constant
        "0.049995",                 # grid step of the 2000-step default (Beat 6)
    }

    def outputs_for(self, fragment):
        chunks = [printed_by(())]
        for command in documented_commands(fragment):
            arguments = command_arguments(command)
            if command not in REJECTED_COMMANDS and "--help" not in arguments:
                chunks.append(printed_by(arguments))
        return "\n".join(chunks)

    def quoted_tokens(self, fragment):
        return set(self.PRINTED.findall(html_text(fragment))) - self.DERIVED_OK

    def test_numbers_quoted_in_each_beat_are_printed_by_that_beat_s_commands(self):
        for number in BEAT_NUMBERS:
            body = section_html(HELP_HTML, f"beat{number}")
            printed = self.outputs_for(body)
            tokens = self.quoted_tokens(body)
            with self.subTest(beat=number):
                self.assertTrue(tokens, "a beat should quote printed numbers")
                self.assertEqual(sorted(t for t in tokens if t not in printed), [])

    def test_numbers_quoted_in_the_reference_sections_are_printed_by_some_command(self):
        every_output = "\n".join(
            printed_by(command_arguments(c))
            for c in dict.fromkeys(documented_commands(HELP_HTML))
            if c not in REJECTED_COMMANDS and "--help" not in c
        )
        for section in ("overview", "beats", "equations", "algorithm", "modules",
                        "quickstart", "parameters", "quantities", "summary", "experiments"):
            with self.subTest(section=section):
                tokens = self.quoted_tokens(section_html(HELP_HTML, section))
                self.assertEqual(sorted(t for t in tokens if t not in every_output), [])


class HelpQuantitativeClaimTests(unittest.TestCase):
    """Independent checks of the statements in the beats that go beyond the printed digits."""

    @staticmethod
    def result(arguments):
        return run_cli(arguments).result

    def test_eq5_roots_and_eq6_wien_constant(self):
        self.assertAlmostEqual(peak_root(5), 4.965114, places=6)
        self.assertAlmostEqual(peak_root(3), 2.821439, places=6)
        wien = phys.H_PLANCK * phys.C_LIGHT / (phys.K_BOLTZMANN * peak_root(5))
        self.assertAlmostEqual(wien, 2.897772e-3, delta=1e-9)
        self.assertAlmostEqual(phys.K_BOLTZMANN * peak_root(3) / phys.H_PLANCK, 5.8789e10, delta=1e6)

    def test_beat1_wavelength_peak_scales_as_inverse_t_and_height_as_t5(self):
        cool = self.result(("--T", "3000"))
        mid = self.result(())
        hot = self.result(("--T", "10000"))
        products = [r.coord_peak * r.T for r in (cool, mid, hot)]
        for product in products:
            self.assertAlmostEqual(product, 2.9010e-3, delta=1e-7)
            self.assertLess(relative_error(product, products[0]), 1e-12)
        self.assertLess(abs(products[0] / 2.897772e-3 - 1.0 - 0.0011), 1e-4)
        self.assertAlmostEqual(mid.y_peak / cool.y_peak, 29.42, places=2)
        self.assertLess(relative_error(mid.y_peak / cool.y_peak, (5900 / 3000) ** 5), 1e-3)
        self.assertAlmostEqual(hot.y_peak / mid.y_peak, (10000 / 5900) ** 5, delta=0.05)
        self.assertEqual({round(r.x_peak, 6) for r in (cool, mid, hot)}, {4.959505})

    def test_beat2_frequency_peak_wavelength_conversion_and_product(self):
        frequency = self.result(("--T", "5900", "--quantity", "frequency"))
        wavelength_of_frequency_peak = phys.C_LIGHT / frequency.coord_peak
        self.assertAlmostEqual(wavelength_of_frequency_peak * 1e9, 867.9, places=1)
        self.assertAlmostEqual(self.result(()).coord_peak * 1e9, 491.70, places=2)
        exact_frequency = phys.K_BOLTZMANN * peak_root(3) * 5900 / phys.H_PLANCK
        self.assertAlmostEqual(exact_frequency, 3.4686e14, delta=1e10)
        self.assertAlmostEqual(phys.K_BOLTZMANN * peak_root(3) * 2.725 / phys.H_PLANCK / 1e9, 160.20, places=2)
        self.assertAlmostEqual(peak_root(3) / peak_root(5), 0.568, places=3)

    def test_beat3_energy_density_is_four_pi_over_c_times_frequency_radiance(self):
        radiance = self.result(("--T", "5900", "--quantity", "frequency"))
        energy = self.result(("--T", "5900", "--quantity", "energy_density"))
        factor = 4.0 * math.pi / phys.C_LIGHT
        self.assertAlmostEqual(factor, 4.1917e-8, delta=5e-13)
        self.assertLess(relative_error(energy.y_peak / radiance.y_peak, factor), 1e-12)
        self.assertLess(relative_error(energy.physical_integral / radiance.physical_integral, factor), 1e-12)
        self.assertEqual(energy.x_peak, radiance.x_peak)
        self.assertEqual(energy.dimensionless_area, radiance.dimensionless_area)
        self.assertEqual(energy.physical_integral_units, "J m^-3")

    def test_beat4_both_radiances_have_the_same_physical_integral_and_error(self):
        wavelength = self.result(())
        frequency = self.result(("--quantity", "frequency"))
        self.assertLess(relative_error(wavelength.physical_integral, frequency.physical_integral), 1e-12)
        for result in (wavelength, frequency):
            error = result.physical_integral / result.exact_physical_integral - 1.0
            self.assertAlmostEqual(error, -6.9e-7, delta=1e-8)
        self.assertAlmostEqual(
            frequency.dimensionless_area - math.pi**4 / 15.0, -4.5e-6, delta=1e-7
        )
        self.assertAlmostEqual(wavelength.dimensionless_area, 8 * math.pi**6 / 63, places=6)

    def test_beat5_temperature_ratios_and_constants(self):
        for quantity, constant in (("frequency", phys.SIGMA_SB / math.pi),
                                   ("energy_density", 4 * phys.SIGMA_SB / phys.C_LIGHT)):
            values = [self.result(("--T", t, "--quantity", quantity)).physical_integral
                      for t in ("3000", "6000", "12000")]
            with self.subTest(quantity=quantity):
                self.assertAlmostEqual(values[1] / values[0], 16.0, places=2)
                self.assertAlmostEqual(values[2] / values[1], 16.0, places=2)
                self.assertAlmostEqual(values[2] / values[0], 256.0, places=1)
                self.assertLess(relative_error(values[0] / 3000.0**4, constant), 1e-5)
        self.assertAlmostEqual(phys.SIGMA_SB / math.pi, 1.8049e-8, delta=5e-13)
        self.assertAlmostEqual(4 * phys.SIGMA_SB / phys.C_LIGHT, 7.5657e-16, delta=5e-21)

    def test_beat6_grid_steps_peaks_and_convergence_rate(self):
        steps = {n: self.result(("--n_steps", str(n))) for n in (10, 100, 1000, 2000, 20000)}
        for n, expected in ((10, 9.999), (100, 0.9999), (1000, 0.09999),
                            (2000, 0.049995), (20000, 0.0049995)):
            self.assertAlmostEqual((100.0 - 0.01) / n, expected, places=7)
        self.assertEqual(steps[100].x_peak, steps[1000].x_peak)
        self.assertAlmostEqual(steps[100].x_peak, 5.0095, places=9)
        self.assertLess(abs(steps[2000].x_peak - 4.965114), abs(steps[1000].x_peak - 4.965114))
        for n, result in steps.items():
            step = (100.0 - 0.01) / n
            self.assertLess(abs(result.x_peak - peak_root(5)), step)
            if n >= 1000:
                self.assertLessEqual(abs(result.x_peak - peak_root(5)), 0.5 * step + 1e-9)
        errors = {n: r.physical_integral / r.exact_physical_integral - 1.0 for n, r in steps.items()}
        self.assertAlmostEqual(errors[10], -0.930, places=3)
        self.assertAlmostEqual(errors[100], -8.9e-4, delta=1e-5)
        self.assertAlmostEqual(errors[1000] / errors[2000], 3.85, delta=0.1)
        self.assertAlmostEqual(errors[20000], 8.7e-8, delta=1e-8)
        self.assertLess(relative_error(steps[10].physical_integral, steps[10].exact_physical_integral),
                        1.0)
        self.assertAlmostEqual(steps[10].physical_integral / steps[10].exact_physical_integral, 0.0696, places=3)
        self.assertAlmostEqual(peak_root(5) / steps[2000].x_peak - 1.0, 0.0011, places=4)

    def test_beat6_and_7_the_error_floor_comes_from_the_switch_and_the_range_not_the_grid(self):
        fine = driver.run_planck2(5900.0, "frequency", 200000)
        floor = fine.physical_integral / fine.exact_physical_integral - 1.0
        self.assertAlmostEqual(floor, 7e-8, delta=1e-8)
        exact_area = math.pi**4 / 15.0
        over = trapezoid_simpson(lambda x: x * x - shape(x, 3), 0.01, 0.05) / exact_area
        omitted = 0.01**3 / 3.0 / exact_area
        self.assertAlmostEqual(over, 1.2e-7, delta=1e-8)
        self.assertAlmostEqual(omitted, 5e-8, delta=1e-8)
        self.assertAlmostEqual(over - omitted, floor, delta=1.5e-8)
        no_switch = driver.run_planck2(
            5900.0, "frequency", 200000, phys.PlanckDomain(0.01, 100.0, 0.01, 100.0)
        )
        self.assertAlmostEqual(
            no_switch.physical_integral / no_switch.exact_physical_integral - 1.0,
            -omitted, delta=1.5e-8,
        )

    def test_beat7_switch_runs_and_the_error_sizes_of_the_two_forms(self):
        exact = self.result(()).exact_physical_integral
        off = self.result(("--x_low", "0.01", "--x_high", "100"))
        self.assertEqual(f"{off.physical_integral:.6e}", f"{self.result(()).physical_integral:.6e}")
        wave = self.result(("--x_low", "1", "--x_high", "3"))
        freq = self.result(("--quantity", "frequency", "--x_low", "1", "--x_high", "3"))
        self.assertAlmostEqual(wave.dimensionless_area / (8 * math.pi**6 / 63) - 1.0, -0.0066, places=4)
        self.assertAlmostEqual(wave.physical_integral / exact - 1.0, 0.0066, places=4)
        self.assertAlmostEqual(freq.dimensionless_area / (math.pi**4 / 15) - 1.0, 0.0066, places=4)
        self.assertLess(relative_error(wave.physical_integral, freq.physical_integral), 1e-12)
        wrong = self.result(("--x_low", "5", "--x_high", "6"))
        self.assertAlmostEqual(wrong.dimensionless_area / (8 * math.pi**6 / 63), 5.65, places=2)
        self.assertAlmostEqual(wrong.physical_integral / exact, 6.60, places=2)
        self.assertEqual(round(wrong.x_peak, 6), round(self.result(()).x_peak, 6))
        self.assertEqual(wrong.x_peak, max(v for v in wrong.x_values if v < 5.0))
        self.assertAlmostEqual(wrong.y_peak / self.result(()).y_peak, 28.5, delta=0.1)
        for x, expected in ((0.05, 0.025), (1.0, 0.72)):
            self.assertAlmostEqual(math.expm1(x) / x - 1.0, expected, delta=0.005)
        self.assertAlmostEqual(math.exp(-3.0), 0.05, delta=0.001)
        self.assertLess(math.exp(-20.0), 3e-9)

    def test_beat8_truncation_estimates_and_hard_cases(self):
        exact = self.result(()).exact_physical_integral
        cut = self.result(("--x_max", "10", "--x_high", "9"))
        upper5 = trapezoid_simpson(lambda x: shape(x, 5), 10.0, 200.0)
        upper3 = trapezoid_simpson(lambda x: shape(x, 3), 10.0, 200.0)
        self.assertAlmostEqual(upper5, 8.05, places=2)
        self.assertAlmostEqual(upper3, 0.062, places=3)
        self.assertAlmostEqual(upper5 / (8 * math.pi**6 / 63), 0.066, places=3)
        self.assertAlmostEqual(upper3 / (math.pi**4 / 15), 0.0096, places=4)
        closed = math.exp(-10.0) * sum(
            math.factorial(5) // math.factorial(5 - j) * 10.0 ** (5 - j) for j in range(6)
        )
        self.assertAlmostEqual(closed, upper5, places=2)
        self.assertAlmostEqual(cut.dimensionless_area / (8 * math.pi**6 / 63) - 1.0, -0.066, places=3)
        self.assertAlmostEqual(cut.physical_integral / exact - 1.0, -0.0096, places=4)
        edge = self.result(("--x_max", "4", "--x_high", "3"))
        self.assertEqual(edge.x_peak, 4.0)
        self.assertAlmostEqual(edge.physical_integral / exact, 0.59, places=2)
        low = self.result(("--x_min", "1", "--x_low", "1"))
        self.assertAlmostEqual(low.physical_integral / exact - 1.0, -0.035, places=3)
        self.assertAlmostEqual(low.dimensionless_area / (8 * math.pi**6 / 63) - 1.0, -0.0011, places=4)
        self.assertAlmostEqual(trapezoid_simpson(lambda x: shape(x, 3), 1e-9, 1.0), 0.2248, places=4)
        self.assertAlmostEqual(trapezoid_simpson(lambda x: shape(x, 5), 1e-9, 1.0), 0.1284, places=4)
        full = self.result(("--y_frac_window", "0"))
        default = self.result(())
        self.assertEqual(full.physical_integral, default.physical_integral)
        self.assertAlmostEqual(min(default.coord_values), 2.4e-8, delta=5e-10)
        self.assertAlmostEqual(max(default.coord_values), 2.4e-4, delta=5e-6)

    def test_zero_window_axis_never_extends_to_negative_coordinates(self):
        result = driver.run_planck2(5900.0, "wavelength", 200)
        with mock.patch.object(plt, "show"):
            plotter.plot_planck2(result, y_frac_window=0.0)
        left, right = plt.gca().get_xlim()
        plt.close("all")
        self.assertEqual((left, right), (min(result.coord_values), max(result.coord_values)))
        self.assertGreater(left, 0.0)

    def test_domain_error_message_lists_the_four_values(self):
        run = run_cli(("--x_max", "10"))
        self.assertEqual(run.exit_code, 2)
        self.assertIn("x_min=0.01, x_low=0.05, x_high=20, x_max=10", run.stderr)


class Audit21Tests(unittest.TestCase):
    """Console peak line, qualified peak bound, scaling-versus-accuracy, descriptor integrity."""

    def test_console_peak_line_matches_the_result_for_every_quantity(self):
        for name, spec in phys.QUANTITY_SPECS.items():
            run = run_cli(("--quantity", name, "--T", "4321"))
            with self.subTest(quantity=name):
                self.assertEqual(
                    run.stdout.splitlines()[1],
                    f"Peak {spec.coordinate} = {run.result.coord_peak:.6e} {spec.coordinate_unit}",
                )
                self.assertIn(f"{run.result.coord_peak:.4e}", run.annotation)

    def test_eq17_bound_holds_for_an_interior_unaltered_peak(self):
        for n_steps in (100, 1000, 2000):
            result = driver.run_planck2(5900.0, "wavelength", n_steps)
            with self.subTest(n_steps=n_steps):
                self.assertLess(abs(result.x_peak - peak_root(5)), (100.0 - 0.01) / n_steps)

    def test_eq17_bound_fails_when_the_range_excludes_the_peak(self):
        run = run_cli(("--x_max", "4", "--x_high", "3"))
        step = (4.0 - 0.01) / 2000
        self.assertEqual(run.result.x_peak, 4.0)
        self.assertGreater(abs(run.result.x_peak - peak_root(5)), 400 * step)

    def test_eq17_bound_fails_when_the_switches_reshape_the_curve_at_the_peak(self):
        run = run_cli(("--quantity", "frequency", "--x_low", "5", "--x_high", "6"))
        step = (100.0 - 0.01) / 2000
        self.assertEqual(round(run.result.x_peak, 6), 4.959505)
        self.assertGreater(abs(run.result.x_peak - peak_root(3)), 40 * step)

    def test_doubling_temperature_gives_sixteen_even_for_a_badly_wrong_sum(self):
        for quantity in phys.QUANTITY_SPECS:
            for extra in ((), ("--x_low", "5", "--x_high", "6")):
                low = run_cli(("--quantity", quantity, "--T", "3000", *extra)).result
                high = run_cli(("--quantity", quantity, "--T", "6000", *extra)).result
                with self.subTest(quantity=quantity, extra=extra):
                    self.assertLess(relative_error(high.physical_integral / low.physical_integral, 16.0), 1e-12)
        wrong = run_cli(("--x_low", "5", "--x_high", "6")).result
        self.assertGreater(wrong.physical_integral / wrong.exact_physical_integral, 6.0)

    def test_help_separates_radiance_integral_from_emitted_flux(self):
        beat0 = html_text(section_html(HELP_HTML, "beat0"))
        beat4 = section_html(HELP_HTML, "beat4")
        self.assertNotIn("total power", beat0)
        self.assertNotIn("the same power", html_text(beat4))
        self.assertIn(r"M=\pi\int_{0}^{\infty}B_\nu\,d\nu=\sigma T^{4}", beat4)
        self.assertIn("stored energy per unit volume", html_text(beat4))
        flux = math.pi * run_cli(()).result.exact_physical_integral
        self.assertLess(relative_error(flux, phys.SIGMA_SB * 5900.0**4), 1e-12)

    def test_descriptor_mappings_are_read_only(self):
        with self.assertRaises(TypeError):
            phys.SHAPE_EXPONENT["wavelength"] = 3
        with self.assertRaises(TypeError):
            phys.QUANTITY_SPECS["wavelength"] = phys.QUANTITY_SPECS["frequency"]

    def test_every_helper_and_the_driver_follow_the_descriptor(self):
        altered = dict(phys.QUANTITY_SPECS)
        altered["wavelength"] = replace(altered["wavelength"], shape_exponent=3)
        domain = phys.PlanckDomain()
        baseline = phys.shape_function(2.0, "wavelength", domain)
        with mock.patch.object(phys, "QUANTITY_SPECS", MappingProxyType(altered)):
            self.assertEqual(
                phys.shape_function(2.0, "wavelength", domain),
                phys.shape_function(2.0, "frequency", domain),
            )
            self.assertNotEqual(phys.shape_function(2.0, "wavelength", domain), baseline)
            self.assertEqual(
                driver.run_planck2(5900.0, "wavelength", 20).y_values[3] / phys.prefactor("wavelength", 5900.0),
                phys.shape_function(driver.run_planck2(5900.0, "wavelength", 20).x_values[3], "wavelength", domain),
            )
            frequency_like = replace(
                phys.QUANTITY_SPECS["wavelength"], coordinate="frequency"
            )
        altered["wavelength"] = frequency_like
        with mock.patch.object(phys, "QUANTITY_SPECS", MappingProxyType(altered)):
            self.assertEqual(
                phys.coordinate_jacobian("wavelength", 2.0, 5900.0),
                phys.coordinate_jacobian("frequency", 2.0, 5900.0),
            )
            result = driver.run_planck2(5900.0, "wavelength", 20)
            self.assertEqual(result.coord_values[3], phys.x_to_frequency(result.x_values[3], 5900.0))


class HelpOriginalCompatibilityTests(unittest.TestCase):
    """The Reference Guide version is optional; these tests never require it."""

    @classmethod
    def setUpClass(cls):
        cls.original = HELP_FILE.with_name("Planck2-original.html")
        if not cls.original.is_file():
            raise unittest.SkipTest("Planck2-original.html is not present in this layout")
        cls.text = cls.original.read_text(encoding="utf-8")

    def test_original_stamp_matches_the_program(self):
        match = re.search(
            r'<p id="version_build"[^>]*>\s*Version\s+([^&<\s]+)(?:&nbsp;)+Build\s+([0-9a-f]{12})',
            self.text,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.groups(), (phys.MODEL_VERSION, phys.BUILD_ID))

    def test_original_parameter_defaults_still_match_the_parser(self):
        parser = HelpStructureParser()
        parser.feed(self.text)
        rows = {r[0]: r[1] for s, r in parser.rows if s == "parameters" and len(r) >= 2 and r[0] != "Parameter"}
        for action in planck2_main.build_parser()._actions:
            if action.option_strings and action.option_strings[0] not in ("-h", "--version"):
                default = action.default
                expected = f'"{default}"' if isinstance(default, str) else str(default)
                self.assertEqual(rows[action.option_strings[0]], expected)

    def test_original_commands_still_run(self):
        commands = []
        for block in re.findall(r'<div class="sig">(.*?)</div>', self.text, re.DOTALL):
            for line in html_module.unescape(re.sub(r"<[^>]+>", "", block)).splitlines():
                line = " ".join(line.split())
                if line.startswith("python main.py"):
                    commands.append(line)
        self.assertGreaterEqual(len(commands), 4)
        for command in dict.fromkeys(commands):
            arguments = command_arguments(command)
            if "--help" in arguments:
                continue
            with self.subTest(command=command):
                self.assertIn(run_cli(arguments).exit_code, (None, 0))


class TestQuantitySpecSynchronization(unittest.TestCase):
    """One immutable descriptor feeds the physics, driver, plotter, and Help."""

    def test_specs_are_immutable_and_cover_exactly_the_quantities(self):
        self.assertEqual(set(phys.QUANTITY_SPECS), {"wavelength", "frequency", "energy_density"})
        spec = phys.quantity_spec("frequency")
        self.assertIsInstance(spec, phys.QuantitySpec)
        with self.assertRaises(FrozenInstanceError):
            spec.shape_exponent = 5
        for name, spec in phys.QUANTITY_SPECS.items():
            self.assertEqual(spec.name, name)
            self.assertIn(spec.coordinate, ("wavelength", "frequency"))
            self.assertIn(spec.coordinate_symbol, ("\u03bb", "\u03bd"))

    def test_quantity_spec_rejects_unknown_quantities(self):
        for bad in ("intensity", "", None, 3):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    phys.quantity_spec(bad)

    def test_module_level_helpers_are_views_of_the_descriptor(self):
        for name, spec in phys.QUANTITY_SPECS.items():
            with self.subTest(quantity=name):
                self.assertEqual(phys.SHAPE_EXPONENT[name], spec.shape_exponent)
                self.assertEqual(phys.units_label(name), (spec.x_label, spec.y_label))
                self.assertEqual(phys.physical_integral_units(name), spec.integral_units)
                self.assertEqual(
                    phys.prefactor(name, 5900.0),
                    spec.prefactor_scale * 5900.0 ** spec.shape_exponent,
                )
                self.assertEqual(
                    phys.exact_physical_integral(name, 5900.0),
                    spec.exact_integral_scale * 5900.0 ** 4,
                )

    def test_result_and_plot_metadata_come_from_the_same_descriptor(self):
        for name, spec in phys.QUANTITY_SPECS.items():
            result = driver.run_planck2(5900.0, name, 20)
            with self.subTest(quantity=name):
                self.assertEqual(result.x_label, spec.x_label)
                self.assertEqual(result.y_label, spec.y_label)
                self.assertEqual(result.physical_integral_units, spec.integral_units)
                convert = (
                    phys.x_to_wavelength if spec.coordinate == "wavelength"
                    else phys.x_to_frequency
                )
                self.assertEqual(
                    result.coord_values, tuple(convert(x, 5900.0) for x in result.x_values)
                )
                with mock.patch.object(plt, "show"):
                    plotter.plot_planck2(result)
                annotation = "\n".join(text.get_text() for text in plt.gca().texts)
                plt.close("all")
                self.assertIn(f"Peak {spec.coordinate_symbol} = ", annotation)
                self.assertEqual(plt.get_fignums(), [])

    def test_plotter_rejects_metadata_that_disagrees_with_the_descriptor(self):
        result = driver.run_planck2(5900.0, "wavelength", 20)
        for field in ("x_label", "y_label", "physical_integral_units"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    plotter.plot_planck2(replace(result, **{field: "changed"}))

    def test_parser_is_available_separately_from_parsing(self):
        parser = planck2_main.build_parser()
        self.assertEqual(parser.prog, "Planck2")
        self.assertEqual(parser.parse_args([]).quantity, "wavelength")
        self.assertEqual(
            tuple(action.choices for action in parser._actions
                  if action.option_strings == ["--quantity"])[0],
            tuple(phys.QUANTITY_SPECS),
        )


class TestMainModule(unittest.TestCase):
    def test_cli_defaults_cover_driver_domain_and_plotter(self):
        args = planck2_main.parse_args([])
        self.assertEqual(
            (args.T, args.quantity, args.n_steps, args.x_min, args.x_max,
             args.x_low, args.x_high, args.corner, args.y_frac_window),
            (5900.0, "wavelength", 2000, 0.01, 100.0, 0.05,
             20.0, "upper_right", 0.003),
        )

    def test_cli_forwards_domain_and_selector_translation(self):
        result = driver.run_planck2(
            T=2.725,
            quantity="energy_density",
            n_steps=4,
            domain=phys.PlanckDomain(0.1, 10.0, 0.2, 8.0),
        )
        arguments = [
            "--T", "2.725", "--quantity", "energy_density", "--n_steps", "4",
            "--x_min", "0.1", "--x_max", "10", "--x_low", "0.2",
            "--x_high", "8", "--corner", "lower_left",
            "--y_frac_window", "0",
        ]
        with (
            mock.patch.object(planck2_main, "run_planck2", return_value=result) as run,
            mock.patch.object(planck2_main, "plot_planck2") as plot,
            mock.patch("builtins.print"),
        ):
            planck2_main.main(arguments)
        run.assert_called_once_with(
            T=2.725, quantity="energy_density", n_steps=4,
            domain=phys.PlanckDomain(0.1, 10.0, 0.2, 8.0),
        )
        plot.assert_called_once_with(
            result, corner="lower left", y_frac_window=0.0
        )

    def test_cli_rejects_invalid_ranges_and_selectors(self):
        for arguments in (
            ["--T", "nan"],
            ["--n_steps", "1000001"],
            ["--x_min", "2", "--x_max", "1"],
            ["--x_low", "21", "--x_high", "20"],
            ["--y_frac_window", "1.1"],
            ["--corner", "upper right"],
            ["--quantity", "intensity"],
        ):
            with self.subTest(arguments=arguments):
                with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                    planck2_main.parse_args(arguments)

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
        self.assertIn("Exact 0..infinity bolometric value", completed.stdout)


if __name__ == "__main__":
    unittest.main()
