"""Regression and scientific-validation tests for the Star module.

The locator deliberately supports both repository layout::

    Star/tests/test_physics_star.py

and a flattened upload in which this test file is placed beside the four
program modules.  The Beats tutorial Help, Star-claude.html (or Star.html once
it is adopted), is required; the Reference Guide Help, Star-original.html, is
optional and its tests skip without it.
"""

import ast
from contextlib import redirect_stdout
import hashlib
import html
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
from types import SimpleNamespace
from typing import NamedTuple, get_args
import unittest
from unittest import mock


CORE_MODULE_FILENAMES = (
    "physics_star.py",
    "driver_star.py",
    "main.py",
    "plot_star.py",
)


def find_module_dir(start):
    """Find the nearest ancestor containing all four Star core modules."""
    path = Path(start).resolve()
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        if all((candidate / name).is_file() for name in CORE_MODULE_FILENAMES):
            return candidate
    raise FileNotFoundError(
        "Could not find a directory containing all Star core modules: "
        + ", ".join(CORE_MODULE_FILENAMES)
    )


MODULE_DIR = find_module_dir(Path(__file__).resolve().parent)
TEST_FILE = Path(__file__).resolve()
HELP_FILENAMES = ("Star-claude.html", "Star.html")


def find_help_file(module_dir):
    """Find Help in a flattened upload or the GFTGU-Documentation tree.

    Documentation folders no longer use chapter-number prefixes, and the
    Help files live under the sibling ``GFTGU-Documentation`` repository
    rather than beside the program modules inside ``GFTGU-Programs``.
    """
    # The Beats Help is named Star-claude.html until it is adopted as the live
    # Help, when it is renamed Star.html; either name is accepted, and the first
    # one found is used.  The Reference Guide version, Star-original.html, is
    # never used here.
    program_name = "Star"
    candidates = [module_dir / name for name in HELP_FILENAMES]
    for ancestor in (module_dir, *module_dir.parents):
        for name in HELP_FILENAMES:
            candidates.append(ancestor / "GFTGU-Documentation" / program_name / name)
            if ancestor.name != program_name:
                candidates.append(ancestor / program_name / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find Star-claude.html (or Star.html) beside the program or in "
        "GFTGU-Documentation/Star/."
    )


HELP_FILE = find_help_file(MODULE_DIR)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

# Select a noninteractive backend before plot_star imports pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

import driver_star  # noqa: E402
import main as star_main  # noqa: E402
import physics_star as phys  # noqa: E402
import plot_star  # noqa: E402


DEFAULT_PARAMETER_NAMES = (
    "p_c",
    "T_c",
    "mu",
    "gamma",
    "max_points",
    "steps_per_scale",
    "output_type",
    "log_y",
)


class InputParameterTableParser(HTMLParser):
    """Map each parameter name to its default cell in the Help table."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_parameters = False
        self.section_depth = 0
        self.in_cell = False
        self.cell_text = []
        self.row = []
        self.defaults = {}

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "section":
            if self.in_parameters:
                self.section_depth += 1
            elif attributes.get("id") == "parameters":
                self.in_parameters = True
                self.section_depth = 1
        elif self.in_parameters and tag == "tr":
            self.row = []
        elif self.in_parameters and tag == "td":
            self.in_cell = True
            self.cell_text = []

    def handle_data(self, data):
        if self.in_cell:
            self.cell_text.append(data)

    def handle_endtag(self, tag):
        if self.in_parameters and tag == "td" and self.in_cell:
            self.row.append("".join(self.cell_text).strip())
            self.in_cell = False
        elif self.in_parameters and tag == "tr":
            if len(self.row) >= 2:
                name = self.row[0]
                if name.startswith("--"):
                    name = name[2:]
                if name in DEFAULT_PARAMETER_NAMES:
                    self.defaults[name] = ast.literal_eval(self.row[1])
        elif self.in_parameters and tag == "section":
            self.section_depth -= 1
            if self.section_depth == 0:
                self.in_parameters = False


def extract_help_defaults(html_text):
    parser = InputParameterTableParser()
    parser.feed(html_text)
    parser.close()
    return parser.defaults


MAIN_DEFAULTS = vars(star_main.parse_args([]))
DEFAULTS = {
    name: MAIN_DEFAULTS[name]
    for name in DEFAULT_PARAMETER_NAMES
    if name != "log_y"
}


def integrate(**changes):
    """Run the documented default model with selected arguments replaced."""
    values = DEFAULTS.copy()
    values.update(changes)
    return driver_star.integrate_star(**values)


def legacy_java_default_profiles():
    """Reproduce Schutz's supplied Java recurrence independently.

    This transcription follows Star-Java(3).html, sections "Preliminary
    definitions of parameters and constants" and "Program code," from the
    declarations through process()'s initialization, while/for integration,
    negative-pressure stop, and final-array truncation.

    The Java output excludes the first negative-pressure point, so the returned
    profiles end at its last positive-pressure grid point.
    """
    p_c = 7.158e15
    T_c = 2.263e7
    mu = 1.285
    gamma = 1.36
    q = 1.67e-27 * mu / 1.38e-23
    rho_c = p_c * q / T_c
    gamma_recip = 1.0 / gamma
    D = rho_c / p_c**gamma_recip
    dr = math.sqrt(p_c / 6.672e-11) / rho_c / 400.0

    radius = [0.0] * 2000
    pressure = [0.0] * 2000
    density = [0.0] * 2000
    temperature = [0.0] * 2000
    mass = [0.0] * 2000
    radius[0] = 0.0
    pressure[0] = p_c
    temperature[0] = T_c
    density[0] = rho_c
    mass[0] = 0.0

    last_step = 0
    used_dr = dr
    while last_step == 0:
        used_dr = dr
        radius[1] = dr
        pressure[1] = p_c
        density[1] = rho_c
        mass[1] = 4.0 * math.pi * dr**3 * rho_c / 3.0
        temperature[1] = q * p_c / rho_c

        for j in range(2, 2000):
            radius[j] = radius[j - 1] + dr
            pressure[j] = (
                pressure[j - 1]
                - 6.672e-11
                * density[j - 1]
                * mass[j - 1]
                * dr
                / (radius[j - 1] * radius[j - 1])
            )
            if pressure[j] < 0.0:
                last_step = j
                break
            mass[j] = (
                mass[j - 1]
                + 4.0
                * math.pi
                * radius[j - 1]
                * radius[j - 1]
                * density[j - 1]
                * dr
            )
            density[j] = D * pressure[j] ** gamma_recip
            temperature[j] = q * pressure[j] / density[j]
        dr *= 2.0

    return {
        "radial_step": used_dr,
        "radius": radius[:last_step],
        "pressure": pressure[:last_step],
        "density": density[:last_step],
        "temperature": temperature[:last_step],
        "mass": mass[:last_step],
    }


class TestLocationAndReleaseMetadata(unittest.TestCase):
    def test_find_module_dir_from_tests_directory(self):
        self.assertEqual(find_module_dir(TEST_FILE.parent), MODULE_DIR)

    def test_find_module_dir_from_module_directory(self):
        self.assertEqual(find_module_dir(MODULE_DIR), MODULE_DIR)

    def test_find_module_dir_chooses_nearest_complete_ancestor(self):
        with tempfile.TemporaryDirectory() as tmp:
            outer = Path(tmp)
            inner = outer / "inner"
            leaf = inner / "nested"
            leaf.mkdir(parents=True)
            for folder in (outer, inner):
                for name in CORE_MODULE_FILENAMES:
                    (folder / name).touch()
            self.assertEqual(find_module_dir(leaf), inner)

    def test_find_module_dir_rejects_incomplete_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                find_module_dir(tmp)

    def test_flattened_import_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            flat = Path(tmp)
            for name in CORE_MODULE_FILENAMES:
                shutil.copy2(MODULE_DIR / name, flat / name)
            shutil.copy2(HELP_FILE, flat / HELP_FILE.name)
            shutil.copy2(TEST_FILE, flat / TEST_FILE.name)
            code = (
                "import importlib.util, pathlib; "
                "p=pathlib.Path('test_physics_star.py').resolve(); "
                "s=importlib.util.spec_from_file_location('flat_test', p); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "print(m.MODULE_DIR)"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=flat,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertEqual(Path(completed.stdout.strip()), flat)

    def test_model_version_is_semantic(self):
        self.assertRegex(phys.MODEL_VERSION, r"^\d+\.\d+\.\d+$")

    def test_build_id_format_and_coverage(self):
        self.assertRegex(phys.BUILD_ID, r"^[0-9a-f]{12}$")
        self.assertEqual(tuple(phys.BUILD_ID_COVERS), CORE_MODULE_FILENAMES)

    def test_build_id_matches_independent_hash(self):
        digest = hashlib.sha256()
        for name in CORE_MODULE_FILENAMES:
            # Match _compute_build_id() explicitly: UTF-8 text plus universal
            # newline conversion before the content is encoded for hashing.
            with open(
                MODULE_DIR / name, "r", encoding="utf-8", newline=None
            ) as source:
                content = source.read().encode("utf-8")
            digest.update(name.encode("utf-8"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        self.assertEqual(phys.BUILD_ID, digest.hexdigest()[:12])

    def test_build_id_returns_unknown_when_source_read_fails(self):
        with mock.patch.object(
            phys, "_read_build_source", side_effect=OSError("induced failure")
        ):
            self.assertEqual(phys._compute_build_id(), "unknown")

    def test_output_choices_come_from_literal_type(self):
        expected = ("pressure", "density", "temperature", "mass")
        self.assertEqual(get_args(driver_star.OutputType), expected)
        self.assertEqual(driver_star.OUTPUT_TYPES, expected)

    def test_python_310_grammar(self):
        for name in CORE_MODULE_FILENAMES:
            source = (MODULE_DIR / name).read_text(encoding="utf-8")
            ast.parse(source, filename=name, feature_version=(3, 10))

class TestPhysicsRelations(unittest.TestCase):
    def test_constants_retain_educational_model_values(self):
        self.assertEqual(phys.k_BOLTZMANN, 1.38e-23)
        self.assertEqual(phys.MPROTON, 1.67e-27)
        self.assertEqual(phys.G_NEWTON, 6.672e-11)

    def test_q_factor(self):
        expected = phys.MPROTON * 1.285 / phys.k_BOLTZMANN
        self.assertAlmostEqual(phys.q_factor(1.285), expected)

    def test_central_density_ideal_gas_relation(self):
        rho = phys.central_density(7.158e15, 2.263e7, 1.285)
        expected = 7.158e15 * phys.MPROTON * 1.285 / (
            phys.k_BOLTZMANN * 2.263e7
        )
        self.assertAlmostEqual(rho, expected, delta=1e-10 * expected)

    def test_central_density_avoids_intermediate_underflow(self):
        p_c, T_c, mu = 1e-300, 1e100, 1e300
        q = phys.q_factor(mu)
        self.assertEqual(p_c / T_c, 0.0)
        expected = (p_c * q) / T_c
        self.assertGreater(expected, 0.0)
        self.assertEqual(phys.central_density(p_c, T_c, mu), expected)

    def test_central_density_rejects_genuine_final_underflow(self):
        with self.assertRaisesRegex(OverflowError, "central density"):
            phys.central_density(5e-324, 1e308, 1.0)

    def test_polytropic_normalization_reconstructs_central_density(self):
        rho_c = 4.9e4
        p_c = 7.1e15
        gamma = 1.36
        D = phys.polytropic_D(rho_c, p_c, gamma)
        self.assertAlmostEqual(
            phys.density_from_pressure(p_c, D, gamma), rho_c, delta=1e-12 * rho_c
        )

    def test_radial_scale_formula_and_alias(self):
        p_c, rho_c = 7.158e15, 4.9187e4
        expected = math.sqrt(p_c / phys.G_NEWTON) / rho_c
        self.assertAlmostEqual(phys.radial_scale(p_c, rho_c), expected)
        self.assertEqual(phys.scale_height(p_c, rho_c), phys.radial_scale(p_c, rho_c))

    def test_radial_scale_avoids_intermediate_overflow(self):
        value = phys.radial_scale(1e300, 1e200)
        self.assertTrue(math.isfinite(value))
        self.assertGreater(value, 0.0)

    def test_hydrostatic_step_formula(self):
        args = (5e14, 2e4, 4e29, 5e8, 1e5)
        expected = args[0] - phys.G_NEWTON * args[1] * args[2] * args[4] / args[3] ** 2
        self.assertEqual(phys.hydrostatic_step(*args), expected)

    def test_mass_step_formula(self):
        args = (4e29, 5e8, 2e4, 1e5)
        expected = args[0] + 4.0 * math.pi * args[1] ** 2 * args[2] * args[3]
        self.assertEqual(phys.mass_step(*args), expected)

    def test_density_zero_at_zero_pressure(self):
        self.assertEqual(phys.density_from_pressure(0.0, 2.0, 1.4), 0.0)

    def test_temperature_relation_and_surface_limit(self):
        p, rho, mu = 1e12, 100.0, 0.8
        self.assertAlmostEqual(
            phys.temperature_from_prho(p, rho, mu), phys.q_factor(mu) * p / rho
        )
        self.assertEqual(phys.temperature_from_prho(0.0, 0.0, mu), 0.0)

    def test_temperature_surface_still_validates_mu(self):
        for bad_mu in (0.0, -1.0, math.nan, math.inf, True, "1"):
            with self.subTest(mu=bad_mu):
                with self.assertRaises(ValueError):
                    phys.temperature_from_prho(0.0, 0.0, bad_mu)

    def test_positive_scalar_functions_reject_bad_mu(self):
        for bad in (0.0, -1.0, math.nan, math.inf, -math.inf, True, "1"):
            with self.subTest(value=bad):
                with self.assertRaises((ValueError, TypeError)):
                    phys.q_factor(bad)

    def test_central_density_rejects_invalid_inputs(self):
        valid = [7e15, 2e7, 1.0]
        for index in range(3):
            for bad in (0.0, -1.0, math.nan, math.inf, True, "bad"):
                args = valid.copy()
                args[index] = bad
                with self.subTest(index=index, value=bad):
                    with self.assertRaises((ValueError, TypeError, OverflowError)):
                        phys.central_density(*args)

    def test_polytropic_functions_reject_gamma_boundary(self):
        for gamma in (1.2, 1.0, 0.0, -1.0):
            with self.subTest(gamma=gamma):
                with self.assertRaises(ValueError):
                    phys.polytropic_D(1.0, 1.0, gamma)
                with self.assertRaises(ValueError):
                    phys.density_from_pressure(1.0, 1.0, gamma)

    def test_step_functions_reject_invalid_domains(self):
        hydro_cases = [
            (-1.0, 1.0, 1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, -1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0, 1.0, 0.0),
        ]
        for args in hydro_cases:
            with self.subTest(function="hydrostatic_step", args=args):
                with self.assertRaises(ValueError):
                    phys.hydrostatic_step(*args)

        mass_cases = [
            (-1.0, 1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0, 1.0),
            (1.0, 1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0, 0.0),
        ]
        for args in mass_cases:
            with self.subTest(function="mass_step", args=args):
                with self.assertRaises(ValueError):
                    phys.mass_step(*args)

    def test_derived_overflow_is_rejected(self):
        with self.assertRaises(OverflowError):
            phys.central_density(1e308, 1.0, 1e100)
        with self.assertRaises(OverflowError):
            phys.mass_step(1e308, 1e200, 1e200, 1e200)

    def test_density_and_temperature_overflow_is_rejected(self):
        with self.assertRaises(OverflowError):
            phys.density_from_pressure(1e308, 1e308, 1.21)
        with self.assertRaises(OverflowError):
            phys.temperature_from_prho(1e308, 1.0, 1e308)


class TestIntegratedStar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default = integrate()

    def test_default_regression_values(self):
        result = self.default
        self.assertEqual(len(result.radius), 1325)
        self.assertEqual(result.surface_index, 1324)
        self.assertEqual(result.restart_count, 0)
        self.assertAlmostEqual(result.radial_step, 526453.7024821984, delta=1e-6)
        self.assertAlmostEqual(result.radius[-1], 6.967372765527942e8, delta=1.0)
        self.assertAlmostEqual(result.mass[-1], 1.9824511309049755e30, delta=2e20)
        self.assertAlmostEqual(result.density[0], 49186.69619012853, delta=1e-8)

    def test_release_metadata_propagates_to_result(self):
        self.assertEqual(self.default.model_version, phys.MODEL_VERSION)
        self.assertEqual(self.default.build_id, phys.BUILD_ID)

    def test_result_arrays_have_consistent_lengths(self):
        lengths = {
            len(self.default.radius),
            len(self.default.pressure),
            len(self.default.density),
            len(self.default.temperature),
            len(self.default.mass),
        }
        self.assertEqual(lengths, {self.default.surface_index + 1})
        self.assertEqual(self.default.last_index, self.default.surface_index)

    def test_center_and_surface_conditions(self):
        result = self.default
        self.assertEqual(result.radius[0], 0.0)
        self.assertEqual(result.pressure[0], DEFAULTS["p_c"])
        self.assertEqual(result.temperature[0], DEFAULTS["T_c"])
        self.assertEqual(result.mass[0], 0.0)
        self.assertEqual(result.pressure[-1], 0.0)
        self.assertEqual(result.density[-1], 0.0)
        self.assertEqual(result.temperature[-1], 0.0)

    def test_profiles_are_monotone(self):
        result = self.default
        self.assertTrue(all(b > a for a, b in zip(result.radius, result.radius[1:])))
        for profile in (result.pressure, result.density, result.temperature):
            self.assertTrue(all(b <= a for a, b in zip(profile, profile[1:])))
        self.assertTrue(all(b >= a for a, b in zip(result.mass, result.mass[1:])))

    def test_first_nonzero_point_uses_central_approximation(self):
        result = self.default
        dr = result.radial_step
        rho_c = result.density[0]
        self.assertEqual(result.radius[1], dr)
        self.assertEqual(result.pressure[1], DEFAULTS["p_c"])
        self.assertEqual(result.density[1], rho_c)
        self.assertEqual(result.temperature[1], DEFAULTS["T_c"])
        self.assertAlmostEqual(result.mass[1], 4 * math.pi * dr**3 * rho_c / 3)

    def test_interior_points_obey_equation_of_state_and_ideal_gas_law(self):
        result = self.default
        D = phys.polytropic_D(result.density[0], result.pressure[0], DEFAULTS["gamma"])
        for index in (2, 10, 100, 500, 1000, result.surface_index - 1):
            self.assertAlmostEqual(
                result.density[index],
                phys.density_from_pressure(result.pressure[index], D, DEFAULTS["gamma"]),
                delta=1e-11 * result.density[index],
            )
            self.assertAlmostEqual(
                result.temperature[index],
                phys.temperature_from_prho(
                    result.pressure[index], result.density[index], DEFAULTS["mu"]
                ),
                delta=1e-11 * result.temperature[index],
            )

    def test_ordinary_euler_recurrences(self):
        result = self.default
        for index in (2, 10, 100, 500, 1000):
            previous = index - 1
            self.assertEqual(
                result.pressure[index],
                phys.hydrostatic_step(
                    result.pressure[previous],
                    result.density[previous],
                    result.mass[previous],
                    result.radius[previous],
                    result.radial_step,
                ),
            )
            self.assertEqual(
                result.mass[index],
                phys.mass_step(
                    result.mass[previous],
                    result.radius[previous],
                    result.density[previous],
                    result.radial_step,
                ),
            )

    def test_surface_is_interpolated_within_last_full_step(self):
        result = self.default
        i = result.surface_index
        previous = i - 1
        p_trial = phys.hydrostatic_step(
            result.pressure[previous],
            result.density[previous],
            result.mass[previous],
            result.radius[previous],
            result.radial_step,
        )
        self.assertLessEqual(p_trial, 0.0)
        self.assertGreater(result.radius[i], result.radius[previous])
        self.assertLessEqual(
            result.radius[i], result.radius[previous] + result.radial_step
        )

    def test_surface_temperature_uses_physics_limit(self):
        with mock.patch.object(
            driver_star,
            "temperature_from_prho",
            wraps=phys.temperature_from_prho,
        ) as temperature:
            result = integrate()
        temperature.assert_any_call(0.0, 0.0, DEFAULTS["mu"])
        self.assertEqual(result.temperature[-1], 0.0)

    def test_python_prefix_matches_supplied_java_recurrence(self):
        legacy = legacy_java_default_profiles()
        result = self.default
        count = len(legacy["radius"])
        self.assertEqual(result.radial_step, legacy["radial_step"])
        self.assertEqual(result.radius[:count], legacy["radius"])
        self.assertEqual(result.pressure[:count], legacy["pressure"])
        self.assertEqual(result.density[:count], legacy["density"])
        self.assertEqual(result.temperature[:count], legacy["temperature"])
        self.assertEqual(result.mass[:count], legacy["mass"])
        self.assertEqual(count + 1, len(result.radius))

    def test_legacy_java_transcription_has_frozen_numeric_anchors(self):
        legacy = legacy_java_default_profiles()
        # Bit-level regression fixture, recorded with CPython (IEEE-754
        # binary64, round to nearest).  The anchors are hexadecimal floats, so
        # the fixture states every bit exactly and does not depend on how a
        # runtime prints decimal numbers.  The tuples are radius, pressure,
        # density, temperature and mass at the listed grid indices.
        expected_hex = {
            1: ('0x1.010eb67abbf28p+19', '0x1.96e29bf4d6000p+52', '0x1.804564730853bp+15',
                '0x1.594e700000000p+24', '0x1.976a42c23fd01p+74'),
            100: ('0x1.91a6fd1fc5ab6p+25', '0x1.67330d836db4fp+52', '0x1.5e9dac0e7ee59p+15',
                  '0x1.4e1932eae4d2bp+24', '0x1.6ab236178d293p+94'),
            500: ('0x1.f610bc67b7125p+27', '0x1.d58e025c56f1cp+48', '0x1.bcb97fd36ec47p+12',
                  '0x1.5852931fa5076p+23', '0x1.fc7a16ea15f86p+99'),
            1000: ('0x1.f610bc67b720ep+28', '0x1.ca7e41661a236p+40', '0x1.da203d3ea9a0bp+6',
                   '0x1.3b5cb7258d08bp+21', '0x1.8caabd24a8d15p+100'),
            1323: ('0x1.4c1dc243125d7p+29', '0x1.8f6956197eee0p+10', '0x1.9b4688befe5aap-16',
                   '0x1.3cb5125b43b3fp+13', '0x1.905a5240abd56p+100'),
        }
        # The decimal values of the earlier fixture, kept as a cross-check.
        self.assertEqual(float.fromhex(expected_hex[1][0]), 526453.7024821984)
        self.assertEqual(float.fromhex(expected_hex[1323][4]), 1.9824511308692555e30)
        expected = {
            index: tuple(float.fromhex(value) for value in values)
            for index, values in expected_hex.items()
        }
        profiles = (
            legacy["radius"],
            legacy["pressure"],
            legacy["density"],
            legacy["temperature"],
            legacy["mass"],
        )
        for index, values in expected.items():
            with self.subTest(index=index):
                self.assertEqual(tuple(profile[index] for profile in profiles), values)

    def test_all_output_modes_compute_identical_profiles(self):
        baseline = self.default
        for mode in ("pressure", "density", "temperature", "mass"):
            with self.subTest(mode=mode):
                result = integrate(output_type=mode)
                self.assertEqual(result.output_type, mode)
                self.assertEqual(result.radius, baseline.radius)
                self.assertEqual(result.pressure, baseline.pressure)
                self.assertEqual(result.density, baseline.density)
                self.assertEqual(result.temperature, baseline.temperature)
                self.assertEqual(result.mass, baseline.mass)

    def test_restart_doubles_step_and_reports_count(self):
        baseline = integrate(max_points=2000, steps_per_scale=400)
        restarted = integrate(max_points=100, steps_per_scale=400)
        self.assertGreater(restarted.restart_count, 0)
        self.assertEqual(
            restarted.radial_step,
            baseline.radial_step * 2 ** restarted.restart_count,
        )
        self.assertLessEqual(len(restarted.radius), 100)

    def test_restart_exhaustion_raises_bounded_runtime_error(self):
        def pressure_never_falls(p_prev, rho_prev, mass_prev, r_prev, dr):
            return p_prev

        with mock.patch.object(
            driver_star, "hydrostatic_step", side_effect=pressure_never_falls
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Unable to reach the zero-pressure surface"
            ):
                integrate(max_points=3, steps_per_scale=400)

    def test_driver_rejects_invalid_physical_inputs(self):
        for name in ("p_c", "T_c", "mu", "gamma"):
            for bad in (math.nan, math.inf, -math.inf, True, "bad"):
                with self.subTest(name=name, value=bad):
                    with self.assertRaises(ValueError):
                        integrate(**{name: bad})
        for name in ("p_c", "T_c", "mu"):
            for bad in (0.0, -1.0):
                with self.subTest(name=name, value=bad):
                    with self.assertRaises(ValueError):
                        integrate(**{name: bad})
        for gamma in (1.2, 1.0, 0.0, -1.0):
            with self.subTest(gamma=gamma):
                with self.assertRaises(ValueError):
                    integrate(gamma=gamma)

    def test_driver_rejects_invalid_numerical_controls(self):
        for bad in (2, 0, -1, 3.0, True, "3"):
            with self.subTest(max_points=bad):
                with self.assertRaises(ValueError):
                    integrate(max_points=bad)
        for bad in (0, -1, 2.0, True, "2"):
            with self.subTest(steps_per_scale=bad):
                with self.assertRaises(ValueError):
                    integrate(steps_per_scale=bad)
        for bad in ("Pressure", "", None, 1):
            with self.subTest(output_type=bad):
                with self.assertRaises(ValueError):
                    integrate(output_type=bad)

    def test_documented_parameter_scalings(self):
        base = self.default
        pressure = integrate(p_c=4 * DEFAULTS["p_c"])
        temperature = integrate(T_c=1.1 * DEFAULTS["T_c"])
        molecular_weight = integrate(mu=1.2 * DEFAULTS["mu"])
        self.assertAlmostEqual(pressure.radius[-1] / base.radius[-1], 0.5, places=12)
        self.assertAlmostEqual(pressure.mass[-1] / base.mass[-1], 0.5, places=12)
        self.assertAlmostEqual(temperature.radius[-1] / base.radius[-1], 1.1, places=12)
        self.assertAlmostEqual(temperature.mass[-1] / base.mass[-1], 1.1**2, places=12)
        self.assertAlmostEqual(molecular_weight.radius[-1] / base.radius[-1], 1 / 1.2, places=12)
        self.assertAlmostEqual(molecular_weight.mass[-1] / base.mass[-1], 1 / 1.2**2, places=12)

    def test_fixed_eos_scaling_documented_in_experiment_six(self):
        base = self.default
        pressure_factor = 4.0
        gamma = DEFAULTS["gamma"]
        sequence = integrate(
            p_c=DEFAULTS["p_c"] * pressure_factor,
            T_c=DEFAULTS["T_c"] * pressure_factor ** (1.0 - 1.0 / gamma),
        )
        expected_r = pressure_factor ** ((gamma - 2.0) / (2.0 * gamma))
        expected_m = pressure_factor ** ((3.0 * gamma - 4.0) / (2.0 * gamma))
        self.assertAlmostEqual(sequence.radius[-1] / base.radius[-1], expected_r, places=12)
        self.assertAlmostEqual(sequence.mass[-1] / base.mass[-1], expected_m, places=12)

    def test_n_equals_one_polytrope_matches_analytic_lane_emden_solution(self):
        p_c, T_c, mu, gamma = 1e16, 1e7, 1.0, 2.0
        rho_c = phys.central_density(p_c, T_c, mu)
        K = p_c / rho_c**2
        a = math.sqrt(K / (2.0 * math.pi * phys.G_NEWTON))
        exact_radius = math.pi * a
        exact_mass = 4.0 * math.pi**2 * a**3 * rho_c
        numerical = driver_star.integrate_star(
            p_c,
            T_c,
            mu,
            gamma,
            max_points=5000,
            steps_per_scale=1600,
        )
        self.assertLess(abs(numerical.radius[-1] / exact_radius - 1.0), 0.002)
        self.assertLess(abs(numerical.mass[-1] / exact_mass - 1.0), 0.002)

    def test_resolution_refinement_reduces_analytic_error(self):
        p_c, T_c, mu, gamma = 1e16, 1e7, 1.0, 2.0
        rho_c = phys.central_density(p_c, T_c, mu)
        K = p_c / rho_c**2
        exact_radius = math.pi * math.sqrt(K / (2.0 * math.pi * phys.G_NEWTON))
        errors = []
        for steps in (200, 400, 800):
            result = driver_star.integrate_star(
                p_c, T_c, mu, gamma, max_points=5000, steps_per_scale=steps
            )
            self.assertEqual(result.restart_count, 0)
            errors.append(abs(result.radius[-1] - exact_radius))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])


class TestProfileCheckpoints(unittest.TestCase):
    def test_default_fractions_and_target_radii(self):
        result = integrate()
        checkpoints = driver_star.interpolate_profile_checkpoints(result)
        self.assertEqual(
            [sample.radius_fraction for sample in checkpoints],
            [0.0, 0.25, 0.50, 0.75, 0.90],
        )
        for sample in checkpoints:
            self.assertAlmostEqual(
                sample.radius,
                sample.radius_fraction * result.radius[-1],
                places=7,
            )
        self.assertEqual(checkpoints[0].pressure, result.pressure[0])
        self.assertEqual(checkpoints[0].density, result.density[0])
        self.assertEqual(checkpoints[0].temperature, result.temperature[0])
        self.assertEqual(checkpoints[0].mass, 0.0)

    def test_interpolation_is_linear_between_stored_points(self):
        result = SimpleNamespace(
            radius=[0.0, 10.0, 20.0],
            pressure=[100.0, 60.0, 0.0],
            density=[10.0, 6.0, 0.0],
            temperature=[1000.0, 600.0, 0.0],
            mass=[0.0, 20.0, 40.0],
        )
        sample = driver_star.interpolate_profile_checkpoints(result, [0.25])[0]
        self.assertEqual(sample.radius, 5.0)
        self.assertEqual(sample.pressure, 80.0)
        self.assertEqual(sample.density, 8.0)
        self.assertEqual(sample.temperature, 800.0)
        self.assertEqual(sample.mass, 10.0)

    def test_checkpoint_validation(self):
        result = integrate()
        for fraction in (-0.1, 1.1, math.nan, math.inf, True, "0.5"):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                driver_star.interpolate_profile_checkpoints(result, [fraction])

        malformed = SimpleNamespace(
            radius=[0.0, 1.0],
            pressure=[1.0],
            density=[1.0, 0.0],
            temperature=[1.0, 0.0],
            mass=[0.0, 1.0],
        )
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            driver_star.interpolate_profile_checkpoints(malformed)


class TestPlottingAndEntryPoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default = integrate()

    def tearDown(self):
        plot_star.plt.close("all")

    def test_each_output_mode_labels_and_plots(self):
        expected_labels = {
            "pressure": "Pressure [Pa]",
            "density": "Density [kg/m³]",
            "temperature": "Temperature [K]",
            "mass": "Enclosed mass [kg]",
        }
        for mode, ylabel in expected_labels.items():
            with self.subTest(mode=mode):
                result = integrate(output_type=mode)
                with mock.patch.object(plot_star.plt, "show") as show:
                    fig, axes = plot_star.plot_star_structure(result)
                self.assertIs(axes.figure, fig)
                self.assertEqual(axes.get_xlabel(), "Radius [m]")
                self.assertEqual(axes.get_ylabel(), ylabel)
                self.assertIn(mode, axes.get_title())
                show.assert_called_once_with()
                plot_star.plt.close("all")

    def test_log_plot_omits_nonpositive_surface_point(self):
        with mock.patch.object(plot_star.plt, "show"):
            plot_star.plot_star_structure(self.default, log_y=True)
        axes = plot_star.plt.gcf().axes[0]
        self.assertEqual(axes.get_yscale(), "log")
        self.assertEqual(len(axes.lines[0].get_ydata()), len(self.default.pressure) - 1)

    def test_log_mass_is_rejected(self):
        figures_before = plot_star.plt.get_fignums()
        with self.assertRaises(ValueError):
            plot_star.plot_star_structure(integrate(output_type="mass"), log_y=True)
        self.assertEqual(plot_star.plt.get_fignums(), figures_before)

    def test_empty_log_data_is_rejected_without_figure_leak(self):
        result = integrate()
        result.pressure = [0.0] * len(result.pressure)
        figures_before = plot_star.plt.get_fignums()
        with self.assertRaisesRegex(ValueError, "No positive values"):
            plot_star.plot_star_structure(result, log_y=True)
        self.assertEqual(plot_star.plt.get_fignums(), figures_before)

    def test_log_y_requires_a_strict_bool_without_figure_leak(self):
        for bad in ("False", 1, 0, None, [], object()):
            with self.subTest(log_y=bad):
                figures_before = plot_star.plt.get_fignums()
                with self.assertRaisesRegex(TypeError, "log_y must be a bool"):
                    plot_star.plot_star_structure(self.default, log_y=bad)
                self.assertEqual(plot_star.plt.get_fignums(), figures_before)

    def test_unknown_plot_type_is_rejected(self):
        result = integrate()
        result.output_type = "bogus"
        with self.assertRaises(ValueError):
            plot_star.plot_star_structure(result)

    def test_main_passes_documented_defaults_to_driver(self):
        sentinel = SimpleNamespace(
            model_version=phys.MODEL_VERSION,
            build_id=phys.BUILD_ID,
        )
        with (
            mock.patch.object(star_main, "integrate_star", return_value=sentinel) as call,
            mock.patch.object(star_main, "print_structure_summary") as summary,
            mock.patch.object(star_main, "plot_star_structure") as plot,
            mock.patch("builtins.print"),
        ):
            star_main.main([])
        call.assert_called_once_with(**DEFAULTS)
        summary.assert_called_once_with(sentinel)
        plot.assert_called_once_with(sentinel, log_y=False)

    def test_command_line_values_are_forwarded(self):
        sentinel = SimpleNamespace(
            model_version=phys.MODEL_VERSION,
            build_id=phys.BUILD_ID,
        )
        with (
            mock.patch.object(star_main, "integrate_star", return_value=sentinel) as call,
            mock.patch.object(star_main, "print_structure_summary"),
            mock.patch.object(star_main, "plot_star_structure") as plot,
            mock.patch("builtins.print"),
        ):
            star_main.main([
                "--p_c", "8e15",
                "--T_c", "2e7",
                "--mu", "1.1",
                "--gamma", "1.5",
                "--max_points", "3000",
                "--steps_per_scale", "600",
                "--output_type", "temperature",
                "--log_y",
            ])
        call.assert_called_once_with(
            p_c=8e15,
            T_c=2e7,
            mu=1.1,
            gamma=1.5,
            max_points=3000,
            steps_per_scale=600,
            output_type="temperature",
        )
        plot.assert_called_once_with(sentinel, log_y=True)

    def test_cli_rejects_bad_values_and_log_mass(self):
        for arguments in (
            ["--p_c", "0"],
            ["--T_c", "nan"],
            ["--gamma", "1.2"],
            ["--max_points", "2"],
            ["--steps_per_scale", "0"],
            ["--output_type", "luminosity"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                star_main.parse_args(arguments)
        with self.assertRaisesRegex(SystemExit, "cannot be used"):
            star_main.main(["--output_type", "mass", "--log_y"])

    def test_structure_summary_uses_requested_checkpoints_and_five_figures(self):
        result = integrate()
        with mock.patch("builtins.print") as printer:
            star_main.print_structure_summary(result)
        text = "\n".join(" ".join(str(item) for item in call.args) for call in printer.call_args_list)
        self.assertIn("Radius (m):     6.9674e+08", text)
        self.assertIn("Total mass (kg): 1.9825e+30", text)
        for label in ("0%", "25%", "50%", "75%", "90%"):
            self.assertRegex(text, rf"(?m)^\s*{label}\s")

    def test_version_command_from_module_directory(self):
        completed = subprocess.run(
            [sys.executable, str(MODULE_DIR / "main.py"), "--version"],
            cwd=MODULE_DIR,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(
            completed.stdout.strip(), f"Star {phys.MODEL_VERSION} (build {phys.BUILD_ID})"
        )


class TestOriginalHelpCompatibility(unittest.TestCase):
    """The Reference Guide version is optional; these tests never require it."""

    @classmethod
    def setUpClass(cls):
        cls.original = HELP_FILE.with_name("Star-original.html")
        if not cls.original.is_file():
            raise unittest.SkipTest("Star-original.html is not present in this layout")
        cls.html = cls.original.read_text(encoding="utf-8")

    def test_help_file_exists_and_has_version_element(self):
        self.assertTrue(self.original.is_file())
        self.assertRegex(self.html, r'<p\s+id="version_build"[^>]*>')

    def test_help_version_and_build_match_code(self):
        match = re.search(
            r'<p\s+id="version_build"[^>]*>\s*Version\s+([^&<\s]+)'
            r'(?:&nbsp;)+Build\s+([0-9a-f]{12})\s*</p>',
            self.html,
            flags=re.IGNORECASE,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), phys.MODEL_VERSION)
        self.assertEqual(match.group(2), phys.BUILD_ID)

    def test_help_default_table_matches_main_assignments_structurally(self):
        self.assertEqual(set(MAIN_DEFAULTS), set(DEFAULT_PARAMETER_NAMES))
        self.assertEqual(extract_help_defaults(self.html), MAIN_DEFAULTS)

    def test_help_lists_every_runtime_output_mode(self):
        for mode in driver_star.OUTPUT_TYPES:
            self.assertIn(f'<code>"{mode}"</code>', self.html)

    def test_help_states_model_scope_and_finite_radius_condition(self):
        self.assertIn("not be confused with a high-accuracy solar-interior model", self.html)
        self.assertIn(r"\(\gamma > 6/5\)", self.html)
        self.assertIn(r"\(\gamma=4/3\)", self.html)

    def test_help_documents_core_only_build_scope_and_plot_return(self):
        self.assertIn("computed from the four executable core modules", self.html)
        self.assertIn("Help file or the regression tests alone", self.html)
        self.assertIn("plot_star_structure(result, log_y=False) → (fig, ax)", self.html)

    def test_mathjax_greater_than_symbols_are_literal(self):
        self.assertNotRegex(self.html, r"\\\([^)]*&gt;[^)]*\\\)")

    def test_experiment_six_distinguishes_fixed_gamma_from_fixed_eos(self):
        self.assertIn("6 · Advanced — Compare Two Kinds of Polytropic Family", self.html)
        self.assertIn(r"not a fixed-\(K\) sequence", self.html)
        self.assertIn("genuinely fixed-equation-of-state family", self.html)
        self.assertIn(r"(\gamma-2)/(2\gamma)", self.html)
        self.assertIn(r"(3\gamma-4)/(2\gamma)", self.html)

    def test_development_history_is_confined_to_license_provenance(self):
        student_content = self.html.split('<section id="license">', 1)[0].lower()
        for phrase in ("ai-generated", "legacy critique", "converted from java"):
            self.assertNotIn(phrase, student_content)

    def test_original_describes_the_new_summary_lines_and_limits(self):
        text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", self.html)).split())
        for phrase in ("Numerical grid", "Restarts (radial step doubled)",
                       "Lane-Emden solution of the same polytrope (exact)",
                       "measure the error of the Euler integration",
                       "from 3 to 1000000"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)
        self.assertNotIn("at least 3.", text)

    def test_original_gives_no_chapter_numbers_and_links_to_no_other_help(self):
        text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", self.html)).split())
        self.assertNotRegex(text, r"\bChapter\s+\d|\bInvestigations?\s+\d")
        links = [h for h in re.findall(r'href="([^"]*)"', self.html)
                 if not h.startswith(("#", "http://", "https://", "mailto:"))]
        self.assertEqual(links, [])

    def test_original_commands_run(self):
        commands = []
        for block in re.findall(r'<div class="sc-params">(.*?)</div>', self.html, re.DOTALL) + \
                re.findall(r"<pre[^>]*>(.*?)</pre>", self.html, re.DOTALL):
            text = html.unescape(re.sub(r"<br\s*/?>", "\n", block))
            text = re.sub(r"<[^>]+>", "", text).replace("\\\n", " ")
            joined = re.sub(r"\n\s+--", " --", text)
            for line in joined.splitlines():
                line = " ".join(line.split())
                if line.startswith("python main.py") and "--help" not in line and "--version" not in line:
                    commands.append(line)
        self.assertGreaterEqual(len(commands), 15)
        for command in dict.fromkeys(commands):
            with self.subTest(command=command):
                run = run_cli(command_arguments(command))
                self.assertIn(run.exit_code, (None, 0), run.stderr)
                self.assertIn("Stellar structure summary", run.stdout)



# ---------------------------------------------------------------------------
# The Lane-Emden comparison, input limits and the printed summary
# ---------------------------------------------------------------------------


class TestLaneEmden(unittest.TestCase):
    # First zeros and mass factors from the standard tables of Lane-Emden
    # functions (e.g. Chandrasekhar, An Introduction to the Study of Stellar
    # Structure, Table 4), to the figures given there.
    TABLE = {
        1.5: (3.65375, 2.71406),
        3.0: (6.89685, 2.01824),
        4.0: (14.97155, 1.79723),
        4.5: (31.83646, 1.73780),
    }

    def test_exact_solutions_for_n_zero_and_one(self):
        xi, omega = phys.lane_emden_surface(0.0)
        self.assertAlmostEqual(xi / math.sqrt(6.0), 1.0, delta=1e-12)
        self.assertAlmostEqual(omega / (2.0 * math.sqrt(6.0)), 1.0, delta=1e-12)
        xi, omega = phys.lane_emden_surface(1.0)
        self.assertAlmostEqual(xi / math.pi, 1.0, delta=1e-10)
        self.assertAlmostEqual(omega / math.pi, 1.0, delta=1e-7)

    def test_tabulated_polytropes(self):
        for n, (xi_table, omega_table) in self.TABLE.items():
            xi, omega = phys.lane_emden_surface(n)
            with self.subTest(n=n):
                self.assertEqual(round(xi, 5), xi_table)
                self.assertEqual(round(omega, 5), omega_table)

    def test_result_is_converged_in_the_step(self):
        for n in (0.5, 2.7777777777777772, 4.9):
            coarse = phys.lane_emden_surface(n)
            with mock.patch.object(phys, "LANE_EMDEN_RELATIVE_STEP", phys.LANE_EMDEN_RELATIVE_STEP / 4):
                fine = phys.lane_emden_surface(n)
            with self.subTest(n=n):
                self.assertAlmostEqual(coarse[0] / fine[0], 1.0, delta=1e-9)
                self.assertAlmostEqual(coarse[1] / fine[1], 1.0, delta=1e-8)

    def test_rejected_indices_and_exponents(self):
        for bad in (-0.1, 5.0, 5.5, math.nan, math.inf, True, "1"):
            with self.subTest(n=bad), self.assertRaises(ValueError):
                phys.lane_emden_surface(bad)
        for bad in (1.2, 1.0, math.nan, True):
            with self.subTest(gamma=bad), self.assertRaises(ValueError):
                phys.polytropic_index(bad)
        self.assertAlmostEqual(phys.polytropic_index(1.36), 1 / 0.36, delta=1e-15)

    def test_surface_beyond_the_largest_xi_is_reported(self):
        with mock.patch.object(phys, "LANE_EMDEN_MAX_XI", 20.0):
            with self.assertRaisesRegex(ValueError, "beyond xi = 20"):
                phys.lane_emden_surface(4.5)
            self.assertAlmostEqual(phys.lane_emden_surface(4.0)[0], 14.97155, delta=1e-5)
        with self.assertRaisesRegex(ValueError, "too close to 6/5"):
            phys.lane_emden_solution(7.158e15, 49186.69619012853, 1.2 + 1e-13)

    def test_physical_radius_and_mass_formulas(self):
        p_c, rho_c = 7.158e15, 49186.69619012853
        exact = phys.lane_emden_solution(p_c, rho_c, 2.0)
        self.assertEqual(exact.n, 1.0)
        radius = math.sqrt(math.pi * p_c / (2.0 * phys.G_NEWTON)) / rho_c
        a = radius / math.pi
        self.assertAlmostEqual(exact.radius / radius, 1.0, delta=1e-10)
        self.assertAlmostEqual(exact.mass / (4.0 * math.pi**2 * a**3 * rho_c), 1.0, delta=1e-7)
        default = phys.lane_emden_solution(p_c, rho_c, 1.36)
        scale = phys.radial_scale(p_c, rho_c) * math.sqrt((default.n + 1) / (4 * math.pi))
        self.assertAlmostEqual(default.length_scale, scale, delta=1e-6)
        self.assertAlmostEqual(default.radius, default.xi_1 * scale, delta=1e-3)

    def test_integration_converges_to_the_exact_solution_at_first_order(self):
        base = integrate()
        exact = phys.lane_emden_solution(base.pressure[0], base.density[0], base.gamma)
        errors = []
        for steps in (400, 800, 1600, 3200):
            result = integrate(steps_per_scale=steps, max_points=20000)
            self.assertEqual(result.restart_count, 0)
            errors.append((result.radius[-1] / exact.radius - 1, result.mass[-1] / exact.mass - 1))
        for coarse, fine in zip(errors, errors[1:]):
            for index in (0, 1):
                with self.subTest(coarse=coarse, index=index):
                    self.assertLess(coarse[index], 0.0)
                    self.assertAlmostEqual(coarse[index] / fine[index], 2.0, delta=0.12)
        self.assertLess(abs(errors[-1][0]), 0.0021)

    def test_exact_solution_scales_like_the_integrated_star(self):
        rho = phys.central_density(DEFAULTS["p_c"], DEFAULTS["T_c"], DEFAULTS["mu"])
        hot_rho = phys.central_density(DEFAULTS["p_c"], 1.1 * DEFAULTS["T_c"], DEFAULTS["mu"])
        base = phys.lane_emden_solution(DEFAULTS["p_c"], rho, 1.36)
        hot = phys.lane_emden_solution(DEFAULTS["p_c"], hot_rho, 1.36)
        self.assertAlmostEqual(hot.radius / base.radius, 1.1, delta=1e-12)
        self.assertAlmostEqual(hot.mass / base.mass, 1.21, delta=1e-12)

    def test_result_records_gamma(self):
        self.assertEqual(integrate().gamma, DEFAULTS["gamma"])
        self.assertIsInstance(integrate(gamma=2).gamma, float)


class TestNumericalRangeAndLimits(unittest.TestCase):
    def test_max_points_limit_in_driver_and_parser(self):
        self.assertEqual(driver_star.MAX_POINTS_LIMIT, 1_000_000)
        with self.assertRaisesRegex(ValueError, "from 3 to 1000000"):
            integrate(max_points=driver_star.MAX_POINTS_LIMIT + 1)
        self.assertEqual(star_main.parse_args(["--max_points", "1000000"]).max_points, 1_000_000)
        with mock.patch("sys.stderr", io.StringIO()) as err, self.assertRaises(SystemExit):
            star_main.parse_args(["--max_points", "1000001"])
        self.assertIn("at most 1000000", err.getvalue())

    def test_radial_scale_keeps_the_default_arithmetic_and_survives_underflow(self):
        p_c, rho_c = 7.158e15, 49186.69619012853
        self.assertEqual(
            phys.radial_scale(p_c, rho_c), math.sqrt(p_c) / (math.sqrt(phys.G_NEWTON) * rho_c)
        )
        tiny = phys.radial_scale(1e-300, 1e-320)
        self.assertTrue(math.isfinite(tiny))
        self.assertAlmostEqual(tiny / ((1e-150 / math.sqrt(phys.G_NEWTON)) / 1e-320), 1.0, delta=1e-12)

    def test_hydrostatic_step_reports_an_underflowing_radius(self):
        with self.assertRaisesRegex(OverflowError, "r_prev\\*\\*2"):
            phys.hydrostatic_step(1.0, 1.0, 1.0, 1e-200, 1e-200)
        self.assertEqual(phys.hydrostatic_step(1.0, 1.0, 1.0, 2.0, 1.0),
                         1.0 - phys.G_NEWTON * 1.0 * 1.0 * 1.0 / 4.0)

    def test_first_sphere_mass_overflow_is_reported(self):
        with self.assertRaisesRegex(OverflowError, "first central sphere"):
            driver_star.integrate_star(1e-300, 2.263e7, 1.285, 1.36)

    def test_surface_within_rounding_of_the_last_point_keeps_radii_increasing(self):
        result = driver_star.integrate_star(1.8424005553576994e-102, 2.263e7,
                                            1.8951299192051218e+204, 1.36, steps_per_scale=383)
        self.assertTrue(all(b > a for a, b in zip(result.radius, result.radius[1:])))
        self.assertEqual(result.pressure[-1], 0.0)
        self.assertEqual(len(driver_star.interpolate_profile_checkpoints(result)), 5)

    def test_extreme_accepted_inputs_end_with_a_message_not_a_traceback(self):
        cases = (
            ["--mu", "1e300"], ["--p_c", "1e-300"], ["--T_c", "1e300"], ["--p_c", "1e300"],
            ["--p_c", "2.92548235035777e-57", "--T_c", "9.100798919071091e+258", "--steps_per_scale", "71"],
            ["--p_c", "1.2942489173649652e-136", "--T_c", "7.559235903906607e+205",
             "--mu", "6.296798463362172e+22", "--gamma", "3.5285756903294248"],
            ["--p_c", "1.8424005553576994e-102", "--mu", "1.8951299192051218e+204",
             "--steps_per_scale", "383"],
            ["--gamma", "1.2000000000001"], ["--gamma", "1e300"], ["--steps_per_scale", "1e23"],
            ["--steps_per_scale", "100000000000000000000000"],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                run = run_cli(tuple(arguments))
                self.assertNotIn("Traceback", run.stderr)
                if run.exit_code in (None, 0):
                    self.assertNotRegex(run.stdout, r"\bnan\b|\binf\b")
                else:
                    message = str(run.exit_code) + run.stderr
                    self.assertNotIn("Errno", message)
                    self.assertNotIn("(34", message)


def summary_value(text, label):
    """The first number printed after a summary label."""
    match = re.search(re.escape(label) + r":?\s+(\S+)", text)
    if match is None:
        raise AssertionError(f"{label!r} not found in the summary")
    return match.group(1)


class TestPrintedSummary(unittest.TestCase):
    def test_default_summary_grid_and_lane_emden_lines(self):
        text = run_cli(()).stdout
        result = integrate()
        exact = phys.lane_emden_solution(result.pressure[0], result.density[0], result.gamma)
        self.assertEqual(summary_value(text, "Radial step (m)"), f"{result.radial_step:.5g}")
        self.assertEqual(summary_value(text, "Grid points, centre to surface"), str(len(result.radius)))
        self.assertEqual(summary_value(text, "Restarts (radial step doubled)"), "0")
        self.assertEqual(summary_value(text, "Polytropic index n = 1/(gamma - 1)"), "2.7778")
        self.assertEqual(summary_value(text, "Lane-Emden radius (m)"), f"{exact.radius:.5g}")
        self.assertEqual(summary_value(text, "Lane-Emden mass (kg)"), f"{exact.mass:.5g}")
        self.assertEqual(summary_value(text, "integrated radius differs by"),
                         f"{result.radius[-1] / exact.radius - 1:.5g}")
        self.assertEqual(summary_value(text, "integrated mass differs by"),
                         f"{result.mass[-1] / exact.mass - 1:.5g}")
        # the lines of the earlier release are unchanged and come first
        lines = text.splitlines()
        self.assertEqual(lines[3], "  Radius (m):     6.9674e+08")
        self.assertEqual(lines[4], "  Total mass (kg): 1.9825e+30")

    def test_restart_is_reported(self):
        text = run_cli(("--max_points", "1000")).stdout
        self.assertEqual(summary_value(text, "Restarts (radial step doubled)"), "1")
        self.assertEqual(summary_value(text, "Radial step (m)"), "1.0529e+06")
        same = run_cli(("--steps_per_scale", "200")).stdout
        self.assertEqual(text.replace("doubled): 1", "doubled): 0"), same)

    def test_lane_emden_not_computed_is_reported(self):
        text = run_cli(("--gamma", "1.2000000000001")).stdout
        self.assertIn("not computed: the Lane-Emden surface lies beyond xi = 1e+12", text)
        self.assertNotIn("Lane-Emden radius", text)

    def test_summary_of_a_minimal_result_prints_only_what_it_can(self):
        minimal = SimpleNamespace(radius=[0.0, 1.0], pressure=[1.0, 0.0], density=[1.0, 0.0],
                                  temperature=[1.0, 0.0], mass=[0.0, 1.0])
        with redirect_stdout(io.StringIO()) as out:
            star_main.print_structure_summary(minimal)
        self.assertNotIn("Numerical grid", out.getvalue())
        self.assertNotIn("Lane-Emden", out.getvalue())
        self.assertIn("Interior checkpoints", out.getvalue())


# ---------------------------------------------------------------------------
# The Beats tutorial Help
# ---------------------------------------------------------------------------

BEAT_NUMBERS = tuple(range(0, 9))
EQUATION_COUNT = 18
VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
# Commands that the Help documents as being rejected by the program.
REJECTED_COMMANDS = {
    "python main.py --gamma 1.2": "gamma must exceed 1.2 for a finite-radius polytrope",
    "python main.py --output_type mass --log_y": "--log_y cannot be used with --output_type mass",
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
    """Run ``main.main()`` in-process with the plot suppressed; runs are cached."""
    key = tuple(arguments)
    if key in _CLI_CACHE:
        return _CLI_CACHE[key]
    out, err = io.StringIO(), io.StringIO()
    exit_code = None
    with mock.patch.object(star_main, "plot_star_structure"), \
            redirect_stdout(out), mock.patch("sys.stderr", err):
        try:
            star_main.main(list(key))
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

    def test_help_version_and_build_match_code(self):
        match = re.search(
            r'<p\s+id="version_build"[^>]*>\s*Version\s+([^&<\s]+)(?:&nbsp;)+Build\s+([0-9a-f]{12})\s*</p>',
            HELP_HTML,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.groups(), (phys.MODEL_VERSION, phys.BUILD_ID))

    def test_help_file_is_the_beats_version_and_not_the_original(self):
        self.assertIn(HELP_FILE.name, HELP_FILENAMES)
        self.assertNotEqual(HELP_FILE.name, "Star-original.html")
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
                self.assertEqual(block.count("{"), block.count("}"))
                self.assertEqual(len(re.findall(r"\\\\", block)), 0)

    def test_student_content_contains_no_ai_or_review_history(self):
        student_text = HELP_HTML.split('<section id="license">', 1)[0]
        for term in ("Claude", "Copilot", "Gemini", "ChatGPT", "Anthropic", "Codex", "Grok",
                     "AI-generated", "audit", "Kickoff", "previous version", "reviewer",
                     "Java"):
            with self.subTest(term=term):
                self.assertNotIn(term, student_text)
        self.assertIn("Java", HELP_HTML.split('<section id="license">', 1)[1])

    def test_no_chapter_or_investigation_numbers(self):
        text = html_text(HELP_HTML)
        self.assertNotRegex(text, r"\bChapter\s+\d|\bInvestigations?\s+\d")

    def test_related_programs_are_named_without_links(self):
        relative = [h for h in STRUCTURE.hrefs
                    if not h.startswith(("#", "http://", "https://", "mailto:"))]
        self.assertEqual(relative, [])
        related = html_text(section_html(HELP_HTML, "related"))
        for name in ("SphereGravity", "Atmosphere", "Binary", "Neutron"):
            self.assertIn(name, related)

    def test_no_other_help_file_links_into_this_folder(self):
        docs_root = HELP_FILE.parent.parent
        if docs_root.name != "GFTGU-Documentation" or not (docs_root / "SphereGravity").is_dir():
            self.skipTest("the sibling documentation folders are not present in this layout")
        inbound = []
        for page in sorted(docs_root.glob("*/*.html")):
            if page.parent == HELP_FILE.parent:
                continue
            for href in re.findall(r'href="([^"#]*)', page.read_text(encoding="utf-8", errors="replace")):
                if "../Star/" in href or href.startswith("Star/"):
                    inbound.append((page.name, href))
        self.assertEqual(inbound, [])


class TestBeatsHelpBeats(unittest.TestCase):
    TAGS = {"LAW": "law", "ODE": "ode", "CLOSURE": "clo", "DEFINITION": "def",
            "DERIVED": "der", "ALGORITHM": "alg"}

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
                self.assertEqual(body.count("<pre>"), 1)
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
        self.assertIn("Eighteen equations are numbered", html_text(section_html(HELP_HTML, "beats")))

    def test_tag_counts_match_the_reading_note(self):
        by_kind = {}
        for number, (kind, _, _) in self.equations().items():
            by_kind.setdefault(kind, []).append(number)
        self.assertEqual(sorted(by_kind["LAW"]), [4])
        self.assertEqual(sorted(by_kind["ODE"]), [2, 3])
        self.assertEqual(sorted(by_kind["CLOSURE"]), [5])
        self.assertEqual(sorted(by_kind["ALGORITHM"]), [14, 15, 16, 17])
        note = html_text(section_html(HELP_HTML, "beats"))
        for phrase in ("Eq. (4), the ideal gas law, is the only equation with this tag", "Eqs. (2) and (3)",
                       "Eq. (5), the polytrope, is the only one", "Eqs. (14) to (17)"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, note)

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
        # every "Experiment N" named in running text exists as well
        for number in re.findall(r"Experiments? (\d+)", html_text(HELP_HTML)):
            self.assertIn(number, experiments)

    def test_experiments_are_ranked_from_introductory_to_advanced(self):
        rank = {"Introductory": 0, "Introductory to Intermediate": 1, "Intermediate": 2,
                "Intermediate to Advanced": 3, "Advanced": 4, "Advanced Programming": 4}
        levels = [
            rank[html_text(title).rsplit("— ", 1)[1]]
            for title in re.findall(r'<h3 id="exp\d+">(.*?)</h3>', section_html(HELP_HTML, "experiments"))
        ]
        self.assertEqual(levels, sorted(levels))


class TestBeatsHelpCommands(unittest.TestCase):
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
                    self.assertNotIn(run.exit_code, (None, 0))
                    self.assertIn(REJECTED_COMMANDS[command], str(run.exit_code) + run.stderr)
                    self.assertEqual(run.stdout, "")
                else:
                    self.assertIn(run.exit_code, (None, 0), run.stderr)
                    lines = run.stdout.splitlines()
                    self.assertEqual(lines[0], f"Star {phys.MODEL_VERSION} (build {phys.BUILD_ID})")
                    self.assertIn("  Interior checkpoints (linear interpolation in radius)", lines)

    def test_rejected_commands_are_documented_in_their_beat(self):
        body = section_html(HELP_HTML, "beat8")
        for command in REJECTED_COMMANDS:
            self.assertIn(command, documented_commands(body))
        self.assertEqual(beat_run("python main.py --gamma 1.2").exit_code, 2)

    def test_documented_options_are_real_options(self):
        with mock.patch.object(sys, "argv", ["main.py"]):
            real = {"--p_c", "--T_c", "--mu", "--gamma", "--max_points", "--steps_per_scale",
                    "--output_type", "--log_y", "--help", "--version"}
        used = set(re.findall(r"(?<![\w-])(--[A-Za-z0-9_-]+)", "\n".join(documented_commands(HELP_HTML))))
        self.assertTrue(used)
        self.assertLessEqual(used, real)
        for option in used - {"--help", "--version"}:
            self.assertIn(option.lstrip("-"), MAIN_DEFAULTS)


class TestBeatsHelpReference(unittest.TestCase):
    def test_parameter_table_matches_the_parser_defaults(self):
        self.assertEqual(extract_help_defaults(HELP_HTML), MAIN_DEFAULTS)

    def test_printed_summary_block_is_exactly_what_the_default_run_prints(self):
        blocks = re.findall(r"<pre>(Star .*?)</pre>", section_html(HELP_HTML, "summary"), re.DOTALL)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(html.unescape(blocks[0]).strip(), run_cli(()).stdout.strip())

    def test_every_summary_label_is_described_in_the_help(self):
        text = html_text(section_html(HELP_HTML, "summary"))
        output = run_cli(()).stdout
        labels = [line.split(":", 1)[0].strip() for line in output.splitlines()[1:]
                  if ":" in line]
        labels += ["integrated radius differs by", "integrated mass differs by",
                   "Interior checkpoints", "r/R", "not computed"]
        for label in labels:
            with self.subTest(label=label):
                self.assertIn(label, text)

    def test_constant_table_matches_the_program(self):
        rows = {
            row[0]: row[1]
            for row in STRUCTURE.table_rows
            if len(row) == 3 and row[0].isupper() or (len(row) == 3 and row[0] == "k_BOLTZMANN")
        }
        for name in ("G_NEWTON", "k_BOLTZMANN", "MPROTON", "LANE_EMDEN_RELATIVE_STEP",
                     "LANE_EMDEN_MAX_XI"):
            with self.subTest(name=name):
                self.assertEqual(float(rows[name]), float(getattr(phys, name)))
        self.assertEqual(float(rows["MAX_POINTS_LIMIT"]), driver_star.MAX_POINTS_LIMIT)
        self.assertEqual([float(v) for v in rows["DEFAULT_CHECKPOINT_FRACTIONS"].split()],
                         list(driver_star.DEFAULT_CHECKPOINT_FRACTIONS))

    def test_output_types_are_all_described(self):
        rows = [row for row in STRUCTURE.table_rows if row and row[0] in driver_star.OUTPUT_TYPES]
        self.assertEqual([row[0] for row in rows], list(driver_star.OUTPUT_TYPES))
        labels = {"pressure": "Pressure [Pa]", "density": "Density [kg/m³]",
                  "temperature": "Temperature [K]", "mass": "Enclosed mass [kg]"}
        for row in rows:
            with self.subTest(output_type=row[0]):
                self.assertEqual(row[1], labels[row[0]])

    def test_code_identifiers_named_in_the_help_exist_in_the_program(self):
        modules = (phys, driver_star, plot_star, star_main)
        names = set()
        for body in re.findall(r"<code>([^<]+)</code>", HELP_HTML):
            body = html.unescape(body)
            match = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*)\(\)", body)
            if match:
                names.add(match.group(1))
            elif re.fullmatch(r"[A-Z][A-Z_0-9]{3,}", body):
                names.add(body)
        self.assertGreater(len(names), 15)
        for name in sorted(names):
            with self.subTest(name=name):
                self.assertTrue(any(hasattr(module, name) for module in modules), name)

    def test_help_mentions_python_version_and_core_formulas(self):
        self.assertRegex(HELP_HTML, r"Python\s+3\.10\s+or\s+later")
        for item in (r"\frac{dp}{dr}=-\frac{G\,m(r)\,\rho(r)}{r^{2}}",
                     r"\frac{dm}{dr}=4\pi r^{2}\rho(r)",
                     r"p=K\rho^{\gamma}",
                     r"\ell=\frac{\sqrt{p_c/G}}{\rho_c}",
                     r"a=\ell\sqrt{\frac{n+1}{4\pi}}",
                     r"\theta=\frac{\sin\xi}{\xi}"):
            with self.subTest(item=item):
                self.assertIn(item, HELP_HTML)

    def test_experiment_eight_temperatures_keep_the_equation_of_state_fixed(self):
        commands = documented_commands(html.unescape(section_html(HELP_HTML, "experiments")))
        family = [c for c in commands if "--T_c" in c and "--p_c" in c and "--gamma" not in c
                  and "2.4893e7" not in c]
        self.assertEqual(len(family), 3)
        gamma = DEFAULTS["gamma"]
        runs = []
        for command in family:
            arguments = command_arguments(command)
            p_c = float(arguments[arguments.index("--p_c") + 1])
            T_c = float(arguments[arguments.index("--T_c") + 1])
            self.assertAlmostEqual(T_c / (DEFAULTS["T_c"] * (p_c / DEFAULTS["p_c"]) ** (1 - 1 / gamma)),
                                   1.0, delta=1e-7)
            runs.append((p_c, integrate(p_c=p_c, T_c=T_c)))
        (p0, r0), (p1, r1) = runs[1], runs[2]
        exponent_r = math.log(r1.radius[-1] / r0.radius[-1]) / math.log(p1 / p0)
        exponent_m = math.log(r1.mass[-1] / r0.mass[-1]) / math.log(p1 / p0)
        self.assertAlmostEqual(exponent_r, (gamma - 2) / (2 * gamma), delta=1e-6)
        self.assertAlmostEqual(exponent_m, (3 * gamma - 4) / (2 * gamma), delta=1e-6)



class TestBeatsHelpQuotedNumbers(unittest.TestCase):
    """Every number the beats quote from a run is the number the run prints."""

    NUMBER = re.compile(r"(?<![\w.])-?\d+\.\d+(?:e[+-]\d+)?(?![\w%]|\.\d)")
    WHOLE = re.compile(r"(?<![\w.+-])\d{4,}(?![\w%]|\.\d)")
    # Numbers a student works out from printed values; each is checked in
    # TestBeatsHelpQuantitativeClaims.
    DERIVED = {
        0: set(),
        1: set(),
        2: {"0.015647", "0.3327"},
        3: {"1399", "35.2"},
        4: {"1.1", "1.1000", "1.2100"},
        5: {"35.2", "36.62"},
        6: {"1.888", "1.995", "2000"},
        7: set(),
        8: {"1.2000000001"},
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
            run = beat_run(command)
            chunks.append(command + "\n" + run.stdout + run.stderr + str(run.exit_code))
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
            if c not in REJECTED_COMMANDS
        )
        allowed = {"0.25", "0.75", "1000000", "2.8e+30"}
        for section in ("overview", "beats", "equations", "algorithm", "modules", "quickstart",
                        "parameters", "output", "summary", "experiments"):
            text = self.visible(section_html(HELP_HTML, section))
            tokens = set(self.NUMBER.findall(text)) | set(self.WHOLE.findall(text))
            with self.subTest(section=section):
                self.assertEqual(sorted(t for t in tokens - allowed if t not in every_output), [])


def _summary(command):
    """The printed numbers of one run, keyed by label or checkpoint row."""
    text = beat_run(command).stdout
    values = {
        "radius": float(summary_value(text, "Radius (m)")),
        "mass": float(summary_value(text, "Total mass (kg)")),
        "step": float(summary_value(text, "Radial step (m)")),
        "points": int(summary_value(text, "Grid points, centre to surface")),
        "restarts": int(summary_value(text, "Restarts (radial step doubled)")),
    }
    if "Lane-Emden radius" in text:
        values.update(
            n=float(summary_value(text, "Polytropic index n = 1/(gamma - 1)")),
            le_radius=float(summary_value(text, "Lane-Emden radius (m)")),
            le_mass=float(summary_value(text, "Lane-Emden mass (kg)")),
            radius_error=float(summary_value(text, "integrated radius differs by")),
            mass_error=float(summary_value(text, "integrated mass differs by")),
        )
    rows = {}
    for match in re.finditer(r"^\s+(\d+)%\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$", text, re.M):
        rows[int(match.group(1))] = tuple(float(v) for v in match.groups()[1:])
    values["rows"] = rows  # radius, pressure, density, temperature, mass
    return values


class TestBeatsHelpQuantitativeClaims(unittest.TestCase):
    SUN_RADIUS = 6.957e8   # m, nominal solar radius quoted in Beat 0
    SUN_MASS = 1.989e30    # kg, solar mass quoted in Beat 0

    def test_beat0_default_star_and_the_sun(self):
        run = _summary("python main.py")
        self.assertEqual((run["radius"], run["mass"]), (6.9674e8, 1.9825e30))
        self.assertLess(abs(run["radius"] / self.SUN_RADIUS - 1), 0.002)
        self.assertLess(abs(run["mass"] / self.SUN_MASS - 1), 0.004)
        result = integrate()
        slopes = [(a - b) for a, b in zip(result.pressure, result.pressure[1:])]
        steepest = max(slopes)
        where = result.radius[slopes.index(steepest)] / result.radius[-1]
        self.assertLess(slopes[1], 0.05 * steepest)       # slow at the very centre
        self.assertEqual(round(1 / where), 7)              # steepest at about a seventh
        self.assertLess(run["rows"][50][1] / DEFAULTS["p_c"], 0.02)  # near zero beyond half

    def test_beat1_central_density_and_the_polytrope(self):
        p_c, T_c, mu, gamma = DEFAULTS["p_c"], DEFAULTS["T_c"], DEFAULTS["mu"], DEFAULTS["gamma"]
        rho_c = p_c * 1.67e-27 * mu / (1.38e-23 * T_c)
        self.assertEqual(f"{rho_c:.5g}", "49187")
        row = _summary("python main.py --output_type density --log_y")["rows"][50]
        self.assertEqual(row[1], 1.12e14)
        self.assertEqual(f"{rho_c * (row[1] / p_c) ** (1 / gamma):.5g}", "2313.2")
        result = integrate()
        positive = [value for value in result.density if value > 0]
        self.assertGreater(positive[0] / positive[-1], 1e9)
        self.assertEqual(result.density[-1], 0.0)

    def test_beat2_temperature_from_the_pressure(self):
        gamma = DEFAULTS["gamma"]
        run = _summary("python main.py --output_type temperature")
        centre, half = run["rows"][0], run["rows"][50]
        self.assertEqual(f"{1 - 1 / gamma:.5g}", "0.26471")
        ratio = half[1] / centre[1]
        self.assertEqual(f"{ratio:.5g}", "0.015647")
        self.assertEqual(f"{ratio ** (1 - 1 / gamma):.4g}", "0.3327")
        self.assertEqual(f"{0.3327 * 2.263e7:.4g}", "7.529e+06")
        # the same number to four figures, from the five-figure printed pressures
        self.assertEqual(f"{centre[3] * ratio ** (1 - 1 / gamma):.4g}", f"{half[3]:.4g}")
        self.assertEqual(round(centre[1] / half[1]), 64)
        self.assertEqual(round(centre[3] / half[3]), 3)
        self.assertLess(abs(half[3] / centre[3] - 1 / 3), 0.01)

    def test_beat3_mass_distribution_and_mean_density(self):
        run = _summary("python main.py --output_type mass")
        mass = run["mass"]
        self.assertEqual(round(100 * run["rows"][25][4] / mass), 32)
        self.assertEqual(round(100 * run["rows"][50][4] / mass), 84)
        self.assertEqual(round(100 * (1 - run["rows"][75][4] / mass)), 1)   # outer quarter
        self.assertEqual((100 * 0.25**3, 100 * 0.5**3), (1.5625, 12.5))
        mean = 3 * mass / (4 * math.pi * run["radius"] ** 3)
        self.assertEqual(round(mean), 1399)
        self.assertEqual(f"{49187 / mean:.3g}", "35.2")
        self.assertLess(abs(mean / 1408 - 1), 0.01)   # the Sun's mean density

    def test_beat4_scaling_with_the_central_temperature(self):
        base, hot = _summary("python main.py"), _summary("python main.py --T_c 2.4893e7")
        self.assertEqual(f"{hot['radius'] / base['radius']:.4f}", "1.1000")
        self.assertEqual(f"{hot['mass'] / base['mass']:.4f}", "1.2100")
        self.assertEqual(f"{hot['step'] / base['step']:.3g}", "1.1")
        self.assertEqual(hot["points"], base["points"])
        self.assertEqual((hot["radius_error"], hot["mass_error"]), (base["radius_error"], base["mass_error"]))
        self.assertEqual((base["radius_error"], base["mass_error"]), (-0.014174, -0.0018494))
        rho_c = phys.central_density(DEFAULTS["p_c"], DEFAULTS["T_c"], DEFAULTS["mu"])
        scale = phys.radial_scale(DEFAULTS["p_c"], rho_c)
        self.assertEqual(f"{scale:.5g}", "2.1058e+08")
        self.assertEqual(f"{scale / 400:.5g}", "5.2645e+05")

    def test_beat5_the_n_equals_one_polytrope_by_hand(self):
        run = _summary("python main.py --gamma 2")
        self.assertEqual(run["n"], 1.0)
        by_hand = math.sqrt(math.pi * DEFAULTS["p_c"] / (2 * phys.G_NEWTON)) / 49187
        self.assertEqual(f"{by_hand:.5g}", "2.6392e+08")
        self.assertEqual(run["le_radius"], 2.6392e8)
        self.assertEqual((run["radius"], run["mass"], run["le_mass"]), (2.6239e8, 1.1565e30, 1.1513e30))
        self.assertEqual((run["radius_error"], run["mass_error"]), (-0.0058198, 0.004502))
        xi, omega = phys.lane_emden_surface(1 / 0.36)
        self.assertEqual((f"{xi:.5g}", f"{omega:.5g}"), ("6.1212", "2.0876"))
        self.assertEqual(f"{xi**3 / (3 * omega):.4g}", "36.62")
        self.assertEqual(f"{1 / 0.36:.5g}", "2.7778")

    def test_beat6_first_order_convergence(self):
        base = _summary("python main.py")
        fine = _summary("python main.py --steps_per_scale 800 --max_points 4000")
        self.assertEqual((fine["step"], fine["points"], fine["restarts"]), (2.6323e5, 2666, 0))
        self.assertGreater(fine["points"], 2000)
        self.assertEqual(f"{base['radius_error'] / fine['radius_error']:.4g}", "1.888")
        self.assertEqual(f"{base['mass_error'] / fine['mass_error']:.4g}", "1.995")
        self.assertEqual(round(-100 * base["radius_error"], 1), 1.4)
        self.assertEqual(base["le_radius"], 7.0676e8)
        self.assertEqual(round(100 * (base["le_radius"] / self.SUN_RADIUS - 1), 1), 1.6)

    def test_beat7_restart_and_surface(self):
        run = _summary("python main.py --max_points 1000")
        base = _summary("python main.py")
        self.assertEqual((run["restarts"], run["points"]), (1, 654))
        self.assertEqual(run["step"], 1.0529e6)
        self.assertEqual(f"{2 * base['step']:.5g}", "1.0529e+06")
        self.assertEqual((run["radius"], run["radius_error"]), (6.8749e8, -0.02726))
        self.assertAlmostEqual(run["radius_error"] / base["radius_error"], 2.0, delta=0.1)
        self.assertEqual(beat_run("python main.py --max_points 1000").stdout.replace("doubled): 1", "doubled): 0"),
                         beat_run("python main.py --steps_per_scale 200").stdout)
        # the surface point: interpolated in the last step, with p = rho = T = 0
        result = integrate()
        self.assertLess(result.radius[-1] - result.radius[-2], result.radial_step)
        self.assertEqual((result.pressure[-1], result.density[-1], result.temperature[-1]), (0.0, 0.0, 0.0))

    def test_beat8_near_the_limit_of_one_polytrope(self):
        run = _summary("python main.py --gamma 1.21")
        self.assertEqual((run["restarts"], run["n"]), (4, 4.7619))
        self.assertEqual(run["step"], 8.4233e6)
        self.assertEqual(f"{16 * integrate().radial_step:.5g}", "8.4233e+06")
        self.assertEqual(2 ** run["restarts"], 16)
        self.assertEqual((run["radius"], run["le_radius"], run["radius_error"], run["mass_error"]),
                         (7.2406e9, 9.9626e9, -0.27322, -0.080272))
        self.assertEqual((run["rows"][25][4], run["mass"]), (2.8287e30, 2.8415e30))
        self.assertGreater(run["rows"][25][4] / run["mass"], 0.99)
        # closer still to 6/5 the printed radius is an artefact of the steps
        artefact = _summary("python main.py --gamma 1.2000000001")
        self.assertEqual(artefact["radius_error"], -1.0)
        self.assertTrue(1e7 < artefact["le_radius"] / artefact["radius"] < 1e8)
        # the Lane-Emden solution is refused only for gamma within about 1e-12 of 6/5
        rho_c = 49186.69619012853
        phys.lane_emden_solution(DEFAULTS["p_c"], rho_c, 1.2 + 1e-12)
        with self.assertRaises(ValueError):
            phys.lane_emden_solution(DEFAULTS["p_c"], rho_c, 1.2 + 5e-13)

    def test_experiment_2_checkpoints_scale_with_the_central_temperature(self):
        cool, base = _summary("python main.py --T_c 2.0367e7"), _summary("python main.py")
        for fraction in (25, 50, 75, 90):
            with self.subTest(fraction=fraction):
                self.assertEqual(cool["rows"][fraction][1], base["rows"][fraction][1])
                self.assertAlmostEqual(cool["rows"][fraction][3] / base["rows"][fraction][3],
                                       0.9, delta=2e-4)

    def test_beat6_radius_and_mass_convergence_rates(self):
        errors = {}
        for gamma in (1.36, 4.0):
            rows = []
            for steps in (400, 800, 1600, 3200):
                result = integrate(gamma=gamma, steps_per_scale=steps, max_points=50000)
                exact = phys.lane_emden_solution(result.pressure[0], result.density[0], gamma)
                rows.append((result.radius[-1] / exact.radius - 1, result.mass[-1] / exact.mass - 1))
            errors[gamma] = [(a[0] / b[0], a[1] / b[1]) for a, b in zip(rows, rows[1:])]
        for radius_ratio, mass_ratio in errors[1.36]:
            self.assertAlmostEqual(mass_ratio, 2.0, delta=0.02)
            self.assertTrue(1.85 < radius_ratio < 1.95)
        for _, mass_ratio in errors[4.0]:
            self.assertAlmostEqual(mass_ratio, 2.0, delta=0.02)
        radius_ratios = [ratio for ratio, _ in errors[4.0]]
        self.assertGreater(max(radius_ratios) - min(radius_ratios), 1.0)   # irregular for n < 1

    def test_experiment_commands_behave_as_described(self):
        experiments = section_html(HELP_HTML, "experiments")
        convergence = documented_commands(re.search(r'<h3 id="exp6">.*?(?=<h3 )', experiments, re.S).group(0))
        for command in convergence[:4]:
            with self.subTest(command=command):
                self.assertEqual(_summary(command)["restarts"], 0)
        near_limit = _summary(convergence[4])
        self.assertEqual(near_limit["restarts"], 0)
        self.assertLess(abs(near_limit["radius_error"]), 0.01)
        shapes = documented_commands(re.search(r'<h3 id="exp7">.*?(?=<h3 )', experiments, re.S).group(0))
        self.assertEqual(len(shapes), 6)
        for command in shapes:
            with self.subTest(command=command):
                self.assertEqual(_summary(command)["restarts"], 0)
        self.assertEqual(_summary("python main.py --gamma 1.25")["restarts"], 1)
        restored = _summary("python main.py --T_c 2.4893e7 --p_c 1.048e16")
        self.assertEqual(f"{7.158e15 * 1.1**4:.4g}", "1.048e+16")
        self.assertEqual(restored["mass"], 1.9825e30)
        self.assertLess(restored["radius"], 6.9674e8)
        giant = _summary("python main.py --gamma 1.335 --p_c 7.158e7 --T_c 2.263e5")
        self.assertTrue(80 < giant["radius"] / 6.9674e8 < 130)
        self.assertLess(abs(giant["mass"] / 1.9825e30 - 1), 0.1)
        neutron = _summary("python main.py --mu 1 --gamma 2 --p_c 1.9888e34 --T_c 1.1188e12 --output_type mass")
        self.assertEqual(neutron["radius"], 10000.0)
        self.assertEqual(f"{neutron['mass']:.2g}", "2.8e+30")
        self.assertEqual(f"{1.4 * self.SUN_MASS:.2g}", "2.8e+30")



class TestEndToEndSubprocess(unittest.TestCase):
    """The program runs as a student would run it, with a non-interactive backend."""

    def test_default_run_prints_the_same_summary_as_in_process(self):
        completed = subprocess.run(
            [sys.executable, "main.py"], cwd=str(MODULE_DIR), capture_output=True, text=True,
            timeout=120, env={**os.environ, "MPLBACKEND": "Agg"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, run_cli(()).stdout)

    def test_rejected_input_exits_with_the_documented_message(self):
        completed = subprocess.run(
            [sys.executable, "main.py", "--gamma", "1.2"], cwd=str(MODULE_DIR),
            capture_output=True, text=True, timeout=120, env={**os.environ, "MPLBACKEND": "Agg"},
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(REJECTED_COMMANDS["python main.py --gamma 1.2"], completed.stderr)


if __name__ == "__main__":
    unittest.main()
