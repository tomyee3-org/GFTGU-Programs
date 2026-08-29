"""Regression and scientific-validation tests for the Star module.

The locator deliberately supports both repository layout::

    Star/tests/test_physics_star.py

and a flattened upload in which this test file is placed beside the four
program modules.
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
from types import SimpleNamespace
from typing import get_args
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
HELP_FILE = MODULE_DIR / "Star.html"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

# Select a noninteractive backend before plot_star imports pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

import driver_star  # noqa: E402
import main as star_main  # noqa: E402
import physics_star as phys  # noqa: E402
import plot_star  # noqa: E402


DEFAULTS = dict(
    p_c=7.158e15,
    T_c=2.263e7,
    mu=1.285,
    gamma=1.36,
    max_points=2000,
    steps_per_scale=400,
    output_type="pressure",
)


def integrate(**changes):
    """Run the documented default model with selected arguments replaced."""
    values = DEFAULTS.copy()
    values.update(changes)
    return driver_star.integrate_star(**values)


def legacy_java_default_profiles():
    """Reproduce Schutz's supplied Java recurrence independently.

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
        with mock.patch("builtins.open", side_effect=OSError("induced failure")):
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
        with self.assertRaises(ValueError):
            plot_star.plot_star_structure(integrate(output_type="mass"), log_y=True)

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
            mock.patch.object(star_main, "parse_args"),
            mock.patch.object(star_main, "integrate_star", return_value=sentinel) as call,
            mock.patch.object(star_main, "plot_star_structure") as plot,
            mock.patch("builtins.print"),
        ):
            star_main.main()
        call.assert_called_once_with(**DEFAULTS)
        plot.assert_called_once_with(sentinel, log_y=False)

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


class TestHelpFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = HELP_FILE.read_text(encoding="utf-8")

    def test_help_file_exists_and_has_version_element(self):
        self.assertTrue(HELP_FILE.is_file())
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

    def test_help_defaults_and_output_modes_match_main(self):
        for value in ("7.158e15", "2.263e7", "1.285", "1.36", "2000", "400"):
            self.assertIn(value, self.html)
        for mode in ("pressure", "density", "temperature", "mass"):
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


if __name__ == "__main__":
    unittest.main()
