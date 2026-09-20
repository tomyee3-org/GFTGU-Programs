"""Regression and unit tests for the Atmosphere teaching model.

The suite uses only Python's standard-library ``unittest`` module so students
do not need an additional test dependency.  Run from the Atmosphere directory:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import hashlib
import html as html_module
import io
import math
import os
from pathlib import Path
import random
import re
import shlex
import subprocess
import sys
import unittest
from unittest.mock import patch


CORE_MODULE_FILENAMES = (
    "physics_atmosphere.py",
    "driver_atmosphere.py",
    "main.py",
    "plot_atmosphere.py",
)


def find_module_dir(start: Path) -> Path:
    """Return the nearest ancestor containing the complete Atmosphere module."""
    start = start.resolve()
    candidates = (start, *start.parents)
    for candidate in candidates:
        if all((candidate / name).is_file() for name in CORE_MODULE_FILENAMES):
            return candidate
    raise RuntimeError(
        "Could not locate the Atmosphere module containing all four core files."
    )


def normalized_utf8_source(raw: bytes) -> bytes:
    """Decode UTF-8 and explicitly normalize CRLF or CR source to LF."""
    text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return text.encode("utf-8")


MODULE_DIR = find_module_dir(Path(__file__).resolve().parent)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import driver_atmosphere as driver  # noqa: E402
import main as entry_point  # noqa: E402
import physics_atmosphere as phys  # noqa: E402
from driver_atmosphere import (  # noqa: E402
    AtmosphereModel,
    AtmosphereParameters,
    AtmosphereResult,
    extract_checkpoints,
    extract_output,
)
from physics_atmosphere import (  # noqa: E402
    TemperatureProfile,
    hydrostatic_step,
    ideal_gas_density,
)


DEFAULT_H = [
    0.0, 11_019.0, 20_063.0, 32_162.0, 47_350.0, 51_412.0,
    71_802.0, 86_000.0, 100_000.0, 150_000.0, 200_000.0,
    250_000.0, 300_000.0, 400_000.0, 500_000.0,
]
DEFAULT_T = [
    288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65,
    186.946, 190.0, 800.0, 1080.0, 1190.0, 1225.0, 1240.0,
    1240.0,
]


def make_params(**overrides):
    values = {
        "planet_name": "Earth",
        "g_accel": 9.81,
        "mu": 28.97,
        "p0": 1.013e5,
        "h_points": DEFAULT_H.copy(),
        "T_points": DEFAULT_T.copy(),
        "output_type": "Pressure",
    }
    values.update(overrides)
    return AtmosphereParameters(**values)


def closest_index(values, target):
    return min(range(len(values)), key=lambda index: abs(values[index] - target))


def find_help_file(module_dir: Path) -> Path:
    """Find Help in a flattened upload, combined ZIP, or documentation tree.

    Documentation folders no longer use chapter-number prefixes, and the
    Help files live under the sibling ``GFTGU-Documentation`` repository
    rather than beside the program modules inside ``GFTGU-Programs``.
    """
    help_filename = "Atmosphere.html"
    program_name = "Atmosphere"
    candidates = [module_dir / help_filename]
    for ancestor in (module_dir, *module_dir.parents):
        candidates.append(ancestor / "Atmosphere-Documentation" / help_filename)
        candidates.append(
            ancestor / "GFTGU-Documentation" / program_name / help_filename
        )
        if ancestor.name != program_name:
            candidates.append(ancestor / program_name / help_filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find Atmosphere.html beside the program or in "
        "GFTGU-Documentation/Atmosphere/ or Atmosphere-Documentation/."
    )


HELP_FILE = find_help_file(MODULE_DIR)


def exact_piecewise_pressure(p0, g_accel, mu, h_points, T_points, target):
    """Exact constant-g, constant-mu hydrostatic pressure within a linear profile."""
    coefficient = g_accel * mu * phys.M_PROTON / phys.K_BOLTZMANN
    pressure = p0
    altitude = h_points[0]
    if altitude != 0.0 or target < 0.0 or target > h_points[-1]:
        raise ValueError("benchmark requires 0 <= target <= final breakpoint")

    for index in range(len(h_points) - 1):
        segment_end = min(target, h_points[index + 1])
        if segment_end <= altitude:
            break
        segment_width = h_points[index + 1] - h_points[index]
        lapse_rate = (T_points[index + 1] - T_points[index]) / segment_width
        t_start = T_points[index] + lapse_rate * (altitude - h_points[index])
        t_end = T_points[index] + lapse_rate * (segment_end - h_points[index])
        if lapse_rate == 0.0:
            pressure *= math.exp(-coefficient * (segment_end - altitude) / t_start)
        else:
            pressure *= (t_end / t_start) ** (-coefficient / lapse_rate)
        altitude = segment_end
        if altitude == target:
            break
    return pressure


def run_main(argv):
    """Run ``main.main(argv)`` headlessly; return its printed text.

    Raises whatever ``main.main`` raises (``SystemExit`` for input errors).
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    captured = io.StringIO()
    try:
        with patch.object(plt, "show"), contextlib.redirect_stdout(captured):
            entry_point.main(list(argv))
    finally:
        plt.close("all")
    return captured.getvalue()


def table_rows(printed):
    """Return the checkpoint-table rows of printed output as lists of cells."""
    rows = []
    for line in printed.splitlines():
        cells = line.split()
        if len(cells) == 5 and re.fullmatch(r"-?[\d.]+(e[+-]\d+)?", cells[0]):
            rows.append(cells)
    return rows


def interpolated_temperature(h_points, T_points, altitude):
    """Independent piecewise-linear temperature (constant below the first point)."""
    if altitude <= h_points[0]:
        return T_points[0]
    for index in range(len(h_points) - 1):
        if h_points[index] <= altitude <= h_points[index + 1]:
            fraction = (altitude - h_points[index]) / (h_points[index + 1] - h_points[index])
            return T_points[index] + fraction * (T_points[index + 1] - T_points[index])
    return None


class BuildMetadataTests(unittest.TestCase):
    def test_module_directory_locator_supports_both_delivery_layouts(self):
        self.assertEqual(find_module_dir(MODULE_DIR), MODULE_DIR)
        self.assertEqual(find_module_dir(MODULE_DIR / "tests"), MODULE_DIR)

    def test_declared_version_is_semantic(self):
        self.assertRegex(phys.MODEL_VERSION, r"^\d+\.\d+\.\d+$")

    def test_build_id_is_twelve_lowercase_hex_digits(self):
        self.assertRegex(phys.BUILD_ID, r"^[0-9a-f]{12}$")

    def test_build_id_covers_exactly_the_four_core_modules(self):
        self.assertEqual(
            phys.BUILD_ID_COVERS,
            (
                "physics_atmosphere.py",
                "driver_atmosphere.py",
                "main.py",
                "plot_atmosphere.py",
            ),
        )

    def test_build_id_matches_core_source_contents(self):
        digest = hashlib.sha256()
        for name in phys.BUILD_ID_COVERS:
            content = normalized_utf8_source((MODULE_DIR / name).read_bytes())
            digest.update(name.encode("utf-8"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        self.assertEqual(phys.BUILD_ID, digest.hexdigest()[:12])

    def test_hash_input_normalization_treats_lf_crlf_and_cr_equally(self):
        lf = b"first\nsecond\n"
        self.assertEqual(normalized_utf8_source(lf), lf)
        self.assertEqual(normalized_utf8_source(b"first\r\nsecond\r\n"), lf)
        self.assertEqual(normalized_utf8_source(b"first\rsecond\r"), lf)


class TemperatureProfileValidationTests(unittest.TestCase):
    def test_valid_profile(self):
        TemperatureProfile([0.0, 1000.0], [280.0, 275.0]).validate()

    def test_tuple_profile_is_a_valid_non_string_sequence(self):
        TemperatureProfile((0.0, 1000.0), (280.0, 275.0)).validate()

    def test_profile_containers_must_be_non_string_sequences(self):
        bad_pairs = (
            (None, [280.0, 275.0]),
            (42, [280.0, 275.0]),
            ("0, 1000", [280.0, 275.0]),
            ([0.0, 1000.0], None),
            ([0.0, 1000.0], 42.0),
            ([0.0, 1000.0], "280, 275"),
        )
        for altitudes, temperatures in bad_pairs:
            with self.subTest(h=altitudes, T=temperatures), self.assertRaisesRegex(
                ValueError, "non-string sequence"
            ):
                TemperatureProfile(altitudes, temperatures).validate()

    def test_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            TemperatureProfile([0.0, 1.0], [280.0]).validate()

    def test_at_least_two_points_are_required(self):
        with self.assertRaisesRegex(ValueError, "At least two"):
            TemperatureProfile([0.0], [280.0]).validate()

    def test_altitudes_must_be_finite_real_numbers(self):
        for bad in (math.nan, math.inf, -math.inf, True, "1000"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "finite"):
                TemperatureProfile([0.0, bad], [280.0, 275.0]).validate()

    def test_temperatures_must_be_finite_positive_real_numbers(self):
        for bad in (0.0, -1.0, math.nan, math.inf, -math.inf, True, "275"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "greater than zero"):
                TemperatureProfile([0.0, 1000.0], [280.0, bad]).validate()

    def test_altitudes_must_be_strictly_increasing(self):
        for altitudes in ([0.0, 0.0], [1000.0, 0.0]):
            with self.subTest(altitudes=altitudes), self.assertRaisesRegex(ValueError, "strictly"):
                TemperatureProfile(list(altitudes), [280.0, 275.0]).validate()

    def test_power_must_be_finite_and_positive(self):
        for bad in (0.0, -0.5, math.nan, math.inf, True):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "power"):
                TemperatureProfile([0.0, 1.0], [280.0, 275.0], power=bad).validate()


class TemperatureInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.profile = TemperatureProfile(
            [100.0, 1100.0, 2100.0],
            [300.0, 250.0, 350.0],
        )
        self.profile.validate()

    def test_below_first_point_uses_first_temperature(self):
        self.assertEqual(self.profile.get_temp(0.0, 100_000.0), 300.0)

    def test_exact_profile_points_are_reproduced(self):
        for altitude, expected in zip(self.profile.h, self.profile.T):
            with self.subTest(altitude=altitude):
                self.assertEqual(self.profile.get_temp(altitude, 90_000.0), expected)

    def test_linear_interpolation_on_rising_and_falling_segments(self):
        self.assertAlmostEqual(self.profile.get_temp(600.0, 95_000.0), 275.0)
        self.assertAlmostEqual(self.profile.get_temp(1600.0, 90_000.0), 300.0)

    def test_first_upper_point_matches_last_supplied_temperature(self):
        pressure = 100.0
        self.assertEqual(self.profile.get_temp(2101.0, pressure), 350.0)
        self.assertTrue(self.profile.reached_top)
        self.assertAlmostEqual(self.profile.beta, 350.0 / math.sqrt(pressure))

    def test_upper_temperature_obeys_power_law_with_fixed_beta(self):
        self.profile.get_temp(2200.0, 100.0)
        self.assertAlmostEqual(self.profile.get_temp(2300.0, 25.0), 175.0)

    def test_zero_pressure_above_profile_returns_last_meaningful_temperature(self):
        self.assertEqual(self.profile.get_temp(2200.0, 0.0), 350.0)

    def test_zero_pressure_query_does_not_corrupt_later_upper_profile_state(self):
        self.assertEqual(self.profile.get_temp(2200.0, 0.0), 350.0)
        self.assertFalse(self.profile.reached_top)
        self.assertEqual(self.profile.beta, 0.0)
        self.assertEqual(self.profile.get_temp(2300.0, 4.0), 350.0)
        self.assertTrue(self.profile.reached_top)
        self.assertEqual(self.profile.beta, 175.0)
        self.assertEqual(self.profile.get_temp(2400.0, 1.0), 175.0)

    def test_upper_power_law_numerical_overflow_is_rejected(self):
        profile = TemperatureProfile([0.0, 1.0], [300.0, 1e308])
        profile.validate()
        with self.assertRaisesRegex(ValueError, "coefficient"):
            profile.get_temp(2.0, 5e-324)

    def test_upper_temperature_underflow_is_rejected(self):
        profile = TemperatureProfile([0.0, 1.0], [300.0, 300.0], power=2.0)
        profile.validate()
        profile.get_temp(2.0, 1.0)
        with self.assertRaisesRegex(ValueError, "temperature"):
            profile.get_temp(3.0, 5e-324)

    def test_invalid_altitude_is_rejected(self):
        for bad in (math.nan, math.inf, True, "0"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "altitude"):
                self.profile.get_temp(bad, 100.0)

    def test_invalid_pressure_is_rejected(self):
        for bad in (-1.0, math.nan, math.inf, True, "100"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "pressure"):
                self.profile.get_temp(0.0, bad)


class IdealGasDensityTests(unittest.TestCase):
    def test_known_earth_surface_density(self):
        expected = 1.013e5 * phys.M_PROTON * 28.97 / (phys.K_BOLTZMANN * 288.15)
        self.assertAlmostEqual(ideal_gas_density(1.013e5, 28.97, 288.15), expected)
        self.assertAlmostEqual(expected, 1.2325, places=4)

    def test_density_is_linear_in_pressure_and_molecular_weight(self):
        reference = ideal_gas_density(100.0, 2.0, 300.0)
        self.assertAlmostEqual(ideal_gas_density(200.0, 2.0, 300.0), 2.0 * reference)
        self.assertAlmostEqual(ideal_gas_density(100.0, 6.0, 300.0), 3.0 * reference)

    def test_density_is_inverse_in_temperature(self):
        reference = ideal_gas_density(100.0, 2.0, 300.0)
        self.assertAlmostEqual(ideal_gas_density(100.0, 2.0, 600.0), 0.5 * reference)

    def test_zero_pressure_has_zero_density(self):
        self.assertEqual(ideal_gas_density(0.0, 28.97, 288.15), 0.0)

    def test_invalid_pressure(self):
        for bad in (-1.0, math.nan, math.inf, True, "1"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "pressure"):
                ideal_gas_density(bad, 28.97, 288.15)

    def test_invalid_molecular_weight(self):
        for bad in (0.0, -1.0, math.nan, math.inf, True, "29"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "mu"):
                ideal_gas_density(1.0, bad, 288.15)

    def test_invalid_temperature(self):
        for bad in (0.0, -1.0, math.nan, math.inf, True, "288"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "temperature"):
                ideal_gas_density(1.0, 28.97, bad)

    def test_non_finite_derived_density_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite density"):
            ideal_gas_density(1e308, 1e308, 1.0)

    def test_positive_input_that_underflows_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "underflows"):
            ideal_gas_density(5e-324, 1.0, 300.0)


class HydrostaticStepTests(unittest.TestCase):
    def test_euler_step(self):
        self.assertAlmostEqual(hydrostatic_step(100_000.0, 1.2, 9.8, 10.0), 99_882.4)

    def test_step_can_signal_boundary_with_negative_pressure(self):
        self.assertLess(hydrostatic_step(1.0, 1.0, 9.8, 1.0), 0.0)

    def test_zero_density_leaves_pressure_unchanged(self):
        self.assertEqual(hydrostatic_step(100.0, 0.0, 9.8, 1.0), 100.0)

    def test_invalid_inputs(self):
        cases = (
            ("pressure_prev", (-1.0, 1.0, 9.8, 1.0)),
            ("rho_prev", (1.0, -1.0, 9.8, 1.0)),
            ("g_accel", (1.0, 1.0, 0.0, 1.0)),
            ("dh", (1.0, 1.0, 9.8, 0.0)),
        )
        for message, args in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                hydrostatic_step(*args)

    def test_non_finite_and_boolean_inputs(self):
        for position in range(4):
            for bad in (math.nan, math.inf, True):
                args = [1.0, 1.0, 1.0, 1.0]
                args[position] = bad
                with self.subTest(position=position, bad=bad), self.assertRaises(ValueError):
                    hydrostatic_step(*args)

    def test_non_finite_derived_pressure_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite pressure"):
            hydrostatic_step(1e308, 1e308, 1e308, 1e308)


class ParameterValidationTests(unittest.TestCase):
    def test_default_parameters_are_valid(self):
        AtmosphereModel(make_params())

    def test_planet_name_must_be_nonempty_string(self):
        for bad in ("", "   ", None, 42):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "planet_name"):
                AtmosphereModel(make_params(planet_name=bad))

    def test_positive_finite_scalar_parameters(self):
        for name in ("g_accel", "mu", "p0"):
            for bad in (0.0, -1.0, math.nan, math.inf, True, "1"):
                with self.subTest(name=name, bad=bad), self.assertRaisesRegex(ValueError, name):
                    AtmosphereModel(make_params(**{name: bad}))

    def test_output_type_is_exactly_one_of_three_choices(self):
        for bad in ("pressure", "PRESSURE", "", None, 1):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "output_type"):
                AtmosphereModel(make_params(output_type=bad))

    def test_model_delegates_profile_validation(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            AtmosphereModel(make_params(h_points=[0.0, 1.0], T_points=[280.0]))


class AtmosphereIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default_result = AtmosphereModel(make_params()).run()

    def test_default_regression_length_step_and_top(self):
        result = self.default_result
        self.assertEqual(len(result.altitudes), 13_653)
        self.assertAlmostEqual(result.altitudes[1], 41.892255239594434, places=10)
        self.assertAlmostEqual(result.altitudes[-1], 571_913.0685308994, places=6)
        self.assertAlmostEqual(result.pressures[-1], 1.48243291309413e-14, delta=1e-25)

    def test_default_pressure_regression_at_key_altitudes(self):
        expected = {
            10_000.0: 26_070.307248623623,
            50_000.0: 70.15205286988557,
            100_000.0: 0.021372282012585772,
            200_000.0: 5.803839221741799e-05,
            500_000.0: 1.1699443144669705e-08,
        }
        for altitude, pressure in expected.items():
            index = closest_index(self.default_result.altitudes, altitude)
            with self.subTest(altitude=altitude):
                self.assertAlmostEqual(self.default_result.pressures[index], pressure, delta=abs(pressure) * 1e-11)

    def test_all_result_arrays_are_coindexed(self):
        result = self.default_result
        self.assertGreater(len(result.altitudes), 1)
        self.assertEqual(len(result.altitudes), len(result.pressures))
        self.assertEqual(len(result.altitudes), len(result.densities))
        self.assertEqual(len(result.altitudes), len(result.temperatures))

    def test_altitude_increases_and_pressure_decreases_strictly(self):
        result = self.default_result
        self.assertTrue(all(b > a for a, b in zip(result.altitudes, result.altitudes[1:])))
        self.assertTrue(all(b < a for a, b in zip(result.pressures, result.pressures[1:])))

    def test_outputs_are_finite_and_physical(self):
        result = self.default_result
        for values in (result.altitudes, result.pressures, result.densities, result.temperatures):
            self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertTrue(all(value >= 0.0 for value in result.altitudes))
        self.assertTrue(all(value > 0.0 for value in result.pressures))
        self.assertTrue(all(value > 0.0 for value in result.densities))
        self.assertTrue(all(value > 0.0 for value in result.temperatures))

    def test_density_matches_ideal_gas_law_throughout_result(self):
        result = self.default_result
        for index in range(0, len(result.altitudes), 137):
            expected = ideal_gas_density(result.pressures[index], 28.97, result.temperatures[index])
            with self.subTest(index=index):
                self.assertAlmostEqual(result.densities[index], expected, delta=abs(expected) * 1e-14)

    def test_metadata_propagates_to_result(self):
        result = self.default_result
        self.assertEqual(result.planet_name, "Earth")
        self.assertEqual(result.output_type, "Pressure")
        self.assertEqual(result.model_version, phys.MODEL_VERSION)
        self.assertEqual(result.build_id, phys.BUILD_ID)

    def test_repeated_runs_are_identical(self):
        model = AtmosphereModel(make_params())
        first = model.run()
        second = model.run()
        self.assertEqual(first.altitudes, second.altitudes)
        self.assertEqual(first.pressures, second.pressures)
        self.assertEqual(first.densities, second.densities)
        self.assertEqual(first.temperatures, second.temperatures)

    def test_reference_temperature_is_interpolated_at_altitude_zero(self):
        params = make_params(h_points=[-1000.0, 1000.0], T_points=[280.0, 300.0])
        result = AtmosphereModel(params).run()
        self.assertEqual(result.altitudes[0], 0.0)
        self.assertAlmostEqual(result.temperatures[0], 290.0)
        self.assertAlmostEqual(result.densities[0], ideal_gas_density(params.p0, params.mu, 290.0))

    def test_profile_starting_above_zero_uses_first_temperature_at_base(self):
        result = AtmosphereModel(
            make_params(h_points=[500.0, 1000.0], T_points=[280.0, 300.0])
        ).run()
        self.assertEqual(result.temperatures[0], 280.0)

    def test_isothermal_euler_solution_and_exact_error(self):
        params = make_params(h_points=[0.0, 100_000.0], T_points=[288.15, 288.15])
        result = AtmosphereModel(params).run()
        scale_height = (
            phys.K_BOLTZMANN * 288.15 / (params.g_accel * params.mu * phys.M_PROTON)
        )
        self.assertAlmostEqual(result.altitudes[1], scale_height / driver.STEPS_PER_SCALE_HEIGHT)

        # Before the end of the supplied isothermal interval, Euler stepping
        # has the exact discrete form p_n = p0 * (1 - 1/N)^n.
        for target in (10_000.0, 50_000.0, 90_000.0):
            index = closest_index(result.altitudes, target)
            discrete = params.p0 * (1.0 - 1.0 / driver.STEPS_PER_SCALE_HEIGHT) ** index
            exact = params.p0 * math.exp(-result.altitudes[index] / scale_height)
            with self.subTest(target=target):
                self.assertAlmostEqual(result.pressures[index], discrete, delta=discrete * 2e-12)
                self.assertLess(abs(result.pressures[index] / exact - 1.0), 0.03)

    def test_linear_lapse_rate_against_independent_analytic_solution(self):
        h_points = [0.0, 100_000.0]
        T_points = [300.0, 400.0]
        params = make_params(h_points=h_points, T_points=T_points)
        result = AtmosphereModel(params).run()
        index = closest_index(result.altitudes, 30_000.0)
        exact = exact_piecewise_pressure(
            params.p0,
            params.g_accel,
            params.mu,
            h_points,
            T_points,
            result.altitudes[index],
        )
        self.assertLess(abs(result.pressures[index] / exact - 1.0), 0.015)

    def test_multilayer_profile_against_exact_piecewise_linear_solution(self):
        params = make_params()
        with (
            patch.object(driver, "STEPS_PER_SCALE_HEIGHT", 800),
            patch.object(driver, "MAX_STEPS", 100_000),
        ):
            result = AtmosphereModel(params).run()
        index = closest_index(result.altitudes, 80_000.0)
        exact = exact_piecewise_pressure(
            params.p0,
            params.g_accel,
            params.mu,
            params.h_points,
            params.T_points,
            result.altitudes[index],
        )
        self.assertLess(abs(result.pressures[index] / exact - 1.0), 0.01)

    def test_surface_pressure_changes_scale_but_not_normalized_shape(self):
        low = AtmosphereModel(make_params(p0=1.013e4)).run()
        high = AtmosphereModel(make_params(p0=1.013e6)).run()
        self.assertEqual(low.altitudes, high.altitudes)
        self.assertEqual(len(low.pressures), len(high.pressures))
        for index in range(0, len(low.pressures), 173):
            with self.subTest(index=index):
                self.assertAlmostEqual(
                    low.pressures[index] / low.pressures[0],
                    high.pressures[index] / high.pressures[0],
                    delta=2e-14,
                )

    def test_larger_gravity_or_molecular_weight_reduces_base_scale_height(self):
        base = AtmosphereModel(make_params()).run().altitudes[1]
        high_g = AtmosphereModel(make_params(g_accel=19.62)).run().altitudes[1]
        high_mu = AtmosphereModel(make_params(mu=57.94)).run().altitudes[1]
        self.assertAlmostEqual(high_g, base / 2.0)
        self.assertAlmostEqual(high_mu, base / 2.0)

    def test_exact_zero_pressure_boundary_is_excluded(self):
        with patch.object(driver, "STEPS_PER_SCALE_HEIGHT", 1):
            result = AtmosphereModel(
                make_params(h_points=[0.0, 1000.0], T_points=[288.15, 288.15])
            ).run()
        self.assertEqual(result.altitudes, [0.0])
        self.assertEqual(result.pressures, [1.013e5])

    def test_restart_guard_raises_instead_of_looping_forever(self):
        with (
            patch.object(driver, "MAX_STEPS", 2),
            patch.object(driver, "MAX_RETRIES", 1),
            self.assertRaisesRegex(RuntimeError, "repeated step-size increases"),
        ):
            AtmosphereModel(
                make_params(h_points=[0.0, 1000.0], T_points=[288.15, 288.15])
            ).run()

    def test_one_successful_restart_matches_direct_coarse_integration(self):
        params = make_params(
            h_points=[0.0, 1000.0],
            T_points=[288.15, 288.15],
        )
        restarted_model = AtmosphereModel(params)
        with (
            patch.object(driver, "STEPS_PER_SCALE_HEIGHT", 200),
            patch.object(driver, "MAX_STEPS", 300),
            patch.object(driver, "MAX_RETRIES", 2),
            patch.object(driver, "hydrostatic_step", wraps=driver.hydrostatic_step) as step_mock,
        ):
            restarted = restarted_model.run()

        self.assertEqual(step_mock.call_count, 299 + len(restarted.altitudes))
        self.assertAlmostEqual(
            restarted.altitudes[1],
            2.0 * (
                phys.K_BOLTZMANN * 288.15
                / (params.g_accel * params.mu * phys.M_PROTON)
                / 200.0
            ),
        )

        direct_model = AtmosphereModel(params)
        with (
            patch.object(driver, "STEPS_PER_SCALE_HEIGHT", 100),
            patch.object(driver, "MAX_STEPS", 300),
            patch.object(driver, "MAX_RETRIES", 2),
        ):
            direct = direct_model.run()

        self.assertEqual(restarted.altitudes, direct.altitudes)
        self.assertEqual(restarted.pressures, direct.pressures)
        self.assertEqual(restarted.temperatures, direct.temperatures)
        self.assertEqual(restarted.densities, direct.densities)
        self.assertEqual(restarted_model.temp_profile.beta, direct_model.temp_profile.beta)

    def test_fixed_seed_randomized_profiles_preserve_invariants(self):
        rng = random.Random(20260827)
        for case_number in range(120):
            g_accel = rng.uniform(1.0, 30.0)
            mu = rng.uniform(2.0, 50.0)
            p0 = 10.0 ** rng.uniform(2.0, 7.0)
            t0 = rng.uniform(180.0, 600.0)
            scale = phys.K_BOLTZMANN * t0 / (g_accel * mu * phys.M_PROTON)
            h_points = [0.0, 2.0 * scale, 5.0 * scale, 8.0 * scale]
            T_points = [
                t0,
                t0 * rng.uniform(0.75, 1.25),
                t0 * rng.uniform(0.75, 1.50),
                t0 * rng.uniform(0.75, 1.50),
            ]
            case = {
                "case": case_number,
                "g_accel": g_accel,
                "mu": mu,
                "p0": p0,
                "h_points": h_points,
                "T_points": T_points,
            }
            with self.subTest(case=case):
                result = AtmosphereModel(
                    make_params(
                        g_accel=g_accel,
                        mu=mu,
                        p0=p0,
                        h_points=h_points,
                        T_points=T_points,
                    )
                ).run()
                self.assertTrue(all(math.isfinite(x) for x in result.altitudes))
                self.assertTrue(all(math.isfinite(x) and x > 0.0 for x in result.pressures))
                self.assertTrue(all(math.isfinite(x) and x > 0.0 for x in result.densities))
                self.assertTrue(all(math.isfinite(x) and x > 0.0 for x in result.temperatures))
                self.assertTrue(all(b > a for a, b in zip(result.altitudes, result.altitudes[1:])))
                self.assertTrue(all(b < a for a, b in zip(result.pressures, result.pressures[1:])))
                stride = max(1, len(result.altitudes) // 12)
                for index in range(0, len(result.altitudes), stride):
                    expected = ideal_gas_density(
                        result.pressures[index], mu, result.temperatures[index]
                    )
                    self.assertAlmostEqual(result.densities[index], expected)

    def test_extreme_values_fail_cleanly(self):
        for overrides in (
            {"g_accel": 1e308},
            {"mu": 1e308},
            {"g_accel": 5e-324},
            {"mu": 5e-324},
            {"p0": 5e-324},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                AtmosphereModel(make_params(**overrides)).run()


class OutputExtractionTests(unittest.TestCase):
    def setUp(self):
        self.result = AtmosphereResult(
            altitudes=[0.0, 1.0],
            pressures=[10.0, 9.0],
            densities=[2.0, 1.8],
            temperatures=[300.0, 299.0],
            output_type="Pressure",
            planet_name="Test",
        )

    def test_each_output_choice_selects_correct_data_and_units(self):
        cases = (
            ("Pressure", self.result.pressures, "Pa"),
            ("Density", self.result.densities, "kg/m^3"),
            ("Temperature", self.result.temperatures, "K"),
        )
        for output_type, expected_y, expected_unit in cases:
            self.result.output_type = output_type
            curve = extract_output(self.result)
            with self.subTest(output_type=output_type):
                self.assertIs(curve.x, self.result.altitudes)
                self.assertIs(curve.y, expected_y)
                self.assertEqual(curve.y_unit, expected_unit)
                self.assertEqual(curve.x_label, "altitude (m)")
                self.assertEqual(curve.y_label, f"{output_type} ({expected_unit})")
                self.assertEqual(curve.title, f"Test atmosphere: {output_type}")

    def test_invalid_result_output_type_is_rejected(self):
        self.result.output_type = "Invalid"
        with self.assertRaisesRegex(ValueError, "output_type"):
            extract_output(self.result)


class CheckpointExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = AtmosphereModel(make_params()).run()

    def test_default_profile_returns_every_checkpoint(self):
        rows = extract_checkpoints(self.result, DEFAULT_H, DEFAULT_T)
        self.assertEqual(len(rows), len(DEFAULT_H))
        self.assertEqual([row.altitude for row in rows], DEFAULT_H)
        self.assertEqual([row.temperature for row in rows], DEFAULT_T)
        self.assertTrue(all(row.pressure is not None for row in rows))
        self.assertTrue(all(row.density is not None for row in rows))
        self.assertTrue(all(row.pressure_over_temperature is not None for row in rows))
        for row in rows:
            self.assertAlmostEqual(
                row.density,
                ideal_gas_density(row.pressure, self.result.mu, row.temperature),
            )

    def test_surface_checkpoint_uses_exact_initial_pressure(self):
        row = extract_checkpoints(self.result, DEFAULT_H, DEFAULT_T)[0]
        self.assertEqual(row.pressure, 1.013e5)
        self.assertEqual(row.density, self.result.densities[0])
        self.assertAlmostEqual(
            row.pressure_over_temperature,
            row.pressure / row.temperature,
        )

    def test_pressure_is_linearly_interpolated(self):
        result = AtmosphereResult(
            altitudes=[0.0, 10.0],
            pressures=[100.0, 80.0],
            densities=[1.0, 0.8],
            temperatures=[200.0, 300.0],
            output_type="Pressure",
            planet_name="Test",
        )
        row = extract_checkpoints(result, [5.0, 8.0], [250.0, 280.0])[0]
        self.assertEqual(row.pressure, 90.0)
        self.assertEqual(row.density, 0.9)
        self.assertEqual(row.pressure_over_temperature, 90.0 / 250.0)
        self.assertEqual(row.temperature, 250.0)

    def test_out_of_domain_checkpoint_is_retained_as_unavailable(self):
        result = AtmosphereResult(
            altitudes=[0.0, 10.0],
            pressures=[100.0, 80.0],
            densities=[1.0, 0.8],
            temperatures=[200.0, 190.0],
            output_type="Pressure",
            planet_name="Test",
        )
        rows = extract_checkpoints(result, [-5.0, 20.0], [220.0, 180.0])
        for row in rows:
            self.assertIsNone(row.pressure)
            self.assertIsNone(row.density)
            self.assertIsNone(row.pressure_over_temperature)

    def test_malformed_result_arrays_are_rejected(self):
        result = AtmosphereResult(
            altitudes=[0.0, 10.0],
            pressures=[100.0],
            densities=[1.0, 0.8],
            temperatures=[200.0, 190.0],
            output_type="Pressure",
            planet_name="Test",
        )
        with self.assertRaisesRegex(ValueError, "co-indexed"):
            extract_checkpoints(result, [0.0, 10.0], [200.0, 190.0])


class CommandLineHelpAndPlotTests(unittest.TestCase):
    def test_command_line_defaults_cover_all_model_inputs(self):
        args = entry_point.parse_args([])
        self.assertEqual(args.planet_name, "Earth")
        self.assertEqual(args.g_accel, 9.81)
        self.assertEqual(args.mu, 28.97)
        self.assertEqual(args.p0, 1.013e5)
        self.assertEqual(args.h_points, list(entry_point.DEFAULT_H_POINTS))
        self.assertEqual(args.T_points, list(entry_point.DEFAULT_T_POINTS))
        self.assertEqual(args.output_type, "pressure")

    def test_command_line_accepts_every_custom_model_input(self):
        args = entry_point.parse_args(
            [
                "--planet_name", "Mars",
                "--g_accel", "3.71",
                "--mu", "44",
                "--p0", "610",
                "--h_points", "0,10000,20000",
                "--T_points", "210,180,160",
                "--output_type", "density",
            ]
        )
        self.assertEqual(args.planet_name, "Mars")
        self.assertEqual(args.g_accel, 3.71)
        self.assertEqual(args.mu, 44.0)
        self.assertEqual(args.p0, 610.0)
        self.assertEqual(args.h_points, [0.0, 10_000.0, 20_000.0])
        self.assertEqual(args.T_points, [210.0, 180.0, 160.0])
        self.assertEqual(args.output_type, "density")

    def test_command_line_rejects_capitalized_selector(self):
        with self.assertRaises(SystemExit):
            entry_point.parse_args(["--output_type", "Pressure"])

    def test_version_command_matches_runtime_metadata(self):
        completed = subprocess.run(
            [sys.executable, str(MODULE_DIR / "main.py"), "--version"],
            cwd=MODULE_DIR,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        self.assertEqual(
            completed.stdout.strip(),
            f"Atmosphere {phys.MODEL_VERSION} (build {phys.BUILD_ID})",
        )

    def test_default_main_run_succeeds_headlessly(self):
        environment = os.environ.copy()
        environment["MPLBACKEND"] = "Agg"
        completed = subprocess.run(
            [sys.executable, str(MODULE_DIR / "main.py")],
            cwd=MODULE_DIR,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        self.assertIn(
            f"Atmosphere {phys.MODEL_VERSION} (build {phys.BUILD_ID})",
            completed.stdout,
        )
        self.assertIn("Atmospheric checkpoints", completed.stdout)
        self.assertIn("pressure (Pa)", completed.stdout)
        self.assertIn("density (kg/m^3)", completed.stdout)
        self.assertIn("p/T (Pa/K)", completed.stdout)
        self.assertIn("temperature (K)", completed.stdout)
        for altitude in entry_point.DEFAULT_H_POINTS:
            self.assertIn(entry_point._format_altitude(altitude), completed.stdout)

    def test_help_version_build_matches_runtime_and_html_parses(self):
        from html.parser import HTMLParser

        html = HELP_FILE.read_text(encoding="utf-8")
        parser = HTMLParser()
        parser.feed(html)
        version_block = re.search(
            r'<p\s+id="version_build"[^>]*>(.*?)</p>', html, re.DOTALL
        )
        self.assertIsNotNone(version_block)
        visible = re.sub(r"<[^>]+>|&nbsp;", " ", version_block.group(1))
        visible = " ".join(visible.split())
        self.assertEqual(
            visible,
            f"Version {phys.MODEL_VERSION} Build {phys.BUILD_ID}",
        )

    def test_help_documents_every_command_line_parameter(self):
        html = HELP_FILE.read_text(encoding="utf-8")
        for option in (
            "--planet_name", "--g_accel", "--mu", "--p0", "--h_points",
            "--T_points", "--output_type", "--version",
        ):
            with self.subTest(option=option):
                self.assertIn(option, html)

    def test_help_documents_checkpoint_report(self):
        html = HELP_FILE.read_text(encoding="utf-8")
        for text in ("pressure", "p/T", "temperature", "checkpoint"):
            with self.subTest(text=text):
                self.assertIn(text, html)

    def test_plotter_uses_curve_labels_and_calls_show(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from plot_atmosphere import plot_atmosphere

        curve = extract_output(
            AtmosphereResult(
                altitudes=[0.0, 1000.0],
                pressures=[100_000.0, 90_000.0],
                densities=[1.2, 1.1],
                temperatures=[288.0, 282.0],
                output_type="Pressure",
                planet_name="Plot Test",
            )
        )
        try:
            with patch.object(plt, "show") as show_mock:
                plot_atmosphere(curve)
            figure = plt.gcf()
            axes = figure.axes[0]
            self.assertEqual(axes.get_xlabel(), curve.x_label)
            self.assertEqual(axes.get_ylabel(), curve.y_label)
            self.assertEqual(axes.get_title(), curve.title)
            self.assertEqual(list(axes.lines[0].get_xdata()), curve.x)
            self.assertEqual(list(axes.lines[0].get_ydata()), curve.y)
            show_mock.assert_called_once_with()
        finally:
            plt.close("all")




class ModelInputSnapshotTests(unittest.TestCase):
    """The model keeps its own copy of its inputs (maintenance item, 1.3.0)."""

    def test_changing_the_callers_lists_after_construction_does_not_change_run(self):
        params = make_params()
        expected = AtmosphereModel(make_params()).run()
        model = AtmosphereModel(params)
        params.h_points[3] = 33_000.0
        params.T_points[3] = 250.0
        params.h_points.append(600_000.0)
        params.T_points.append(1_240.0)
        result = model.run()
        # Compared with ``==`` rather than assertEqual: a failing assertEqual
        # on 16,000-element lists spends minutes building its diff.
        self.assertTrue(result.altitudes == expected.altitudes, "altitudes changed")
        self.assertTrue(result.pressures == expected.pressures, "pressures changed")
        self.assertTrue(result.temperatures == expected.temperatures, "temperatures changed")

    def test_changing_the_callers_parameter_object_after_construction_has_no_effect(self):
        params = make_params()
        expected = AtmosphereModel(make_params()).run()
        model = AtmosphereModel(params)
        params.g_accel = 1.0
        params.mu = 2.0
        params.p0 = 5.0
        params.planet_name = "Changed"
        params.output_type = "Density"
        result = model.run()
        self.assertTrue(result.pressures == expected.pressures, "pressures changed")
        self.assertEqual(result.planet_name, "Earth")
        self.assertEqual(result.output_type, "Pressure")

    def test_models_built_from_one_parameter_object_are_independent(self):
        params = make_params(h_points=[0.0, 20_000.0], T_points=[288.15, 216.65])
        first = AtmosphereModel(params)
        params.T_points[1] = 400.0
        second = AtmosphereModel(params)
        first_result = first.run()
        second_result = second.run()
        self.assertEqual(first_result.temperatures[0], 288.15)
        self.assertNotEqual(first_result.pressures[-1], second_result.pressures[-1])
        self.assertEqual(first.params.T_points, [288.15, 216.65])
        self.assertEqual(second.params.T_points, [288.15, 400.0])

    def test_model_holds_list_copies_even_for_tuple_input(self):
        params = make_params(h_points=(0.0, 1000.0), T_points=(288.15, 280.0))
        model = AtmosphereModel(params)
        self.assertIsInstance(model.params.h_points, list)
        self.assertIsInstance(model.params.T_points, list)
        self.assertIsNot(model.params.h_points, params.h_points)
        self.assertEqual(model.params.h_points, [0.0, 1000.0])

    def test_run_leaves_the_callers_parameters_unchanged(self):
        params = make_params()
        before = (params.h_points.copy(), params.T_points.copy(), params.g_accel, params.mu, params.p0)
        AtmosphereModel(params).run()
        self.assertEqual(
            before,
            (params.h_points, params.T_points, params.g_accel, params.mu, params.p0),
        )

    def test_construction_still_rejects_bad_containers_before_copying(self):
        for bad in (None, 42, "0,1000"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "non-string sequence"):
                AtmosphereModel(make_params(h_points=bad))

    def test_an_invalid_edit_to_the_models_own_copy_is_rejected_at_run(self):
        cases = (
            ("T_points", [300.0, -1.0], "greater than zero"),
            ("h_points", [1000.0, 0.0], "strictly"),
            ("g_accel", -9.81, "g_accel"),
            ("mu", 0.0, "mu"),
            ("p0", math.nan, "p0"),
            ("output_type", "Speed", "output_type"),
        )
        for name, bad, message in cases:
            model = AtmosphereModel(make_params(h_points=[0.0, 1000.0], T_points=[300.0, 290.0]))
            setattr(model.params, name, bad)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                model.run()

    def test_a_valid_edit_to_the_models_own_copy_is_used_by_the_next_run(self):
        model = AtmosphereModel(make_params(h_points=[0.0, 10_000.0], T_points=[288.15, 250.0]))
        model.run()
        model.params.h_points = [0.0, 20_000.0]
        model.params.T_points = [288.15, 216.65]
        edited = model.run()
        fresh = AtmosphereModel(
            make_params(h_points=[0.0, 20_000.0], T_points=[288.15, 216.65])
        ).run()
        self.assertEqual(edited.pressures, fresh.pressures)
        self.assertEqual(edited.temperatures, fresh.temperatures)


class LeadingMinusListArgumentTests(unittest.TestCase):
    """A list that starts with a negative altitude can be typed the natural way."""

    def test_space_separated_list_with_leading_negative_altitude_parses(self):
        args = entry_point.parse_args(
            ["--h_points", "-1000,0,10000", "--T_points", "280,288,250"]
        )
        self.assertEqual(args.h_points, [-1000.0, 0.0, 10_000.0])
        self.assertEqual(args.T_points, [280.0, 288.0, 250.0])

    def test_equals_form_and_decimal_forms_still_parse(self):
        args = entry_point.parse_args(["--h_points=-1000,0,10000", "--T_points=280,288,250"])
        self.assertEqual(args.h_points, [-1000.0, 0.0, 10_000.0])
        args = entry_point.parse_args(["--h_points", "-.5,0,10", "--T_points", "280,288,250"])
        self.assertEqual(args.h_points, [-0.5, 0.0, 10.0])

    def test_ordinary_arguments_are_not_rewritten(self):
        argv = ["--planet_name", "X", "--h_points", "0,1000", "--T_points", "288,280"]
        self.assertEqual(entry_point._join_leading_minus_list_values(argv), argv)
        self.assertEqual(entry_point._join_leading_minus_list_values([]), [])
        self.assertEqual(
            entry_point._join_leading_minus_list_values(["--h_points"]), ["--h_points"]
        )

    def test_only_the_two_list_options_are_rewritten(self):
        self.assertEqual(
            entry_point._join_leading_minus_list_values(["--g_accel", "-3"]),
            ["--g_accel", "-3"],
        )
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(io.StringIO()):
            entry_point.parse_args(["--mu", "-5"])
        self.assertEqual(raised.exception.code, 2)

    def test_arguments_are_taken_from_sys_argv_when_none_is_given(self):
        with patch.object(sys, "argv", ["main.py", "--h_points", "-5,5", "--T_points", "280,290"]):
            args = entry_point.parse_args()
        self.assertEqual(args.h_points, [-5.0, 5.0])

    def test_end_to_end_row_below_altitude_zero_is_reported_unavailable(self):
        printed = run_main(
            ["--h_points", "-1000,0,10000", "--T_points", "280,288,250"]
        )
        rows = table_rows(printed)
        self.assertEqual([row[0] for row in rows], ["-1000", "0", "10000"])
        self.assertEqual(rows[0][1:4], ["unavailable"] * 3)
        self.assertEqual(rows[0][4], "280")
        self.assertNotEqual(rows[1][1], "unavailable")
        self.assertNotEqual(rows[2][1], "unavailable")

    def test_a_single_leading_negative_value_gets_the_model_error_not_argparses(self):
        with self.assertRaises(SystemExit) as raised:
            run_main(["--h_points", "-5", "--T_points", "280"])
        self.assertIn("At least two", str(raised.exception.code))
        with self.assertRaises(SystemExit) as raised:
            run_main(["--h_points", "0,10", "--T_points", "-5,280"])
        self.assertNotEqual(raised.exception.code, 2)


class EdgeProfileFamilyTests(unittest.TestCase):
    """Fixed-seed families of unusual but valid temperature profiles.

    Every generated case must satisfy the physical and bookkeeping invariants,
    and, wherever the checkpoint lies inside the computed domain, agree with an
    independent exact integral of Eq. (7.3) to within a first-order Euler
    global-error estimate.
    """

    CASES_PER_FAMILY = 12

    @staticmethod
    def scales(rng):
        g_accel = rng.uniform(1.0, 30.0)
        mu = rng.uniform(2.0, 50.0)
        p0 = 10.0 ** rng.uniform(2.0, 7.0)
        t0 = rng.uniform(150.0, 600.0)
        q = phys.M_PROTON * mu / phys.K_BOLTZMANN
        return g_accel, mu, p0, t0, t0 / (g_accel * q)

    @staticmethod
    def exact_pressure(p0, g_accel, mu, h_points, T_points, target):
        """Exact pressure at ``target`` >= 0 for the piecewise-linear profile."""
        q = phys.M_PROTON * mu / phys.K_BOLTZMANN
        edges = [0.0] + [h for h in h_points if 0.0 < h < target] + [target]
        integral = 0.0
        for lower, upper in zip(edges, edges[1:]):
            t_lower = interpolated_temperature(h_points, T_points, lower)
            t_upper = interpolated_temperature(h_points, T_points, upper)
            if abs(t_upper - t_lower) <= 1e-12 * t_lower:
                integral += (upper - lower) / t_lower
            else:
                integral += (upper - lower) * math.log(t_upper / t_lower) / (t_upper - t_lower)
        return p0 * math.exp(-g_accel * q * integral)

    @staticmethod
    def euler_error_estimate(g_accel, mu, h_points, T_points, target, step):
        """Generous first-order estimate of the relative Euler error at ``target``."""
        q = phys.M_PROTON * mu / phys.K_BOLTZMANN
        count = 4000
        total = 0.0
        previous = None
        for index in range(count + 1):
            altitude = min(target * index / count, target)
            u = g_accel * q / interpolated_temperature(h_points, T_points, altitude)
            if previous is not None:
                total += 0.5 * (u * u + previous * previous) * (target / count)
            previous = u
        u_start = g_accel * q / interpolated_temperature(h_points, T_points, 0.0)
        return 2.0 * 0.5 * step * (total + abs(previous - u_start)) + 1e-4

    def family_profiles(self):
        """Yield (family, parameters, h_points, T_points)."""
        for family_index, family in enumerate(
            (
                "spans_zero", "starts_above_zero", "entirely_below_zero", "many_points",
                "steep_jumps", "tiny_spacing", "cooling_to_cold", "two_points",
            )
        ):
            rng = random.Random(20260920 + family_index)
            for case in range(self.CASES_PER_FAMILY):
                g_accel, mu, p0, t0, h_scale = self.scales(rng)
                if family == "spans_zero":
                    h = [-rng.uniform(0.1, 3.0) * h_scale, 0.0]
                    h += sorted(rng.uniform(0.2, 10.0) * h_scale for _ in range(rng.randint(2, 6)))
                    t = [t0 * rng.uniform(0.6, 1.6) for _ in h]
                elif family == "starts_above_zero":
                    h = [rng.uniform(0.1, 2.0) * h_scale]
                    h += sorted(rng.uniform(2.5, 10.0) * h_scale for _ in range(rng.randint(1, 5)))
                    t = [t0 * rng.uniform(0.6, 1.6) for _ in h]
                elif family == "entirely_below_zero":
                    h = sorted(-rng.uniform(0.1, 5.0) * h_scale for _ in range(rng.randint(2, 4)))
                    t = [t0 * rng.uniform(0.6, 1.6) for _ in h]
                elif family == "many_points":
                    h, altitude = [0.0], 0.0
                    for _ in range(59):
                        altitude += rng.uniform(0.02, 0.3) * h_scale
                        h.append(altitude)
                    t, value = [t0], t0
                    for _ in range(59):
                        value = min(max(value * rng.uniform(0.93, 1.07), 0.5 * t0), 1.8 * t0)
                        t.append(value)
                elif family == "steep_jumps":
                    h = [0.0, 1.5 * h_scale, 3.0 * h_scale, 4.5 * h_scale, 6.0 * h_scale]
                    t = [t0, t0 / 20.0, t0, t0 / 20.0, t0]
                elif family == "tiny_spacing":
                    h = [0.0, 1e-3, 2e-3, rng.uniform(4.0, 10.0) * h_scale]
                    t = [t0, t0 * 1.001, t0 * 0.999, t0 * rng.uniform(0.6, 1.6)]
                elif family == "cooling_to_cold":
                    h = [0.0, 2.0 * h_scale, 5.0 * h_scale]
                    t = [t0, t0 / rng.uniform(5.0, 15.0), t0 / rng.uniform(20.0, 30.0)]
                else:
                    h = [0.0, rng.uniform(1.0, 8.0) * h_scale]
                    t = [t0, t0 * rng.choice((1.0, 0.5, 1.5))]
                yield family, (g_accel, mu, p0), h, t

    def test_generated_profiles_preserve_invariants_and_match_exact_integrals(self):
        cases_run = 0
        checkpoints_compared = 0
        for family, (g_accel, mu, p0), h_points, T_points in self.family_profiles():
            case = {"family": family, "g": g_accel, "mu": mu, "p0": p0, "h": h_points, "T": T_points}
            with self.subTest(case=case):
                result = AtmosphereModel(
                    make_params(g_accel=g_accel, mu=mu, p0=p0, h_points=h_points, T_points=T_points)
                ).run()
                cases_run += 1

                self.assertEqual(result.altitudes[0], 0.0)
                self.assertEqual(result.pressures[0], p0)
                for values in (result.altitudes, result.pressures, result.densities, result.temperatures):
                    self.assertEqual(len(values), len(result.altitudes))
                    self.assertTrue(all(math.isfinite(v) for v in values))
                self.assertTrue(all(v > 0.0 for v in result.pressures))
                self.assertTrue(all(v > 0.0 for v in result.densities))
                self.assertTrue(all(v > 0.0 for v in result.temperatures))
                self.assertTrue(all(b > a for a, b in zip(result.altitudes, result.altitudes[1:])))
                self.assertTrue(all(b < a for a, b in zip(result.pressures, result.pressures[1:])))
                stride = max(1, len(result.altitudes) // 40)
                for index in range(0, len(result.altitudes), stride):
                    self.assertAlmostEqual(
                        result.densities[index],
                        ideal_gas_density(result.pressures[index], mu, result.temperatures[index]),
                        delta=1e-12 * result.densities[index],
                    )

                # The reference temperature and every stored temperature inside
                # the supplied altitude range follow the supplied profile.
                if h_points[-1] >= 0.0:
                    self.assertAlmostEqual(
                        result.temperatures[0],
                        interpolated_temperature(h_points, T_points, 0.0),
                        delta=1e-9 * T_points[0],
                    )
                else:
                    # Every supplied altitude lies below the reference level, so
                    # the reference temperature comes from the upper-atmosphere
                    # law, whose coefficient is fixed to reproduce T_points[-1]
                    # (equal to within one rounding step).
                    self.assertAlmostEqual(
                        result.temperatures[0], T_points[-1], delta=1e-12 * T_points[-1]
                    )
                for index in range(0, len(result.altitudes), stride):
                    altitude = result.altitudes[index]
                    if altitude <= h_points[-1]:
                        self.assertAlmostEqual(
                            result.temperatures[index],
                            interpolated_temperature(h_points, T_points, altitude),
                            delta=1e-9 * max(T_points),
                        )

                rows = extract_checkpoints(result, h_points, T_points)
                self.assertEqual(len(rows), len(h_points))
                for row, altitude, temperature in zip(rows, h_points, T_points):
                    self.assertEqual(row.altitude, altitude)
                    self.assertEqual(row.temperature, temperature)
                    if altitude < 0.0:
                        self.assertIsNone(row.pressure)
                        self.assertIsNone(row.density)
                        self.assertIsNone(row.pressure_over_temperature)
                        continue
                    self.assertIsNotNone(row.pressure)
                    self.assertAlmostEqual(
                        row.density,
                        ideal_gas_density(row.pressure, mu, temperature),
                        delta=1e-12 * row.density,
                    )
                    self.assertAlmostEqual(
                        row.pressure_over_temperature, row.pressure / temperature,
                        delta=1e-12 * row.pressure_over_temperature,
                    )
                    step = result.altitudes[1] - result.altitudes[0]
                    exact = self.exact_pressure(p0, g_accel, mu, h_points, T_points, altitude)
                    bound = self.euler_error_estimate(g_accel, mu, h_points, T_points, altitude, step)
                    self.assertLessEqual(abs(row.pressure / exact - 1.0), bound)
                    checkpoints_compared += 1
        self.assertEqual(cases_run, 8 * self.CASES_PER_FAMILY)
        self.assertGreater(checkpoints_compared, 200)

    def test_entirely_negative_profile_reports_every_checkpoint_unavailable(self):
        h_points, T_points = [-3000.0, -1000.0], [250.0, 300.0]
        result = AtmosphereModel(make_params(h_points=h_points, T_points=T_points)).run()
        self.assertEqual(result.temperatures[0], 300.0)
        rows = extract_checkpoints(result, h_points, T_points)
        self.assertTrue(all(row.pressure is None for row in rows))
        self.assertEqual([row.temperature for row in rows], T_points)


HELP_HTML = HELP_FILE.read_text(encoding="utf-8")


def help_layout(html):
    """Classify a Help page as ``"beats"``, ``"classic"`` or ``"unrecognised"``."""
    ids = set(re.findall(r'<section id="([^"]+)"', html))
    beat_ids = {name for name in ids if re.fullmatch(r"beat\d+", name)}
    if "beats" in ids and beat_ids:
        return "beats"
    if "physics" in ids and "algorithm" in ids and not beat_ids and "beats" not in ids:
        return "classic"
    return "unrecognised"


HELP_LAYOUT = help_layout(HELP_HTML)
needs_beats = unittest.skipUnless(
    HELP_LAYOUT == "beats", "the Beats-only assertions apply to the Beats layout"
)


def html_text(fragment):
    """Visible text of an HTML fragment with entities decoded and spaces collapsed."""
    text = re.sub(r"<[^>]+>", " ", fragment)
    return " ".join(html_module.unescape(text).split())


def section_html(html, section_id):
    match = re.search(rf'<section id="{re.escape(section_id)}">(.*?)</section>', html, re.DOTALL)
    if match is None:
        raise AssertionError(f"section {section_id!r} not found in the Help file")
    return match.group(1)


def beat_text(number):
    return html_text(section_html(HELP_HTML, f"beat{number}"))


class HelpStructure:
    """Ids, links, sidebar links and tag balance of a Help page."""

    VOID = {"br", "hr", "img", "meta", "link", "input"}

    def __init__(self, html):
        from html.parser import HTMLParser

        outer = self
        self.ids, self.hrefs, self.sidebar_hrefs, self.problems = [], [], [], []
        self._stack = []
        self._in_sidebar = 0

        class Parser(HTMLParser):
            def handle_starttag(self, tag, attrs):
                attributes = dict(attrs)
                if "id" in attributes:
                    outer.ids.append(attributes["id"])
                if tag == "a" and attributes.get("href", "").startswith("#"):
                    outer.hrefs.append(attributes["href"][1:])
                    if outer._in_sidebar:
                        outer.sidebar_hrefs.append(attributes["href"][1:])
                if tag not in outer.VOID:
                    outer._stack.append(tag)
                    if tag == "nav" and attributes.get("id") == "sidebar":
                        outer._in_sidebar = len(outer._stack)

            def handle_startendtag(self, tag, attrs):
                if "id" in dict(attrs):
                    outer.ids.append(dict(attrs)["id"])

            def handle_endtag(self, tag):
                if tag in outer.VOID:
                    return
                if not outer._stack or outer._stack[-1] != tag:
                    outer.problems.append(f"unexpected </{tag}> (open: {outer._stack[-3:]})")
                    if tag in outer._stack:
                        while outer._stack and outer._stack.pop() != tag:
                            pass
                    return
                if outer._in_sidebar == len(outer._stack):
                    outer._in_sidebar = 0
                outer._stack.pop()

        parser = Parser(convert_charrefs=True)
        parser.feed(html)
        parser.close()
        if self._stack:
            self.problems.append(f"unclosed tags at end of file: {self._stack}")


def documented_commands(html):
    """Every ``python main.py ...`` command shown in a code block, as argv lists."""
    blocks = re.findall(r"<pre[^>]*>(.*?)</pre>", html, re.DOTALL)
    blocks += re.findall(r'<div class="params">(.*?)</div>', html, re.DOTALL)
    commands = []
    for block in blocks:
        text = html_module.unescape(re.sub(r"<[^>]+>", "", block))
        text = re.sub(r"\\\n\s*", " ", text)
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("python main.py"):
                argv = shlex.split(line)
                assert argv[:2] == ["python", "main.py"], line
                commands.append(argv[2:])
    return commands


class HelpLayoutRecogniserTests(unittest.TestCase):
    def test_recogniser_classifies_synthetic_pages(self):
        beats = '<section id="description"></section><section id="beats"></section><section id="beat0"></section>'
        classic = '<section id="description"></section><section id="physics"></section><section id="algorithm"></section>'
        self.assertEqual(help_layout(beats), "beats")
        self.assertEqual(help_layout(classic), "classic")
        self.assertEqual(help_layout(classic + '<section id="beat0"></section>'), "unrecognised")
        self.assertEqual(help_layout('<section id="description"></section>'), "unrecognised")
        self.assertEqual(help_layout(""), "unrecognised")

    def test_shipped_help_has_a_recognised_layout(self):
        self.assertIn(HELP_LAYOUT, ("beats", "classic"))


class HelpStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structure = HelpStructure(HELP_HTML)

    def test_tags_are_balanced(self):
        self.assertEqual(self.structure.problems, [])

    def test_ids_are_unique(self):
        duplicates = sorted({name for name in self.structure.ids if self.structure.ids.count(name) > 1})
        self.assertEqual(duplicates, [])

    def test_every_in_page_link_resolves(self):
        unresolved = sorted({name for name in self.structure.hrefs if name not in self.structure.ids})
        self.assertEqual(unresolved, [])

    def test_sidebar_lists_every_named_section_in_document_order(self):
        section_ids = re.findall(r'<section id="([^"]+)"', HELP_HTML)
        self.assertEqual(self.structure.sidebar_hrefs, section_ids)

    def test_page_loads_mathjax_from_the_cdn_and_no_other_external_script(self):
        sources = re.findall(r'<script[^>]*\ssrc="([^"]+)"', HELP_HTML)
        self.assertEqual(sources, ["https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"])

    def test_every_governing_equation_is_tagged_exactly_once(self):
        tags = re.findall(r"\\tag\{(7\.\d)\}", HELP_HTML)
        self.assertEqual(sorted(tags), [f"7.{n}" for n in range(1, 8)])

    def test_parenthesised_equation_references_name_defined_equations(self):
        cited = set(re.findall(r"\(7\.(\d+)\)", html_text(HELP_HTML)))
        self.assertTrue(cited)
        self.assertTrue(cited <= {str(n) for n in range(1, 8)}, cited)

    def test_experiments_are_numbered_consecutively_and_titled_in_order(self):
        titles = re.findall(r'<div class="scenario-card"[^>]*>\s*<h4>(\d+) ·', HELP_HTML)
        self.assertEqual(titles, [str(n) for n in range(1, len(titles) + 1)])
        self.assertGreaterEqual(len(titles), 9)
        if HELP_LAYOUT == "beats":
            # The Beats pointers link to the cards, so each card carries an id.
            ids = [name for name in self.structure.ids if re.fullmatch(r"exp\d+", name)]
            self.assertEqual(ids, [f"exp{n}" for n in range(1, len(titles) + 1)])

    def test_help_states_defaults_that_match_the_program(self):
        args = entry_point.parse_args([])
        table = section_html(HELP_HTML, "parameters")
        rows = {
            html_text(match.group(1)): match.group(2)
            for match in re.finditer(r"<tr>\s*<td>(.*?)</td>\s*<td>(.*?)</td>", table, re.DOTALL)
        }

        def cell(option_prefix):
            for key, value in rows.items():
                if key.startswith(option_prefix):
                    return html_text(value)
            self.fail(f"{option_prefix} not in the Parameters table")

        self.assertEqual(cell("--g_accel"), repr(args.g_accel))
        self.assertEqual(cell("--mu"), repr(args.mu))
        self.assertEqual(float(cell("--p0")), args.p0)
        self.assertEqual(cell("--output_type"), args.output_type.lower())
        for option, values in (("--h_points", args.h_points), ("--T_points", args.T_points)):
            listed = [float(item) for item in cell(option).replace(" ", "").split(",")]
            self.assertEqual(listed, list(values), option)

    def test_help_constants_and_step_rules_match_the_code(self):
        flat = re.sub(r"\s+", " ", html_module.unescape(re.sub(r"<[^>]+>", "", HELP_HTML)))
        for name, value in (("K_BOLTZMANN", phys.K_BOLTZMANN), ("M_PROTON", phys.M_PROTON)):
            with self.subTest(constant=name):
                self.assertIn(f"{name} = {value!r}", flat)
                mantissa, exponent = f"{value:.2e}".split("e")
                self.assertIn(rf"{mantissa}\times10^{{{int(exponent)}}}", HELP_HTML)
        self.assertEqual(driver.STEPS_PER_SCALE_HEIGHT, 200)
        self.assertIn("scale / 200.0", html_module.unescape(HELP_HTML))
        self.assertIn("H/200", flat)
        self.assertEqual(driver.MAX_STEPS, 50_000)
        self.assertIn("50 000", flat)


class HelpCommandTests(unittest.TestCase):
    """Every command printed in the Help must parse and run."""

    @classmethod
    def setUpClass(cls):
        cls.commands = documented_commands(HELP_HTML)

    def test_help_shows_a_substantial_number_of_commands(self):
        self.assertGreaterEqual(len(self.commands), 15)

    def test_every_documented_command_parses_with_matching_list_lengths(self):
        for argv in self.commands:
            with self.subTest(argv=argv):
                args = entry_point.parse_args(argv)
                self.assertEqual(len(args.h_points), len(args.T_points))

    def test_every_distinct_documented_command_runs_and_prints_its_table(self):
        seen = []
        for argv in self.commands:
            if argv in seen:
                continue
            seen.append(argv)
            with self.subTest(argv=argv):
                args = entry_point.parse_args(argv)
                printed = run_main(argv)
                self.assertIn("Atmospheric checkpoints", printed)
                self.assertEqual(len(table_rows(printed)), len(args.h_points))
        self.assertGreaterEqual(len(seen), 15)

    def test_negative_first_altitude_example_is_documented_and_runs(self):
        self.assertGreaterEqual(HELP_HTML.count("--h_points -1000,0,10000"), 3)
        printed = run_main(["--h_points", "-1000,0,10000", "--T_points", "295,288.15,223"])
        rows = table_rows(printed)
        self.assertEqual([row[0] for row in rows], ["-1000", "0", "10000"])
        self.assertEqual(rows[0][1:4], ["unavailable"] * 3)

    def test_checkpoint_excerpt_matches_the_real_default_output(self):
        block = re.search(r"<pre><code>(Atmospheric checkpoints.*?)</code></pre>", HELP_HTML, re.DOTALL)
        self.assertIsNotNone(block)
        shown = [line.split() for line in html_module.unescape(block.group(1)).splitlines()
                 if line.strip() and line.strip() != "..."]
        printed = [line for line in run_main([]).splitlines() if line.strip()]
        start = printed.index("Atmospheric checkpoints")
        actual = [line.split() for line in printed[start:]]
        self.assertGreaterEqual(len(shown), 4)
        self.assertEqual(shown, actual[: len(shown)])

    def test_table_is_the_same_for_every_output_type_and_has_five_significant_figures(self):
        tables = {mode: table_rows(run_main(["--output_type", mode]))
                  for mode in ("pressure", "density", "temperature")}
        self.assertEqual(tables["pressure"], tables["density"])
        self.assertEqual(tables["pressure"], tables["temperature"])
        self.assertEqual(len(tables["pressure"]), len(DEFAULT_H))
        for row in tables["pressure"]:
            for cell in row[1:]:
                digits = re.sub(r"e[+-]\d+$", "", cell.lstrip("-")).replace(".", "").lstrip("0")
                self.assertLessEqual(len(digits), 5, cell)


@needs_beats
class BeatStructureTests(unittest.TestCase):
    def test_beats_are_numbered_from_zero_in_document_order(self):
        ids = re.findall(r'<section id="(beat\d+)"', HELP_HTML)
        self.assertEqual(ids, [f"beat{n}" for n in range(len(ids))])
        self.assertEqual(len(ids), 8)
        self.assertLess(HELP_HTML.index('id="beats"'), HELP_HTML.index('id="beat0"'))
        self.assertLess(HELP_HTML.index('id="beat7"'), HELP_HTML.index('id="equations"'))

    def test_every_beat_has_the_same_parts_in_the_same_order(self):
        for number in range(8):
            with self.subTest(beat=number):
                body = section_html(HELP_HTML, f"beat{number}")
                self.assertRegex(body, rf"<h2>Beat {number} · ")
                self.assertEqual(body.count("<pre>"), 1)
                self.assertEqual(body.count("Three tasks, in order."), 1)
                self.assertEqual(body.count("<em>Then</em>"), 1)
                self.assertEqual(body.count("Experiments that go with this beat"), 1)
                self.assertRegex(body, r'<a href="#exp\d">')
                between = body[body.index("</pre>"):body.index("Three tasks, in order.")]
                self.assertRegex(between, r"[Ll]ook at")
                self.assertLess(body.index("<pre>"), body.index("Three tasks, in order."))
                self.assertLess(body.index("Three tasks, in order."), body.index("<em>Then</em>"))
                self.assertLess(body.index("<em>Then</em>"), body.index("Experiments that go with"))

    def test_sidebar_labels_carry_the_beat_numbers(self):
        for number in range(8):
            with self.subTest(beat=number):
                self.assertRegex(HELP_HTML, rf'<a href="#beat{number}">{number} · ')

    def test_every_experiment_is_pointed_to_by_at_least_one_beat(self):
        cited = set()
        for number in range(8):
            cited |= set(re.findall(r'href="#(exp\d+)"', section_html(HELP_HTML, f"beat{number}")))
        every = set(re.findall(r'id="(exp\d+)"', HELP_HTML))
        self.assertEqual(cited, every)

    def test_equation_index_lists_every_equation(self):
        index = section_html(HELP_HTML, "equations")
        for number in range(1, 8):
            with self.subTest(equation=number):
                self.assertRegex(index, rf"<td>\(7\.{number}\)</td>")

    def test_each_beat_command_block_holds_only_commands(self):
        for number in range(8):
            with self.subTest(beat=number):
                body = section_html(HELP_HTML, f"beat{number}")
                block = html_module.unescape(re.search(r"<pre>(.*?)</pre>", body, re.DOTALL).group(1))
                block = re.sub(r"\\\n\s*", " ", block)
                lines = [line for line in block.splitlines() if line.strip()]
                self.assertTrue(lines)
                for line in lines:
                    self.assertTrue(line.startswith("python main.py"), line)


@needs_beats
class BeatQuotedNumberTests(unittest.TestCase):
    """The numbers quoted in each Beat are recomputed from real runs."""

    @staticmethod
    def rows(argv):
        return {float(row[0]): row for row in table_rows(run_main(argv))}

    def assertQuoted(self, number, *strings):
        text = beat_text(number)
        for string in strings:
            with self.subTest(beat=number, quoted=string):
                self.assertIn(string, text)

    def test_beat_0_weight_of_the_air(self):
        rows = self.rows([])
        p0, p11 = float(rows[0.0][1]), float(rows[11019.0][1])
        self.assertQuoted(0, rows[0.0][1], rows[11019.0][1], "10,300", "22%", "78%")
        self.assertAlmostEqual(p0 / 9.81, 10_300, delta=50)
        self.assertEqual(round(100 * p11 / p0), 22)

    def test_beat_1_density_is_proportional_to_p_over_t(self):
        rows = self.rows(["--output_type", "density"])
        self.assertEqual(rows, self.rows([]))
        constant = 28.97 * phys.M_PROTON / phys.K_BOLTZMANN
        for altitude, row in rows.items():
            with self.subTest(altitude=altitude):
                self.assertAlmostEqual(float(row[2]) / float(row[3]), constant, delta=3e-4 * constant)
        self.assertQuoted(1, rows[0.0][2], rows[0.0][3], rows[100000.0][2], rows[100000.0][3],
                          f"{constant * 1e3:.3f}")
        low, high = rows[71802.0], rows[86000.0]
        self.assertQuoted(
            1,
            f"{float(low[1]) / float(high[1]):.1f}",
            f"{float(low[2]) / float(high[2]):.1f}",
            f"{float(low[4]) / float(high[4]):.3f}",
        )

    def test_beat_2_equal_steps_give_equal_factors_and_the_scale_height(self):
        argv = ["--h_points", "0,8000,16000,24000,32000", "--T_points", "288.15,288.15,288.15,288.15,288.15"]
        rows = self.rows(argv)
        pressures = [float(rows[h][1]) for h in (0.0, 8000.0, 16000.0, 24000.0, 32000.0)]
        for a, b in zip(pressures, pressures[1:]):
            self.assertEqual(f"{b / a:.4f}", "0.3840")
        self.assertQuoted(2, "0.3840", rows[8000.0][1], rows[16000.0][1], rows[24000.0][1])
        self.assertAlmostEqual(8000.0 / math.log(1.0 / 0.3840), 8360.0, delta=10.0)
        self.assertQuoted(2, f"{math.log(1 / 0.3840):.3f}")
        scale_from_rows = pressures[0] / (9.81 * float(rows[0.0][2]))
        self.assertQuoted(2, f"{scale_from_rows:.0f}")
        self.assertAlmostEqual(
            scale_from_rows, 288.15 * phys.K_BOLTZMANN / (9.81 * 28.97 * phys.M_PROTON), delta=1.0
        )

    def test_beat_3_one_dial_at_a_time(self):
        runs = {
            "default": [],
            "mu18": ["--mu", "18"],
            "mu44": ["--mu", "44"],
            "g19": ["--g_accel", "19.62"],
            "p0": ["--p0", "1.013e6"],
        }
        fractions, cells = {}, {}
        for name, argv in runs.items():
            rows = self.rows(argv)
            fractions[name] = float(rows[11019.0][1]) / float(rows[0.0][1])
            cells[name] = rows[11019.0][1]
        self.assertQuoted(3, cells["mu18"], cells["mu44"], cells["g19"], cells["p0"], "22263")
        for name, quoted in (("default", "0.2198"), ("mu18", "0.390"), ("mu44", "0.100"),
                             ("g19", "0.0483"), ("p0", "0.2198")):
            with self.subTest(run=name):
                decimals = len(quoted.split(".")[1])
                self.assertEqual(f"{fractions[name]:.{decimals}f}", quoted)
                self.assertIn(quoted, beat_text(3))
        log_default = math.log(fractions["default"])
        self.assertQuoted(3, f"{log_default:.3f}".replace("-", "\u2212"))
        self.assertAlmostEqual(math.exp(log_default * 44 / 28.97), fractions["mu44"],
                               delta=1e-3 * fractions["mu44"])
        self.assertAlmostEqual(fractions["default"] ** 2, fractions["g19"], delta=1e-3 * fractions["g19"])
        self.assertQuoted(
            3,
            f"44/28.97) = {math.exp(log_default * 44 / 28.97):.3f}",
            f"{fractions['default']:.4f}^2 = {fractions['default'] ** 2:.4f}",
        )
        self.assertAlmostEqual(fractions["default"], fractions["p0"], delta=1e-4)

        def scale_km(mu, g):
            return 288.15 * phys.K_BOLTZMANN / (g * mu * phys.M_PROTON) / 1000.0

        self.assertQuoted(3, f"{scale_km(28.97, 9.81):.3f}", f"{scale_km(18, 9.81):.2f}",
                          f"{scale_km(44, 9.81):.3f}", f"{scale_km(28.97, 19.62):.3f}")

    def test_beat_4_layers_and_the_cold_layer_below_a_warm_one(self):
        rows = self.rows(["--output_type", "temperature"])
        self.assertQuoted(4, rows[200000.0][1], rows[500000.0][1])
        p0, p11 = float(rows[0.0][1]), float(rows[11019.0][1])
        self.assertQuoted(4, f"{p0 / p11:.2f}")
        ratio = float(rows[200000.0][1]) / float(rows[500000.0][1])
        self.assertTrue(4500 < ratio < 5500)
        per_km_cold = math.log(p0 / p11) / 11.019
        per_km_hot = math.log(ratio) / 300.0
        self.assertTrue(4.0 < per_km_cold / per_km_hot < 5.5)
        base = phys.K_BOLTZMANN / (9.81 * 28.97 * phys.M_PROTON) / 1000.0
        self.assertQuoted(4, f"{216.65 * base:.2f}", f"{1240.0 * base:.1f}", "186.95")
        self.assertEqual(rows[86000.0][4], "186.95")
        second = self.rows(["--h_points", "0,50000,100000", "--T_points", "288,200,350"])
        pressure_ratio = float(second[50000.0][1]) / float(second[100000.0][1])
        density_ratio = float(second[50000.0][2]) / float(second[100000.0][2])
        self.assertQuoted(4, f"{pressure_ratio:.0f}", f"{density_ratio:.0f}")
        self.assertAlmostEqual(density_ratio, pressure_ratio * 350.0 / 200.0, delta=1e-3 * density_ratio)

    def test_beat_5_other_worlds(self):
        mars = self.rows(["--planet_name", "Mars", "--g_accel", "3.72", "--mu", "44", "--p0", "610",
                          "--h_points", "0,10000,25000,45000", "--T_points", "210,170,145,130"])
        venus = self.rows(["--planet_name", "Venus", "--g_accel", "8.87", "--mu", "44", "--p0", "9.2e6",
                           "--h_points", "0,10000,20000,30000,40000,50000,60000",
                           "--T_points", "735,658,579,495,416,348,263"])
        jupiter = self.rows(["--planet_name", "Jupiter", "--g_accel", "24.8", "--mu", "2.2", "--p0", "1e5",
                             "--h_points", "0,20000,50000,100000", "--T_points", "165,130,110,150"])
        self.assertQuoted(5, mars[0.0][2], venus[0.0][2], jupiter[0.0][2], "1.2325")
        for name, rows, p0, g, quoted in (
            ("Mars", mars, 610.0, 3.72, "10.6"),
            ("Venus", venus, 9.2e6, 8.87, "15.6"),
            ("Jupiter", jupiter, 1e5, 24.8, "25.0"),
        ):
            with self.subTest(world=name):
                scale_km = p0 / (g * float(rows[0.0][2])) / 1000.0
                self.assertEqual(f"{scale_km:.1f}", quoted)
                self.assertQuoted(5, quoted)
        self.assertTrue(150 < 1.013e5 / 610.0 < 180)
        self.assertQuoted(5, venus[10000.0][1], venus[60000.0][1])
        low_10 = round(100 * (1 - float(venus[10000.0][1]) / 4.80e6))
        low_60 = round(100 * (1 - float(venus[60000.0][1]) / 23.9e3))
        self.assertQuoted(5, f"({low_10}% low)", f"({low_60}% low)")
        self.assertEqual((low_10, low_60), (3, 20))

    def test_beat_6_euler_error_and_the_comparison_table(self):
        argv = ["--h_points", "0,100000", "--T_points", "288.15,288.15"]
        exact_scale = 288.15 * phys.K_BOLTZMANN / (9.81 * 28.97 * phys.M_PROTON)
        exact = 1.013e5 * math.exp(-100_000.0 / exact_scale)
        self.assertQuoted(6, f"{exact:.4f}", f"{exact_scale:.0f}")
        errors = []
        for steps in (200, 400, 800, 1600):
            with patch.object(driver, "STEPS_PER_SCALE_HEIGHT", steps):
                rows = self.rows(argv)
            errors.append(100.0 * (1.0 - float(rows[100000.0][1]) / exact))
            if steps == 200:
                self.assertQuoted(6, rows[100000.0][1])
        for coarse, fine in zip(errors, errors[1:]):
            self.assertAlmostEqual(coarse / fine, 2.0, delta=0.05)
        self.assertQuoted(6, f"{errors[0]:.2f}%", f"{errors[1]:.1f}%", f"{errors[2]:.2f}%", f"{errors[3]:.2f}%")
        height_in_scales = 100_000.0 / exact_scale
        self.assertQuoted(6, f"{height_in_scales:.2f}", f"{1 - height_in_scales / 400:.3f}")

        table = section_html(HELP_HTML, "beat6")
        rows = self.rows([])
        compared = re.findall(
            r'<tr><td class="num">(\d+)</td><td class="num">([^<]+)</td>'
            r'<td class="num">([^<]+)</td><td class="num">([^<]+)</td></tr>', table)
        self.assertEqual([int(row[0]) for row in compared],
                         [11019, 20063, 32162, 47350, 51412, 71802, 86000])
        for altitude, program, standard, ratio in compared:
            with self.subTest(altitude=altitude):
                self.assertEqual(program, rows[float(altitude)][1])
                self.assertEqual(f"{float(program) / float(standard):.3f}", ratio)

    def test_beat_6_accounts_for_the_deficit_against_the_standard_atmosphere(self):
        """Removing the model's simplifications one at a time closes the 86 km gap."""

        def exact_pressure(g_of_h, q, p0, altitude):
            # Simpson integration of d(ln p)/dh = -g(h) q / T(h) with 5 m panels.
            steps = int(math.ceil(altitude / 5.0 / 2.0)) * 2
            width = altitude / steps
            total = 0.0
            for index in range(steps + 1):
                h = index * width
                weight = 1 if index in (0, steps) else (4 if index % 2 else 2)
                total += weight * g_of_h(h) * q / interpolated_temperature(DEFAULT_H, DEFAULT_T, h)
            return p0 * math.exp(-total * width / 3.0)

        table = section_html(HELP_HTML, "beat6")
        standard_86 = float(re.search(
            r'<td class="num">86000</td><td class="num">[^<]+</td><td class="num">([^<]+)</td>', table).group(1))
        program_86 = float(self.rows([])[86000.0][1])
        radius = 6_371_000.0
        q_program = 28.97 * phys.M_PROTON / phys.K_BOLTZMANN
        atomic_mass_unit, boltzmann = 1.66053907e-27, 1.380649e-23
        q_standard = 28.9644 * atomic_mass_unit / boltzmann

        ratios = [
            program_86 / standard_86,
            exact_pressure(lambda h: 9.81, q_program, 1.013e5, 86_000.0) / standard_86,
            exact_pressure(lambda h: 9.81 * (radius / (radius + h)) ** 2, q_program, 1.013e5, 86_000.0)
            / standard_86,
            exact_pressure(lambda h: 9.80665 * (radius / (radius + h)) ** 2, q_standard, 101_325.0, 86_000.0)
            / standard_86,
        ]
        self.assertQuoted(6, *(f"{ratio:.3f}" for ratio in ratios))
        self.assertEqual([f"{ratio:.3f}" for ratio in ratios], ["0.743", "0.773", "0.919", "1.000"])
        self.assertTrue(ratios[0] < ratios[1] < ratios[2] < ratios[3])
        gains = [ratios[1] - ratios[0], ratios[2] - ratios[1], ratios[3] - ratios[2]]
        self.assertTrue(gains[1] > gains[2] > gains[0], gains)   # gravity, constants, Euler
        self.assertQuoted(6, "Constant gravity is the largest source, the constants come second and the Euler step third.")

        excess = 100.0 * (q_program / q_standard - 1.0)
        self.assertQuoted(6, f"{excess:.2f}%")
        fall = 1.013e5 / program_86
        self.assertQuoted(6, f"{fall / 1e5:.1f}&times;10".replace("&times;", "\u00d7"), f"e^{{{math.log(fall):.1f}}}")
        self.assertTrue(7.5 < 100.0 * (math.exp(excess / 100.0 * math.log(fall)) - 1.0) < 9.0)
        self.assertQuoted(6, "roughly 8%", "9.80665", "101325", "28.9644", "6371")
        self.assertEqual(f"{100 * (1 - ratios[0]):.0f}%", "26%")
        self.assertQuoted(6, "26% at 86")

    def test_beat_7_where_the_curve_ends(self):
        default = AtmosphereModel(make_params()).run()
        self.assertEqual(round(default.altitudes[-1] / 1000.0), 572)
        self.assertAlmostEqual(default.temperatures[-1], 1.4, delta=0.05)
        short = AtmosphereModel(make_params(h_points=[0.0, 20_000.0], T_points=[288.15, 216.65])).run()
        self.assertEqual(round(short.altitudes[-1] / 1000.0), 32)
        self.assertEqual(round(short.altitudes[-1] / 1000.0) - 20, 12)
        self.assertQuoted(7, "1240 K", "572 km", "216.65 K", "32 km", "12 km", "1.4 K")
        rows = self.rows(["--output_type", "temperature"])
        self.assertEqual(rows[500000.0][4], "1240")


if __name__ == "__main__":
    unittest.main()
