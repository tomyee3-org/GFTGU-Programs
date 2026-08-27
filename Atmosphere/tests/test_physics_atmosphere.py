"""Regression and unit tests for the Atmosphere teaching model.

The suite uses only Python's standard-library ``unittest`` module so students
do not need an additional test dependency.  Run from the Atmosphere directory:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import random
import re
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
import physics_atmosphere as phys  # noqa: E402
from driver_atmosphere import (  # noqa: E402
    AtmosphereModel,
    AtmosphereParameters,
    AtmosphereResult,
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


class CommandLineHelpAndPlotTests(unittest.TestCase):
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

    def test_help_version_build_matches_runtime_and_html_parses(self):
        from html.parser import HTMLParser

        html = (MODULE_DIR / "Atmosphere.html").read_text(encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
