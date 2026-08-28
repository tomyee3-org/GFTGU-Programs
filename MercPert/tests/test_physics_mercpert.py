"""Regression tests for the complete MercPert program module.

The discovery logic deliberately supports both repository layouts used during
review: this file may live in ``MercPert/tests`` or may be flattened beside the
four core modules by an upload/download system.
"""

from __future__ import annotations

import ast
import base64
import hashlib
from html.parser import HTMLParser
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
    "physics_mercpert.py",
    "driver_mercpert.py",
    "main.py",
    "plot_mercpert.py",
)
HELP_FILENAME = "MercPert.html"


def find_module_dir(start: Path) -> Path:
    """Find the nearest ancestor containing all four MercPert modules."""
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if all((directory / name).is_file()
               for name in CORE_MODULE_FILENAMES):
            return directory
    names = ", ".join(CORE_MODULE_FILENAMES)
    raise FileNotFoundError(
        f"Could not find a directory containing all MercPert modules: {names}"
    )


MODULE_DIR = find_module_dir(Path(__file__))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import driver_mercpert as driver  # noqa: E402
import physics_mercpert as physics  # noqa: E402
import plot_mercpert as plots  # noqa: E402


def default_binary() -> physics.BinarySystemParams:
    return physics.BinarySystemParams(1.0, 0.1, 0.7 * physics.AU)


def default_ic() -> physics.MercuryInitialConditions:
    return physics.MercuryInitialConditions(
        0.3 * physics.AU, 0.0, 0.0, 59220.0
    )


def short_run(steps: int = 8) -> driver.MercPertOutput:
    return driver.run_mercpert(
        default_binary(),
        default_ic(),
        driver.MercPertRunParams(2000.0, steps, 0.05, 1.0e-4),
    )


class _HelpInspector(HTMLParser):
    """Collect the small amount of HTML structure used by documentation tests."""

    def __init__(self) -> None:
        super().__init__()
        self.ids = []
        self.fragment_links = []
        self.images = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.append(attributes["id"])
        href = attributes.get("href", "")
        if tag == "a" and href.startswith("#"):
            self.fragment_links.append(href[1:])
        if tag == "img":
            self.images.append(attributes)


class DiscoveryTests(unittest.TestCase):
    def test_module_dir_contains_all_core_files(self) -> None:
        for name in CORE_MODULE_FILENAMES:
            self.assertTrue((MODULE_DIR / name).is_file())

    def test_find_module_dir_from_module_directory(self) -> None:
        self.assertEqual(find_module_dir(MODULE_DIR), MODULE_DIR)

    def test_find_module_dir_from_nested_tests_directory(self) -> None:
        self.assertEqual(find_module_dir(Path(__file__).parent), MODULE_DIR)

    def test_find_module_dir_prefers_nearest_complete_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outer = root / "outer"
            inner = outer / "inner"
            nested = inner / "tests"
            nested.mkdir(parents=True)
            for directory in (outer, inner):
                for name in CORE_MODULE_FILENAMES:
                    (directory / name).write_text("", encoding="utf-8")
            self.assertEqual(find_module_dir(nested), inner.resolve())

    def test_find_module_dir_rejects_incomplete_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            start = Path(temporary) / "nested"
            start.mkdir()
            with self.assertRaises(FileNotFoundError):
                find_module_dir(start)


class MetadataAndCompatibilityTests(unittest.TestCase):
    def test_declared_version_is_semantic(self) -> None:
        self.assertRegex(physics.MODEL_VERSION, r"^\d+\.\d+\.\d+$")

    def test_build_coverage_is_exact(self) -> None:
        self.assertEqual(physics.BUILD_ID_COVERS, CORE_MODULE_FILENAMES)

    def test_build_id_matches_independent_hash(self) -> None:
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
        self.assertNotEqual(physics.BUILD_ID, "unknown")

    def test_all_core_sources_parse_as_python_310(self) -> None:
        for name in CORE_MODULE_FILENAMES:
            source = (MODULE_DIR / name).read_text(encoding="utf-8")
            try:
                ast.parse(source, filename=name, feature_version=(3, 10))
            except SyntaxError as exc:  # pragma: no cover - assertion detail
                self.fail(f"{name} is not Python 3.10 syntax: {exc}")

    def test_help_file_exists_beside_modules(self) -> None:
        self.assertTrue((MODULE_DIR / HELP_FILENAME).is_file())

    def test_help_version_and_build_match_program(self) -> None:
        help_text = (MODULE_DIR / HELP_FILENAME).read_text(encoding="utf-8")
        match = re.search(
            r'<p\s+id="version_build"[^>]*>\s*'
            r'Version\s+([0-9]+\.[0-9]+\.[0-9]+)'
            r'(?:&nbsp;|\s)+Build\s+([0-9a-f]{12})',
            help_text,
            flags=re.IGNORECASE,
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), physics.MODEL_VERSION)
        self.assertEqual(match.group(2), physics.BUILD_ID)

    def test_help_contains_required_scientific_sections(self) -> None:
        help_text = (MODULE_DIR / HELP_FILENAME).read_text(encoding="utf-8")
        inspector = _HelpInspector()
        inspector.feed(help_text)
        for section_id in (
            "three-body", "energy", "convergence", "output", "suggestions"
        ):
            self.assertIn(section_id, inspector.ids)

        def section(section_id: str) -> str:
            match = re.search(
                rf'<section id="{section_id}">(.*?)</section>',
                help_text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            self.assertIsNotNone(match)
            assert match is not None
            return " ".join(match.group(1).lower().split())

        self.assertIn("barycentric", section("three-body"))
        energy = section("energy")
        for concept in ("collision", "jacobi", "linear interpolation"):
            self.assertIn(concept, energy)
        self.assertIn("barycentric", section("output"))

    def test_student_help_has_no_development_history(self) -> None:
        help_text = (MODULE_DIR / HELP_FILENAME).read_text(
            encoding="utf-8"
        ).lower()
        for phrase in (
            "copilot",
            "gemini",
            "codex",
            "ai-generated",
            "ported from java",
            "development history",
        ):
            self.assertNotIn(phrase, help_text)

    def test_help_internal_links_and_ids_are_consistent(self) -> None:
        inspector = _HelpInspector()
        inspector.feed(
            (MODULE_DIR / HELP_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(len(inspector.ids), len(set(inspector.ids)))
        self.assertTrue(set(inspector.fragment_links).issubset(inspector.ids))

    def test_help_contains_six_valid_embedded_gallery_images(self) -> None:
        inspector = _HelpInspector()
        inspector.feed(
            (MODULE_DIR / HELP_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(len(inspector.images), 6)
        for attributes in inspector.images:
            self.assertTrue(attributes.get("alt"))
            prefix = "data:image/png;base64,"
            source = attributes.get("src", "")
            self.assertTrue(source.startswith(prefix))
            decoded = base64.b64decode(source[len(prefix):], validate=True)
            self.assertTrue(decoded.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_exercises_are_ranked_in_approximate_difficulty_order(self) -> None:
        help_text = (MODULE_DIR / HELP_FILENAME).read_text(encoding="utf-8")
        suggestions = re.search(
            r'<section id="suggestions">(.*?)</section>',
            help_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        self.assertIsNotNone(suggestions)
        assert suggestions is not None
        headings = re.findall(
            r"<h3>\s*([1-8])\s*·.*?<small>\((.*?)\)</small>\s*</h3>",
            suggestions.group(1),
            flags=re.DOTALL,
        )
        self.assertEqual([number for number, _ in headings], list("12345678"))
        levels = " ".join(level for _, level in headings)
        for label in ("Introductory", "Intermediate", "Advanced", "Capstone"):
            self.assertIn(label, levels)

    def test_help_omits_obsolete_java_listing(self) -> None:
        help_text = (MODULE_DIR / HELP_FILENAME).read_text(
            encoding="utf-8"
        ).lower()
        self.assertNotIn("listing of the java code", help_text)
        self.assertNotIn("private double", help_text)


class ValidationTests(unittest.TestCase):
    def test_valid_binary_and_initial_conditions(self) -> None:
        physics.validate_binary_params(default_binary())
        physics.validate_mercury_ic(default_ic())

    def test_binary_rejects_nonpositive_mass_or_separation(self) -> None:
        cases = (
            physics.BinarySystemParams(0.0, 0.1, 1.0),
            physics.BinarySystemParams(-1.0, 0.1, 1.0),
            physics.BinarySystemParams(1.0, 0.0, 1.0),
            physics.BinarySystemParams(1.0, -0.1, 1.0),
            physics.BinarySystemParams(1.0, 0.1, 0.0),
            physics.BinarySystemParams(1.0, 0.1, -1.0),
        )
        for params in cases:
            with self.subTest(params=params), self.assertRaises(ValueError):
                physics.validate_binary_params(params)

    def test_binary_rejects_nan_infinity_and_boolean(self) -> None:
        bad_values = (math.nan, math.inf, -math.inf, True, "1.0", None)
        for bad in bad_values:
            with self.subTest(value=bad), self.assertRaises(ValueError):
                physics.validate_binary_params(
                    physics.BinarySystemParams(bad, 0.1, 1.0)
                )

    def test_binary_rejects_overflowing_total_mass(self) -> None:
        with self.assertRaises(ValueError):
            physics.validate_binary_params(
                physics.BinarySystemParams(1.0e308, 1.0e308, physics.AU)
            )

    def test_initial_conditions_reject_nonfinite_and_boolean(self) -> None:
        for bad in (math.nan, math.inf, -math.inf, True, "0", None):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                physics.validate_mercury_ic(
                    physics.MercuryInitialConditions(bad, 0.0, 0.0, 0.0)
                )

    def test_angular_velocity_rejects_unrepresentable_scale(self) -> None:
        with self.assertRaises(ValueError):
            physics.compute_binary_angular_velocity(
                physics.BinarySystemParams(1.0, 1.0, 5.0e-324)
            )
        with self.assertRaises(ValueError):
            physics.compute_binary_angular_velocity(
                physics.BinarySystemParams(1.0, 1.0, 1.0e308)
            )

    def test_position_velocity_and_jacobi_reject_nonfinite_state(self) -> None:
        params = default_binary()
        with self.assertRaises(ValueError):
            physics.binary_positions(math.inf, params)
        with self.assertRaises(ValueError):
            physics.binary_velocities(math.nan, params)
        with self.assertRaises(ValueError):
            physics.mercury_acceleration(0.0, math.inf, 0.0, params)
        with self.assertRaises(ValueError):
            physics.jacobi_constant(0.0, 0.0, 0.0, math.nan, 0.0, params)


class BinaryGeometryTests(unittest.TestCase):
    def test_nominal_constants(self) -> None:
        self.assertEqual(physics.GM_SUN, 1.3271244e20)
        self.assertEqual(physics.R_SUN, 6.957e8)
        self.assertEqual(physics.AU, 1.495978707e11)

    def test_binary_radii_sum_to_separation(self) -> None:
        params = default_binary()
        r_sun, r_companion = physics.compute_binary_radii(params)
        self.assertAlmostEqual(r_sun + r_companion, params.binary_separation)

    def test_binary_centre_of_mass_is_origin(self) -> None:
        params = default_binary()
        for fraction in (0.0, 0.13, 0.25, 0.5, 1.0):
            omega = physics.compute_binary_angular_velocity(params)
            t = fraction * 2.0 * math.pi / omega
            sun, companion = physics.binary_positions(t, params)
            for component in (0, 1):
                moment = (
                    params.m_sun_solar * sun[component]
                    + params.m_planet_solar * companion[component]
                )
                self.assertAlmostEqual(moment, 0.0, delta=1.0e-3)

    def test_positions_at_zero_and_quarter_period(self) -> None:
        params = default_binary()
        r_sun, r_companion = physics.compute_binary_radii(params)
        omega = physics.compute_binary_angular_velocity(params)
        sun0, companion0 = physics.binary_positions(0.0, params)
        self.assertEqual(sun0, (-r_sun, -0.0))
        self.assertEqual(companion0, (r_companion, 0.0))
        sunq, companionq = physics.binary_positions(
            math.pi / (2.0 * omega), params
        )
        self.assertAlmostEqual(sunq[0], 0.0, delta=1.0e-3)
        self.assertAlmostEqual(sunq[1], -r_sun, delta=1.0e-3)
        self.assertAlmostEqual(companionq[0], 0.0, delta=1.0e-3)
        self.assertAlmostEqual(companionq[1], r_companion, delta=1.0e-3)

    def test_velocities_are_tangent_with_expected_speed(self) -> None:
        params = default_binary()
        omega = physics.compute_binary_angular_velocity(params)
        r_sun, r_companion = physics.compute_binary_radii(params)
        for t in (0.0, 12345.0, 543210.0):
            positions = physics.binary_positions(t, params)
            velocities = physics.binary_velocities(t, params)
            for position, velocity, radius in zip(
                positions, velocities, (r_sun, r_companion)
            ):
                dot = position[0] * velocity[0] + position[1] * velocity[1]
                self.assertAlmostEqual(dot, 0.0, delta=1.0e6)
                self.assertAlmostEqual(math.hypot(*velocity), omega * radius)

    def test_velocity_matches_position_derivative(self) -> None:
        params = default_binary()
        t = 100000.0
        h = 0.1
        before = physics.binary_positions(t - h, params)
        after = physics.binary_positions(t + h, params)
        exact = physics.binary_velocities(t, params)
        for body in (0, 1):
            for component in (0, 1):
                numerical = (
                    after[body][component] - before[body][component]
                ) / (2.0 * h)
                self.assertAlmostEqual(numerical, exact[body][component], delta=0.02)

    def test_initial_state_conversion_adds_sun_state(self) -> None:
        params = default_binary()
        ic = default_ic()
        state = physics.mercury_initial_barycentric_state(params, ic)
        sun, _ = physics.binary_positions(0.0, params)
        sun_v, _ = physics.binary_velocities(0.0, params)
        expected = (
            sun[0] + ic.x_init,
            sun[1] + ic.y_init,
            sun_v[0] + ic.vx_init,
            sun_v[1] + ic.vy_init,
        )
        self.assertEqual(state, expected)


class ForceAndJacobiTests(unittest.TestCase):
    def test_distances_match_direct_geometry(self) -> None:
        params = default_binary()
        t = 123456.0
        x, y = 0.2 * physics.AU, -0.1 * physics.AU
        sun, companion = physics.binary_positions(t, params)
        distances = physics.distances_to_primaries(t, x, y, params)
        self.assertAlmostEqual(distances[0], math.hypot(x - sun[0], y - sun[1]))
        self.assertAlmostEqual(
            distances[1], math.hypot(x - companion[0], y - companion[1])
        )

    def test_acceleration_matches_direct_newtonian_sum(self) -> None:
        params = default_binary()
        t = 34567.0
        x, y = 0.12 * physics.AU, 0.07 * physics.AU
        sun, companion = physics.binary_positions(t, params)
        expected = [0.0, 0.0]
        for mass, body in (
            (params.m_sun_solar, sun),
            (params.m_planet_solar, companion),
        ):
            dx, dy = body[0] - x, body[1] - y
            radius = math.hypot(dx, dy)
            expected[0] += physics.GM_SUN * mass * dx / radius ** 3
            expected[1] += physics.GM_SUN * mass * dy / radius ** 3
        actual = physics.mercury_acceleration(t, x, y, params)
        self.assertAlmostEqual(actual[0], expected[0], places=12)
        self.assertAlmostEqual(actual[1], expected[1], places=12)

    def test_acceleration_is_finite_at_very_large_distance(self) -> None:
        acceleration = physics.mercury_acceleration(
            0.0, 1.0e150, -1.0e150, default_binary()
        )
        self.assertTrue(all(math.isfinite(value) for value in acceleration))

    def test_acceleration_rejects_each_point_mass_singularity(self) -> None:
        params = default_binary()
        sun, companion = physics.binary_positions(0.0, params)
        with self.assertRaisesRegex(ValueError, "Sun's point-mass"):
            physics.mercury_acceleration(0.0, *sun, params)
        with self.assertRaisesRegex(ValueError, "companion's point-mass"):
            physics.mercury_acceleration(0.0, *companion, params)

    def test_l4_corotating_state_has_centripetal_acceleration(self) -> None:
        params = default_binary()
        sun, companion = physics.binary_positions(0.0, params)
        x = 0.5 * (sun[0] + companion[0])
        y = math.sqrt(3.0) * params.binary_separation / 2.0
        omega = physics.compute_binary_angular_velocity(params)
        ax, ay = physics.mercury_acceleration(0.0, x, y, params)
        self.assertAlmostEqual(ax, -omega * omega * x, delta=1.0e-10)
        self.assertAlmostEqual(ay, -omega * omega * y, delta=1.0e-10)

    def test_jacobi_constant_is_rotation_invariant(self) -> None:
        params = default_binary()
        omega = physics.compute_binary_angular_velocity(params)
        x0, y0 = 0.18 * physics.AU, -0.11 * physics.AU
        vx0, vy0 = 1300.0, 42000.0
        c0 = physics.jacobi_constant(0.0, x0, y0, vx0, vy0, params)
        t = 0.37 * 2.0 * math.pi / omega
        angle = omega * t
        cosine, sine = math.cos(angle), math.sin(angle)
        x = cosine * x0 - sine * y0
        y = sine * x0 + cosine * y0
        vx = cosine * vx0 - sine * vy0
        vy = sine * vx0 + cosine * vy0
        c1 = physics.jacobi_constant(t, x, y, vx, vy, params)
        self.assertAlmostEqual(c1, c0, delta=1.0e-3)


class DriverHelperTests(unittest.TestCase):
    def test_vector_relative_change_zero_vector(self) -> None:
        self.assertEqual(driver._vector_relative_change(0, 0, 0, 0), 0.0)

    def test_vector_relative_change_is_scale_invariant(self) -> None:
        first = driver._vector_relative_change(3, 4, 6, 8)
        second = driver._vector_relative_change(30, 40, 60, 80)
        self.assertAlmostEqual(first, second)

    def test_moving_circle_detects_initial_interior(self) -> None:
        fraction = driver._moving_circle_crossing_fraction(
            (0.5, 0.0), (2.0, 0.0), (0.0, 0.0), (0.0, 0.0), 1.0
        )
        self.assertEqual(fraction, 0.0)

    def test_moving_circle_detects_crossing_between_outside_endpoints(self) -> None:
        fraction = driver._moving_circle_crossing_fraction(
            (-2.0, 0.0), (2.0, 0.0), (0.0, 0.0), (0.0, 0.0), 1.0
        )
        self.assertAlmostEqual(fraction, 0.25)

    def test_moving_circle_accounts_for_primary_motion(self) -> None:
        fraction = driver._moving_circle_crossing_fraction(
            (0.0, 0.0), (0.0, 0.0), (2.0, 0.0), (-2.0, 0.0), 0.5
        )
        self.assertAlmostEqual(fraction, 0.375)

    def test_moving_circle_returns_none_for_miss_or_disabled_radius(self) -> None:
        miss = driver._moving_circle_crossing_fraction(
            (-2.0, 2.0), (2.0, 2.0), (0.0, 0.0), (0.0, 0.0), 1.0
        )
        disabled = driver._moving_circle_crossing_fraction(
            (-2.0, 0.0), (2.0, 0.0), (0.0, 0.0), (0.0, 0.0), 0.0
        )
        self.assertIsNone(miss)
        self.assertIsNone(disabled)


class DriverRunTests(unittest.TestCase):
    def test_run_parameter_validation_matrix(self) -> None:
        invalid = (
            driver.MercPertRunParams(0.0, 1, 0.05, 1.0e-4),
            driver.MercPertRunParams(math.inf, 1, 0.05, 1.0e-4),
            driver.MercPertRunParams(True, 1, 0.05, 1.0e-4),
            driver.MercPertRunParams("1", 1, 0.05, 1.0e-4),
            driver.MercPertRunParams(1.0, 0, 0.05, 1.0e-4),
            driver.MercPertRunParams(1.0, True, 0.05, 1.0e-4),
            driver.MercPertRunParams(1.0, 1, 0.0, 1.0e-4),
            driver.MercPertRunParams(1.0, 1, "0.05", 1.0e-4),
            driver.MercPertRunParams(1.0, 1, 1.0, 1.0e-4),
            driver.MercPertRunParams(1.0, 1, 0.05, 0.0),
            driver.MercPertRunParams(1.0, 1, 0.05, 0.05),
            driver.MercPertRunParams(1.0, 1, 0.05, 1.0e-4, -1.0),
            driver.MercPertRunParams(1.0, 1, 0.05, 1.0e-4, 0.0, math.nan),
        )
        for params in invalid:
            with self.subTest(params=params), self.assertRaises(ValueError):
                driver._validate_run_params(params)

    def test_run_rejects_overlapping_collision_surfaces(self) -> None:
        binary = physics.BinarySystemParams(1.0, 1.0, 10.0)
        run = driver.MercPertRunParams(0.01, 1, 0.05, 1.0e-4, 6.0, 4.0)
        with self.assertRaisesRegex(ValueError, "sum to less"):
            driver.run_mercpert(binary, default_ic(), run)

    def test_short_run_shapes_metadata_and_monotonic_time(self) -> None:
        output = short_run(12)
        series = (
            output.times, output.sun_x, output.sun_y,
            output.planet_x, output.planet_y,
            output.merc_x, output.merc_y,
            output.merc_vx, output.merc_vy,
            output.jacobi, output.dt_used,
        )
        self.assertTrue(all(len(values) == 13 for values in series))
        self.assertEqual(output.accepted_steps, 12)
        self.assertEqual(output.termination_reason, "max_steps reached")
        self.assertIsNone(output.collision_body)
        self.assertEqual(output.model_version, physics.MODEL_VERSION)
        self.assertEqual(output.build_id, physics.BUILD_ID)
        self.assertEqual(output.times[0], 0.0)
        self.assertEqual(output.dt_used[0], 0.0)
        self.assertTrue(all(b > a for a, b in zip(output.times, output.times[1:])))
        self.assertTrue(all(0.0 < dt <= 2000.0 for dt in output.dt_used[1:]))
        self.assertTrue(all(
            math.isfinite(value) for values in series for value in values
        ))

    def test_default_initial_sample_is_exact_conversion(self) -> None:
        output = short_run(1)
        expected = physics.mercury_initial_barycentric_state(
            default_binary(), default_ic()
        )
        actual = (
            output.merc_x[0], output.merc_y[0],
            output.merc_vx[0], output.merc_vy[0],
        )
        self.assertEqual(actual, expected)

    def test_initial_sun_collision_stops_without_steps(self) -> None:
        binary = default_binary()
        ic = physics.MercuryInitialConditions(0.5 * physics.R_SUN, 0, 0, 0)
        run = driver.MercPertRunParams(
            1.0, 10, 0.05, 1.0e-4, physics.R_SUN, 0.0
        )
        output = driver.run_mercpert(binary, ic, run)
        self.assertEqual(output.accepted_steps, 0)
        self.assertEqual(output.collision_body, "Sun")
        self.assertEqual(len(output.times), 1)

    def test_initial_companion_collision_stops_without_steps(self) -> None:
        binary = default_binary()
        # At t=0 the companion is one separation to the right of the Sun.
        ic = physics.MercuryInitialConditions(
            binary.binary_separation - 1.0, 0.0, 0.0, 0.0
        )
        run = driver.MercPertRunParams(1.0, 10, 0.05, 1.0e-4, 0.0, 2.0)
        output = driver.run_mercpert(binary, ic, run)
        self.assertEqual(output.accepted_steps, 0)
        self.assertEqual(output.collision_body, "companion")

    def test_inward_trajectory_stops_at_solar_boundary(self) -> None:
        binary = physics.BinarySystemParams(1.0, 1.0e-10, 100.0 * physics.AU)
        ic = physics.MercuryInitialConditions(
            2.0 * physics.R_SUN, 0.0, -20000.0, 0.0
        )
        run = driver.MercPertRunParams(
            2000.0, 500, 0.02, 1.0e-5, physics.R_SUN, 0.0
        )
        output = driver.run_mercpert(binary, ic, run)
        self.assertEqual(output.collision_body, "Sun")
        final_distance = physics.distances_to_primaries(
            output.times[-1], output.merc_x[-1], output.merc_y[-1], binary
        )[0]
        self.assertAlmostEqual(final_distance, physics.R_SUN, delta=5.0e4)

    def test_default_jacobi_drift_regression_bound(self) -> None:
        output = driver.run_mercpert(
            default_binary(),
            default_ic(),
            driver.MercPertRunParams(2000.0, 10000, 0.05, 1.0e-4),
        )
        initial = output.jacobi[0]
        drift = max(abs(value - initial) for value in output.jacobi) / abs(initial)
        self.assertLess(drift, 2.0e-6)

    def test_halving_timestep_improves_finite_time_endpoint(self) -> None:
        binary, ic = default_binary(), default_ic()
        outputs = []
        for dt, steps in ((2000.0, 1000), (1000.0, 2000), (500.0, 4000)):
            outputs.append(driver.run_mercpert(
                binary, ic, driver.MercPertRunParams(dt, steps, 0.05, 1.0e-4)
            ))
        coarse, medium, fine = outputs
        error_coarse = math.hypot(
            coarse.merc_x[-1] - fine.merc_x[-1],
            coarse.merc_y[-1] - fine.merc_y[-1],
        )
        error_medium = math.hypot(
            medium.merc_x[-1] - fine.merc_x[-1],
            medium.merc_y[-1] - fine.merc_y[-1],
        )
        self.assertAlmostEqual(coarse.times[-1], medium.times[-1])
        self.assertAlmostEqual(medium.times[-1], fine.times[-1])
        self.assertLess(error_medium, error_coarse)

    def test_extreme_trial_state_fails_cleanly(self) -> None:
        binary = default_binary()
        ic = physics.MercuryInitialConditions(1.0, 1.0, 1.0e150, 1.0e150)
        run = driver.MercPertRunParams(1.0e308, 1, 0.05, 1.0e-4)
        with self.assertRaisesRegex(RuntimeError, "could not find"):
            driver.run_mercpert(binary, ic, run)


class PlotTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_display_option_validation(self) -> None:
        for corner in plots._CORNER_TO_XY:
            plots._validate_display_options(corner, "m")
            plots._validate_display_options(corner, "AU")
        with self.assertRaises(ValueError):
            plots._validate_display_options("middle", "m")
        with self.assertRaises(ValueError):
            plots._validate_display_options("upper left", "km")

    def test_orbit_plot_styles_labels_annotation_and_equal_aspect(self) -> None:
        output = short_run(5)
        with mock.patch.object(plt, "show") as show:
            plots.plot_orbits(
                output,
                merc_ic=default_ic(),
                binary_params=default_binary(),
                corner="upper left",
                position_unit="AU",
            )
        show.assert_called_once()
        axis = plt.gcf().axes[0]
        labelled = {line.get_label(): line for line in axis.lines}
        self.assertEqual(labelled["Sun's orbit"].get_linestyle(), "--")
        self.assertEqual(labelled["Companion's orbit"].get_linestyle(), "--")
        self.assertEqual(labelled["Mercury"].get_linestyle(), "-")
        self.assertEqual(axis.get_xlabel(), "barycentric x (AU)")
        self.assertEqual(axis.get_ylabel(), "barycentric y (AU)")
        self.assertTrue(any("relative to Sun" in text.get_text()
                            for text in axis.texts))
        self.assertEqual(axis.get_aspect(), 1.0)

    def test_collision_plot_reserves_black_x_for_collision(self) -> None:
        output = short_run(1)
        output.collision_body = "Sun"
        output.termination_reason = "collision with Sun"
        with mock.patch.object(plt, "show"):
            plots.plot_orbits(output)
        axis = plt.gcf().axes[0]
        collision = [line for line in axis.lines
                     if line.get_label() == "Collision: Sun"]
        self.assertEqual(len(collision), 1)
        self.assertEqual(collision[0].get_marker(), "x")

    def test_primary_reference_tracks_stop_after_one_period(self) -> None:
        binary = default_binary()
        omega = physics.compute_binary_angular_velocity(binary)
        period = 2.0 * math.pi / omega
        times = [0.0, 0.5 * period, period, 1.5 * period]
        positions = [physics.binary_positions(t, binary) for t in times]
        output = driver.MercPertOutput(
            times=times,
            sun_x=[p[0][0] for p in positions],
            sun_y=[p[0][1] for p in positions],
            planet_x=[p[1][0] for p in positions],
            planet_y=[p[1][1] for p in positions],
            merc_x=[0.0] * 4,
            merc_y=[0.0] * 4,
            merc_vx=[0.0] * 4,
            merc_vy=[0.0] * 4,
            jacobi=[1.0] * 4,
            dt_used=[0.0, 1.0, 1.0, 1.0],
            accepted_steps=3,
            termination_reason="max_steps reached",
        )
        with mock.patch.object(plt, "show"):
            plots.plot_orbits(output, binary_params=binary)
        axis = plt.gcf().axes[0]
        labelled = {line.get_label(): line for line in axis.lines}
        self.assertEqual(len(labelled["Sun's orbit"].get_xdata()), 3)
        self.assertEqual(len(labelled["Mercury"].get_xdata()), 4)

    def test_jacobi_plot_values(self) -> None:
        output = short_run(3)
        with mock.patch.object(plt, "show") as show:
            plots.plot_jacobi_drift(output)
        show.assert_called_once()
        axis = plt.gcf().axes[0]
        expected = [
            (value - output.jacobi[0]) / abs(output.jacobi[0])
            for value in output.jacobi
        ]
        self.assertEqual(list(axis.lines[0].get_ydata()), expected)

    def test_jacobi_plot_rejects_missing_data(self) -> None:
        output = short_run(1)
        output.jacobi = []
        with self.assertRaises(ValueError):
            plots.plot_jacobi_drift(output)


class CommandLineTests(unittest.TestCase):
    def test_version_command_reports_synchronized_metadata(self) -> None:
        completed = subprocess.run(
            [sys.executable, "main.py", "--version"],
            cwd=MODULE_DIR,
            text=True,
            capture_output=True,
            check=True,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        expected = f"MercPert {physics.MODEL_VERSION} (build {physics.BUILD_ID})"
        self.assertEqual(completed.stdout.strip(), expected)

    def test_default_command_smoke_run(self) -> None:
        completed = subprocess.run(
            [sys.executable, "main.py"],
            cwd=MODULE_DIR,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        self.assertIn("accepted steps: 10000", completed.stdout)
        self.assertIn("termination: max_steps reached", completed.stdout)
        self.assertRegex(
            completed.stdout,
            r"Maximum fractional Jacobi drift: [0-9.e+-]+",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
