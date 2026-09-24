"""Regression tests for the GFTGU Neutron module.

The discovery code deliberately supports both repository layouts used during
review: this file may live in ``tests/`` or be flattened beside the four core
program modules during upload.
"""

from __future__ import annotations

import ast
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


CORE_MODULE_FILENAMES = (
    "physics_neutron.py",
    "driver_neutron.py",
    "main.py",
    "plot_neutron.py",
)
# The Beats-format tutorial is the Help file this suite scrapes and requires.
# ``Neutron-original.html`` (the pre-Beats Reference Guide) is not looked for
# here and a passing run never depends on it existing.
PROGRAM_NAME = "Neutron"
HELP_FILENAME = "Neutron-claude.html"


def find_module_dir(start: Path) -> Path:
    """Find the nearest ancestor containing all four core module files."""
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        if all((directory / name).is_file() for name in CORE_MODULE_FILENAMES):
            return directory

    names = ", ".join(CORE_MODULE_FILENAMES)
    raise FileNotFoundError(
        f"Could not find a directory containing all core modules: {names}"
    )


MODULE_DIR = find_module_dir(Path(__file__))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import driver_neutron as driver  # noqa: E402
import physics_neutron as phys  # noqa: E402
import plot_neutron as plotter  # noqa: E402


DEFAULT_GAMMA = 1.666667
DEFAULT_PC = 1.26e35
DEFAULT_K = 5.3802e3

# Independent high-precision reference for the default model (gamma=
# 1.666667, pC=1.26e35, K=5.3802e3). This fixed constant predates the
# _q_reference_model() helper defined below and was not regenerated from
# it; both were established independently of the production solver and
# protect its surface-location regression (test_default_model_matches_
# independent_surface_reference, test_refinement_improves_surface_
# reference_agreement) against a change there. _q_reference_model() itself
# -- the same tangent-based extrapolation used for the parametrized
# cross-checks further below, just run at this model's own parameters --
# agrees with this constant to within about a micrometre in radius, not
# bit-for-bit: see test_q_reference_model_matches_default_reference_constant,
# which checks that agreement directly instead of leaving it asserted only
# in comments or in the Beats Help file.
REFERENCE_RADIUS_M = 7802.758706219183
REFERENCE_MASS_KG = 1.9140826174028036e30


def _q_reference_model(
    gamma: float,
    p_c: float,
    K: float,
    *,
    steps_per_scale: int = 8000,
    max_steps: int = 500_000,
) -> tuple[float, float]:
    """Independent fine-grid reference using q=p**((gamma-1)/gamma).

    This test integrator evolves q directly, unlike the production solver,
    which evolves pressure and uses q only for its final surface estimate.
    """
    rho_c = math.exp((math.log(p_c) - math.log(K)) / gamma)
    scale = math.exp(0.5 * (math.log(p_c) - math.log(phys.G)) - math.log(rho_c))
    h_nominal = scale / steps_per_scale
    r = h_nominal
    coefficient = (
        2.0
        * math.pi
        * phys.G
        * (rho_c + p_c / phys.C2)
        * (rho_c / 3.0 + p_c / phys.C2)
    )
    p = p_c - coefficient * r * r
    mass = 4.0 * math.pi * rho_c * r**3 / 3.0

    exponent = (gamma - 1.0) / gamma
    q = p**exponent
    inverse_K_power = K ** (-1.0 / gamma)
    pressure_power = gamma / (gamma - 1.0)
    density_power = 1.0 / (gamma - 1.0)

    def derivatives(rr: float, qq: float, mm: float) -> tuple[float, float]:
        if qq < 0.0:
            raise ArithmeticError("q crossed the reference surface")
        pressure = qq**pressure_power
        density = inverse_K_power * qq**density_power
        denominator = rr - 2.0 * phys.G * mm / phys.C2
        dqdr = (
            -exponent
            * phys.G
            * (inverse_K_power + qq / phys.C2)
            * (mm + 4.0 * math.pi * rr**3 * pressure / phys.C2)
            / (rr * denominator)
        )
        dmdr = 4.0 * math.pi * rr * rr * density
        return dqdr, dmdr

    for _ in range(max_steps):
        dqdr, dmdr = derivatives(r, q, mass)
        surface_distance = -q / dqdr
        if surface_distance <= h_nominal:
            # rho is proportional to q**(1/(gamma-1)); integrating that
            # leading falloff multiplies the rectangular tail by exponent.
            return (
                r + surface_distance,
                mass + dmdr * exponent * surface_distance,
            )

        h = h_nominal
        while True:
            try:
                k1q, k1m = derivatives(r, q, mass)
                k2q, k2m = derivatives(
                    r + 0.5 * h,
                    q + 0.5 * h * k1q,
                    mass + 0.5 * h * k1m,
                )
                k3q, k3m = derivatives(
                    r + 0.5 * h,
                    q + 0.5 * h * k2q,
                    mass + 0.5 * h * k2m,
                )
                k4q, k4m = derivatives(
                    r + h, q + h * k3q, mass + h * k3m
                )
                q_new = q + h * (k1q + 2.0 * k2q + 2.0 * k3q + k4q) / 6.0
                mass_new = mass + h * (
                    k1m + 2.0 * k2m + 2.0 * k3m + k4m
                ) / 6.0
                if q_new < 0.0:
                    raise ArithmeticError("q crossed the reference surface")
                break
            except ArithmeticError:
                h *= 0.5

        r += h
        q = q_new
        mass = mass_new

    raise AssertionError("Independent q-reference integration did not finish")


@pytest.fixture(scope="module")
def default_model() -> dict:
    return driver.compute_neutron_star(DEFAULT_GAMMA, DEFAULT_PC, DEFAULT_K)


def _independent_build_id() -> str:
    digest = hashlib.sha256()
    for name in CORE_MODULE_FILENAMES:
        with (MODULE_DIR / name).open(
            "r", encoding="utf-8", newline=None
        ) as source:
            content = source.read().encode("utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()[:12]


def find_help_file(module_dir: Path) -> Path:
    """Find the Beats Help file in a flattened upload or the
    GFTGU-Documentation tree.

    Documentation folders no longer use chapter-number prefixes, and the
    Help files live under the sibling ``GFTGU-Documentation`` repository
    rather than beside the program modules inside ``GFTGU-Programs``. The
    program's documentation directory is still named ``Neutron`` even
    though the file sought inside it is ``Neutron-claude.html``, so the
    directory name is kept separate from the Help filename.
    """
    candidates = [module_dir / HELP_FILENAME]
    for ancestor in (module_dir, *module_dir.parents):
        candidates.append(
            ancestor / "GFTGU-Documentation" / PROGRAM_NAME / HELP_FILENAME
        )
        if ancestor.name != PROGRAM_NAME:
            candidates.append(ancestor / PROGRAM_NAME / HELP_FILENAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {HELP_FILENAME} beside the program or in "
        f"GFTGU-Documentation/{PROGRAM_NAME}/."
    )


def _help_text() -> str:
    return find_help_file(MODULE_DIR).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Layout, compatibility, release metadata, and end-to-end execution
# ---------------------------------------------------------------------------


def test_find_module_dir_from_module_and_nested_test_directory() -> None:
    assert find_module_dir(MODULE_DIR) == MODULE_DIR
    assert find_module_dir(Path(__file__)) == MODULE_DIR


def test_find_module_dir_rejects_tree_without_core_files(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="all core modules"):
        find_module_dir(nested)


def test_all_core_sources_parse_as_python_310() -> None:
    for name in CORE_MODULE_FILENAMES:
        source = (MODULE_DIR / name).read_text(encoding="utf-8")
        ast.parse(source, filename=name, feature_version=(3, 10))


def test_build_id_covers_exactly_the_four_core_modules() -> None:
    assert phys.BUILD_ID_COVERS == CORE_MODULE_FILENAMES
    assert phys.BUILD_ID == _independent_build_id()
    assert re.fullmatch(r"[0-9a-f]{12}", phys.BUILD_ID)


def test_help_version_and_build_are_in_sync() -> None:
    html = _help_text()
    match = re.search(
        r'<p\s+id="version_build"[^>]*>\s*Version\s+([^&<\s]+)'
        r'(?:&nbsp;|\s)+Build\s+([0-9a-f]{12})',
        html,
        flags=re.IGNORECASE,
    )
    assert match, "Help file lacks a parseable version_build paragraph"
    assert match.group(1) == phys.MODEL_VERSION
    assert match.group(2) == phys.BUILD_ID


def test_main_version_cli() -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "--version"],
        cwd=MODULE_DIR,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        f"Neutron {phys.MODEL_VERSION} (build {phys.BUILD_ID})"
    )


def test_default_program_runs_headlessly() -> None:
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=MODULE_DIR,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"Neutron {phys.MODEL_VERSION} (build {phys.BUILD_ID})" in result.stdout
    assert "Neutron-star model summary" in result.stdout
    assert "causality check: satisfied" in result.stdout


# ---------------------------------------------------------------------------
# Equation of state and TOV physics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pressure", "K", "gamma"),
    [(1.0, 2.0, 2.0), (1.26e35, 5.3802e3, 1.666667), (1e-200, 1e-100, 3.0)],
)
def test_eos_density_inverts_polytrope(
    pressure: float, K: float, gamma: float
) -> None:
    rho = phys.eos_density(pressure, K, gamma)
    assert K * rho**gamma == pytest.approx(pressure, rel=2e-13)


def test_eos_zero_pressure_has_zero_density() -> None:
    assert phys.eos_density(0.0, DEFAULT_K, DEFAULT_GAMMA) == 0.0


@pytest.mark.parametrize(
    ("pressure", "K", "gamma"),
    [
        (-1.0, 1.0, 2.0),
        (1.0, 0.0, 2.0),
        (1.0, -1.0, 2.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 0.5),
        (math.nan, 1.0, 2.0),
        (math.inf, 1.0, 2.0),
        (1.0, math.nan, 2.0),
        (1.0, 1.0, math.inf),
        (True, 1.0, 2.0),
    ],
)
def test_eos_rejects_invalid_inputs(
    pressure: float, K: float, gamma: float
) -> None:
    with pytest.raises(ValueError):
        phys.eos_density(pressure, K, gamma)


def test_eos_rejects_unrepresentable_density() -> None:
    with pytest.raises(ValueError, match="floating-point range"):
        phys.eos_density(1e308, 1e-308, 1.000001)


def test_structure_derivatives_match_tov_formula() -> None:
    r = 12_000.0
    p = 4.0e33
    m = 1.2e30
    K = DEFAULT_K
    gamma = DEFAULT_GAMMA
    rho = phys.eos_density(p, K, gamma)
    expected_dpdr = (
        -phys.G
        * (rho + p / phys.C2)
        * (m + 4.0 * math.pi * r**3 * p / phys.C2)
        / (r * (r - 2.0 * phys.G * m / phys.C2))
    )
    expected_dmdr = 4.0 * math.pi * r**2 * rho
    dpdr, dmdr = phys.structure_derivatives(r, p, m, K, gamma)
    assert dpdr == pytest.approx(expected_dpdr, rel=2e-15)
    assert dmdr == pytest.approx(expected_dmdr, rel=2e-15)
    assert dpdr < 0.0
    assert dmdr > 0.0


def test_tov_reduces_to_newtonian_hydrostatics_in_weak_field() -> None:
    r = 1.0e6
    p = 1.0
    m = 1.0e20
    K = 1.0e-12
    gamma = 2.0
    rho = phys.eos_density(p, K, gamma)
    newtonian = -phys.G * rho * m / r**2
    relativistic, _ = phys.structure_derivatives(r, p, m, K, gamma)
    assert relativistic == pytest.approx(newtonian, rel=2e-6)


@pytest.mark.parametrize(
    ("r", "p", "m"),
    [
        (0.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, -1.0),
        (math.nan, 1.0, 0.0),
        (1.0, math.inf, 0.0),
        (1.0, 1.0, math.nan),
    ],
)
def test_structure_derivatives_reject_invalid_state(
    r: float, p: float, m: float
) -> None:
    with pytest.raises(ValueError):
        phys.structure_derivatives(r, p, m, 1.0, 2.0)


def test_structure_derivatives_reject_horizon_denominator() -> None:
    r = 1000.0
    horizon_mass = 1.0001 * r * phys.C2 / (2.0 * phys.G)
    with pytest.raises(RuntimeError, match="TOV denominator is singular"):
        phys.structure_derivatives(r, 1.0e20, horizon_mass, 1.0, 2.0)


def test_structure_derivatives_reject_underflowed_full_denominator() -> None:
    with pytest.raises(ValueError, match="full TOV denominator"):
        phys.structure_derivatives(1.0e-200, 1.0e-300, 0.0, 1.0e-300, 2.0)


def test_structure_derivatives_reject_overflowed_compound_terms() -> None:
    with pytest.raises(ValueError, match="floating-point range"):
        phys.structure_derivatives(1.0e200, 1.0, 0.0, 1.0, 2.0)


def test_central_state_matches_regular_series() -> None:
    p_c = DEFAULT_PC
    r0 = 10.0
    p0, rho0, m0 = phys.central_state(p_c, DEFAULT_K, DEFAULT_GAMMA, r0)
    rho_c = phys.eos_density(p_c, DEFAULT_K, DEFAULT_GAMMA)
    expected_mass = 4.0 * math.pi * rho_c * r0**3 / 3.0
    expected_coeff = (
        2.0
        * math.pi
        * phys.G
        * (rho_c + p_c / phys.C2)
        * (rho_c / 3.0 + p_c / phys.C2)
    )
    assert m0 == pytest.approx(expected_mass, rel=2e-15)
    assert p0 == pytest.approx(p_c - expected_coeff * r0**2, rel=2e-15)
    assert rho0 == pytest.approx(
        phys.eos_density(p0, DEFAULT_K, DEFAULT_GAMMA), rel=2e-15
    )
    assert 0.0 < p0 < p_c
    assert 0.0 < rho0 < rho_c


@pytest.mark.parametrize("r0", [0.0, -1.0, math.nan, math.inf])
def test_central_state_rejects_invalid_radius(r0: float) -> None:
    with pytest.raises(ValueError):
        phys.central_state(DEFAULT_PC, DEFAULT_K, DEFAULT_GAMMA, r0)


def test_central_state_rejects_step_that_crosses_surface() -> None:
    with pytest.raises(ValueError, match="initial radial step is too large"):
        phys.central_state(DEFAULT_PC, DEFAULT_K, DEFAULT_GAMMA, 1.0e9)


def test_central_state_rejects_cubic_radius_overflow_cleanly() -> None:
    with pytest.raises(ValueError, match="central expansion"):
        phys.central_state(1.0, 1.0, 2.0, 1.0e200)


def test_rk4_step_advances_pressure_down_and_mass_up() -> None:
    r = 100.0
    p, _, m = phys.central_state(
        DEFAULT_PC, DEFAULT_K, DEFAULT_GAMMA, r
    )
    p_new, m_new = phys.rk4_step(
        r, p, m, 1.0, DEFAULT_K, DEFAULT_GAMMA
    )
    assert 0.0 < p_new < p
    assert m_new > m


def test_rk4_step_refines_consistently() -> None:
    r = 100.0
    p, _, m = phys.central_state(
        DEFAULT_PC, DEFAULT_K, DEFAULT_GAMMA, r
    )
    p_full, m_full = phys.rk4_step(
        r, p, m, 2.0, DEFAULT_K, DEFAULT_GAMMA
    )
    p_half, m_half = phys.rk4_step(
        r, p, m, 1.0, DEFAULT_K, DEFAULT_GAMMA
    )
    p_two, m_two = phys.rk4_step(
        r + 1.0, p_half, m_half, 1.0, DEFAULT_K, DEFAULT_GAMMA
    )
    assert p_full == pytest.approx(p_two, rel=2e-10)
    assert m_full == pytest.approx(m_two, rel=2e-10)


@pytest.mark.parametrize("h", [0.0, -1.0, math.nan, math.inf, True])
def test_rk4_rejects_invalid_step(h: float) -> None:
    with pytest.raises(ValueError):
        phys.rk4_step(1.0, 1.0, 0.0, h, 1.0, 2.0)


def test_rk4_rejects_nonfinite_combined_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def enormous_derivatives(*_args: object) -> tuple[float, float]:
        return sys.float_info.max, sys.float_info.max

    monkeypatch.setattr(phys, "structure_derivatives", enormous_derivatives)
    with pytest.raises(ValueError, match="finite floating-point range"):
        phys.rk4_step(1.0, 1.0, 1.0, 2.0, 1.0, 2.0)


# ---------------------------------------------------------------------------
# Integrated model, surface, convergence, diagnostics, and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gamma", "p_c", "K", "steps", "max_steps"),
    [
        (1.0, DEFAULT_PC, DEFAULT_K, 400, 200_000),
        (math.nan, DEFAULT_PC, DEFAULT_K, 400, 200_000),
        (DEFAULT_GAMMA, 0.0, DEFAULT_K, 400, 200_000),
        (DEFAULT_GAMMA, math.inf, DEFAULT_K, 400, 200_000),
        (DEFAULT_GAMMA, DEFAULT_PC, 0.0, 400, 200_000),
        (DEFAULT_GAMMA, DEFAULT_PC, math.nan, 400, 200_000),
        (DEFAULT_GAMMA, DEFAULT_PC, DEFAULT_K, 49, 200_000),
        (DEFAULT_GAMMA, DEFAULT_PC, DEFAULT_K, 10_000_001, 200_000),
        (DEFAULT_GAMMA, DEFAULT_PC, DEFAULT_K, 50.0, 200_000),
        (DEFAULT_GAMMA, DEFAULT_PC, DEFAULT_K, True, 200_000),
        (DEFAULT_GAMMA, DEFAULT_PC, DEFAULT_K, 400, 99),
        (DEFAULT_GAMMA, DEFAULT_PC, DEFAULT_K, 400, 10_000_001),
        (DEFAULT_GAMMA, DEFAULT_PC, DEFAULT_K, 400, 100.0),
        (DEFAULT_GAMMA, DEFAULT_PC, DEFAULT_K, 400, False),
    ],
)
def test_compute_rejects_invalid_inputs(
    gamma: float, p_c: float, K: float, steps: int, max_steps: int
) -> None:
    with pytest.raises(ValueError):
        driver.compute_neutron_star(
            gamma,
            p_c,
            K,
            steps_per_scale=steps,
            max_steps=max_steps,
        )


def test_compute_reports_step_limit_failure() -> None:
    with pytest.raises(RuntimeError, match="surface was not reached"):
        driver.compute_neutron_star(
            DEFAULT_GAMMA, DEFAULT_PC, DEFAULT_K, max_steps=100
        )


@pytest.mark.parametrize(
    ("gamma", "p_c", "K"),
    [
        (
            1.0001513808512184,
            4.2308654942826733e-48,
            1.3627252898262582e96,
        ),
        (
            1.0047723049491972,
            3.022685144472793e-169,
            2.261623147e-315,
        ),
    ],
)
def test_copilot_extreme_finite_reproducers_fail_cleanly(
    gamma: float, p_c: float, K: float
) -> None:
    with pytest.raises(ValueError, match="floating-point range"):
        driver.compute_neutron_star(
            gamma,
            p_c,
            K,
            steps_per_scale=50,
            max_steps=10_000,
        )


def test_driver_does_not_relabel_unrelated_value_error_as_surface_crossing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unrelated_failure(*_args: object, **_kwargs: object) -> tuple[float, float]:
        raise ValueError("sentinel arithmetic failure")

    monkeypatch.setattr(driver, "rk4_step", unrelated_failure)
    with pytest.raises(ValueError, match="sentinel arithmetic failure"):
        driver.compute_neutron_star(DEFAULT_GAMMA, DEFAULT_PC, DEFAULT_K)


def test_deterministic_extreme_range_sweep_has_only_controlled_outcomes() -> None:
    """Seed and ranges are retained so the advertised sweep is reproducible."""
    import random

    generator = random.Random(20260828)
    outcomes = {"success": 0, "controlled": 0}
    for _ in range(80):
        gamma = 1.0 + 10.0 ** generator.uniform(-15.0, 0.3)
        p_c = 10.0 ** generator.uniform(-300.0, 300.0)
        K = 10.0 ** generator.uniform(-300.0, 300.0)
        try:
            data = driver.compute_neutron_star(
                gamma,
                p_c,
                K,
                steps_per_scale=50,
                max_steps=100,
            )
            assert all(
                math.isfinite(value)
                for key in ("radius", "pressure", "density", "mass")
                for value in data[key]
            )
            outcomes["success"] += 1
        except (ValueError, RuntimeError):
            outcomes["controlled"] += 1

    assert sum(outcomes.values()) == 80


def test_default_model_regression(default_model: dict) -> None:
    assert default_model["model_version"] == phys.MODEL_VERSION
    assert default_model["build_id"] == phys.BUILD_ID
    assert default_model["surface_radius_m"] == pytest.approx(
        7803.563177583, abs=2e-6
    )
    assert default_model["total_mass_kg"] == pytest.approx(
        1.914082624724e30, rel=2e-12
    )
    assert default_model["total_mass_solar"] == pytest.approx(
        0.962590647445, rel=2e-12
    )


def test_default_profile_invariants(default_model: dict) -> None:
    keys = ("radius", "pressure", "density", "mass")
    lengths = {len(default_model[key]) for key in keys}
    assert lengths == {default_model["last_step"] + 1}
    assert len(default_model["radius"]) >= 100

    radius = default_model["radius"]
    pressure = default_model["pressure"]
    density = default_model["density"]
    mass = default_model["mass"]

    assert radius[0] == pressure[-1] == density[-1] == mass[0] == 0.0
    assert all(math.isfinite(value) for key in keys for value in default_model[key])
    assert all(b > a for a, b in zip(radius, radius[1:]))
    assert all(b < a for a, b in zip(pressure[:-2], pressure[1:-1]))
    assert all(b < a for a, b in zip(density[:-2], density[1:-1]))
    assert all(b > a for a, b in zip(mass, mass[1:]))
    assert all(value >= 0.0 for value in pressure + density + mass)


def test_integrated_density_obeys_eos(default_model: dict) -> None:
    for pressure, density in zip(
        default_model["pressure"][:-1], default_model["density"][:-1]
    ):
        assert density == pytest.approx(
            phys.eos_density(pressure, DEFAULT_K, DEFAULT_GAMMA), rel=3e-14
        )


def test_summary_quantities_are_self_consistent(default_model: dict) -> None:
    radius = default_model["surface_radius_m"]
    mass = default_model["total_mass_kg"]
    assert default_model["surface_radius_km"] == pytest.approx(radius / 1000.0)
    assert default_model["total_mass_solar"] == pytest.approx(mass / phys.M_SUN)
    assert default_model["compactness"] == pytest.approx(
        2.0 * phys.G * mass / (radius * phys.C2)
    )
    assert default_model["buchdahl_satisfied"] == (
        default_model["compactness"] <= 8.0 / 9.0
    )


def test_surface_uses_regular_polytropic_variable(default_model: dict) -> None:
    r = default_model["radius"][-2]
    p = default_model["pressure"][-2]
    m = default_model["mass"][-2]
    dpdr, dmdr = phys.structure_derivatives(
        r, p, m, DEFAULT_K, DEFAULT_GAMMA
    )
    pressure_scale_distance = -p / dpdr
    expected_distance = (
        DEFAULT_GAMMA / (DEFAULT_GAMMA - 1.0) * pressure_scale_distance
    )
    assert default_model["surface_radius_m"] - r == pytest.approx(
        expected_distance, rel=2e-13
    )
    assert default_model["total_mass_kg"] - m == pytest.approx(
        dmdr * pressure_scale_distance, rel=5e-10
    )


def test_default_model_matches_independent_surface_reference(
    default_model: dict,
) -> None:
    assert abs(default_model["surface_radius_m"] - REFERENCE_RADIUS_M) < 2.0
    assert abs(default_model["total_mass_kg"] / REFERENCE_MASS_KG - 1.0) < 1e-7


def test_refinement_improves_surface_reference_agreement() -> None:
    coarse = driver.compute_neutron_star(
        DEFAULT_GAMMA, DEFAULT_PC, DEFAULT_K, steps_per_scale=200
    )
    fine = driver.compute_neutron_star(
        DEFAULT_GAMMA, DEFAULT_PC, DEFAULT_K, steps_per_scale=1600
    )
    coarse_error = abs(coarse["surface_radius_m"] - REFERENCE_RADIUS_M)
    fine_error = abs(fine["surface_radius_m"] - REFERENCE_RADIUS_M)
    assert fine_error < coarse_error
    assert fine_error < 0.5
    assert abs(fine["total_mass_kg"] / REFERENCE_MASS_KG - 1.0) < 1e-9


def test_q_reference_model_matches_default_reference_constant() -> None:
    """Make the relationship between the fixed REFERENCE_RADIUS_M/
    REFERENCE_MASS_KG constants and this file's own _q_reference_model()
    helper executable rather than only asserted in a comment or in the
    Beats Help file (Audit24 Codex #2). The constants predate this helper
    and are not regenerated from it, so running the helper at the default
    model's own parameters is expected to land close to, but not exactly
    on, the fixed values: within a generous few-micrometre radius
    tolerance and a tiny relative mass tolerance, both well above observed
    floating-point noise and well below the production solver's own
    roughly 0.8 m surface-location error that these constants benchmark.
    """
    radius, mass = _q_reference_model(DEFAULT_GAMMA, DEFAULT_PC, DEFAULT_K)
    # Observed gap is about 1.01e-6 m (~1 micrometre) in radius and about
    # 9.4e-12 in relative mass; both tolerances below are generously wider.
    assert abs(radius - REFERENCE_RADIUS_M) < 5e-6
    assert abs(mass / REFERENCE_MASS_KG - 1.0) < 1e-9


@pytest.mark.parametrize(
    ("gamma", "p_c", "rho_c", "radius_tolerance", "mass_tolerance"),
    [
        (1.21, 1.0e20, 1.0e15, 1.0e-5, 1.0e-7),
        (1.25, 1.0e25, 1.0e17, 4.0e-5, 5.0e-7),
        (1.4, 1.0e32, 1.0e18, 6.0e-5, 5.0e-7),
        (2.0, 1.0e34, 1.0e18, 2.0e-4, 2.0e-6),
    ],
)
def test_surface_against_independent_q_reference_across_models(
    gamma: float,
    p_c: float,
    rho_c: float,
    radius_tolerance: float,
    mass_tolerance: float,
) -> None:
    K = p_c / rho_c**gamma
    production = driver.compute_neutron_star(gamma, p_c, K)
    reference_radius, reference_mass = _q_reference_model(gamma, p_c, K)
    assert abs(production["surface_radius_m"] / reference_radius - 1.0) < (
        radius_tolerance
    )
    assert abs(production["total_mass_kg"] / reference_mass - 1.0) < (
        mass_tolerance
    )


def test_central_sound_speed_and_causality_diagnostic(default_model: dict) -> None:
    expected_squared = (
        DEFAULT_GAMMA
        * DEFAULT_PC
        / (default_model["rhoC"] * phys.C2)
    )
    assert default_model["central_sound_speed_squared_over_c2"] == pytest.approx(
        expected_squared
    )
    assert default_model["central_sound_speed_over_c"] == pytest.approx(
        math.sqrt(expected_squared)
    )
    assert default_model["causality_satisfied"] is True


def test_acausal_toy_model_is_flagged_without_hiding_result() -> None:
    data = driver.compute_neutron_star(DEFAULT_GAMMA, 1.0e37, DEFAULT_K)
    assert data["central_sound_speed_over_c"] > 1.0
    assert data["causality_satisfied"] is False
    assert data["surface_radius_m"] > 0.0


def test_default_eos_sequence_has_a_mass_turnover() -> None:
    low = driver.compute_neutron_star(DEFAULT_GAMMA, 1.0e34, DEFAULT_K)
    middle = driver.compute_neutron_star(DEFAULT_GAMMA, 1.0e35, DEFAULT_K)
    high = driver.compute_neutron_star(DEFAULT_GAMMA, 1.0e36, DEFAULT_K)
    assert middle["total_mass_solar"] > low["total_mass_solar"]
    assert middle["total_mass_solar"] > high["total_mass_solar"]


def test_simple_polytrope_exhibits_artificial_low_mass_branch() -> None:
    data = driver.compute_neutron_star(DEFAULT_GAMMA, 1.0e25, DEFAULT_K)
    assert data["total_mass_solar"] < 0.02


# ---------------------------------------------------------------------------
# Plotting and textual summary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output_type", "expected_ylabel"),
    [
        ("Pressure", "Pressure (Pa)"),
        ("Density", "Density (kg/m$^3$)"),
        ("Mass", "Enclosed mass ($M_\\odot$)"),
    ],
)
def test_plot_modes(
    default_model: dict,
    output_type: str,
    expected_ylabel: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    plotter.plot_neutron(default_model, output_type)
    ax = plt.gcf().axes[0]
    assert ax.get_xlabel() == "Radius (km)"
    assert ax.get_ylabel() == expected_ylabel
    assert ax.get_title() == f"Neutron Star: {output_type}"
    assert len(ax.lines[0].get_xdata()) == len(default_model["radius"])
    plt.close("all")


def test_log_plot_omits_zero_surface_point(
    default_model: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    plotter.plot_neutron(default_model, "Pressure", log_y=True)
    ax = plt.gcf().axes[0]
    assert ax.get_yscale() == "log"
    assert len(ax.lines[0].get_ydata()) == len(default_model["pressure"]) - 1
    assert all(value > 0.0 for value in ax.lines[0].get_ydata())
    plt.close("all")


@pytest.mark.parametrize("output_type", ["", "Energy", None])
def test_plot_rejects_invalid_output_type(
    default_model: dict, output_type: str
) -> None:
    with pytest.raises(ValueError, match="output_type"):
        plotter.plot_neutron(default_model, output_type)


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("pressure", "Pressure"), (" DENSITY ", "Density"), ("mAsS", "Mass")],
)
def test_plot_accepts_case_insensitive_student_input(
    default_model: dict,
    alias: str,
    canonical: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    plotter.plot_neutron(default_model, alias)
    assert plt.gcf().axes[0].get_title() == f"Neutron Star: {canonical}"
    plt.close("all")


def test_mass_log_request_warns_and_remains_linear(
    default_model: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    with pytest.warns(UserWarning, match="Mass profile will use a linear"):
        plotter.plot_neutron(default_model, "Mass", log_y=True)
    assert plt.gcf().axes[0].get_yscale() == "linear"
    plt.close("all")


@pytest.mark.parametrize("log_y", [0, 1, "yes", None])
def test_plot_rejects_nonboolean_log_setting(
    default_model: dict, log_y: bool
) -> None:
    with pytest.raises(ValueError, match="log_y"):
        plotter.plot_neutron(default_model, "Pressure", log_y=log_y)


def test_model_summary_reports_global_and_validity_fields(
    default_model: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    plotter.print_model_summary(default_model)
    output = capsys.readouterr().out
    for phrase in (
        "surface radius",
        "total mass",
        "central pressure",
        "central density",
        "central sound",
        "compactness",
        "causality check",
        "Buchdahl check",
        "radial samples",
        "Radial checkpoints",
    ):
        assert phrase in output


# ---------------------------------------------------------------------------
# Student Help content and exercise regression checks
# ---------------------------------------------------------------------------


def test_help_documents_defaults_and_output_modes() -> None:
    html = _help_text()
    for text in (
        "1.666667",
        "1.26e35",
        "5.3802e3",
        "steps_per_scale",
        "200000",
        '"Pressure"',
        '"Density"',
        '"Mass"',
        "log_y",
    ):
        assert text in html


def test_help_documents_correct_tov_terms_and_mass_equation() -> None:
    html = _help_text()
    assert r"\rho+\dfrac{p}{c^2}" in html
    assert r"m+\dfrac{4\pi r^3p}{c^2}" in html
    assert r"\frac{dm}{dr} = 4\pi r^2 \rho(r)" in html
    assert r"2GM}{Rc^2} \le \frac{8}{9}" in html
    assert "not a separately tracked baryonic or rest mass" in html


def test_help_documents_surface_variable_and_causality_diagnostic() -> None:
    html = _help_text()
    assert r"p^{(\gamma-1)/\gamma}" in html
    assert "central sound speed" in html.lower()
    assert "causality" in html.lower()


def test_help_explains_gamma_one_restriction_and_plot_normalization() -> None:
    html = _help_text()
    assert "gamma must be finite and greater than 1." in html
    assert "pressure only approaches zero asymptotically" in html
    assert "protects the model's physical and numerical definition" in html
    assert "Capitalization and surrounding spaces are ignored" in html
    assert "prints a warning" in html


def test_help_contains_no_development_history_commentary() -> None:
    lowered = _help_text().lower()
    forbidden = (
        "historical implementation",
        "original java source omitted",
        "old numerical results",
        "forward-euler march",
        "fixed 2000-element",
        "resolution-degrading step doubling",
        "earlier release",
        "previous release",
        "previous version",
    )
    for phrase in forbidden:
        assert phrase not in lowered


def test_help_tags_the_equation_of_state_as_closure() -> None:
    """The polytropic equation of state (Eq. 3) is a physically motivated but
    chosen relation, not a quantity defined purely by a formula, so it carries
    a distinct CLOSURE tag rather than DEFINITION (Audit21 Codex #4)."""
    html = _help_text()
    assert 'Eq. 3 &mdash; Polytropic equation of state <span class="kind kind-clo">CLOSURE</span>' in html
    assert '<tr><td>Eq. 3</td><td><span class="kind kind-clo">CLOSURE</span></td>' in html
    assert "kind-clo" in html
    # The unnumbered Newtonian comparison equation is never integrated by the
    # program, so it must not carry the ODE tag (which the legend defines as
    # "a differential equation the program integrates").
    assert 'Newtonian hydrostatic equilibrium (for later comparison, unnumbered, untagged' in html


def test_help_beat3_states_the_correct_radius_ratio() -> None:
    """16.375 m / 1 m = 16.375, not 'ten times closer' (Audit21 Codex #7)."""
    html = _help_text()
    assert "16.375 times closer" in html
    assert "ten times closer" not in html.lower()


def test_help_beat4_quotes_digits_the_console_actually_prints() -> None:
    """Beat 4's headline numbers must match what the console's .3f/.6f
    formatting shows, with the extra precision clearly attributed to a
    script that prints the result dictionary directly (Audit21 Grok item 1)."""
    html = _help_text()
    assert "7.804, 7.804, 7.804 and 7.803 km" in html
    assert "0.962591 solar masses all four times" in html
    assert "convergence_check.py" in html
    assert 'd["surface_radius_km"]' in html or "surface_radius_km" in html


def test_help_distinguishes_rk4_local_and_global_error_order() -> None:
    """A single RK4 step's local error is O(h^5); the accumulated error over
    a fixed interval is O(h^4) (Audit21 Codex #6)."""
    html = _help_text()
    assert "local truncation error" in html
    assert "global error over that span" in html
    assert "factor of about 32" in html
    assert "factor of about 16 per halving" in html


def test_help_does_not_generalize_the_weak_field_ranking() -> None:
    """The weak-field state's ranking (compactness >> pressure corrections)
    is specific to that hand-picked, low-pressure state; the default star's
    interior shows the opposite ranking close to the center, reversing at
    r/R~0.287 and r/R~0.420 (Audit22 Codex #1; radii pinned down precisely
    in Audit23 Codex #1, which found compactness dominant from there out to
    the surface -- i.e. through most of the star's volume, not the pressure
    corrections)."""
    html = _help_text()
    assert "typically the largest of the three corrections" not in html
    # The interior factors at r=16.375 m and r~1637.5 m must be shown so the
    # reversed ranking is demonstrated from the actual model, not asserted.
    assert "0.2113" in html and "0.6340" in html
    assert "0.10076" in html and "0.53470" in html
    assert "ranking reverses" in html or "the ranking reverses" in html


def test_help_does_not_claim_the_sweep_measures_rk4_global_order() -> None:
    """The steps_per_scale sweep's radius/mass errors do not shrink by the
    textbook O(h^4) factor of 16 per halving -- that convergence is masked
    by the lower-order surface-location estimate near the star's edge
    (Audit22 Codex #2)."""
    html = _help_text()
    assert (
        "steps_per_scale</code> sweep above or Beat 5's error table actually "
        "show" in html
    )
    # The actually observed, non-uniform ratios must be stated explicitly.
    assert "1.5, 1.0 and 2.1" in html
    assert "4.2, 4.4 and 4.2" in html


def test_help_describes_surface_estimate_as_extrapolation_not_bracket() -> None:
    """The usual surface estimate extrapolates from the last positive-
    pressure sample using its local slope; it does not bracket the surface
    between two already-stored samples (Audit22 Codex #3)."""
    html = _help_text()
    assert "extrapolates forward from the last positive-pressure sample" in html
    assert "generally falls between two stored radial samples" not in html
    assert "estimates where between the last two samples" not in html
    # The independent q-reference is a well-converged numerical estimate,
    # not an exact q=0 crossing (reworded in Audit23 to "fixed benchmark"
    # once its own provenance and grid-doubling precision were spelled out;
    # check the substance rather than one exact phrasing of it).
    assert "well-converged" in html
    assert "not the exact physical surface" in html
    assert "stops when \\(q\\) crosses zero" not in html


def test_help_labels_the_step_doubling_agreement_as_relative() -> None:
    """The 1.4e-13 / 9.2e-13 step-doubling figures are relative differences,
    not absolute differences in a physical quantity (Audit22 Codex #4)."""
    html = _help_text()
    assert "agree to a relative 1.4" in html


def test_help_does_not_claim_the_center_is_newtonian() -> None:
    """Compactness vanishing at the center does not make the classical
    equation exact there: the TOV/Newtonian gradient ratio tends to
    (1+eta)(1+3eta) as r->0, not to 1, where eta=p_C/(rho_C c^2). The page
    must not claim the classical equation is valid 'very close to the
    center' of the default (high-central-pressure) model (Audit24 Codex #1).
    """
    html = _help_text()
    assert (
        "not what the program solves anywhere but very close to the "
        "center" not in html
    )
    assert "1.979374" in html
    assert "(1+\\eta)(1+3\\eta)" in html or "1+\\eta" in html
    assert "eta" in html.lower() or "\\eta" in html


def test_help_scopes_the_interior_ranking_to_where_it_holds() -> None:
    """The pressure corrections dominate only close to the center; compactness
    overtakes each of them in turn and stays largest out to the surface, where
    the pressure corrections (not compactness) fall to zero. The old claim
    that 'none of the three corrections is close to negligible' at the
    surface contradicted this and is gone (Audit23 Codex #1)."""
    html = _help_text()
    assert "none of the three corrections is close to negligible" not in html
    assert "r/R" in html and "0.287" in html and "0.420" in html
    assert "0.3700" in html and "0.1091" in html and "0.2092" in html
    # The toy equation of state's limitations are already emphasized
    # elsewhere on this page; the interior demonstration should not imply
    # this ranking is a fact about real neutron-star matter.
    assert "real stellar interior" not in html
    assert "toy-star model" in html


def test_help_calls_the_surface_extrapolation_second_order_not_first() -> None:
    """A tangent-line estimate of a smooth function's zero is second-order
    accurate in the remaining distance, even though the extrapolation
    formula itself is linear; the page must not call it first-order, and
    must not claim sole proven causation for the observed convergence rates
    without a controlled comparison (Audit23 Codex #2)."""
    html = _help_text()
    assert "linear (first-order) extrapolation" not in html
    assert "second-order accurate" in html or "second-order" in html
    assert "would need its own controlled comparison" in html


def test_help_states_the_surface_reference_constants_provenance() -> None:
    """REF_R is a fixed benchmark checked against the production solver's
    own regression, not something a literal steps_per_scale=8000 run of the
    shown reference integrator reproduces bit-for-bit; the page should say
    so, name the executable test that checks that specific gap
    (Audit24 Codex #2), and quantify the reference's own grid-doubling
    convergence instead of implying the printed digits are exact
    (Audit23 Codex #3)."""
    html = _help_text()
    assert "REFERENCE_RADIUS_M" in html
    assert "bit-for-bit" in html
    assert "test_q_reference_model_matches_default_reference_constant" in html
    assert "0.99, 0.26 and 0.065 micrometres" in html
    # Audit24 Codex #2: the constant is checked against production output,
    # not described as the q-reference integrator's own self-check.
    assert "reference integrator's own regression check" not in html


def test_help_exercises_are_numbered_in_increasing_difficulty() -> None:
    """The Beats file lists experiments as ``experiment-card`` blocks rather
    than the classic Reference Guide's numbered ``<h3>`` headings; check the
    same kind of thing (an unbroken, increasing sequence with the expected
    early/late titles) against that markup instead."""
    html = _help_text()
    cards = re.findall(
        r'<div class="experiment-card" id="exp(\d+)">\s*'
        r'<div class="exp-num">Experiment \1[^<]*</div>\s*'
        r'<div class="exp-title">([^<]+)</div>',
        html,
    )
    assert [int(number) for number, _ in cards] == list(range(1, 9))
    titles = [title for _, title in cards]
    assert titles[0].startswith("Profile shapes")
    assert "Numerical convergence" in titles[1]
    assert "maximum mass" in titles[3]
    assert "Buchdahl" in titles[4]
    assert "minimum-mass puzzle" in titles[-1].lower()


def test_help_preserves_textbook_minimum_mass_question() -> None:
    html = _help_text()
    assert "0.02" in html
    assert "Investigation 20.1" in html
    assert "low-density" in html


def test_help_has_restoration_and_repository_links() -> None:
    html = _help_text()
    assert "Restoring Default Parameter Values" in html
    assert "tomyee3-org/GFTGU-Programs" in html
    assert "tomyee3-org/GFTGU-Documentation" in html
