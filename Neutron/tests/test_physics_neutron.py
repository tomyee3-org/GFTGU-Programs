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
HELP_FILENAME = "Neutron.html"


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

# Independent high-accuracy reference obtained by integrating
# q=p**((gamma-1)/gamma) with a terminal q=0 event. These values are not copied
# from the production solver and protect the surface-location regression.
REFERENCE_RADIUS_M = 7802.758706219183
REFERENCE_MASS_KG = 1.9140826174028036e30


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


def _help_text() -> str:
    return (MODULE_DIR / HELP_FILENAME).read_text(encoding="utf-8")


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
        (DEFAULT_GAMMA, DEFAULT_PC, DEFAULT_K, 50.0, 200_000),
        (DEFAULT_GAMMA, DEFAULT_PC, DEFAULT_K, True, 200_000),
        (DEFAULT_GAMMA, DEFAULT_PC, DEFAULT_K, 400, 99),
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


@pytest.mark.parametrize("output_type", ["pressure", "", "Energy", None])
def test_plot_rejects_invalid_output_type(
    default_model: dict, output_type: str
) -> None:
    with pytest.raises(ValueError, match="output_type"):
        plotter.plot_neutron(default_model, output_type)


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
        "central density",
        "central sound",
        "compactness",
        "causality check",
        "Buchdahl check",
        "radial samples",
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
    assert r"\frac{dm}{dr}=4\pi r^2\rho" in html
    assert r"2GM/(Rc^2)\le8/9" in html
    assert "does not separately compute baryonic (rest) mass" in html


def test_help_documents_surface_variable_and_causality_diagnostic() -> None:
    html = _help_text()
    assert r"p^{(\gamma-1)/\gamma}" in html
    assert "central sound speed" in html.lower()
    assert "causality" in html.lower()


def test_help_contains_no_development_history_commentary() -> None:
    lowered = _help_text().lower()
    forbidden = (
        "historical implementation",
        "original java source omitted",
        "old numerical results",
        "forward-euler march",
        "fixed 2000-element",
        "resolution-degrading step doubling",
    )
    for phrase in forbidden:
        assert phrase not in lowered


def test_help_exercises_are_numbered_in_increasing_difficulty() -> None:
    html = _help_text()
    headings = re.findall(r"<h3>(\d+)\.\s*([^<]+)</h3>", html)
    assert [int(number) for number, _ in headings] == list(range(1, 9))
    titles = [title for _, title in headings]
    assert titles[0].startswith("Profile Shapes")
    assert "Numerical Convergence" in titles[1]
    assert "Mass–Central-Pressure Sequence" in titles[3]
    assert "Minimum-Mass" in titles[4]
    assert "Limits of a Single Polytrope" in titles[-1]


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
