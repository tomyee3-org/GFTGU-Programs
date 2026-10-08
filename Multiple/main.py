"""Multiple: Newtonian N-body gravity in three dimensions.

Run ``python main.py --help`` for defaults and examples. Inputs:
  --n_bodies: number of bodies (at least two).
  --masses_solar: comma-separated positive masses in solar units.
  --positions_init: semicolon-separated x,y,z triples in metres; quote the
      complete argument in your shell, e.g. "1e10,0,0;-1e10,0,0" (double
      quotes work in both cmd.exe and POSIX shells; a single-quoted
      argument is passed through literally by cmd.exe and will not parse).
  --velocities_init: matching x,y,z triples of velocities [m/s].
  --dt: largest integration step [s]; --max_steps: accepted-step ceiling.
  --eps1: per-body predictor acceleration-change threshold (0 to 1).
  --eps2: corrector velocity-increment threshold (0 to eps1).
  --output_type: trajectories (complete static paths) or animation
      (uniformly timed playback frames; default).
  --animation_mode: trails (recent path) or current_positions (markers only).
  --frame_time: simulated seconds between frames; --frame_interval_ms:
      real milliseconds per frame; --trail_time: recent trail duration [s].
  --projection: xy, xz, yz plane; --axis_mode: fixed full-frame limits or
      auto zoom tracking the visible positions (animation only).
  --display_frame: com (recenter plotted positions at their instantaneous
      center of mass) or user (plot the input inertial coordinates).
  --show_energy_diagnostic: add an energy drift graph (trajectories only).
  --test_positions_init, --test_velocities_init: optional massless test
      particles (x,y,z triples, as for the massive bodies). Test particles
      feel the massive bodies but pull on nothing.
  --test_ring: "R_MIN,R_MAX,N" places test particles on circular orbits at N
      radii evenly spaced from R_MIN to R_MAX metres in the x-y plane.
  --test_ring_per_radius: particles at each ring radius (default 1).
  --test_phases: golden (each particle 137.5 degrees round from the last) or
      random (uniform starting angles from --test_seed, reproducible).
  --test_center: com or a body number; the centre of --test_ring and the
      reference for each test particle's starting distance, a and e.
  --removal_radii: one radius per massive body [m]; a test particle that
      comes this close is removed as an encounter (at t = 0 if it starts
      that close).
  --escape_radius: a test particle farther than this from the massive
      bodies' centre of mass is removed as escaped [m].
  --output_type survival: no plot of paths; record when each test particle
      is removed and plot survival time against starting distance.
  --test_csv: write each test particle's starting orbit and fate to a CSV.

Console totals use input inertial coordinates in both display frames and
describe the massive bodies only. Eleven
locally quadratically interpolated sets of E, E_internal, K, P and L span the run.
"""

import argparse
import csv
import math
import sys

import numpy as np

import physics_multiple as phys
from driver_multiple import SimulationParams, run_simulation
from plot_multiple import (animate_multiple, plot_energy_drift, plot_survival,
                           plot_survivor_positions, plot_trajectories)

AU_M = 1.495978707e11          # astronomical unit [m]
YEAR_S = 365.25 * 86400.0      # Julian year [s]

DEFAULTS = {
    "n_bodies": 3,
    "masses_solar": [1.0, 1.0, 1.0],
    "positions_init": [[4.6e10, 0., 0.], [-4.6e10, 0., 0.], [0., 4.6e10, 0.]],
    "velocities_init": [[0., 0., 0.], [0., -30000., 0.], [-30000., 0., 0.]],
    "dt": 2000., "max_steps": 60000, "eps1": .005, "eps2": 1e-7,
    "output_type": "animation", "animation_mode": "trails", "frame_time": 2e5,
    "frame_interval_ms": 50, "trail_time": 6e5, "projection": "xy",
    "axis_mode": "fixed", "display_frame": "com",
    "show_energy_diagnostic": False,
    "test_positions_init": None, "test_velocities_init": None,
    "test_ring": None, "test_ring_per_radius": 1, "test_phases": "golden",
    "test_seed": 0, "test_center": "com", "removal_radii": None,
    "escape_radius": None, "test_csv": None,
}


def finite_float(text):
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("enter a finite number") from error
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("enter a finite number")
    return value


def positive_float(text):
    value = finite_float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("enter a number greater than zero")
    return value


def nonnegative_float(text):
    value = finite_float(text)
    if value < 0:
        raise argparse.ArgumentTypeError("enter a non-negative number")
    return value


def fraction(text):
    value = finite_float(text)
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError("enter a number strictly between 0 and 1")
    return value


def positive_int(text):
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("enter a positive integer") from error
    if value < 1:
        raise argparse.ArgumentTypeError("enter a positive integer")
    return value


def masses(text):
    try:
        result = [positive_float(x.strip()) for x in text.split(",")]
    except argparse.ArgumentTypeError as error:
        raise argparse.ArgumentTypeError("masses_solar must list positive finite masses") from error
    if len(result) < 2:
        raise argparse.ArgumentTypeError("masses_solar needs at least two masses")
    return result


def triples(text):
    try:
        result = [[finite_float(value.strip()) for value in item.split(",")]
                  for item in text.split(";")]
    except argparse.ArgumentTypeError as error:
        raise argparse.ArgumentTypeError("enter semicolon-separated finite x,y,z triples") from error
    if any(len(row) != 3 for row in result):
        raise argparse.ArgumentTypeError("each body needs exactly three x,y,z components")
    return result


def nonnegative_int(text):
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("enter a non-negative integer") from error
    if value < 0:
        raise argparse.ArgumentTypeError("enter a non-negative integer")
    return value


def nonnegative_list(text):
    try:
        result = [nonnegative_float(x.strip()) for x in text.split(",")]
    except argparse.ArgumentTypeError as error:
        raise argparse.ArgumentTypeError("removal_radii must list non-negative finite radii") from error
    return result


def ring(text):
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("test_ring needs R_MIN,R_MAX,N")
    r_min, r_max = positive_float(parts[0]), positive_float(parts[1])
    count = positive_int(parts[2])
    if r_max < r_min:
        raise argparse.ArgumentTypeError("test_ring needs R_MAX >= R_MIN")
    return r_min, r_max, count


def center(text):
    if text.lower() == "com":
        return "com"
    try:
        return positive_int(text)
    except argparse.ArgumentTypeError as error:
        raise argparse.ArgumentTypeError("test_center must be com or a body number") from error


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="Multiple", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Simulate Newtonian N-body motion; choose initial states, numerical controls and plot options.",
    )
    p.add_argument("--version", action="version",
                   version=f"Multiple {phys.MODEL_VERSION} (build {phys.BUILD_ID})")
    p.add_argument("--n_bodies", type=positive_int, default=DEFAULTS["n_bodies"],
                   help="number of gravitating point masses, at least two")
    p.add_argument("--masses_solar", type=masses, default=DEFAULTS["masses_solar"],
                   metavar="M1,M2,...", help="one positive solar mass per body")
    p.add_argument("--positions_init", type=triples, default=DEFAULTS["positions_init"],
                   metavar='"X,Y,Z;X,Y,Z;..."', help="one Cartesian position triple per body [m]; quote semicolons with double quotes")
    p.add_argument("--velocities_init", type=triples, default=DEFAULTS["velocities_init"],
                   metavar='"VX,VY,VZ;VX,VY,VZ;..."', help="one velocity triple per body [m/s]; quote semicolons with double quotes")
    for name, value_type, description in (
        ("dt", positive_float, "maximum adaptive integration timestep [s]"),
        ("max_steps", positive_int, "maximum accepted integration steps"),
        ("eps1", fraction, "per-body acceleration-change threshold (0 to 1)"),
        ("eps2", fraction, "corrector velocity-increment threshold (0 to eps1)"),
        ("frame_time", positive_float, "simulated seconds between animation frames"),
        ("frame_interval_ms", positive_int, "real milliseconds between animation frames"),
        ("trail_time", nonnegative_float, "recent trajectory duration shown in animation [s]"),
    ):
        p.add_argument(f"--{name}", type=value_type, default=DEFAULTS[name], help=description)
    p.add_argument("--output_type", choices=("animation", "trajectories", "survival"),
                   default=DEFAULTS["output_type"],
                   help="animation: playback at uniform simulated times; trajectories: full static paths; "
                        "survival: test-particle survival times only")
    p.add_argument("--animation_mode", choices=("trails", "current_positions"),
                   default=DEFAULTS["animation_mode"],
                   help="trails: show recent motion; current_positions: markers only")
    p.add_argument("--projection", choices=("xy", "xz", "yz"),
                   default=DEFAULTS["projection"], help="two spatial axes to display")
    p.add_argument("--axis_mode", choices=("fixed", "auto"),
                   default=DEFAULTS["axis_mode"],
                   help="fixed: full animation limits; auto: follow visible bodies and trails")
    p.add_argument("--display_frame", choices=("com", "user"),
                   default=DEFAULTS["display_frame"],
                   help="com: centered plot; user: input coordinates (printed totals always use user frame)")
    p.add_argument("--show_energy_diagnostic", action=argparse.BooleanOptionalAction,
                   default=DEFAULTS["show_energy_diagnostic"],
                   help="also show energy drift graph when output_type=trajectories")
    p.add_argument("--test_positions_init", type=triples, default=DEFAULTS["test_positions_init"],
                   metavar='"X,Y,Z;X,Y,Z;..."', help="massless test-particle positions [m]")
    p.add_argument("--test_velocities_init", type=triples, default=DEFAULTS["test_velocities_init"],
                   metavar='"VX,VY,VZ;VX,VY,VZ;..."', help="matching test-particle velocities [m/s]")
    p.add_argument("--test_ring", type=ring, default=DEFAULTS["test_ring"], metavar="R_MIN,R_MAX,N",
                   help="circular x-y test-particle orbits at N radii from R_MIN to R_MAX [m] about --test_center")
    p.add_argument("--test_ring_per_radius", type=positive_int,
                   default=DEFAULTS["test_ring_per_radius"],
                   help="number of test particles at each --test_ring radius")
    p.add_argument("--test_phases", choices=("golden", "random"), default=DEFAULTS["test_phases"],
                   help="starting angles round the ring: golden (137.5 degrees apart) or random")
    p.add_argument("--test_seed", type=nonnegative_int, default=DEFAULTS["test_seed"],
                   help="seed for --test_phases random; the same seed gives the same angles")
    p.add_argument("--test_center", type=center, default=DEFAULTS["test_center"], metavar="com|BODY",
                   help="centre of --test_ring and reference for each test particle's starting orbit")
    p.add_argument("--removal_radii", type=nonnegative_list, default=DEFAULTS["removal_radii"],
                   metavar="R1,R2,...", help="remove a test particle this close to each massive body [m]")
    p.add_argument("--escape_radius", type=positive_float, default=DEFAULTS["escape_radius"],
                   help="remove a test particle this far from the massive bodies' centre of mass [m]")
    p.add_argument("--test_csv", default=DEFAULTS["test_csv"], metavar="PATH",
                   help="write each test particle's starting orbit, angle path and fate to this CSV file")
    # Bind negative exponent-form scalars and vectors with a negative first
    # component so argparse validates values rather than treating them as flags.
    numeric = {"n_bodies", "dt", "max_steps", "eps1", "eps2", "frame_time",
               "frame_interval_ms", "trail_time", "escape_radius"}
    vector_options = {"masses_solar", "positions_init", "velocities_init",
                      "test_positions_init", "test_velocities_init",
                      "test_ring", "removal_radii"}
    raw = list(sys.argv[1:] if argv is None else argv)
    normalized = []
    index = 0
    while index < len(raw):
        word = raw[index]
        if (word.startswith("--") and word[2:] in numeric | vector_options
                and index + 1 < len(raw)
                and raw[index + 1].startswith("-")
                and not raw[index + 1].startswith("--")):
            if word[2:] in vector_options:
                normalized.append(f"{word}={raw[index + 1]}")
                index += 2
                continue
            try:
                float(raw[index + 1])
            except ValueError:
                pass
            else:
                normalized.append(f"{word}={raw[index + 1]}")
                index += 2
                continue
        normalized.append(word)
        index += 1
    args = p.parse_args(normalized)
    if args.n_bodies < 2:
        p.error("--n_bodies must be at least two")
    for name in ("masses_solar", "positions_init", "velocities_init"):
        if len(getattr(args, name)) != args.n_bodies:
            p.error(f"--{name} must contain exactly --n_bodies entries")
    if args.eps2 >= args.eps1:
        p.error("--eps2 must be smaller than --eps1")
    if args.show_energy_diagnostic and args.output_type != "trajectories":
        p.error("--show_energy_diagnostic requires --output_type trajectories")
    explicit = (args.test_positions_init, args.test_velocities_init)
    if (explicit[0] is None) != (explicit[1] is None):
        p.error("--test_positions_init and --test_velocities_init must be given together")
    if explicit[0] is not None and len(explicit[0]) != len(explicit[1]):
        p.error("--test_velocities_init needs one triple per test particle")
    if explicit[0] is not None and args.test_ring is not None:
        p.error("use either --test_ring or --test_positions_init, not both")
    if args.test_ring is None and (args.test_ring_per_radius != 1
                                   or args.test_phases != "golden"):
        p.error("--test_ring_per_radius and --test_phases need --test_ring")
    if args.test_center != "com" and args.test_center > args.n_bodies:
        p.error("--test_center must be com or a body number from 1 to --n_bodies")
    if args.removal_radii is not None and len(args.removal_radii) != args.n_bodies:
        p.error("--removal_radii must contain exactly --n_bodies entries")
    if args.output_type == "survival" and explicit[0] is None and args.test_ring is None:
        p.error("--output_type survival needs --test_ring or --test_positions_init")
    return args


def test_particle_initial_states(args):
    """Return explicit or ring-generated test-particle states, or (None, None)."""
    if args.test_ring is None:
        return args.test_positions_init, args.test_velocities_init
    r_min, r_max, count = args.test_ring
    per_radius = args.test_ring_per_radius
    radii = [r_min + (r_max - r_min) * k / (count - 1) if count > 1 else r_min
             for k in range(count) for _ in range(per_radius)]
    if args.test_phases == "random":
        rng = np.random.default_rng(args.test_seed)
        phases = rng.uniform(0.0, 2.0 * math.pi, size=len(radii))
    else:
        phases = None   # k times the golden angle
    if args.test_center == "com":
        position = phys.center_of_mass(args.positions_init, args.masses_solar)
        velocity = phys.center_of_mass_velocity(args.velocities_init, args.masses_solar)
        mass = sum(args.masses_solar)
    else:
        body = args.test_center - 1
        position = args.positions_init[body]
        velocity = args.velocities_init[body]
        mass = args.masses_solar[body]
    positions, velocities = phys.ring_test_particle_states(
        radii, mass, position, velocity, phases_rad=phases)
    return positions.tolist(), velocities.tolist()


def _fate_text(fate, body):
    if fate == "encounter":
        return f"encounter (body {body})"
    return fate


def fewest_steps_per_orbit(result, dt):
    """Return the smallest starting orbital period divided by dt, or None."""
    info = result["test_particles"]
    mu = phys.GM_SUN * info["center_mass_solar"]
    # Particles removed at the start are never integrated, so they do not count.
    periods = [2.0 * math.pi * math.sqrt(a ** 3 / mu)
               for a, t in zip(info["initial_semi_major_axis_m"], info["removal_time_s"])
               if math.isfinite(a) and t != 0.0]
    return min(periods) / dt if periods else None


def print_test_particle_table(result, dt=None):
    """Print each test particle's starting orbit and fate."""
    info = result["test_particles"]
    fates = info["fate"]
    counts = {name: fates.count(name) for name in ("survived", "escaped", "encounter", "numerical")}
    at_start = sum(1 for t in info["removal_time_s"] if t == 0.0)
    paths = [p for p, f in zip(info["phase_motion"], fates) if f == "survived"]
    print(f"Test particles: {result['n_test_particles']}; survived {counts['survived']}; "
          f"escaped {counts['escaped']}; encounter {counts['encounter']}"
          + (f"; numerical {counts['numerical']}" if counts["numerical"] else "")
          + (f" (removed at the start: {at_start})" if at_start else ""))
    print("Angle paths of the survivors: " + "; ".join(
        f"{name} {paths.count(name)}" for name in
        ("tadpole L4", "tadpole L5", "horseshoe", "librating", "circulating",
         "passed body", "unfinished") if paths.count(name)))
    center = "the massive bodies' centre of mass" if info["center"] == "com" else info["center"]
    print(f"Starting orbits are measured from {center} "
          f"(point mass {info['center_mass_solar']:.6g} solar masses); "
          f"phi0 is the starting angle from body {info['phase_reference_body']}.")
    print("   #   r0 [AU]   a0 [AU]       e0  phi0 [deg]  phi range [deg]  angle path   "
          "fate                 t_removed [yr]")
    for index in range(result["n_test_particles"]):
        a0 = info["initial_semi_major_axis_m"][index]
        if math.isfinite(a0):
            a_text = f"{a0 / AU_M:9.4f}"
        else:   # inf: unbound; NaN: undefined (particle exactly at the centre)
            a_text = "  unbound" if a0 == math.inf else "        -"
        e0 = info["initial_eccentricity"][index]
        time = info["removal_time_s"][index]
        t_text = f"{time / YEAR_S:.4e}" if math.isfinite(time) else "-"
        e_text = f"{e0:8.5f}" if math.isfinite(e0) else "       -"
        print(f"{index + 1:4d} {info['initial_distance_m'][index] / AU_M:9.4f} {a_text} "
              f"{e_text}  "
              f"{info['initial_phase_deg'][index]:10.2f}  "
              f"{info['phase_min_deg'][index]:7.1f}..{info['phase_max_deg'][index]:<7.1f} "
              f"{info['phase_motion'][index]:12s} "
              f"{_fate_text(fates[index], info['removal_body'][index]):20s} {t_text}")
    if dt is not None:
        steps = fewest_steps_per_orbit(result, dt)
        if steps is not None:
            print(f"Fewest dt steps per starting test-particle orbit: {steps:.1f}")
            if steps < 100.0:
                print("Warning: fewer than 100 dt steps in a starting test-particle orbit. "
                      "Particles shorten their own steps only near close approaches, so "
                      "their ordinary orbits may be inaccurate; reduce --dt.")


def write_test_particle_csv(result, path):
    """Write starting states, starting orbits and fates in SI units."""
    info = result["test_particles"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["particle", "x0_m", "y0_m", "z0_m", "vx0_m_s", "vy0_m_s", "vz0_m_s",
                         "r0_m", "a0_m", "e0", "phi0_deg", "phi_min_deg", "phi_max_deg",
                         "angle_path", "fate", "removal_body",
                         "removal_time_s"])
        for index in range(result["n_test_particles"]):
            time = info["removal_time_s"][index]
            writer.writerow([index + 1,
                             *(repr(float(v)) for v in info["initial_positions"][index]),
                             *(repr(float(v)) for v in info["initial_velocities"][index]),
                             repr(float(info["initial_distance_m"][index])),
                             repr(float(info["initial_semi_major_axis_m"][index])),
                             repr(float(info["initial_eccentricity"][index])),
                             repr(float(info["initial_phase_deg"][index])),
                             repr(float(info["phase_min_deg"][index])),
                             repr(float(info["phase_max_deg"][index])),
                             info["phase_motion"][index],
                             info["fate"][index], int(info["removal_body"][index]),
                             repr(float(time)) if math.isfinite(time) else ""])


def five(value):
    return f"{float(value):.4e}"


def _format_conservation_totals(label: str, state: dict, masses_solar) -> str:
    """Format user-frame E, P and L, with boost-independent internal energy."""
    energy = float(state["energy"])
    momentum = state["momentum"]
    angular = state["angular_momentum"]
    total_mass = float(sum(masses_solar))
    p_squared = sum(float(part) ** 2 for part in momentum)
    internal = energy - p_squared / (2.0 * total_mass)
    return (
        f"{label} user-frame totals (scaled by one solar mass): "
        f"E={five(energy)} m^2/s^2; "
        f"E_internal={five(internal)} m^2/s^2; "
        f"P=({', '.join(five(part) for part in momentum)}) m/s; "
        f"L=({', '.join(five(part) for part in angular)}) m^2/s"
    )


def print_conservation_samples(result):
    """Print aligned conservation samples, reserving a sign in each value."""
    rows = []
    for sample in result["conservation_samples"]:
        scalar_values = (sample["time"] / 86400, sample["energy"],
                         sample["internal_energy"], sample["kinetic_energy"])
        rows.append((f"{sample['fraction']:.1f}",
                     [f"{float(value): .4e}" for value in scalar_values],
                     sample["momentum"], sample["angular_momentum"]))

    # Widen columns for three-digit exponents rather than shifting later
    # columns. Ordinary positive and negative values occupy equal widths.
    scalar_widths = [max([12] + [len(row[1][i]) for row in rows])
                     for i in range(4)]
    component_width = max([11] + [len(f"{float(value): .4e}")
                                 for row in rows for vector in row[2:]
                                 for value in vector])

    def vector_text(values):
        return "(" + ", ".join(f"{float(value): {component_width}.4e}"
                               for value in values) + ")"

    vector_width = 3 * component_width + 6
    widths = [8, *scalar_widths, vector_width, vector_width]
    headings = ("Fraction", "t", "E", "E_internal", "K", "P", "L")
    print("Conservation over ten equal time intervals (user coordinates, scaled by one solar mass):")
    print("t [days]; E, E_internal, K [m^2/s^2]; P [m/s]; L [m^2/s]")
    print(" ".join(label.ljust(width) for label, width in zip(headings, widths)).rstrip())
    for fraction, scalars, momentum, angular in rows:
        cells = [fraction.ljust(widths[0]),
                 *(value.rjust(width) for value, width in zip(scalars, scalar_widths)),
                 vector_text(momentum), vector_text(angular)]
        print(" ".join(cells))


def main(argv=None):
    args = parse_args(argv)
    try:
        test_positions, test_velocities = test_particle_initial_states(args)
    except ValueError as error:
        raise SystemExit(f"Multiple error: {error}") from error
    params = SimulationParams(
        n_bodies=args.n_bodies, masses_solar=args.masses_solar,
        positions_init=args.positions_init, velocities_init=args.velocities_init,
        dt=args.dt, max_steps=args.max_steps, eps1=args.eps1, eps2=args.eps2,
        output_type=args.output_type,
        animation_mode=args.animation_mode.replace("_", " "),
        frame_time=args.frame_time, frame_interval_ms=args.frame_interval_ms,
        trail_time=args.trail_time, projection=args.projection,
        axis_mode=args.axis_mode, display_frame=args.display_frame,
        test_positions_init=test_positions, test_velocities_init=test_velocities,
        removal_radii=args.removal_radii, escape_radius=args.escape_radius,
        test_center=args.test_center,
    )
    try:
        result = run_simulation(params)
    except (ValueError, RuntimeError, OverflowError) as error:
        raise SystemExit(f"Multiple error: {error}") from error

    if result["display_frame"] == "com":
        frame_note = ("Display frame: COM (center of mass). Plots subtract the instantaneous "
                      "center-of-mass position R_CM; printed E, P and L use input coordinates.")
    else:
        frame_note = ("Display frame: user (input coordinates). A large total linear "
                      "momentum makes the plot drift across its axes.")
    print(f"Multiple {result['model_version']} (build {result['build_id']}) — "
          f"accepted steps: {result['accepted_steps']}; "
          f"simulated time: {five(result['final_time'] / 86400)} days")
    print(frame_note)
    print(_format_conservation_totals("Initial", result["initial_conservation"], params.masses_solar))
    print(_format_conservation_totals("Final", result["final_conservation"], params.masses_solar))
    print("Maximum conservation drift: "
          f"energy={five(result['max_fractional_energy_drift'])}, "
          f"momentum={five(result['max_momentum_drift'])}, "
          f"angular momentum={five(result['max_angular_momentum_drift'])}")
    print_conservation_samples(result)
    if result["n_test_particles"]:
        print(f"Simulated time: {five(result['final_time'] / YEAR_S)} yr")
        print_test_particle_table(result, params.dt)
        if args.test_csv:
            try:
                write_test_particle_csv(result, args.test_csv)
            except OSError as error:
                raise SystemExit(f"Multiple error: could not write {args.test_csv}: {error}") from error
            print(f"Test-particle table written to {args.test_csv}")
    elif args.test_csv:
        print("No test particles: --test_csv was ignored.")

    try:
        if result["type"] == "survival":
            plot_survival(result)
            info = result["test_particles"]
            secure = [p for p, f in zip(info["phase_motion"], info["fate"])
                      if f == "survived" and p in ("tadpole L4", "tadpole L5", "circulating")]
            if secure:
                plot_survivor_positions(result)
            else:
                print("No tadpole or circulating survivors: the end-position window is skipped.")
        elif result["type"] == "trajectories":
            plot_trajectories(result, projection=params.projection)
            if args.show_energy_diagnostic:
                plot_energy_drift(result)
        else:
            n_frames = len(result["frame_times"])
            playback_seconds = n_frames * params.frame_interval_ms / 1000.0
            print(f"Animation frames: {n_frames}; "
                  f"frame spacing: {five(params.frame_time)} simulated s; "
                  f"approximate playback time: {five(playback_seconds)} real s")
            animate_multiple(result)
    except (ValueError, RuntimeError, OverflowError) as error:
        raise SystemExit(f"Multiple plot error: {error}") from error


if __name__ == "__main__":
    main()
