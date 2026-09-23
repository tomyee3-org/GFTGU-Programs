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

Console totals use input inertial coordinates in both display frames. Eleven
locally quadratically interpolated sets of E, E_internal, K, P and L span the run.
"""

import argparse
import math
import sys

import physics_multiple as phys
from driver_multiple import SimulationParams, run_simulation
from plot_multiple import animate_multiple, plot_energy_drift, plot_trajectories

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
    p.add_argument("--output_type", choices=("animation", "trajectories"),
                   default=DEFAULTS["output_type"],
                   help="animation: playback at uniform simulated times; trajectories: full static paths")
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
    # Bind negative exponent-form scalars and vectors with a negative first
    # component so argparse validates values rather than treating them as flags.
    numeric = {"n_bodies", "dt", "max_steps", "eps1", "eps2", "frame_time",
               "frame_interval_ms", "trail_time"}
    vector_options = {"masses_solar", "positions_init", "velocities_init"}
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
    return args


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
    """Print eleven locally interpolated accepted-step diagnostic sets."""
    print("Conservation over ten equal time intervals (user coordinates, scaled by one solar mass):")
    print("E, E_internal, K [m^2/s^2]; P [m/s]; L [m^2/s]")
    for sample in result["conservation_samples"]:
        print(f"  Fraction {sample['fraction']:.1f}; t={five(sample['time'] / 86400)} days: "
              f"E={five(sample['energy'])}; E_internal={five(sample['internal_energy'])}; "
              f"K={five(sample['kinetic_energy'])}")
        print(f"    P=({', '.join(five(value) for value in sample['momentum'])}); "
              f"L=({', '.join(five(value) for value in sample['angular_momentum'])})")


def main(argv=None):
    args = parse_args(argv)
    params = SimulationParams(
        n_bodies=args.n_bodies, masses_solar=args.masses_solar,
        positions_init=args.positions_init, velocities_init=args.velocities_init,
        dt=args.dt, max_steps=args.max_steps, eps1=args.eps1, eps2=args.eps2,
        output_type=args.output_type,
        animation_mode=args.animation_mode.replace("_", " "),
        frame_time=args.frame_time, frame_interval_ms=args.frame_interval_ms,
        trail_time=args.trail_time, projection=args.projection,
        axis_mode=args.axis_mode, display_frame=args.display_frame,
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

    try:
        if result["type"] == "trajectories":
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
