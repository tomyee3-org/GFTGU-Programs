"""Command-line entry point for Random2.

Choose one display with ``--display``:

``scaled_distance``
    Run many three-dimensional component-based random walks and plot average
    displacement, scaled by each walk's mean step length, against step count.

``walk2d``
    Draw fixed-step isotropic two-dimensional walks inside a circular
    schematic star until they escape or reach a safety cap.

Every user-facing input can be set at the command line:

``--max_steps`` and ``--n_trials``
    Largest scaled-distance walk (default 4096) and independent walks averaged
    at each step count (default 100).

``--step_distribution {uniform,gaussian}``
    ``uniform`` uses independent components on [-1, 1]; ``gaussian`` uses
    independent standard-normal components.

``--reference_steps`` and ``--n_walks``
    Reference count used to size the default star (default 2000) and number of
    displayed two-dimensional paths (default 4).

``--mean_free_path``, ``--radius_factor``, and ``--radius``
    Positive geometric inputs. If ``--radius`` is omitted, the radius is
    radius_factor * mean_free_path * sqrt(reference_steps).

``--ray_length_factor`` and ``--step_cap``
    Nonnegative outgoing-ray length as a fraction of the radius (default 0.6;
    use 0 to suppress it), and positive per-walk safety cap (default 200000).

``--corner {upper_right,upper_left,lower_right,lower_left}``
    Location of the results annotation on the two-dimensional plot. Underscore
    spellings are command-line-safe; they map to the plotter's spaced labels.

``--seed``
    Nonnegative integer that makes a run repeatable. Omit it (the default) and
    every run uses different random numbers.

Every run prints a summary before it opens the plot: for scaled_distance, the
mean scaled distance, its standard error and its ratio to sqrt(N) at each step
count, the large-N prediction for that ratio, and the fitted log-log slope; for
walk2d, the star radius, (R / mean free path)^2, each walk's escape step count
(for at most 12 walks) and the escape-step statistics.

Examples
--------
  python main.py --display scaled_distance --max_steps 8192 --n_trials 500
  python main.py --display scaled_distance --step_distribution gaussian
  python main.py --display walk2d --n_walks 6 --radius 50 --step_cap 100000
  python main.py --display walk2d --corner lower_left --ray_length_factor 0
  python main.py --seed 1
"""

import argparse
import math

import random2_physics
from random2_driver import (
    escape_statistics,
    run_scaled_distance_statistics,
    run_walk2d,
)
from random2_physics import (
    diffusion_step_scale,
    fitted_loglog_slope,
    large_n_scaled_distance_ratio,
    length_text,
    seed_generator,
)
from random2_plot import plot_scaled_distance, plot_walk2d


DISPLAY_CHOICES = ("scaled_distance", "walk2d")
# Walks are listed one per line in the summary only up to this many.
MAX_LISTED_WALKS = 12
STEP_DISTRIBUTIONS = ("uniform", "gaussian")
CORNER_CHOICES = {
    "upper_right": "upper right",
    "upper_left": "upper left",
    "lower_right": "lower right",
    "lower_left": "lower left",
}


def _positive_int(text):
    """Parse a positive command-line integer."""
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer.") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer.")
    return value


def _step_count_at_least_two(text):
    """Parse the largest scaled-distance step count (at least 2)."""
    value = _positive_int(text)
    if value < 2:
        raise argparse.ArgumentTypeError(
            "max_steps must be at least 2, because the smallest walk has 2 steps."
        )
    return value


def _seed(text):
    """Parse a nonnegative integer seed."""
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer.") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("seed must be a nonnegative integer.")
    return value


def _positive_float(text):
    """Parse a positive finite command-line number."""
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number.") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite number.")
    return value


def _nonnegative_float(text):
    """Parse a nonnegative finite command-line number."""
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number.") from exc
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError(
            "value must be a nonnegative finite number."
        )
    return value


def build_parser():
    """Return the command-line parser (separate so tests can inspect it)."""
    parser = argparse.ArgumentParser(
        prog="Random2",
        description=(
            "Explore square-root random-walk scaling or display isotropic "
            "two-dimensional diffusion in a circular schematic star."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"Random2 {random2_physics.MODEL_VERSION} "
            f"(build {random2_physics.BUILD_ID})"
        ),
    )
    parser.add_argument(
        "--display",
        choices=DISPLAY_CHOICES,
        default="scaled_distance",
        help=(
            "calculation to show: scaled_distance tests sqrt(N) scaling in "
            "3D; walk2d draws isotropic fixed-step paths in a circular star"
        ),
    )

    scaled = parser.add_argument_group("Scaled-distance display")
    scaled.add_argument(
        "--max_steps",
        "--maxSteps",
        dest="max_steps",
        type=_step_count_at_least_two,
        default=4096,
        metavar="N",
        help=(
            "largest step count (at least 2); successive plotted counts are "
            "integer halves"
        ),
    )
    scaled.add_argument(
        "--n_trials",
        "--nTrials",
        dest="n_trials",
        type=_positive_int,
        default=100,
        metavar="N",
        help="independent walks averaged at each plotted step count",
    )
    scaled.add_argument(
        "--step_distribution",
        choices=STEP_DISTRIBUTIONS,
        default="uniform",
        metavar="KIND",
        help=(
            "3D component distribution: uniform samples [-1,1]; gaussian "
            "samples the standard normal distribution"
        ),
    )

    walk = parser.add_argument_group("Two-dimensional star display")
    walk.add_argument(
        "--reference_steps",
        type=_positive_int,
        default=2000,
        metavar="N",
        help="reference count used only to calculate the default star radius",
    )
    walk.add_argument(
        "--n_walks",
        type=_positive_int,
        default=4,
        metavar="N",
        help="number of independent two-dimensional paths to draw",
    )
    walk.add_argument(
        "--mean_free_path",
        type=_positive_float,
        default=1.0,
        metavar="LENGTH",
        help="fixed positive length of every isotropic two-dimensional step",
    )
    walk.add_argument(
        "--radius_factor",
        type=_positive_float,
        default=2.0,
        metavar="FACTOR",
        help="positive multiplier in the default-radius relation",
    )
    walk.add_argument(
        "--radius",
        type=_positive_float,
        default=None,
        metavar="LENGTH",
        help="explicit positive star radius; omit to use the default relation",
    )
    walk.add_argument(
        "--ray_length_factor",
        type=_nonnegative_float,
        default=0.6,
        metavar="FACTOR",
        help="outgoing-ray length divided by radius; use 0 to suppress rays",
    )
    walk.add_argument(
        "--step_cap",
        type=_positive_int,
        default=200_000,
        metavar="N",
        help="maximum number of steps allowed for each two-dimensional walk",
    )
    walk.add_argument(
        "--corner",
        choices=tuple(CORNER_CHOICES),
        default="upper_right",
        metavar="POSITION",
        help=(
            "annotation location: upper_right, upper_left, lower_right, or "
            "lower_left"
        ),
    )
    parser.add_argument(
        "--seed",
        type=_seed,
        default=None,
        metavar="N",
        help=(
            "nonnegative integer that makes a run repeatable; omit it for "
            "different random numbers on every run"
        ),
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def _seed_text(seed):
    return "none (results differ from run to run)" if seed is None else str(seed)


def scaled_distance_summary(result, seed=None):
    """Return the console summary lines for a scaled-distance run."""
    prediction = large_n_scaled_distance_ratio(result.step_distribution)
    lines = [
        "display: scaled_distance",
        f"step components: {result.step_distribution}",
        f"trials per step count: {result.n_trials}",
        f"seed: {_seed_text(seed)}",
        f"{'N':>8}  {'mean scaled distance':>20}  {'standard error':>14}  "
        f"{'ratio to sqrt(N)':>16}",
    ]
    for n_steps, mean, error in zip(
        result.lengths, result.averages, result.standard_errors
    ):
        error_text = "n/a" if error is None else f"{error:.4f}"
        lines.append(
            f"{int(n_steps):>8}  {mean:>20.4f}  {error_text:>14}  "
            f"{mean / math.sqrt(n_steps):>16.4f}"
        )
    lines.append(f"large-N prediction for ratio to sqrt(N): {prediction:.4f}")
    if len(result.lengths) >= 2:
        slope = fitted_loglog_slope(result.lengths, result.averages)
        lines.append(f"fitted log-log slope: {slope:.4f}")
    else:
        lines.append("fitted log-log slope: n/a (only one step count)")
    return lines


def walk2d_summary(result, seed=None):
    """Return the console summary lines for a two-dimensional star run."""
    stats = escape_statistics(result)
    diffusion_steps = diffusion_step_scale(result.radius, result.mean_free_path)
    lines = [
        "display: walk2d",
        f"star radius R: {length_text(result.radius)}",
        f"mean free path: {result.mean_free_path:g}",
        "(R / mean free path)^2: "
        + (
            f"{diffusion_steps:.1f}"
            if 0.1 <= diffusion_steps < 1.0e9
            else f"{diffusion_steps:.6g}"
        ),
        f"step cap: {result.step_cap}",
        f"seed: {_seed_text(seed)}",
    ]
    if len(result.walks) <= MAX_LISTED_WALKS:
        for number, walk in enumerate(result.walks, start=1):
            if walk.escaped:
                lines.append(f"walk {number}: escaped after {walk.steps_taken} steps")
            else:
                lines.append(
                    f"walk {number}: stopped at the step cap after "
                    f"{walk.steps_taken} steps, not escaped"
                )
    lines.append(f"escaped walks: {stats.n_escaped} of {stats.n_walks}")
    if stats.mean is None:
        lines.append("escape steps: n/a (no walk escaped)")
        return lines
    error_text = (
        "n/a" if stats.standard_error is None else f"{stats.standard_error:.1f}"
    )
    lines.append(
        f"escape steps: mean {stats.mean:.1f} (standard error {error_text}), "
        f"median {stats.median:.1f}, min {stats.minimum}, max {stats.maximum}"
    )
    lines.append(
        f"mean escape steps / (R / mean free path)^2: "
        f"{stats.mean / diffusion_steps:.4f}"
    )
    return lines


def main(argv=None):
    args = parse_args(argv)

    print(
        f"Random2 {random2_physics.MODEL_VERSION} "
        f"(build {random2_physics.BUILD_ID})"
    )

    try:
        seed_generator(args.seed)
        if args.display == "scaled_distance":
            result = run_scaled_distance_statistics(
                args.max_steps,
                args.n_trials,
                step_distribution=args.step_distribution,
            )
            print("\n".join(scaled_distance_summary(result, args.seed)))
            plot_scaled_distance(result.lengths, result.averages)
        else:
            result = run_walk2d(
                reference_steps=args.reference_steps,
                n_walks=args.n_walks,
                radius=args.radius,
                mean_free_path=args.mean_free_path,
                radius_factor=args.radius_factor,
                ray_length_factor=args.ray_length_factor,
                step_cap=args.step_cap,
            )
            print("\n".join(walk2d_summary(result, args.seed)))
            plot_walk2d(result, corner=CORNER_CHOICES[args.corner])
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"Random2 input/model error: {exc}") from exc


if __name__ == "__main__":
    main()
