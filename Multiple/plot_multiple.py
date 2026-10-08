"""
Static plotting, conservation diagnostics, and animation for Multiple.
"""

from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from physics_multiple import center_of_mass, positions_in_display_frame


_COLORS = ["red", "green", "blue", "orange", "purple", "brown"]
_TEST_COLOR = "0.45"   # grey for massless test particles
# Survival plots: survivors in blues and greens, horseshoes in gold (they
# survived the run but are not secure), removed particles in reds.
_TADPOLE_COLOR = "#1f3fbf"
_HORSESHOE_COLOR = "#e6b800"
_CIRCULATING_COLOR = "#2ca02c"
_EDGE = {_HORSESHOE_COLOR: "#b38f00"}
AU_M = 1.495978707e11
YEAR_S = 365.25 * 86400.0


def _test_positions_for_display(result, key, massive_positions, frame):
    """Return test-particle positions (NaN once removed) in the display frame.

    Returns None when the result has no test particles. In the COM frame the
    massive bodies' centre of mass is subtracted, since test particles have
    no mass of their own.
    """
    if key not in result:
        return None
    try:
        test = np.asarray(result[key], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a numeric array.") from exc
    if (test.ndim != 3 or test.shape[0] != massive_positions.shape[0]
            or test.shape[2] != 3):
        raise ValueError(
            f"{key} must have shape (number of states, number of test "
            "particles, 3) matching the massive bodies' states."
        )
    if test.shape[1] == 0:
        return None
    if frame == "com":
        com = center_of_mass(massive_positions, result["masses_solar"])
        test = test - com[:, None, :]
    return test


def _projection_indices(projection: str):
    if not isinstance(projection, str):
        raise ValueError('projection must be "xy", "xz", or "yz".')
    mapping = {
        "xy": (0, 1, "x", "y"),
        "xz": (0, 2, "x", "z"),
        "yz": (1, 2, "y", "z"),
    }
    try:
        return mapping[projection.lower()]
    except KeyError as exc:
        raise ValueError('projection must be "xy", "xz", or "yz".') from exc


def _normalized_display_frame(value) -> str:
    if not isinstance(value, str):
        raise ValueError('display_frame must be "com" or "user".')
    frame = value.lower()
    if frame not in ("com", "user"):
        raise ValueError('display_frame must be "com" or "user".')
    return frame


def _require_result_mapping(result: Any, expected_type: str) -> None:
    """Reject a non-dict result before any ``.get``/``[...]`` access.

    ``run_simulation()`` always returns a ``dict``, so this only fires on a
    caller error (e.g. passing a raw array or a string). Without it, the
    first field lookup below raises a bare ``AttributeError`` instead of the
    same clear ``ValueError`` every other malformed-result case already
    gets.
    """
    if not isinstance(result, dict):
        raise ValueError(
            f"{expected_type} result must be a dict, as returned by "
            "run_simulation()."
        )


def _resolve_display_frame(result: Dict[str, Any]) -> str:
    """
    Choose a display frame that can actually be computed from result.

    Legacy results that omit both display_frame and masses_solar are treated
    as user-frame data. An explicit COM request without masses is rejected
    rather than labeled COM while raw coordinates are drawn.
    """
    requested = result.get("display_frame")
    masses = result.get("masses_solar")
    if requested is None:
        if masses is None:
            return "user"
        return "com"
    frame = _normalized_display_frame(requested)
    if frame == "com" and masses is None:
        raise ValueError(
            "COM display requires masses_solar in the result dictionary."
        )
    return frame


def _frame_label(frame: str) -> str:
    if frame == "com":
        return "COM frame"
    return "user frame"


def _positions_for_display(result: Dict[str, Any], positions, frame: str):
    masses = result.get("masses_solar")
    if frame == "com":
        if masses is None:
            raise ValueError(
                "COM display requires masses_solar in the result dictionary."
            )
        return positions_in_display_frame(positions, masses, "com")
    return np.asarray(positions, dtype=float)


def _fixed_limits(projected):
    x = projected[..., 0]
    y = projected[..., 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    span = max(xmax - xmin, ymax - ymin, 1.0)
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    half = 0.55 * span
    return (xmid - half, xmid + half), (ymid - half, ymid + half)


def _validated_result_positions(result: Dict[str, Any]) -> np.ndarray:
    """Return result['positions'] as a finite (n_states, n_bodies, 3) array."""
    try:
        raw_positions = result["positions"]
    except KeyError as exc:
        raise ValueError("Trajectory results must contain positions.") from exc
    try:
        positions = np.asarray(raw_positions, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("positions must be a numeric array.") from exc
    if (positions.ndim != 3 or positions.shape[0] == 0
            or positions.shape[1] == 0 or positions.shape[2] != 3):
        raise ValueError(
            "positions must have shape (number of states, number of "
            "bodies, 3), with at least one state and one body."
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must contain only finite values.")
    return positions


def plot_trajectories(
    result: Dict[str, Any],
    projection: str = "xy",
) -> None:
    """Plot complete trajectories in the selected 2-D projection."""
    _require_result_mapping(result, "plot_trajectories")
    if result.get("type") != "trajectories":
        raise ValueError("plot_trajectories requires a trajectories result.")

    raw_positions = _validated_result_positions(result)
    frame = _resolve_display_frame(result)
    positions = _positions_for_display(result, raw_positions, frame)
    _, n_bodies, _ = positions.shape
    i1, i2, label1, label2 = _projection_indices(projection)

    test = _test_positions_for_display(result, "test_positions", raw_positions, frame)

    fig, ax = plt.subplots()
    if test is not None:
        for k in range(test.shape[1]):
            ax.plot(test[:, k, i1], test[:, k, i2], color=_TEST_COLOR,
                    linewidth=0.5, alpha=0.6,
                    label="Test particles" if k == 0 else None)
    for i in range(n_bodies):
        ax.plot(
            positions[:, i, i1],
            positions[:, i, i2],
            color=_COLORS[i % len(_COLORS)],
            label=f"Body {i + 1}",
        )

    ax.set_xlabel(f"{label1} (m)")
    ax.set_ylabel(f"{label2} (m)")
    ax.set_title(
        f"Multiple trajectories ({projection.lower()} projection, "
        f"{_frame_label(frame)})"
    )
    ax.legend()
    ax.set_aspect("equal", "box")
    plt.tight_layout()
    plt.show()


def plot_energy_drift(result: Dict[str, Any]) -> None:
    """Plot fractional total-energy drift for trajectory output."""
    _require_result_mapping(result, "plot_energy_drift")
    if result.get("type") != "trajectories":
        raise ValueError(
            "Energy-drift history is available in trajectories mode."
        )

    try:
        energies = np.asarray(result["energies"], dtype=float)
        times = np.asarray(result["times"], dtype=float)
    except KeyError as exc:
        raise ValueError(
            "Trajectory results must contain energies and times."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("energies and times must be numeric arrays.") from exc

    if (
        energies.ndim != 1
        or times.ndim != 1
        or energies.size == 0
        or energies.shape != times.shape
    ):
        raise ValueError(
            "energies and times must be non-empty one-dimensional arrays "
            "with matching lengths."
        )
    if not (np.all(np.isfinite(energies)) and np.all(np.isfinite(times))):
        raise ValueError("energies and times must contain only finite values.")

    times_days = times / 86400.0
    e0 = float(energies[0])
    normalization = result.get("energy_drift_normalization")
    scale = result.get("energy_drift_scale")

    if normalization == "characteristic_energy":
        scale = _validated_energy_scale(scale)
        drift = (energies - e0) / scale
        ylabel = r"$(E-E_0)/(K_0+|U_0|)$"
    elif normalization == "initial_energy":
        scale = _validated_energy_scale(scale)
        drift = (energies - e0) / scale
        ylabel = r"$(E-E_0)/|E_0|$"
    elif normalization is not None:
        raise ValueError(
            "energy_drift_normalization must be 'initial_energy' or "
            "'characteristic_energy'."
        )
    elif e0 != 0.0:
        # Backward-compatible handling for trajectory results produced before
        # the normalization metadata was added.
        drift = (energies - e0) / abs(e0)
        ylabel = r"$(E-E_0)/|E_0|$"
    else:
        # Legacy result with exactly zero E0 and no characteristic scale.
        drift = energies - e0
        ylabel = r"$E-E_0$ (scaled m$^2$/s$^2$)"

    fig, ax = plt.subplots()
    ax.plot(times_days, drift)
    ax.set_xlabel("time (days)")
    ax.set_ylabel(ylabel)
    ax.set_title("Multiple total-energy drift")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _validated_energy_scale(value) -> float:
    """Return a positive finite energy scale or raise a public ValueError."""
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "energy_drift_scale must be a positive finite number when "
            "normalization metadata is present."
        ) from exc
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "energy_drift_scale must be a positive finite number when "
            "normalization metadata is present."
        )
    return scale


def animate_multiple(result: Dict[str, Any]):
    """
    Play a precomputed Multiple animation.

    Keyboard controls:
      Space       pause / resume
      Right arrow advance one frame while paused
      Left arrow  go back one frame while paused
      Home        jump to first frame and pause
      End         jump to final frame and pause
      f           toggle user / COM display frame

    Space resumes from the currently displayed frame. If the final frame is
    displayed, Space replays from the beginning.
    """
    _require_result_mapping(result, "animate_multiple")
    if result.get("type") != "animation":
        raise ValueError("animate_multiple requires an animation result.")

    try:
        raw_frame_times = result["frame_times"]
        raw_frame_positions = result["frame_positions"]
    except KeyError as exc:
        raise ValueError(
            "Animation results must contain frame_times and frame_positions."
        ) from exc

    try:
        frame_times = np.asarray(raw_frame_times, dtype=float)
        source_positions = np.asarray(raw_frame_positions, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "frame_times and frame_positions must be numeric arrays."
        ) from exc

    # Reject a 0-d array before indexing .shape[0]: a scalar or None
    # frame_times/frame_positions (e.g. frame_positions=7) becomes a 0-d
    # array whose .shape is (), so indexing .shape[0] below would raise a
    # bare IndexError instead of the clear ValueError every other
    # malformed case gets. Anything with ndim >= 1 is safe to index and is
    # still fully checked by the two blocks below, unchanged.
    if frame_times.ndim == 0 or source_positions.ndim == 0:
        raise ValueError(
            "frame_positions must have shape (number of frame_times, "
            "number of bodies, 3)."
        )
    if frame_times.size == 0 or source_positions.shape[0] == 0:
        raise ValueError("No animation frames are available.")
    if (
        frame_times.ndim != 1
        or source_positions.ndim != 3
        or source_positions.shape[0] != frame_times.shape[0]
        or source_positions.shape[1] == 0
        or source_positions.shape[2] != 3
    ):
        raise ValueError(
            "frame_positions must have shape (number of frame_times, "
            "number of bodies, 3), with at least one body."
        )
    if not (
        np.all(np.isfinite(frame_times))
        and np.all(np.isfinite(source_positions))
    ):
        raise ValueError(
            "frame_times and frame_positions must contain only finite "
            "values."
        )

    try:
        mode = result["animation_mode"]
        raw_frame_time = result["frame_time"]
        raw_interval_ms = result["frame_interval_ms"]
        raw_trail_time = result["trail_time"]
        projection = result["projection"]
        axis_mode = result["axis_mode"]
    except KeyError as exc:
        raise ValueError(
            "Animation results must contain animation_mode, frame_time, "
            "frame_interval_ms, trail_time, projection, and axis_mode."
        ) from exc

    try:
        frame_time = float(raw_frame_time)
        interval_ms = int(raw_interval_ms)
        trail_time = float(raw_trail_time)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "frame_time, frame_interval_ms, and trail_time must be "
            "numeric."
        ) from exc
    if not (np.isfinite(frame_time) and frame_time > 0.0):
        raise ValueError("frame_time must be a positive finite number.")
    if interval_ms <= 0:
        raise ValueError("frame_interval_ms must be a positive integer.")
    if not (np.isfinite(trail_time) and trail_time >= 0.0):
        raise ValueError("trail_time must be a non-negative finite number.")
    if not isinstance(mode, str) or mode not in ("current positions", "trails"):
        raise ValueError(
            'animation_mode must be "current positions" or "trails".'
        )
    if not isinstance(axis_mode, str) or axis_mode not in ("fixed", "auto"):
        raise ValueError('axis_mode must be "fixed" or "auto".')
    display_frame = _resolve_display_frame(result)
    i1, i2, label1, label2 = _projection_indices(projection)
    n_frames, n_bodies, _ = source_positions.shape

    has_test = _test_positions_for_display(
        result, "test_frame_positions", source_positions, "user"
    ) is not None

    fig, ax = plt.subplots()

    test_marker = None
    if has_test:
        test_marker, = ax.plot([], [], marker=".", linestyle="none",
                               color=_TEST_COLOR, markersize=3,
                               label="Test particles")

    lines = []
    markers = []
    for i in range(n_bodies):
        color = _COLORS[i % len(_COLORS)]
        line, = ax.plot([], [], color=color, linewidth=1.2)
        marker, = ax.plot(
            [], [],
            marker="o",
            linestyle="none",
            color=color,
            markersize=6,
            label=f"Body {i + 1}",
        )
        lines.append(line)
        markers.append(marker)

    time_text = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        ha="left", va="top",
        family="monospace",
    )
    ax.set_xlabel(f"{label1} (m)")
    ax.set_ylabel(f"{label2} (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")

    trail_frames = 0
    if mode == "trails" and trail_time > 0.0:
        trail_frames = max(1, int(np.ceil(trail_time / frame_time)))

    state = {
        "paused": False,
        "index": 0,
        "finished": False,
        "display_frame": display_frame,
        "projected": None,
        "projected_tests": None,
    }

    def _projected_for_frame(frame_name):
        displayed = _positions_for_display(result, source_positions, frame_name)
        return displayed[:, :, [i1, i2]]

    def _projected_tests(frame_name):
        if not has_test:
            return None
        shown = _test_positions_for_display(
            result, "test_frame_positions", source_positions, frame_name
        )
        return shown[:, :, [i1, i2]]

    def _apply_fixed_limits():
        if axis_mode != "fixed":
            return
        extent = state["projected"]
        if state.get("projected_tests") is not None:
            # Frame the massive bodies and the test particles' starting places;
            # escaping particles are allowed to leave the view.
            first = state["projected_tests"][0]
            first = first[np.all(np.isfinite(first), axis=1)]
            extent = np.concatenate((extent.reshape(-1, 2), first), axis=0)
        xlim, ylim = _fixed_limits(extent)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    def _set_frame(frame_name):
        state["display_frame"] = frame_name
        state["projected"] = _projected_for_frame(frame_name)
        state["projected_tests"] = _projected_tests(frame_name)
        ax.set_title(
            f"Multiple animation ({projection} projection, "
            f"{_frame_label(frame_name)})"
        )
        _apply_fixed_limits()

    _set_frame(display_frame)

    def _auto_limits(i):
        if axis_mode != "auto":
            return

        start = max(0, i - trail_frames) if mode == "trails" else i
        visible = state["projected"][start:i + 1]

        x = visible[..., 0]
        y = visible[..., 1]
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))

        # One common span preserves a true 1:1 spatial scale while zooming.
        span = max(xmax - xmin, ymax - ymin, 1.0)
        half = 0.60 * span
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)

        ax.set_xlim(xmid - half, xmid + half)
        ax.set_ylim(ymid - half, ymid + half)

    def draw_frame(i):
        i = max(0, min(n_frames - 1, int(i)))
        state["index"] = i
        state["finished"] = (i >= n_frames - 1)

        start = max(0, i - trail_frames) if mode == "trails" else i
        projected = state["projected"]

        for body in range(n_bodies):
            markers[body].set_data(
                [projected[i, body, 0]],
                [projected[i, body, 1]],
            )

            if mode == "trails":
                lines[body].set_data(
                    projected[start:i + 1, body, 0],
                    projected[start:i + 1, body, 1],
                )
            else:
                lines[body].set_data([], [])

        if test_marker is not None:
            tests = state["projected_tests"][i]
            test_marker.set_data(tests[:, 0], tests[:, 1])
        _auto_limits(i)
        time_text.set_text(
            f"t = {frame_times[i]:.4e} s\n"
            f"frame {i + 1} / {n_frames}"
        )
        artists = [*lines, *markers, time_text]
        if test_marker is not None:
            artists.append(test_marker)
        return artists

    # Persistent canvas timer: remains valid after the last displayed frame.
    timer = fig.canvas.new_timer(interval=interval_ms)

    def advance():
        if state["paused"]:
            return

        next_index = state["index"] + 1
        if next_index >= n_frames:
            state["finished"] = True
            state["paused"] = True
            timer.stop()
            return

        draw_frame(next_index)
        fig.canvas.draw_idle()

        if state["index"] >= n_frames - 1:
            state["finished"] = True
            state["paused"] = True
            timer.stop()

    timer.add_callback(advance)

    def pause():
        state["paused"] = True
        timer.stop()

    def resume_from_displayed_frame():
        if state["index"] >= n_frames - 1:
            draw_frame(0)
        state["paused"] = False
        state["finished"] = False
        timer.start()

    def on_key(event):
        key = event.key

        if key == " ":
            if state["paused"] or state["finished"]:
                resume_from_displayed_frame()
            else:
                pause()
            fig.canvas.draw_idle()
            return

        if key in ("f", "F"):
            next_frame = "user" if state["display_frame"] == "com" else "com"
            _set_frame(next_frame)
            draw_frame(state["index"])
            fig.canvas.draw_idle()
            return

        if key in ("left", "right", "home", "end"):
            pause()

            if key == "left":
                i = state["index"] - 1
            elif key == "right":
                i = state["index"] + 1
            elif key == "home":
                i = 0
            else:
                i = n_frames - 1

            draw_frame(i)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_frame(0)
    plt.tight_layout()

    state["paused"] = False
    state["finished"] = False
    timer.start()
    plt.show()

    return {
        "timer": timer,
        "state": state,
        "figure": fig,
        "on_key": on_key,
    }


def plot_survival(result: Dict[str, Any]) -> None:
    """Plot test-particle survival against starting distance and angle.

    Left panel: survival time against starting distance. Removed particles
    are plotted at their removal time; survivors at the end of the run with
    an upward triangle, since they lasted at least that long; particles
    removed at the start sit on the bottom edge with a cross.

    Survivors are drawn in greens and blues, removed particles in reds.

    Right panel: starting angle, measured from the reference body, against
    starting distance, marked by fate. Bands of survivors at particular
    angles (such as near +60 and -60 degrees at a planet's own distance)
    show where position along the orbit, not just distance, decides
    survival.
    """
    _require_result_mapping(result, "plot_survival")
    info = result.get("test_particles")
    if not isinstance(info, dict):
        raise ValueError("plot_survival requires a result with test particles.")
    try:
        distance = np.asarray(info["initial_distance_m"], dtype=float) / AU_M
        removal = np.asarray(info["removal_time_s"], dtype=float) / YEAR_S
        fates = list(info["fate"])
        final_time = float(result["final_time"]) / YEAR_S
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "test_particles must contain initial_distance_m, removal_time_s "
            "and fate, and the result must contain final_time."
        ) from exc
    if distance.ndim != 1 or distance.size == 0 or removal.shape != distance.shape \
            or len(fates) != distance.size:
        raise ValueError("test-particle arrays must be one-dimensional and matching.")
    phase = info.get("initial_phase_deg")
    if phase is not None:
        phase = np.asarray(phase, dtype=float)
        if phase.shape != distance.shape:
            raise ValueError("initial_phase_deg must match the other test-particle arrays.")

    fate_array = np.array(fates)
    at_start = (removal == 0.0) & (fate_array != "survived")
    styles = {
        "escaped": ("#ff7f0e", "o", "escaped"),
        "encounter": ("#d62728", "s", "encounter"),
        "numerical": ("#8b0000", "x", "numerical failure"),
        "survived": ("#2ca02c", "^", "survived the whole run"),
    }
    marker_size = 6 if distance.size <= 100 else 3
    if phase is None:
        fig, ax = plt.subplots()
        axes = (ax,)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    for fate, (color, marker, label) in styles.items():
        chosen = (fate_array == fate) & ~at_start
        if not np.any(chosen):
            continue
        times = np.full(int(np.sum(chosen)), final_time) if fate == "survived" \
            else removal[chosen]
        ax.plot(distance[chosen], times, linestyle="none", marker=marker,
                markersize=marker_size, color=color, label=label)
    ax.set_yscale("log")
    if np.any(at_start):
        bottom = ax.get_ylim()[0]
        ax.plot(distance[at_start], np.full(int(np.sum(at_start)), bottom),
                linestyle="none", marker="x", markersize=marker_size,
                color="#ff9896", label="removed at the start", clip_on=False)
        ax.set_ylim(bottom=bottom)
    center = info.get("center", "the centre")
    if center == "com":
        center = "the centre of mass"
    ax.set_xlabel(f"starting distance from {center} (AU)")
    ax.set_ylabel("survival time (years)")
    ax.set_title(f"Survival time ({final_time:.4g}-year run)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize="small")

    if phase is not None:
        ax = axes[1]
        motion = np.array(info.get("phase_motion", ["-"] * distance.size))
        if motion.shape != distance.shape:
            raise ValueError("phase_motion must match the other test-particle arrays.")
        for fate, (color, marker, label) in styles.items():
            if fate == "survived":
                continue
            chosen = (fate_array == fate) & ~at_start
            if np.any(chosen):
                ax.plot(distance[chosen], phase[chosen], linestyle="none",
                        marker=marker, markersize=marker_size, color=color,
                        label=label)
        survivor_styles = (
            (("tadpole L4", "tadpole L5"), _TADPOLE_COLOR, "D", "survived: tadpole"),
            (("horseshoe",), _HORSESHOE_COLOR, "v", "survived: horseshoe"),
            (("circulating", "-"), _CIRCULATING_COLOR, "^", "survived: circulating"),
        )
        for names, color, marker, label in survivor_styles:
            chosen = (fate_array == "survived") & np.isin(motion, names)
            if np.any(chosen):
                ax.plot(distance[chosen], phase[chosen], linestyle="none",
                        marker=marker, markersize=marker_size, color=color,
                        markeredgecolor=_EDGE.get(color, color),
                        markeredgewidth=0.5, label=label)
        if np.any(at_start):
            ax.plot(distance[at_start], phase[at_start], linestyle="none",
                    marker="x", markersize=marker_size, color="#ff9896",
                    label="removed at the start")
        reference = info.get("phase_reference_body", "?")
        ax.set_xlabel(f"starting distance from {center} (AU)")
        ax.set_ylabel(f"starting angle from body {reference} (degrees)")
        ax.set_ylim(-180.0, 180.0)
        ax.set_yticks(np.arange(-180, 181, 60))
        ax.set_title("Fate by starting position")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Multiple test-particle survival")
    plt.tight_layout()
    plt.show()


def plot_survivor_positions(result: Dict[str, Any]) -> None:
    """Show where the secure survivors are at the end of a run.

    Only survivors whose angle path is a tadpole (blue) or circulating
    (green) are drawn; removed particles and every horseshoe are left out.
    Angles are measured from the reference body, so both panels are views
    in the frame that turns with it.

    Left: ending angle against ending distance, on the same axes as the
    starting-angle panel of plot_survival. Right: the same points seen from
    above, with the centre in the middle, the reference body fixed on the
    right, and its L4 and L5 points 60 degrees ahead and behind.
    """
    _require_result_mapping(result, "plot_survivor_positions")
    info = result.get("test_particles")
    if not isinstance(info, dict):
        raise ValueError("plot_survivor_positions requires a result with test particles.")
    try:
        angle = np.asarray(info["final_phase_deg"], dtype=float)
        distance = np.asarray(info["final_distance_m"], dtype=float) / AU_M
        motion = np.asarray(info["phase_motion"])
        fates = np.asarray(info["fate"])
        reference_distance = float(info["reference_final_distance_m"]) / AU_M
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "test_particles must contain final_phase_deg, final_distance_m, "
            "phase_motion, fate and reference_final_distance_m."
        ) from exc
    if not (angle.shape == distance.shape == motion.shape == fates.shape) \
            or angle.ndim != 1:
        raise ValueError("test-particle arrays must be one-dimensional and matching.")
    tadpole = (fates == "survived") & np.isin(motion, ("tadpole L4", "tadpole L5"))
    circulating = (fates == "survived") & (motion == "circulating")
    if not np.any(tadpole | circulating):
        raise ValueError("No tadpole or circulating survivors to show.")

    reference = info.get("phase_reference_body", "?")
    center = info.get("center", "the centre")
    if center == "com":
        center = "the centre of mass"
    size = 6 if np.sum(tadpole | circulating) <= 100 else 3
    groups = ((tadpole, _TADPOLE_COLOR, "D", "tadpole (Trojan)"),
              (circulating, _CIRCULATING_COLOR, "^", "circulating"))

    fig = plt.figure(figsize=(13, 5.6))
    flat = fig.add_subplot(1, 2, 1)
    polar = fig.add_subplot(1, 2, 2, projection="polar")
    for chosen, color, marker, label in groups:
        if not np.any(chosen):
            continue
        flat.plot(distance[chosen], angle[chosen], linestyle="none", marker=marker,
                  markersize=size, color=color, label=label)
        polar.plot(np.radians(angle[chosen]), distance[chosen], linestyle="none",
                   marker=marker, markersize=size, color=color, label=label)
    for lagrange, name in ((60.0, "L4"), (-60.0, "L5")):
        flat.axhline(lagrange, color="0.5", linewidth=0.8, linestyle="--")
        flat.annotate(name, (1.0, lagrange), xycoords=("axes fraction", "data"),
                      xytext=(4, 0), textcoords="offset points", va="center",
                      color="0.35")
        polar.plot([np.radians(lagrange)], [reference_distance], marker="o",
                   markersize=9, markerfacecolor="none", markeredgecolor="0.35",
                   linestyle="none")
        polar.annotate(name, (np.radians(lagrange), 1.12 * reference_distance),
                       ha="center", va="center", color="0.25", fontweight="bold")
    polar.plot([0.0], [reference_distance], marker="*", markersize=14, color="black",
               linestyle="none", label=f"body {reference}")
    polar.plot([0.0], [0.0], marker="+", markersize=10, color="black", linestyle="none")
    flat.set_xlabel(f"ending distance from {center} (AU)")
    flat.set_ylabel(f"ending angle from body {reference} (degrees)")
    flat.set_ylim(-180.0, 180.0)
    flat.set_yticks(np.arange(-180, 181, 60))
    flat.grid(True, alpha=0.3)
    flat.set_title("Ending angle against ending distance")
    flat.legend(fontsize="small", loc="lower left")
    polar.set_rlim(0.0, 1.1 * max(float(np.nanmax(distance)), reference_distance))
    polar.set_title(f"Seen from above, turning with body {reference} (distances in AU)",
                    pad=18)
    polar.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.05, 1.0))
    final_time = float(result.get("final_time", np.nan)) / YEAR_S
    fig.suptitle(f"Secure survivors at the end of the {final_time:.4g}-year run "
                 "(horseshoes and removed particles not shown)")
    plt.tight_layout()
    plt.show()
