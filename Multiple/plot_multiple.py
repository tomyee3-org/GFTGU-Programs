"""
Static plotting, conservation diagnostics, and animation for Multiple.
"""

from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from physics_multiple import positions_in_display_frame


_COLORS = ["red", "green", "blue", "orange", "purple", "brown"]


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
    if positions.ndim != 3 or positions.shape[0] == 0 or positions.shape[2] != 3:
        raise ValueError(
            "positions must have shape (number of states, number of "
            "bodies, 3), with at least one state."
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

    fig, ax = plt.subplots()
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

    if frame_times.size == 0 or source_positions.shape[0] == 0:
        raise ValueError("No animation frames are available.")
    if (
        frame_times.ndim != 1
        or source_positions.ndim != 3
        or source_positions.shape[0] != frame_times.shape[0]
        or source_positions.shape[2] != 3
    ):
        raise ValueError(
            "frame_positions must have shape (number of frame_times, "
            "number of bodies, 3)."
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

    fig, ax = plt.subplots()

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
    }

    def _projected_for_frame(frame_name):
        displayed = _positions_for_display(result, source_positions, frame_name)
        return displayed[:, :, [i1, i2]]

    def _apply_fixed_limits():
        if axis_mode != "fixed":
            return
        xlim, ylim = _fixed_limits(state["projected"])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    def _set_frame(frame_name):
        state["display_frame"] = frame_name
        state["projected"] = _projected_for_frame(frame_name)
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

        _auto_limits(i)
        time_text.set_text(
            f"t = {frame_times[i]:.4e} s\n"
            f"frame {i + 1} / {n_frames}"
        )
        return [*lines, *markers, time_text]

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
