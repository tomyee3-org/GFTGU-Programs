"""
Static plotting, conservation diagnostics, and animation for Multiple.
"""

from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np


_COLORS = ["red", "green", "blue", "orange", "purple", "brown"]


def _projection_indices(projection: str):
    mapping = {
        "xy": (0, 1, "x", "y"),
        "xz": (0, 2, "x", "z"),
        "yz": (1, 2, "y", "z"),
    }
    try:
        return mapping[projection.lower()]
    except KeyError as exc:
        raise ValueError('projection must be "xy", "xz", or "yz".') from exc


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


def plot_trajectories(
    result: Dict[str, Any],
    projection: str = "xy",
) -> None:
    """Plot complete trajectories in the selected 2-D projection."""
    if result.get("type") != "trajectories":
        raise ValueError("plot_trajectories requires a trajectories result.")

    positions = result["positions"]
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
    ax.set_title(f"Multiple trajectories ({projection.lower()} projection)")
    ax.legend()
    ax.set_aspect("equal", "box")
    plt.tight_layout()
    plt.show()


def plot_energy_drift(result: Dict[str, Any]) -> None:
    """Plot fractional total-energy drift for trajectory output."""
    if result.get("type") != "trajectories":
        raise ValueError(
            "Energy-drift history is available in trajectories mode."
        )

    energies = np.asarray(result["energies"], dtype=float)
    times_days = np.asarray(result["times"], dtype=float) / 86400.0
    e0 = float(energies[0])
    normalization = result.get("energy_drift_normalization")
    scale = result.get("energy_drift_scale")

    if normalization == "characteristic_energy":
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("energy_drift_scale must be positive and finite.")
        drift = (energies - e0) / scale
        ylabel = r"$(E-E_0)/(K_0+|U_0|)$"
    elif normalization == "initial_energy":
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("energy_drift_scale must be positive and finite.")
        drift = (energies - e0) / scale
        ylabel = r"$(E-E_0)/|E_0|$"
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


def animate_multiple(result: Dict[str, Any]):
    """
    Play a precomputed Multiple animation.

    Keyboard controls:
      Space       pause / resume
      Right arrow advance one frame while paused
      Left arrow  go back one frame while paused
      Home        jump to first frame and pause
      End         jump to final frame and pause

    Space resumes from the currently displayed frame. If the final frame is
    displayed, Space replays from the beginning.
    """
    if result.get("type") != "animation":
        raise ValueError("animate_multiple requires an animation result.")

    frame_times = np.asarray(result["frame_times"], dtype=float)
    positions = np.asarray(result["frame_positions"], dtype=float)

    if frame_times.size == 0 or positions.shape[0] == 0:
        raise ValueError("No animation frames are available.")

    mode = result["animation_mode"]
    frame_time = float(result["frame_time"])
    interval_ms = int(result["frame_interval_ms"])
    trail_time = float(result["trail_time"])
    projection = result["projection"]
    axis_mode = result["axis_mode"]

    i1, i2, label1, label2 = _projection_indices(projection)
    projected = positions[:, :, [i1, i2]]
    n_frames, n_bodies, _ = projected.shape

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
    ax.set_title(f"Multiple animation ({projection} projection)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")

    if axis_mode == "fixed":
        xlim, ylim = _fixed_limits(projected)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    trail_frames = 0
    if mode == "trails" and trail_time > 0.0:
        trail_frames = max(1, int(np.ceil(trail_time / frame_time)))

    state = {"paused": False, "index": 0, "finished": False}

    def _auto_limits(i):
        if axis_mode != "auto":
            return

        start = max(0, i - trail_frames) if mode == "trails" else i
        visible = projected[start:i + 1]

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
            f"t = {frame_times[i]:.4g} s\n"
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

    return {"timer": timer, "state": state}
