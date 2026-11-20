#!/usr/bin/env python3
"""Plot robot and pedestrian behavior from a saved episode replay NPZ.

Use a named preset to reproduce the paper figures, or pass an arbitrary replay
with explicit phase times. Times are seconds from the start unless --time-to-goal
is set. Use --combined to save all phases in one PNG with a shared colorbar.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from plot_replay_common import (
    CBF_VECTOR_COLOR, EVALUATION_MEAN_PEDESTRIAN_RADIUS_M, GOAL_COLOR,
    NAVIGATION_VECTOR_COLOR, PEDESTRIAN_COLOR, _body_velocity_to_world,
    _navigation_goal_marker, _nearest_index, _plot_pedestrian_trajectories,
    _plot_robot_speed_trajectory, _robot_marker, _velocity_arrow,
)

SCRIPT_DIR = Path(__file__).resolve().parent
COLORS = ("#0072B2", "#E69F00", "#CC79A7", "#009E73")


@dataclass(frozen=True)
class Preset:
    filename: str
    output_name: str
    title: str
    phases: tuple[tuple[float, str, str], ...]
    highlights: tuple[int, ...]
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    time_to_goal: bool = True
    history_s: float | None = None
    dpi: int = 350


PRESETS = {
    "crossing-episode": Preset(
        "crossing_9705bc8caf1f467f94a4e236d26eafa7.npz", "crossing_9705bc8", "Crossing",
        ((3.0, "Assert", "Assert across the path of pedestrian 15"),
         (5.6, "Yield", "Assert past pedestrian 3 while yielding to pedestrian 11")),
        (15, 3, 11), (64.5, 75.5), (-33.5, -22.5), False,
    ),
    "crossing-success": Preset(
        "crossing_success_000034.npz", "crossing", "Crossing",
        ((-7.52, "yield", "Yield: concede space to the approaching stream"),
         (-3.92, "assert", "Assert: commit to the gap and cross the stream"),
         (-2.00, "yield", "Yield: restore clearance after the crossing")),
        (1, 4, 9), (26.0, 36.0), (-35.0, -25.0),
    ),
    "with-flow": Preset(
        "with_flow_slow_leader_success_000014.npz", "withflow", "With flow — slow leader",
        ((-6.48, "overtake", "Overtake: pass the slow leader"),
         (-2.32, "wait", "Wait: let the slow leader clear the goal first"),
         (0.00, "leader_clears_goal", "Leader clears goal before the robot reaches its own goal")),
        (0,), (28.0, 38.0), (10.0, 20.0),
    ),
    "against-flow": Preset(
        "against_flow_success_000082.npz", "againstflow", "Against flow",
        ((-9.28, "side_step", "Side-step: yield laterally to the oncoming pedestrians"),
         (-6.32, "find_gap", "Find gap: identify the opening between two pedestrians"),
         (-4.48, "exploit_gap", "Exploit gap: complete the pass between the two pedestrians")),
        (7, 10, 1, 3), (41.0, 51.0), (-6.0, 4.0), True, 3.0,
    ),
    "against-flow-69831": Preset(
        "against_flow_69831eeee017455981e6e180c2f45518.npz",
        "against_flow_69831_behavior", "Against flow",
        ((4.40, "move sideway", ""),
         (7.04, "stop and yield", "")),
        (10, 1, 15), (62.0, 76.0), (-7.0, 6.0), False, None, 180,
    ),
}


def plot_case(
    input_path: Path, output_stem: Path, *, phases: tuple[tuple[float, str, str], ...],
    highlights: tuple[int, ...] = (), xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None, time_to_goal: bool = False,
    history_s: float | None = None, title: str = "Behavior", dpi: int = 350,
    combined: bool = False,
) -> list[Path]:
    with np.load(input_path) as data:
        time_s = data["time_s"]
        robot_xy = data["robot_position_xy"]
        robot_yaw = data["robot_yaw"]
        robot_speed = np.linalg.norm(data["robot_velocity_xy_world"], axis=1)
        goal_xy = data["goal_position_xy"]
        pedestrian_xy = data["pedestrian_position_xy"]
        active = data["pedestrian_active_mask"]
        pedestrian_velocity = data["pedestrian_velocity_xy_world"]
        navigation_velocity = data["navigation_policy_velocity_body"]
        cbf_velocity = data["cbf_filtered_command_velocity_body"] if "cbf_filtered_command_velocity_body" in data else None

    if len(time_s) < 2:
        raise ValueError("Replay must contain at least two frames.")
    if not phases:
        raise ValueError("At least one phase is required.")
    if any(agent_id < 0 or agent_id >= active.shape[1] for agent_id in highlights):
        raise ValueError(f"Highlighted pedestrian IDs must be in [0, {active.shape[1] - 1}].")
    dt = float(np.median(np.diff(time_s)))
    goal_time = float(time_s[-1] + dt)
    highlight_colors = {agent_id: COLORS[index % len(COLORS)] for index, agent_id in enumerate(highlights)}
    norm = Normalize(vmin=0.15, vmax=max(1.55, float(np.percentile(robot_speed, 98))))
    if xlim is None:
        xlim = (float(np.nanmin(robot_xy[:, 0])) - 2, float(np.nanmax(robot_xy[:, 0])) + 2)
    if ylim is None:
        ylim = (float(np.nanmin(robot_xy[:, 1])) - 2, float(np.nanmax(robot_xy[:, 1])) + 2)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.spines.top": False, "axes.spines.right": False})
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    handles = [Line2D([0], [0], color=PEDESTRIAN_COLOR, lw=1.7, label="Ped. traj.")]
    handles += [Line2D([0], [0], color=color, lw=2.5, label=f"Ped. {agent_id}")
                for agent_id, color in highlight_colors.items()]
    handles += [Line2D([0], [0], color=GOAL_COLOR, marker=">", lw=1.8, linestyle="--", label="Nav. goal"),
                Line2D([0], [0], color="#718096", lw=2, linestyle="--", label="Robot (future)"),
                Line2D([0], [0], color=NAVIGATION_VECTOR_COLOR, marker=">", lw=2.5, label="Nav. cmd.")]
    if cbf_velocity is not None:
        handles.append(Line2D([0], [0], color=CBF_VECTOR_COLOR, marker=">", lw=2.5, label="CBF cmd."))

    def draw_phase(ax: plt.Axes, phase: tuple[float, str, str]):
        requested_time, slug, description = phase
        frame = _nearest_index(time_s, goal_time + requested_time if time_to_goal else requested_time)
        history = pedestrian_xy.copy()
        if history_s is not None:
            history[:max(0, frame - int(round(history_s / dt)) + 1)] = np.nan
        _plot_pedestrian_trajectories(
            ax, history, active, frame=frame, highlighted_pedestrians=highlight_colors,
            alpha=0.28, linewidth=1.0,
        )
        _navigation_goal_marker(ax, goal_xy[frame], float(robot_yaw[-1]))
        ax.plot(robot_xy[frame:, 0], robot_xy[frame:, 1], color="#718096",
                linewidth=2, linestyle="--", alpha=0.65, zorder=3)
        collection = _plot_robot_speed_trajectory(ax, robot_xy[:frame + 1], robot_speed[:frame + 1], norm)
        for agent_id in np.flatnonzero(active[frame]):
            position = pedestrian_xy[frame, agent_id]
            velocity = pedestrian_velocity[frame, agent_id]
            color = highlight_colors.get(int(agent_id), PEDESTRIAN_COLOR)
            marked = agent_id in highlight_colors
            if marked:
                ax.add_patch(Circle(position, EVALUATION_MEAN_PEDESTRIAN_RADIUS_M,
                                    facecolor="none", edgecolor=color, linewidth=1.35,
                                    linestyle=(0, (3, 2)), zorder=6))
            ax.scatter(*position, s=60 if marked else 22, color=color, edgecolor="white",
                       linewidth=0.8 if marked else 0.35, zorder=6)
            ax.arrow(position[0], position[1], velocity[0] * 0.35, velocity[1] * 0.35,
                     width=0.018 if marked else 0.012, head_width=0.09 if marked else 0.065,
                     head_length=0.08 if marked else 0.06, color=color, zorder=6)
        yaw = float(robot_yaw[frame])
        _robot_marker(ax, robot_xy[frame], yaw)
        _velocity_arrow(ax, robot_xy[frame], _body_velocity_to_world(navigation_velocity[frame, :2], yaw),
                        NAVIGATION_VECTOR_COLOR)
        if cbf_velocity is not None:
            _velocity_arrow(ax, robot_xy[frame], _body_velocity_to_world(cbf_velocity[frame, :2], yaw),
                            CBF_VECTOR_COLOR)
        time_label = f"{requested_time:.2f} s to goal" if time_to_goal else f"t = {time_s[frame]:.2f} s"
        ax.set(xlim=xlim, ylim=ylim, aspect="equal", xlabel="x (m)", ylabel="y (m)",
               title=f"{title} | {slug.replace('_', ' ').capitalize()} | {time_label}")
        ax.grid(color="#DCE3EA", linewidth=0.6, zorder=0)
        if description:
            ax.text(0.02, 0.02, description, transform=ax.transAxes, fontsize=11.5,
                    color="#4A5568", va="bottom")
        return collection

    if combined:
        columns = min(3, len(phases))
        rows = ceil(len(phases) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(6.8 * columns, 6.5 * rows),
                                 constrained_layout=True, squeeze=False)
        panels = list(axes.flat)
        for ax, phase in zip(panels, phases):
            collection = draw_phase(ax, phase)
        for ax in panels[len(phases):]:
            ax.set_visible(False)
        fig.colorbar(collection, ax=panels[:len(phases)], location="right", pad=0.02,
                     shrink=0.78, label="Robot speed (m/s): slow → fast")
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0),
                   ncol=min(4, len(handles)), frameon=True, fontsize=10.5)
        path = output_stem.parent / f"{output_stem.name}_combined.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return [path]

    outputs = []
    for number, phase in enumerate(phases, 1):
        fig, ax = plt.subplots(figsize=(7.6, 7.1), constrained_layout=True)
        collection = draw_phase(ax, phase)
        fig.colorbar(collection, ax=ax, location="right", pad=0.02, shrink=0.78,
                     label="Robot speed (m/s): slow → fast")
        ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.92, fontsize=10.5)
        path = output_stem.parent / f"{output_stem.name}_{number}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="Saved episode replay NPZ.")
    parser.add_argument("--preset", choices=PRESETS, help="Reproduce one of the existing paper figure sets.")
    parser.add_argument("--output", type=Path, help="Output filename prefix.")
    parser.add_argument("--times", type=float, nargs="+", help="Phase times in seconds.")
    parser.add_argument("--combined", action="store_true", help="Plot all phases in one PNG with a shared colorbar.")
    parser.add_argument("--time-to-goal", action="store_true", help="Interpret --times relative to the goal.")
    parser.add_argument("--highlight", type=int, nargs="*", help="Pedestrian IDs to highlight.")
    parser.add_argument("--x-limits", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--y-limits", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--history-s", type=float, help="Length of pedestrian history in seconds.")
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, help="Output resolution (default: preset value or 350).")
    args = parser.parse_args(argv)
    if args.input is None and args.preset is None:
        parser.error("Provide an input NPZ or --preset.")
    if args.preset is None and args.times is None:
        parser.error("Provide --times for a replay without a preset.")
    if (args.dpi is not None and args.dpi < 1) or (args.history_s is not None and args.history_s <= 0):
        parser.error("--dpi and --history-s must be positive.")
    for limits in (args.x_limits, args.y_limits):
        if limits is not None and (not np.all(np.isfinite(limits)) or limits[0] >= limits[1]):
            parser.error("View limits must be finite and increasing.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    preset = PRESETS.get(args.preset)
    input_path = args.input or SCRIPT_DIR / "paper_figures" / preset.filename
    output = args.output or (SCRIPT_DIR / "paper_figures" / preset.output_name if preset else input_path.with_suffix(""))
    phases = (tuple((time, f"phase_{index}", "") for index, time in enumerate(args.times, 1))
              if args.times is not None else preset.phases)
    for path in plot_case(
        input_path, output, phases=phases,
        highlights=tuple(args.highlight) if args.highlight is not None else (preset.highlights if preset else ()),
        xlim=tuple(args.x_limits) if args.x_limits else (preset.xlim if preset else None),
        ylim=tuple(args.y_limits) if args.y_limits else (preset.ylim if preset else None),
        time_to_goal=args.time_to_goal or (preset.time_to_goal if preset and args.times is None else False),
        history_s=args.history_s if args.history_s is not None else (preset.history_s if preset else None),
        title=args.title or (preset.title if preset else "Behavior"),
        dpi=args.dpi if args.dpi is not None else (350 if args.combined else preset.dpi if preset else 350),
        combined=args.combined,
    ):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
