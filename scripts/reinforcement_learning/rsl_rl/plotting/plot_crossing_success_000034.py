#!/usr/bin/env python3
"""Create standalone paper-ready top-view figures for Crossing · success_000034.

The robot trajectory is colored by measured world-frame speed (red = slow,
green = fast). Each phase is written as an independent PNG so it can be placed
separately in a paper.

Usage:
    python plot_crossing_success_000034.py
    python plot_crossing_success_000034.py --input /path/to/success_000034.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from plot_replay_common import (
    CBF_VECTOR_COLOR,
    EVALUATION_MEAN_PEDESTRIAN_RADIUS_M,
    GOAL_COLOR,
    NAVIGATION_VECTOR_COLOR,
    PEDESTRIAN_COLOR,
    _body_velocity_to_world,
    _navigation_goal_marker,
    _nearest_index,
    _plot_pedestrian_trajectories,
    _plot_robot_speed_trajectory,
    _robot_marker,
    _velocity_arrow,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "paper_figures" / "crossing_success_000034.npz"
DEFAULT_OUTPUT = SCRIPT_DIR / "paper_figures" / "crossing"

# The episode reaches its goal on the control step immediately following the
# last saved sample. Phase timing is expressed in the requested time-to-goal
# convention and resolved to the nearest recorded frame below.
PHASES = (
    (-7.52, "yield", "Yield: concede space to the approaching stream"),
    (-3.92, "assert", "Assert: commit to the gap and cross the stream"),
    (-2.00, "yield", "Yield: restore clearance after the crossing"),
)
# The three agents that shape the crossing decision.  Keep their colors fixed
# across the yield, assert, and final-yield panels: ID 1 is the initial
# approaching pedestrian, ID 4 constrains the crossing entry, and ID 9 is the
# pedestrian that motivates the final clearance-restoring yield.
INTERACTIVE_PEDESTRIAN_IDS = (1, 4, 9)
INTERACTIVE_PEDESTRIAN_COLORS = ("#0072B2", "#E69F00", "#CC79A7")
# One shared world-frame 10 x 10 m window for the three decision points.  It
# contains the crossing interaction and terminal goal, while earlier approach
# history can naturally clip at the edge.
WORLD_VIEW_X_LIMITS = (26.0, 36.0)
WORLD_VIEW_Y_LIMITS = (-35.0, -25.0)


def _plot_phase(
    ax: plt.Axes, *, frame: int, phase: tuple[float, str, str], robot_xy: np.ndarray,
    robot_yaw: np.ndarray, robot_speed: np.ndarray, goal_xy: np.ndarray, goal_yaw: float, pedestrian_xy: np.ndarray,
    pedestrian_active: np.ndarray, pedestrian_velocity: np.ndarray,
    navigation_velocity_body: np.ndarray, cbf_velocity_body: np.ndarray, norm: Normalize,
) -> LineCollection:
    active = pedestrian_active[frame]
    interactive_pedestrians = dict(zip(INTERACTIVE_PEDESTRIAN_IDS, INTERACTIVE_PEDESTRIAN_COLORS))
    _plot_pedestrian_trajectories(
        ax, pedestrian_xy, pedestrian_active, frame=frame, highlighted_pedestrians=interactive_pedestrians,
        alpha=0.28, linewidth=1.0,
    )
    goal = goal_xy[frame]
    _navigation_goal_marker(ax, goal, goal_yaw)
    # The completed route provides context, while the color-coded section is
    # restricted to motion available at this decision phase.
    ax.plot(robot_xy[frame:, 0], robot_xy[frame:, 1], color="#718096", linewidth=2.0,
            linestyle="--", alpha=0.65, zorder=3, label="Remaining robot route")
    collection = _plot_robot_speed_trajectory(ax, robot_xy[: frame + 1], robot_speed[: frame + 1], norm)
    other_ids = np.flatnonzero(active)
    other_ids = other_ids[~np.isin(other_ids, INTERACTIVE_PEDESTRIAN_IDS)]
    if other_ids.size:
        others = pedestrian_xy[frame, other_ids]
        other_velocities = pedestrian_velocity[frame, other_ids]
        ax.scatter(others[:, 0], others[:, 1], s=22, color=PEDESTRIAN_COLOR, edgecolor="white", linewidth=0.35, zorder=5)
        for agent, velocity in zip(others, other_velocities):
            ax.arrow(
                agent[0], agent[1], velocity[0] * 0.35, velocity[1] * 0.35,
                width=0.012, head_width=0.065, head_length=0.06, color=PEDESTRIAN_COLOR, zorder=5,
            )
    for pedestrian_id, color in interactive_pedestrians.items():
        if not active[pedestrian_id]:
            continue
        agent = pedestrian_xy[frame, pedestrian_id]
        velocity = pedestrian_velocity[frame, pedestrian_id]
        ax.add_patch(Circle(
            agent, EVALUATION_MEAN_PEDESTRIAN_RADIUS_M, facecolor="none",
            edgecolor=color, linewidth=1.35, linestyle=(0, (3.0, 2.0)), zorder=6,
        ))
        ax.scatter(*agent, s=60, color=color, edgecolor="white", linewidth=0.8, zorder=6)
        ax.arrow(agent[0], agent[1], velocity[0] * 0.35, velocity[1] * 0.35,
                 width=0.018, head_width=0.09, head_length=0.08, color=color, zorder=6)
    _robot_marker(ax, robot_xy[frame], float(robot_yaw[frame]))
    yaw = float(robot_yaw[frame])
    _velocity_arrow(
        ax, robot_xy[frame], _body_velocity_to_world(navigation_velocity_body[frame, :2], yaw),
        NAVIGATION_VECTOR_COLOR,
    )
    _velocity_arrow(
        ax, robot_xy[frame], _body_velocity_to_world(cbf_velocity_body[frame, :2], yaw), CBF_VECTOR_COLOR,
    )
    return collection


def plot_case(input_path: Path, output_stem: Path) -> list[Path]:
    data = np.load(input_path)
    time_s = data["time_s"]
    robot_xy = data["robot_position_xy"]
    robot_yaw = data["robot_yaw"]
    robot_speed = np.linalg.norm(data["robot_velocity_xy_world"], axis=1)
    goal_xy = data["goal_position_xy"]
    # Replays store the target position but not its sampled heading.  Since a
    # successful episode terminates only after orienting at the target, the
    # final robot yaw is the closest available target-heading estimate.
    goal_yaw = float(robot_yaw[-1])
    pedestrian_xy = data["pedestrian_position_xy"]
    pedestrian_active = data["pedestrian_active_mask"]
    pedestrian_velocity = data["pedestrian_velocity_xy_world"]
    navigation_velocity_body = data["navigation_policy_velocity_body"]
    cbf_velocity_body = data["cbf_filtered_command_velocity_body"]

    # Make colors comparable across panels without letting the stationary reset
    # frame dominate the color scale.
    norm = Normalize(vmin=0.50, vmax=max(1.55, float(np.percentile(robot_speed, 98))))
    goal_time_s = float(time_s[-1] + np.median(np.diff(time_s)))
    phase_frames = [_nearest_index(time_s, goal_time_s + phase[0]) for phase in PHASES]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13.0,
        "axes.labelsize": 14.5,
        "axes.titlesize": 18.0,
        "xtick.labelsize": 12.0,
        "ytick.labelsize": 12.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#374151",
        "xtick.color": "#4B5563",
        "ytick.color": "#4B5563",
    })
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for panel_number, (frame, (time_to_goal_s, slug, description)) in enumerate(zip(phase_frames, PHASES), start=1):
        fig, axis = plt.subplots(figsize=(7.6, 7.1), constrained_layout=True)
        robot_collection = _plot_phase(
            axis, frame=frame, phase=(time_to_goal_s, slug, description), robot_xy=robot_xy,
            robot_yaw=robot_yaw, robot_speed=robot_speed, goal_xy=goal_xy, goal_yaw=goal_yaw, pedestrian_xy=pedestrian_xy,
            pedestrian_active=pedestrian_active, pedestrian_velocity=pedestrian_velocity,
            navigation_velocity_body=navigation_velocity_body, cbf_velocity_body=cbf_velocity_body, norm=norm,
        )
        phase_name = slug.capitalize()
        axis.set(
            xlim=WORLD_VIEW_X_LIMITS, ylim=WORLD_VIEW_Y_LIMITS, aspect="equal", xlabel="x (m)", ylabel="y (m)",
            title=f"Crossing | {phase_name} | {time_to_goal_s:.2f} s to goal",
        )
        axis.grid(color="#DCE3EA", linewidth=0.6, zorder=0)
        axis.set_xticks(np.arange(WORLD_VIEW_X_LIMITS[0], WORLD_VIEW_X_LIMITS[1] + 1.0, 1.0))
        axis.set_yticks(np.arange(WORLD_VIEW_Y_LIMITS[0], WORLD_VIEW_Y_LIMITS[1] + 1.0, 1.0))
        axis.text(0.02, 0.02, description, transform=axis.transAxes, fontsize=12.5, color="#4A5568", va="bottom")
        colorbar = fig.colorbar(robot_collection, ax=axis, location="right", pad=0.02, shrink=0.78)
        colorbar.set_label("Robot speed (m/s): slow → fast", fontsize=12.5)
        colorbar.ax.tick_params(labelsize=11.5)
        axis.legend(handles=[
            Line2D([0], [0], color=PEDESTRIAN_COLOR, lw=1.7, label="Ped. traj."),
            Line2D([0], [0], color=INTERACTIVE_PEDESTRIAN_COLORS[0], lw=2.5, label="Int. peds."),
            Line2D([0], [0], color=INTERACTIVE_PEDESTRIAN_COLORS[0], lw=1.35, linestyle="--",
                   label="Agent rad."),
            Line2D([0], [0], color=GOAL_COLOR, marker=">", markersize=10, lw=1.8, linestyle="--", label="Nav. goal"),
            Line2D([0], [0], color="#718096", lw=2.0, linestyle="--", label="Robot (future)"),
            Line2D([0], [0], color=NAVIGATION_VECTOR_COLOR, marker=">", lw=2.5, label="Nav. cmd."),
            Line2D([0], [0], color=CBF_VECTOR_COLOR, marker=">", lw=2.5, label="CBF cmd."),
        ], loc="upper left", frameon=True, framealpha=0.92, fontsize=12.5)
        path = output_stem.parent / f"{output_stem.name}_{panel_number}.png"
        fig.savefig(path, dpi=350, bbox_inches="tight")
        outputs.append(path)
        plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Saved success_000034 NPZ replay.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output filename prefix.")
    args = parser.parse_args()
    for path in plot_case(args.input, args.output):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
