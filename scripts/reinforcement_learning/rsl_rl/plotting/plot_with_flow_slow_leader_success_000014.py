#!/usr/bin/env python3
"""Create paper-ready phases for With flow — slow leader · success_000014.

The replay records a successful overtake followed by a deliberate wait: the
robot lets the slow leader cross its goal before completing its own approach.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from plot_crossing_success_000034 import (
    CBF_VECTOR_COLOR,
    EVALUATION_MEAN_PEDESTRIAN_RADIUS_M,
    GOAL_COLOR,
    HIGHLIGHT_COLOR,
    NAVIGATION_VECTOR_COLOR,
    PEDESTRIAN_COLOR,
    _body_velocity_to_world,
    _nearest_index,
    _plot_pedestrian_trajectories,
    _plot_robot_speed_trajectory,
    _navigation_goal_marker,
    _robot_marker,
    _velocity_arrow,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "paper_figures" / "with_flow_slow_leader_success_000014.npz"
DEFAULT_OUTPUT = SCRIPT_DIR / "paper_figures" / "withflow"

# The saved trace ends one control step before success.  These phases are
# resolved to trace frames using that same time-to-goal convention.
PHASES = (
    (-6.48, "overtake", "Overtake: pass the slow leader"),
    (-2.32, "wait", "Wait: let the slow leader clear the goal first"),
    (0.00, "leader_clears_goal", "Leader clears goal before the robot reaches its own goal"),
)
SLOW_LEADER_ID = 0
WORLD_VIEW_X_LIMITS = (28.0, 38.0)
WORLD_VIEW_Y_LIMITS = (10.0, 20.0)


def _plot_phase(
    ax: plt.Axes, *, frame: int, robot_xy: np.ndarray, robot_yaw: np.ndarray,
    robot_speed: np.ndarray, goal_xy: np.ndarray, goal_yaw: float, pedestrian_xy: np.ndarray,
    pedestrian_active: np.ndarray, pedestrian_velocity: np.ndarray,
    navigation_velocity_body: np.ndarray, cbf_velocity_body: np.ndarray, norm: Normalize,
):
    _plot_pedestrian_trajectories(
        ax, pedestrian_xy, pedestrian_active, frame=frame, highlight_pedestrian=SLOW_LEADER_ID,
        alpha=0.28, linewidth=1.0,
    )
    goal = goal_xy[frame]
    _navigation_goal_marker(ax, goal, goal_yaw)
    ax.plot(
        robot_xy[frame:, 0], robot_xy[frame:, 1], color="#718096", linewidth=2.0,
        linestyle="--", alpha=0.65, zorder=3,
    )
    collection = _plot_robot_speed_trajectory(ax, robot_xy[: frame + 1], robot_speed[: frame + 1], norm)

    active = pedestrian_active[frame]
    other_ids = np.flatnonzero(active & (np.arange(active.size) != SLOW_LEADER_ID))
    if other_ids.size:
        others = pedestrian_xy[frame, other_ids]
        other_velocities = pedestrian_velocity[frame, other_ids]
        ax.scatter(others[:, 0], others[:, 1], s=22, color=PEDESTRIAN_COLOR, edgecolor="white", linewidth=0.35, zorder=5)
        for agent, velocity in zip(others, other_velocities):
            ax.arrow(
                agent[0], agent[1], velocity[0] * 0.35, velocity[1] * 0.35,
                width=0.012, head_width=0.065, head_length=0.06, color=PEDESTRIAN_COLOR, zorder=5,
            )
    if active[SLOW_LEADER_ID]:
        leader = pedestrian_xy[frame, SLOW_LEADER_ID]
        leader_velocity = pedestrian_velocity[frame, SLOW_LEADER_ID]
        ax.add_patch(Circle(
            leader, EVALUATION_MEAN_PEDESTRIAN_RADIUS_M, facecolor="none", edgecolor=HIGHLIGHT_COLOR,
            linewidth=1.35, linestyle=(0, (3.0, 2.0)), zorder=6,
        ))
        ax.scatter(*leader, s=60, color=HIGHLIGHT_COLOR, edgecolor="white", linewidth=0.8, zorder=7)
        ax.arrow(
            leader[0], leader[1], leader_velocity[0] * 0.35, leader_velocity[1] * 0.35,
            width=0.018, head_width=0.09, head_length=0.08, color=HIGHLIGHT_COLOR, zorder=7,
        )

    yaw = float(robot_yaw[frame])
    _robot_marker(ax, robot_xy[frame], yaw)
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
    # The stored replay has target position only; infer the terminal target
    # heading from the final successful robot pose.
    goal_yaw = float(robot_yaw[-1])
    pedestrian_xy = data["pedestrian_position_xy"]
    pedestrian_active = data["pedestrian_active_mask"]
    pedestrian_velocity = data["pedestrian_velocity_xy_world"]
    navigation_velocity_body = data["navigation_policy_velocity_body"]
    cbf_velocity_body = data["cbf_filtered_command_velocity_body"]

    norm = Normalize(vmin=0.15, vmax=max(1.55, float(np.percentile(robot_speed, 98))))
    goal_time_s = float(time_s[-1] + np.median(np.diff(time_s)))
    phase_frames = [_nearest_index(time_s, goal_time_s + phase[0]) for phase in PHASES]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13.0, "axes.labelsize": 14.5,
        "axes.titlesize": 18.0, "xtick.labelsize": 12.0, "ytick.labelsize": 12.0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": "#374151",
        "xtick.color": "#4B5563", "ytick.color": "#4B5563",
    })
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for panel_number, (frame, (time_to_goal_s, slug, description)) in enumerate(zip(phase_frames, PHASES), start=1):
        fig, axis = plt.subplots(figsize=(7.6, 7.1), constrained_layout=True)
        robot_collection = _plot_phase(
            axis, frame=frame, robot_xy=robot_xy, robot_yaw=robot_yaw, robot_speed=robot_speed,
            goal_xy=goal_xy, goal_yaw=goal_yaw, pedestrian_xy=pedestrian_xy, pedestrian_active=pedestrian_active,
            pedestrian_velocity=pedestrian_velocity, navigation_velocity_body=navigation_velocity_body,
            cbf_velocity_body=cbf_velocity_body, norm=norm,
        )
        phase_name = slug.replace("_", " ").capitalize()
        axis.set(
            xlim=WORLD_VIEW_X_LIMITS, ylim=WORLD_VIEW_Y_LIMITS, aspect="equal", xlabel="x (m)", ylabel="y (m)",
            title=f"With flow — slow leader | {phase_name} | {time_to_goal_s:.2f} s to goal",
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
            Line2D([0], [0], color=HIGHLIGHT_COLOR, lw=2.5, label="Slow lead."),
            Line2D([0], [0], color=HIGHLIGHT_COLOR, lw=1.35, linestyle="--", label="Agent rad."),
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
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for output in plot_case(args.input, args.output):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
