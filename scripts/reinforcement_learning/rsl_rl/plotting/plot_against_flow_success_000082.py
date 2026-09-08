#!/usr/bin/env python3
"""Create paper-ready top-view phases for Against flow · success_000082."""

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
    NAVIGATION_VECTOR_COLOR,
    PEDESTRIAN_COLOR,
    _body_velocity_to_world,
    _continuous_past_track,
    _navigation_goal_marker,
    _nearest_index,
    _plot_pedestrian_trajectories,
    _plot_robot_speed_trajectory,
    _robot_marker,
    _velocity_arrow,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "paper_figures" / "against_flow_success_000082.npz"
DEFAULT_OUTPUT = SCRIPT_DIR / "paper_figures" / "againstflow"

# The replay reaches success on the next control step after its final saved
# frame.  Keep all four interaction agents visibly and chromatically stable:
# IDs 7/10 prompt the side-step, while 1/3 bound the later gap.
PHASES = (
    (-9.28, "side_step", "Side-step: yield laterally to the oncoming pedestrians"),
    (-6.32, "find_gap", "Find gap: identify the opening between two pedestrians"),
    (-4.48, "exploit_gap", "Exploit gap: complete the pass between the two pedestrians"),
)
INTERACTIVE_PEDESTRIAN_IDS = (7, 10, 1, 3)
INTERACTIVE_PEDESTRIAN_COLORS = ("#0072B2", "#E69F00", "#CC79A7", "#009E73")
PEDESTRIAN_HISTORY_S = 3.0
WORLD_VIEW_X_LIMITS = (41.0, 51.0)
WORLD_VIEW_Y_LIMITS = (-6.0, 4.0)


def _plot_marked_pedestrians(
    ax: plt.Axes, *, pedestrian_ids: tuple[int, ...], frame: int, pedestrian_xy: np.ndarray,
    pedestrian_active: np.ndarray, pedestrian_velocity: np.ndarray, history_frames: int,
) -> None:
    """Overlay the interaction agents so a multi-agent gap is explicit."""
    for pedestrian_id, color in zip(pedestrian_ids, INTERACTIVE_PEDESTRIAN_COLORS):
        track = _continuous_past_track(
            pedestrian_xy[:, pedestrian_id], pedestrian_active[:, pedestrian_id], frame
        )
        track[: max(0, frame - history_frames + 1)] = np.nan
        ax.plot(track[:, 0], track[:, 1], color=color, linewidth=2.5, alpha=0.94, zorder=2)
        if not pedestrian_active[frame, pedestrian_id]:
            continue
        agent = pedestrian_xy[frame, pedestrian_id]
        velocity = pedestrian_velocity[frame, pedestrian_id]
        ax.add_patch(Circle(
            agent, EVALUATION_MEAN_PEDESTRIAN_RADIUS_M, facecolor="none", edgecolor=color,
            linewidth=1.35, linestyle=(0, (3.0, 2.0)), zorder=6,
        ))
        ax.scatter(*agent, s=60, color=color, edgecolor="white", linewidth=0.8, zorder=7)
        ax.arrow(
            agent[0], agent[1], velocity[0] * 0.35, velocity[1] * 0.35,
            width=0.018, head_width=0.09, head_length=0.08, color=color, zorder=7,
        )


def _plot_phase(
    ax: plt.Axes, *, frame: int, robot_xy: np.ndarray,
    robot_yaw: np.ndarray, robot_speed: np.ndarray, goal_xy: np.ndarray, goal_yaw: float,
    pedestrian_xy: np.ndarray, pedestrian_active: np.ndarray, pedestrian_velocity: np.ndarray,
    navigation_velocity_body: np.ndarray, cbf_velocity_body: np.ndarray, history_frames: int, norm: Normalize,
):
    pedestrian_history = pedestrian_xy.copy()
    pedestrian_history[: max(0, frame - history_frames + 1)] = np.nan
    _plot_pedestrian_trajectories(
        ax, pedestrian_history, pedestrian_active, frame=frame, alpha=0.28, linewidth=1.0,
    )
    _plot_marked_pedestrians(
        ax, pedestrian_ids=INTERACTIVE_PEDESTRIAN_IDS, frame=frame, pedestrian_xy=pedestrian_xy,
        pedestrian_active=pedestrian_active, pedestrian_velocity=pedestrian_velocity, history_frames=history_frames,
    )
    _navigation_goal_marker(ax, goal_xy[frame], goal_yaw)
    ax.plot(
        robot_xy[frame:, 0], robot_xy[frame:, 1], color="#718096", linewidth=2.0,
        linestyle="--", alpha=0.65, zorder=3,
    )
    collection = _plot_robot_speed_trajectory(ax, robot_xy[: frame + 1], robot_speed[: frame + 1], norm)

    active = pedestrian_active[frame]
    marked_mask = np.zeros(active.size, dtype=bool)
    marked_mask[list(INTERACTIVE_PEDESTRIAN_IDS)] = True
    other_ids = np.flatnonzero(active & ~marked_mask)
    if other_ids.size:
        others = pedestrian_xy[frame, other_ids]
        other_velocities = pedestrian_velocity[frame, other_ids]
        ax.scatter(
            others[:, 0], others[:, 1], s=22, color=PEDESTRIAN_COLOR, edgecolor="white",
            linewidth=0.35, zorder=5,
        )
        for agent, velocity in zip(others, other_velocities):
            ax.arrow(
                agent[0], agent[1], velocity[0] * 0.35, velocity[1] * 0.35,
                width=0.012, head_width=0.065, head_length=0.06, color=PEDESTRIAN_COLOR, zorder=5,
            )

    yaw = float(robot_yaw[frame])
    _robot_marker(ax, robot_xy[frame], yaw)
    _velocity_arrow(
        ax, robot_xy[frame], _body_velocity_to_world(navigation_velocity_body[frame, :2], yaw),
        NAVIGATION_VECTOR_COLOR,
    )
    _velocity_arrow(
        ax, robot_xy[frame], _body_velocity_to_world(cbf_velocity_body[frame, :2], yaw),
        CBF_VECTOR_COLOR,
    )
    return collection


def plot_case(input_path: Path, output_stem: Path) -> list[Path]:
    data = np.load(input_path)
    time_s = data["time_s"]
    robot_xy = data["robot_position_xy"]
    robot_yaw = data["robot_yaw"]
    robot_speed = np.linalg.norm(data["robot_velocity_xy_world"], axis=1)
    goal_xy = data["goal_position_xy"]
    # Target heading was not saved in this replay.  The terminal successful
    # robot yaw is its closest available estimate.
    goal_yaw = float(robot_yaw[-1])
    pedestrian_xy = data["pedestrian_position_xy"]
    pedestrian_active = data["pedestrian_active_mask"]
    pedestrian_velocity = data["pedestrian_velocity_xy_world"]
    navigation_velocity_body = data["navigation_policy_velocity_body"]
    cbf_velocity_body = data["cbf_filtered_command_velocity_body"]

    norm = Normalize(vmin=0.15, vmax=max(1.55, float(np.percentile(robot_speed, 98))))
    goal_time_s = float(time_s[-1] + np.median(np.diff(time_s)))
    phase_frames = [_nearest_index(time_s, goal_time_s + phase[0]) for phase in PHASES]
    history_frames = int(round(PEDESTRIAN_HISTORY_S / np.median(np.diff(time_s))))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13.0, "axes.labelsize": 14.5,
        "axes.titlesize": 18.0, "xtick.labelsize": 12.0, "ytick.labelsize": 12.0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": "#374151",
        "xtick.color": "#4B5563", "ytick.color": "#4B5563",
    })
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for panel_number, (frame, (time_to_goal_s, slug, description)) in enumerate(
        zip(phase_frames, PHASES), start=1
    ):
        fig, axis = plt.subplots(figsize=(7.6, 7.1), constrained_layout=True)
        robot_collection = _plot_phase(
            axis, frame=frame, robot_xy=robot_xy,
            robot_yaw=robot_yaw, robot_speed=robot_speed, goal_xy=goal_xy, goal_yaw=goal_yaw,
            pedestrian_xy=pedestrian_xy, pedestrian_active=pedestrian_active,
            pedestrian_velocity=pedestrian_velocity, navigation_velocity_body=navigation_velocity_body,
            cbf_velocity_body=cbf_velocity_body, history_frames=history_frames, norm=norm,
        )
        axis.set(
            xlim=WORLD_VIEW_X_LIMITS, ylim=WORLD_VIEW_Y_LIMITS, aspect="equal", xlabel="x (m)", ylabel="y (m)",
            title=f"Against flow | {slug.replace('_', ' ').capitalize()} | {time_to_goal_s:.2f} s to goal",
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
            Line2D([0], [0], color=INTERACTIVE_PEDESTRIAN_COLORS[0], lw=1.35, linestyle="--", label="Agent rad."),
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
