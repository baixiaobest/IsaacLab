#!/usr/bin/env python3
"""Plot three top-view phases of crossing episode 9705bc8caf1f467f94a4e236d26eafa7."""

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

from plot_replay_common import (
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
EPISODE_ID = "9705bc8caf1f467f94a4e236d26eafa7"
DEFAULT_INPUT = SCRIPT_DIR / "paper_figures" / f"crossing_{EPISODE_ID}.npz"
DEFAULT_OUTPUT = SCRIPT_DIR / "paper_figures" / f"crossing_{EPISODE_ID[:8]}"

# Centers of detected interactions: assert with agent 15 (0.16–2.32 s),
# assert with agent 3 (1.52–5.28 s), yield to agent 11 (3.60–5.28 s).
PHASES = (
    (1.20, "assert_15", "Assert across the path of pedestrian 15"),
    (4.40, "assert_yield", "Assert past pedestrian 3 while yielding to pedestrian 11"),
    (8.00, "final_approach", "Final approach to the navigation goal"),
)
HIGHLIGHTED_PEDESTRIANS = {15: "#0072B2", 3: "#E69F00", 11: "#CC79A7"}
VIEW_X = (64.5, 75.5)
VIEW_Y = (-33.5, -22.5)


def plot_case(input_path: Path, output_stem: Path) -> list[Path]:
    data = np.load(input_path)
    time_s = data["time_s"]
    robot_xy = data["robot_position_xy"]
    robot_yaw = data["robot_yaw"]
    robot_speed = np.linalg.norm(data["robot_velocity_xy_world"], axis=1)
    goal_xy = data["goal_position_xy"]
    pedestrian_xy = data["pedestrian_position_xy"]
    pedestrian_active = data["pedestrian_active_mask"]
    pedestrian_velocity = data["pedestrian_velocity_xy_world"]
    navigation_velocity = data["navigation_policy_velocity_body"]
    norm = Normalize(vmin=0.5, vmax=max(1.55, float(np.percentile(robot_speed, 98))))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for number, (requested_time, slug, description) in enumerate(PHASES, 1):
        frame = _nearest_index(time_s, requested_time)
        fig, ax = plt.subplots(figsize=(7.6, 7.1), constrained_layout=True)
        _plot_pedestrian_trajectories(
            ax, pedestrian_xy, pedestrian_active, frame=frame,
            highlighted_pedestrians=HIGHLIGHTED_PEDESTRIANS, alpha=0.28,
        )
        _navigation_goal_marker(ax, goal_xy[frame], float(robot_yaw[-1]))
        ax.plot(robot_xy[frame:, 0], robot_xy[frame:, 1], color="#718096",
                linewidth=2, linestyle="--", alpha=0.65, zorder=3)
        collection = _plot_robot_speed_trajectory(
            ax, robot_xy[:frame + 1], robot_speed[:frame + 1], norm)
        for agent_id in np.flatnonzero(pedestrian_active[frame]):
            position = pedestrian_xy[frame, agent_id]
            velocity = pedestrian_velocity[frame, agent_id]
            color = HIGHLIGHTED_PEDESTRIANS.get(int(agent_id), PEDESTRIAN_COLOR)
            highlighted = agent_id in HIGHLIGHTED_PEDESTRIANS
            if highlighted:
                ax.add_patch(Circle(position, EVALUATION_MEAN_PEDESTRIAN_RADIUS_M,
                                    facecolor="none", edgecolor=color, linewidth=1.35,
                                    linestyle=(0, (3, 2)), zorder=6))
            ax.scatter(*position, s=60 if highlighted else 22, color=color,
                       edgecolor="white", linewidth=0.8 if highlighted else 0.35, zorder=6)
            ax.arrow(position[0], position[1], velocity[0] * 0.35, velocity[1] * 0.35,
                     width=0.018 if highlighted else 0.012,
                     head_width=0.09 if highlighted else 0.065,
                     head_length=0.08 if highlighted else 0.06, color=color, zorder=6)
        _robot_marker(ax, robot_xy[frame], float(robot_yaw[frame]))
        _velocity_arrow(ax, robot_xy[frame],
                        _body_velocity_to_world(navigation_velocity[frame, :2], float(robot_yaw[frame])),
                        NAVIGATION_VECTOR_COLOR)
        ax.set(xlim=VIEW_X, ylim=VIEW_Y, aspect="equal", xlabel="x (m)", ylabel="y (m)",
               title=f"Crossing | {slug.replace('_', ' ').title()} | t = {time_s[frame]:.2f} s")
        ax.grid(color="#DCE3EA", linewidth=0.6, zorder=0)
        ax.text(0.02, 0.02, description, transform=ax.transAxes, fontsize=11.5,
                color="#4A5568", va="bottom")
        fig.colorbar(collection, ax=ax, location="right", pad=0.02, shrink=0.78,
                     label="Robot speed (m/s): slow → fast")
        ax.legend(handles=[
            Line2D([0], [0], color=PEDESTRIAN_COLOR, lw=1.7, label="Ped. traj."),
            *[Line2D([0], [0], color=color, lw=2.5, label=f"Ped. {agent_id}")
              for agent_id, color in HIGHLIGHTED_PEDESTRIANS.items()],
            Line2D([0], [0], color=GOAL_COLOR, marker=">", lw=1.8,
                   linestyle="--", label="Nav. goal"),
            Line2D([0], [0], color="#718096", lw=2, linestyle="--", label="Robot (future)"),
            Line2D([0], [0], color=NAVIGATION_VECTOR_COLOR, marker=">", lw=2.5,
                   label="Nav. cmd."),
        ], loc="upper left", frameon=True, framealpha=0.92, fontsize=10.5)
        output = output_stem.parent / f"{output_stem.name}_{number}.png"
        fig.savefig(output, dpi=350, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)
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
