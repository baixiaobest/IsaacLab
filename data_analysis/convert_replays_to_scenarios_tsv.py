#!/usr/bin/env python3
"""Convert evaluate.py episode-case replays (.npz + failure_cases.json) into the
scenarios.txt TSV schema consumed by the Crowd Navigation Replay artifact.

Usage:
    python convert_replays_to_scenarios_tsv.py <eval_output_run_dir> [-o out.txt]

<eval_output_run_dir> is the timestamped directory evaluate.py prints at the end,
e.g. .../output3/2026-09-12_19-53-55 (the one containing episode_cases/).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROBOT_RADIUS_M = 0.4  # matches SocialForceCrowdCfg.robot_radius / RVO2 eval default

COLUMNS = [
    "scenario", "dataset_id", "episode_id", "episode_number", "sample_bucket", "step",
    "time_s", "pedestrian_count", "robot_x_world", "robot_y_world", "robot_yaw",
    "robot_radius_m", "robot_vx_world", "robot_vy_world", "goal_x_world", "goal_y_world",
    "reward", "terminal_timeout", "agent_id", "agent_x_world", "agent_y_world",
    "agent_vx_world", "agent_vy_world", "agent_radius_m", "pedestrian_active",
]


def convert(run_dir: Path, out_path: Path) -> int:
    index_path = run_dir / "episode_cases" / "failure_cases.json"
    with index_path.open() as f:
        index = json.load(f)

    dataset_id = f"replay-{run_dir.name}"
    rows: list[str] = ["\t".join(COLUMNS)]
    episode_number = 0

    for case in index["cases"]:
        episode_number += 1
        episode_id = case["case_id"]
        scenario = case["scenario"]
        outcome = case["outcome"]
        npz_path = run_dir / "episode_cases" / case["replay_file"]
        data = np.load(npz_path)

        frame_count = int(case["frame_count"])
        time_s = data["time_s"]
        robot_xy = data["robot_position_xy"]
        robot_yaw = data["robot_yaw"]
        robot_vel = data["robot_velocity_xy_world"]
        goal_xy = data["goal_position_xy"]
        reward = data["reward"]
        ped_xy = data["pedestrian_position_xy"]
        ped_vel = data["pedestrian_velocity_xy_world"]
        ped_radius = data["pedestrian_radius"]
        ped_active = data["pedestrian_active_mask"]

        for i in range(frame_count):
            is_last = i == frame_count - 1
            terminal_timeout = "true" if (is_last and outcome == "timeout") else "false"
            active_ids = [j for j in range(ped_active.shape[1]) if bool(ped_active[i, j])]

            base = [
                scenario, dataset_id, episode_id, str(episode_number), str(i + 1), str(i),
                f"{time_s[i]:.6f}", str(len(active_ids)),
                f"{robot_xy[i, 0]:.6f}", f"{robot_xy[i, 1]:.6f}", f"{robot_yaw[i]:.6f}",
                f"{ROBOT_RADIUS_M:.6f}", f"{robot_vel[i, 0]:.6f}", f"{robot_vel[i, 1]:.6f}",
                f"{goal_xy[i, 0]:.6f}", f"{goal_xy[i, 1]:.6f}", f"{reward[i]:.6f}", terminal_timeout,
            ]

            if not active_ids:
                rows.append("\t".join(base + [""] * 6))
                continue

            for j in active_ids:
                row = base + [
                    str(j), f"{ped_xy[i, j, 0]:.6f}", f"{ped_xy[i, j, 1]:.6f}",
                    f"{ped_vel[i, j, 0]:.6f}", f"{ped_vel[i, j, 1]:.6f}",
                    f"{ped_radius[i, j]:.6f}", "true",
                ]
                rows.append("\t".join(row))

    out_path.write_text("\n".join(rows) + "\n")
    return episode_number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="evaluate.py timestamped output run directory")
    parser.add_argument("-o", "--output", type=Path, default=None, help="output .txt path")
    args = parser.parse_args()

    out_path = args.output or (args.run_dir / "scenarios.txt")
    n = convert(args.run_dir, out_path)
    print(f"Wrote {n} episodes to {out_path}")


if __name__ == "__main__":
    main()
