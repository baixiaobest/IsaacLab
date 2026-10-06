"""Regenerate projection_fixture_v3.json for ROS parity tests."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.projection import CONTRACT_VERSION, project_history


def main() -> None:
    hits = np.array([
        [[2, 0], [3, 0], [1, 1], [0, 0]],
        [[2, 0], [3, 0], [1, 1], [0, 0]],
        [[2, 0], [3, 0], [1, 1], [0, 0]],
        [[2, 0], [3, 0], [1, 1], [0, 0]],
    ], dtype=np.float64)
    states = np.array([[2, 2, 1, 0]] * 4, dtype=np.uint8)
    pose = [1.0, 0.0]
    yaw = 0.0
    tensor, winner = project_history(hits, states, np.array(pose), yaw)
    fixture = {
        "contract": CONTRACT_VERSION,
        "world_hit_xy_newest_first": hits.tolist(),
        "ray_state_newest_first": states.tolist(),
        "evaluation_xy": pose,
        "evaluation_yaw": yaw,
        "expected_input_2x4x128": tensor.tolist(),
        "expected_newest_winner_ray": winner.tolist(),
        "expected_newest_reflection_mask": (winner >= 0).tolist(),
    }
    (Path(__file__).parent / "projection_fixture_v3.json").write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    main()
