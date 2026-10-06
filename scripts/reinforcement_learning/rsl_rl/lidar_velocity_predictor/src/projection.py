"""Version 3 temporal LiDAR projection contract for ROS and Isaac Lab parity.

World XY endpoints and ray states are newest first. State 0 is unavailable,
1 is a measured no-return direction, and 2 is a surface reflection.
"""

from __future__ import annotations

import numpy as np

CONTRACT_VERSION = "temporal_lidar_v3"
WORLD_BINS = 256
FOV_BINS = 128
HISTORY = 4
MAX_DISTANCE_M = 20.0


def project_history(
    hit_xy: np.ndarray, ray_state: np.ndarray, evaluation_xy: np.ndarray, evaluation_yaw: float,
    max_distance: float = MAX_DISTANCE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (2,4,128) distance/validity and newest reflection winner ray IDs.

    Duplicate hits in a bin select the nearest range. Equal ranges select the
    lowest ray index. Unavailable bins have normalized distance 1, validity 0.
    """
    hit_xy = np.asarray(hit_xy, dtype=np.float64)
    ray_state = np.asarray(ray_state, dtype=np.uint8)
    evaluation_xy = np.asarray(evaluation_xy, dtype=np.float64)
    if hit_xy.ndim != 3 or hit_xy.shape[0] != HISTORY or hit_xy.shape[2] != 2:
        raise ValueError("hit_xy must have shape (4, rays, 2).")
    if ray_state.shape != hit_xy.shape[:2] or evaluation_xy.shape != (2,) or max_distance <= 0:
        raise ValueError("Invalid ray states, evaluation pose, or maximum range.")
    center = int((evaluation_yaw + np.pi) / (2 * np.pi) * WORLD_BINS) % WORLD_BINS
    arc = (center + np.arange(-FOV_BINS // 2, FOV_BINS // 2)) % WORLD_BINS
    output = np.zeros((2, HISTORY, FOV_BINS), dtype=np.float32)
    output[0] = 1.0
    newest_winner = np.full(FOV_BINS, -1, dtype=np.int32)
    for age in range(HISTORY):
        delta = hit_xy[age] - evaluation_xy
        distance = np.minimum(np.linalg.norm(delta, axis=-1), max_distance)
        angle = np.arctan2(delta[:, 1], delta[:, 0])
        world_bin = (np.floor((angle + np.pi) / (2 * np.pi) * WORLD_BINS).astype(np.int64)) % WORLD_BINS
        for local_bin, global_bin in enumerate(arc):
            rays = np.flatnonzero((world_bin == global_bin) & (ray_state[age] > 0))
            if rays.size == 0:
                continue
            output[1, age, local_bin] = 1.0
            hits = rays[ray_state[age, rays] == 2]
            if hits.size:
                winner = hits[np.argmin(distance[hits])]
                output[0, age, local_bin] = distance[winner] / max_distance
                if age == 0:
                    newest_winner[local_bin] = winner
    return output, newest_winner
