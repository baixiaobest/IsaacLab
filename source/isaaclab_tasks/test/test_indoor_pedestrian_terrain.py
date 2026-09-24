"""Unit coverage for deterministic indoor pedestrian terrain layouts."""

from __future__ import annotations

import numpy as np

from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.pedestrian_terrains import (
    INDOOR_MAX_OBSTACLES,
    INDOOR_WALL_INNER_Y_M,
    build_mixed_static_pedestrian_corridor,
    indoor_curriculum_band,
    indoor_obstacle_layout,
    indoor_position_is_clear,
    indoor_surface_points,
)


def test_indoor_layout_is_deterministic_and_matches_curriculum_bands():
    for level in range(10):
        boxes = indoor_obstacle_layout(level)
        assert np.array_equal(boxes, indoor_obstacle_layout(level))
        count_range, short_range, long_range, height_range = indoor_curriculum_band(level)
        assert count_range[0] <= len(boxes) <= min(count_range[1], INDOOR_MAX_OBSTACLES)
        short = np.minimum(boxes[:, 2], boxes[:, 3])
        long = np.maximum(boxes[:, 2], boxes[:, 3])
        assert np.all((short >= short_range[0]) & (short <= short_range[1]))
        assert np.all((long >= long_range[0]) & (long <= long_range[1]))
        assert np.all((boxes[:, 4] >= height_range[0]) & (boxes[:, 4] <= height_range[1]))
        # Protected flow/crossing lanes and endpoint goal regions stay empty.
        assert np.all(np.abs(boxes[:, 0]) > 1.75)
        assert np.all(np.abs(boxes[:, 1]) > 1.25)
        assert np.all(np.abs(boxes[:, 0]) < 6.5)
        assert np.all(np.abs(boxes[:, 1]) < 4.25)


def test_indoor_surface_points_cover_walls_and_thin_blocks_once():
    points, weights = indoor_surface_points(9)
    assert len(points) == len(weights)
    assert len(points) <= 512
    assert np.any(np.isclose(points[:, 1], -INDOOR_WALL_INNER_Y_M))
    assert np.any(np.isclose(points[:, 1], INDOOR_WALL_INNER_Y_M))
    assert np.all(weights > 0.0)


def test_indoor_clearance_rejects_blocks_and_walls():
    boxes = indoor_obstacle_layout(0)
    assert indoor_position_is_clear(0, np.array([0.0, 0.0]), 0.9, 0.6)
    assert not indoor_position_is_clear(0, boxes[0, :2], 0.9, 0.6)
    assert not indoor_position_is_clear(0, np.array([0.0, INDOOR_WALL_INNER_Y_M]), 0.9, 0.6)


def test_mixed_builder_allocates_four_equal_terrain_families():
    cfg = build_mixed_static_pedestrian_corridor()
    assert cfg.num_cols == 4
    assert tuple(cfg.sub_terrains) == (
        "discrete_obstacles", "concentric_maze", "ped_corridor", "indoor_ped_corridor"
    )

