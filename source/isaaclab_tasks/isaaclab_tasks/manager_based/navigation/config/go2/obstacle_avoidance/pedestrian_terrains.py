"""Corridor terrain generator for pedestrian obstacle-avoidance scenarios.

A single corridor terrain hosts BOTH the flow (with/against) and crossing scenarios — the
pedestrian crowd always flows along the corridor's local-x axis, and only the robot's
spawn/goal placement differs between scenarios (along-x for flow, across in local-y for
crossing). The per-episode scenario is selected per-env at reset (see
``reset_pedestrian_scenario_robot`` / ``CorridorPedestrianPose2dCommand``), so both scenarios
co-train on this one terrain.

The terrain uses a single sub-terrain type repeated across ``num_rows`` difficulty levels with
identical geometry — the per-env :attr:`terrain.terrain_levels` value (driven by the existing
``pose_2d_command_terrain_curriculum``) does not change the terrain shape, only the pedestrian
density/speed via :func:`isaaclab_tasks.manager_based.navigation.mdp.curriculums
.pedestrian_crowd_curriculum`.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from isaaclab.terrains import (
    FlatPatchSamplingCfg,
    HfConcentricMazeTerrainCfg,
    HfDiscretePositiveObstaclesTerrainCfg,
    TerrainGeneratorCfg,
)
from isaaclab.terrains.config.rough import FLAT_PATCH_HEIGHT_LIMITTED_CFG
from isaaclab.terrains.height_field.hf_terrains import discrete_positive_obstacles_terrain
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

# Number of curriculum difficulty levels (rows). Pedestrian count/speed ranges are
# linearly interpolated over terrain_levels in [0, PEDESTRIAN_CURRICULUM_MAX_LEVEL].
PEDESTRIAN_CURRICULUM_NUM_LEVELS = 10
PEDESTRIAN_CURRICULUM_MAX_LEVEL = PEDESTRIAN_CURRICULUM_NUM_LEVELS - 1

# Fixed benchmark rows shared by the static-obstacle and dynamic-crowd columns.
EVALUATION_LEVEL_COUNTS = (2, 4, 6, 8, 10, 12, 14, 16)
EVALUATION_NUM_LEVELS = len(EVALUATION_LEVEL_COUNTS)

# Sparse static obstacles shared by both corridor terrains (lidar remains meaningfully
# exercised alongside the dynamic pedestrians).
_SPARSE_OBSTACLE_KWARGS = dict(
    min_num_low_obstacles=0,
    max_num_low_obstacles=0,
    min_num_high_obstacles=0,
    max_num_high_obstacles=0,
    low_obstacle_max_height=0,
    high_obstacle_height_range=(1.0, 1.5),
    obstacle_width_range=(0.3, 0.6),
    platform_width=1.1,
)


# ---------------------------------------------------------------------------
# Indoor dynamic-pedestrian corridor
# ---------------------------------------------------------------------------
# Layout coordinates are terrain-local with the origin at the corridor centre.
# The layout is deliberately deterministic per curriculum row: the terrain mesh,
# pedestrian spawn checks, and social-force surface samples must describe exactly
# the same physical objects.
INDOOR_LAYOUT_SEED = 20260924
INDOOR_WALL_THICKNESS_M = 0.5
INDOOR_WALL_HEIGHT_M = 1.5
INDOOR_WALL_INNER_Y_M = 6.5
INDOOR_MAX_OBSTACLES = 8
INDOOR_OBSTACLE_CLEARANCE_M = 0.5


def indoor_curriculum_band(level: int) -> tuple[tuple[int, int], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return count, short-side, long-side, and height ranges for one indoor level."""
    bands = (
        ((1, 2), (0.3, 0.5), (0.5, 0.9), (1.0, 1.3)),
        ((2, 3), (0.3, 0.6), (0.5, 1.2), (1.0, 1.5)),
        ((3, 4), (0.3, 0.7), (0.6, 1.5), (1.0, 1.7)),
        ((4, 6), (0.3, 0.8), (0.6, 1.8), (1.0, 1.8)),
        ((6, 8), (0.3, 1.0), (0.6, 2.0), (1.0, 2.0)),
    )
    return bands[min(max(int(level), 0) // 2, len(bands) - 1)]


def indoor_layout_level(difficulty: float, num_levels: int = PEDESTRIAN_CURRICULUM_NUM_LEVELS) -> int:
    """Map generator difficulty to its curriculum row without row-jitter ambiguity."""
    return min(max(int(difficulty * num_levels), 0), num_levels - 1)


def indoor_obstacle_layout(level: int, seed: int = INDOOR_LAYOUT_SEED) -> np.ndarray:
    """Generate deterministic ``[center_x, center_y, size_x, size_y, height]`` rectangles.

    Obstacles are allocated among the four outer quadrants.  The central flow/crossing
    lanes and endpoint goal regions are never populated, so reset-time validation has
    a feasible resampling region at every difficulty.
    """
    count_range, short_range, long_range, height_range = indoor_curriculum_band(level)
    rng = np.random.default_rng(seed + int(level))
    count = int(rng.integers(count_range[0], count_range[1] + 1))
    boxes: list[tuple[float, float, float, float, float]] = []
    # Four safe outer regions: x leaves the crossing lane, y leaves the flow lane.
    regions = ((-6.25, -1.9, -4.0, -1.45), (-6.25, -1.9, 1.45, 4.0),
               (1.9, 6.25, -4.0, -1.45), (1.9, 6.25, 1.45, 4.0))
    order = rng.permutation(len(regions))
    for index in range(count):
        placed = False
        # Cycle through quadrants first; retry all quadrants if a dense row needs it.
        for retry in range(128):
            region = regions[order[(index + retry) % len(regions)]]
            short = float(rng.uniform(*short_range))
            long = float(rng.uniform(*long_range))
            size_x, size_y = (long, short) if rng.random() < 0.5 else (short, long)
            min_x, max_x, min_y, max_y = region
            if max_x - min_x <= size_x or max_y - min_y <= size_y:
                continue
            x = float(rng.uniform(min_x + size_x / 2, max_x - size_x / 2))
            y = float(rng.uniform(min_y + size_y / 2, max_y - size_y / 2))
            candidate = np.array([x, y, size_x, size_y], dtype=np.float64)
            collision = False
            for other in boxes:
                dx = abs(candidate[0] - other[0]) - (candidate[2] + other[2]) / 2
                dy = abs(candidate[1] - other[1]) - (candidate[3] + other[3]) / 2
                if max(dx, dy) < INDOOR_OBSTACLE_CLEARANCE_M:
                    collision = True
                    break
            if not collision:
                boxes.append((x, y, size_x, size_y, float(rng.uniform(*height_range))))
                placed = True
                break
        if not placed:
            raise RuntimeError(f"Unable to place indoor obstacle {index + 1}/{count} for level {level}.")
    return np.asarray(boxes, dtype=np.float32)


def indoor_surface_points(
    level: int, spacing_m: float = 0.2, thin_threshold_m: float = 0.4, seed: int = INDOOR_LAYOUT_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Return local XY social-force points and represented arc-length weights.

    Walls expose their inward faces.  A thin rectangle contributes one representative
    long edge, avoiding nearly coincident duplicate force samples.
    """
    if spacing_m <= 0.0:
        raise ValueError("Indoor surface-point spacing must be positive.")
    points: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    def _segment(start: tuple[float, float], end: tuple[float, float]):
        start_a, end_a = np.asarray(start, dtype=np.float32), np.asarray(end, dtype=np.float32)
        length = float(np.linalg.norm(end_a - start_a))
        count = max(2, int(np.ceil(length / spacing_m)) + 1)
        t = np.linspace(0.0, 1.0, count, dtype=np.float32)
        points.append(start_a[None, :] + t[:, None] * (end_a - start_a)[None, :])
        # Endpoint half-weights make the discrete sum approximate a line integral.
        sample_weight = np.full(count, length / (count - 1), dtype=np.float32)
        sample_weight[[0, -1]] *= 0.5
        weights.append(sample_weight)

    _segment((-10.0, -INDOOR_WALL_INNER_Y_M), (10.0, -INDOOR_WALL_INNER_Y_M))
    _segment((-10.0, +INDOOR_WALL_INNER_Y_M), (10.0, +INDOOR_WALL_INNER_Y_M))
    for x, y, size_x, size_y, _ in indoor_obstacle_layout(level, seed):
        x0, x1 = x - size_x / 2, x + size_x / 2
        y0, y1 = y - size_y / 2, y + size_y / 2
        if min(size_x, size_y) <= thin_threshold_m:
            # A deterministic single long edge preserves the requested thin-object behavior.
            _segment((x0, y1), (x1, y1)) if size_x >= size_y else _segment((x1, y0), (x1, y1))
        else:
            _segment((x0, y0), (x1, y0))
            _segment((x1, y0), (x1, y1))
            _segment((x1, y1), (x0, y1))
            _segment((x0, y1), (x0, y0))
    return np.concatenate(points, axis=0), np.concatenate(weights, axis=0)


def indoor_position_is_clear(
    level: int, xy: np.ndarray, obstacle_clearance_m: float, wall_clearance_m: float, seed: int = INDOOR_LAYOUT_SEED
) -> bool:
    """Whether a corridor-local point has the requested obstacle and wall clearance."""
    if abs(float(xy[1])) > INDOOR_WALL_INNER_Y_M - wall_clearance_m:
        return False
    for x, y, size_x, size_y, _ in indoor_obstacle_layout(level, seed):
        outside = np.maximum(np.abs(np.asarray(xy) - np.asarray((x, y))) - np.asarray((size_x, size_y)) / 2, 0.0)
        if float(np.linalg.norm(outside)) < obstacle_clearance_m:
            return False
    return True


@height_field_to_mesh
def indoor_pedestrian_corridor_terrain(difficulty: float, cfg: HfTerrainBaseCfg) -> np.ndarray:
    """Build side walls and the deterministic level-specific rectangular obstacle layout."""
    if cfg.layout_level_map:
        row = min(max(int(difficulty * len(cfg.layout_level_map)), 0), len(cfg.layout_level_map) - 1)
        level = cfg.layout_level_map[row]
    else:
        level = indoor_layout_level(difficulty)
    boxes = indoor_obstacle_layout(level, cfg.layout_seed)
    scale = cfg.horizontal_scale
    # ``height_field_to_mesh`` has already removed its one-cell outer border
    # before calling us.  Return exactly that interior shape (not the mesh-grid
    # vertex shape, which would be one cell larger in each dimension).
    height = np.zeros((int(round(cfg.size[0] / scale)), int(round(cfg.size[1] / scale))), dtype=np.int16)

    def _fill_box(center_x: float, center_y: float, size_x: float, size_y: float, box_height: float):
        x0 = max(0, int(np.floor((center_x - size_x / 2 + cfg.size[0] / 2) / scale)))
        x1 = min(height.shape[0], int(np.ceil((center_x + size_x / 2 + cfg.size[0] / 2) / scale)))
        y0 = max(0, int(np.floor((center_y - size_y / 2 + cfg.size[1] / 2) / scale)))
        y1 = min(height.shape[1], int(np.ceil((center_y + size_y / 2 + cfg.size[1] / 2) / scale)))
        height[x0:x1, y0:y1] = max(1, int(round(box_height / cfg.vertical_scale)))

    # Walls only occupy the lateral corridor boundaries; the longitudinal ends remain open.
    _fill_box(0.0, -(INDOOR_WALL_INNER_Y_M + INDOOR_WALL_THICKNESS_M / 2), cfg.size[0], INDOOR_WALL_THICKNESS_M, INDOOR_WALL_HEIGHT_M)
    _fill_box(0.0, +(INDOOR_WALL_INNER_Y_M + INDOOR_WALL_THICKNESS_M / 2), cfg.size[0], INDOOR_WALL_THICKNESS_M, INDOOR_WALL_HEIGHT_M)
    for box in boxes:
        _fill_box(*box)
    return height


@configclass
class IndoorPedestrianCorridorTerrainCfg(HfTerrainBaseCfg):
    """Height-field configuration for the deterministic indoor dynamic corridor."""

    function = indoor_pedestrian_corridor_terrain
    proportion: float = 1.0
    layout_seed: int = INDOOR_LAYOUT_SEED
    layout_level_map: tuple[int, ...] = ()
    """Optional evaluation-row to training-level mapping."""


# ---------------------------------------------------------------------------
# Unified corridor (hosts both the flow a/b and crossing c scenarios)
# ---------------------------------------------------------------------------
# Pedestrians always flow along local-x in roughly [-10, 10] (20 m long). The corridor is
# wide enough (local-y in roughly [-7, 7], 14 m) that the crossing scenario's robot can
# spawn near one edge (y ~ -5) and reach a goal near the other (y ~ +5) across the flow,
# while the flow scenario places its goal up/downstream along local-x. The named flat
# patches cover both scenarios' goal regions (goals are sampled analytically by
# ``CorridorPedestrianPose2dCommand``; the patches are kept for debug/inspection parity).

PEDESTRIAN_CORRIDOR = TerrainGeneratorCfg(
    size=(20.0, 14.0),
    border_width=10.0,
    num_rows=PEDESTRIAN_CURRICULUM_NUM_LEVELS,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "ped_corridor": HfDiscretePositiveObstaclesTerrainCfg(
            proportion=1.0,
            **_SPARSE_OBSTACLE_KWARGS,
            flat_patch_sampling={
                "target": FLAT_PATCH_HEIGHT_LIMITTED_CFG,
                # Flow scenario goal regions (up/downstream along local-x).
                "goal_west": FlatPatchSamplingCfg(
                    num_patches=200, patch_radius=0.4, x_range=(-9.0, -7.0), y_range=(-2.0, 2.0),
                    z_range=(-0.2, 0.2), max_height_diff=0.1,
                ),
                "goal_east": FlatPatchSamplingCfg(
                    num_patches=200, patch_radius=0.4, x_range=(7.0, 9.0), y_range=(-2.0, 2.0),
                    z_range=(-0.2, 0.2), max_height_diff=0.1,
                ),
                # Crossing scenario spawn/goal regions (across the flow along local-y).
                "spawn_south": FlatPatchSamplingCfg(
                    num_patches=200, patch_radius=0.4, x_range=(-1.5, 1.5), y_range=(-6.0, -4.0),
                    z_range=(-0.2, 0.2), max_height_diff=0.1,
                ),
                "goal_north": FlatPatchSamplingCfg(
                    num_patches=200, patch_radius=0.4, x_range=(-1.5, 1.5), y_range=(4.0, 6.0),
                    z_range=(-0.2, 0.2), max_height_diff=0.1,
                ),
            },
        ),
    },
)


# ---------------------------------------------------------------------------
# Mixed static + pedestrian corridor (co-trains static-obstacle columns with
# pedestrian-corridor columns on one terrain generator)
# ---------------------------------------------------------------------------
# The static sub-terrains reuse the "discrete_obstacles"/"concentric_maze" kwargs from
# DISCRETE_OBSTACLES_MAZE (12x12m) — the Hf* generators are size-agnostic, so they're reused
# here at the corridor's (20, 14) size.
_DISCRETE_OBSTACLES_KWARGS = dict(
    min_num_low_obstacles=0,
    max_num_low_obstacles=0,
    min_num_high_obstacles=0,
    max_num_high_obstacles=15,
    low_obstacle_max_height=0.3,
    high_obstacle_height_range=(1.0, 2.0),
    obstacle_width_range=(0.3, 1.5),
    platform_width=1.1,
)
_CONCENTRIC_MAZE_KWARGS = dict(
    fence_height_range=(0.5, 1.5),
    fence_spacing_range=(2.0, 3.0),
    opening_width_range=(1.0, 2.0),
    num_openings_range=(1, 3),
)


def _fixed_evaluation_discrete_obstacles_terrain(difficulty: float, cfg) -> object:
    """Generate one static benchmark row with its exact high-obstacle count.

    Curriculum terrain generation samples a small random offset within each row. Mapping that
    sampled difficulty back to its row preserves randomized obstacle placement while making the
    count itself exactly reproducible for every evaluation level.
    """
    row = min(int(difficulty * EVALUATION_NUM_LEVELS), EVALUATION_NUM_LEVELS - 1)
    count = EVALUATION_LEVEL_COUNTS[row]
    fixed_cfg = replace(cfg, min_num_high_obstacles=count, max_num_high_obstacles=count)
    return discrete_positive_obstacles_terrain(difficulty, fixed_cfg)

# Flat-patch sampling shared by the "ped_corridor" sub-terrain in PEDESTRIAN_CORRIDOR and in
# build_mixed_static_pedestrian_corridor (goal regions for the flow/crossing scenarios).
_PED_CORRIDOR_FLAT_PATCH_SAMPLING = {
    "target": FLAT_PATCH_HEIGHT_LIMITTED_CFG,
    "goal_west": FlatPatchSamplingCfg(
        num_patches=200, patch_radius=0.4, x_range=(-9.0, -7.0), y_range=(-2.0, 2.0),
        z_range=(-0.2, 0.2), max_height_diff=0.1,
    ),
    "goal_east": FlatPatchSamplingCfg(
        num_patches=200, patch_radius=0.4, x_range=(7.0, 9.0), y_range=(-2.0, 2.0),
        z_range=(-0.2, 0.2), max_height_diff=0.1,
    ),
    "spawn_south": FlatPatchSamplingCfg(
        num_patches=200, patch_radius=0.4, x_range=(-1.5, 1.5), y_range=(-6.0, -4.0),
        z_range=(-0.2, 0.2), max_height_diff=0.1,
    ),
    "goal_north": FlatPatchSamplingCfg(
        num_patches=200, patch_radius=0.4, x_range=(-1.5, 1.5), y_range=(4.0, 6.0),
        z_range=(-0.2, 0.2), max_height_diff=0.1,
    ),
}


def build_mixed_static_pedestrian_corridor(
    discrete_obstacles_proportion: float = 1.0,
    concentric_maze_proportion: float = 1.0,
    ped_corridor_proportion: float = 1.0,
    indoor_ped_corridor_proportion: float = 1.0,
    num_cols: int = 4,
) -> TerrainGeneratorCfg:
    """Build a terrain generator that splits columns between static obstacle/maze terrain
    (no pedestrians) and the pedestrian corridor (social-force crowd).

    Defaults assign one column to each family: discrete static obstacles, static maze,
    open pedestrian corridor, and indoor pedestrian corridor.  This gives 50% static,
    25% open dynamic, and 25% indoor dynamic rollouts.
    """
    return TerrainGeneratorCfg(
        size=(20.0, 14.0),
        border_width=10.0,
        num_rows=PEDESTRIAN_CURRICULUM_NUM_LEVELS,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains={
            "discrete_obstacles": HfDiscretePositiveObstaclesTerrainCfg(
                proportion=discrete_obstacles_proportion,
                **_DISCRETE_OBSTACLES_KWARGS,
                flat_patch_sampling={"target": FLAT_PATCH_HEIGHT_LIMITTED_CFG},
            ),
            "concentric_maze": HfConcentricMazeTerrainCfg(
                proportion=concentric_maze_proportion,
                **_CONCENTRIC_MAZE_KWARGS,
                flat_patch_sampling={"target": FLAT_PATCH_HEIGHT_LIMITTED_CFG},
            ),
            "ped_corridor": HfDiscretePositiveObstaclesTerrainCfg(
                proportion=ped_corridor_proportion,
                **_SPARSE_OBSTACLE_KWARGS,
                flat_patch_sampling=_PED_CORRIDOR_FLAT_PATCH_SAMPLING,
            ),
            "indoor_ped_corridor": IndoorPedestrianCorridorTerrainCfg(
                proportion=indoor_ped_corridor_proportion,
                flat_patch_sampling=_PED_CORRIDOR_FLAT_PATCH_SAMPLING,
            ),
        },
    )


def build_static_dynamic_evaluation_terrain() -> TerrainGeneratorCfg:
    """Return the fixed 8-row x 7-column static-plus-dynamic evaluation terrain.

    The first column is the training-compatible discrete-obstacle family. The remaining six
    columns are identical pedestrian corridors, one assigned to each benchmark scenario by the
    evaluator. Deliberately omit the maze family from this benchmark.
    """
    return TerrainGeneratorCfg(
        size=(20.0, 14.0),
        border_width=10.0,
        num_rows=EVALUATION_NUM_LEVELS,
        num_cols=7,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains={
            "discrete_obstacles": HfDiscretePositiveObstaclesTerrainCfg(
                proportion=1.0,
                function=_fixed_evaluation_discrete_obstacles_terrain,
                **_DISCRETE_OBSTACLES_KWARGS,
                flat_patch_sampling={"target": FLAT_PATCH_HEIGHT_LIMITTED_CFG},
            ),
            "ped_corridor": HfDiscretePositiveObstaclesTerrainCfg(
                proportion=6.0,
                **_SPARSE_OBSTACLE_KWARGS,
                flat_patch_sampling=_PED_CORRIDOR_FLAT_PATCH_SAMPLING,
            ),
        },
    )


def build_indoor_dynamic_evaluation_terrain() -> TerrainGeneratorCfg:
    """Return the 4-count x 3-scenario fixed indoor dynamic benchmark terrain."""
    return TerrainGeneratorCfg(
        size=(20.0, 14.0),
        border_width=10.0,
        num_rows=4,
        num_cols=3,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains={
            "indoor_ped_corridor": IndoorPedestrianCorridorTerrainCfg(
                proportion=1.0,
                layout_level_map=(0, 5, 7, 9),
                flat_patch_sampling=_PED_CORRIDOR_FLAT_PATCH_SAMPLING,
            ),
        },
    )
