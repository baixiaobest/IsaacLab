"""Tests for fixed terrain coverage used by LiDAR velocity collection."""

import pytest
import torch

from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.lidar_velocity_data_env import (
    FixedCoverageTerrainImporter,
)


def _importer() -> FixedCoverageTerrainImporter:
    importer = object.__new__(FixedCoverageTerrainImporter)
    importer.device = "cpu"
    return importer


def test_fixed_coverage_assigns_one_of_every_tile() -> None:
    importer = _importer()
    origins = torch.zeros(10, 4, 3)
    importer._compute_env_origins_curriculum(40, origins)
    tiles = importer.terrain_levels * 4 + importer.terrain_types
    assert torch.equal(torch.sort(tiles).values, torch.arange(40))
    assert torch.equal(importer.tile_replicas, torch.zeros(40, dtype=torch.long))


def test_fixed_coverage_repeats_tiles_for_larger_collection() -> None:
    importer = _importer()
    origins = torch.zeros(10, 4, 3)
    importer._compute_env_origins_curriculum(80, origins)
    tiles = importer.terrain_levels * 4 + importer.terrain_types
    assert torch.equal(tiles[:40], tiles[40:])
    assert torch.equal(importer.tile_replicas[:40], torch.zeros(40, dtype=torch.long))
    assert torch.equal(importer.tile_replicas[40:], torch.ones(40, dtype=torch.long))


def test_fixed_coverage_rejects_partial_replica() -> None:
    with pytest.raises(ValueError, match="positive multiple"):
        _importer()._compute_env_origins_curriculum(41, torch.zeros(10, 4, 3))
