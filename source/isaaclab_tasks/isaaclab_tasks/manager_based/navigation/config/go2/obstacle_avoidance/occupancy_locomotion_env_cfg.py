"""Occupancy-grid variants of the Go2 low-level locomotion environment.

These configurations live with occupancy navigation because they own the L2
LiDAR sensor and the derived occupancy observation.  The normal locomotion
configuration intentionally remains policy-only.
"""

from __future__ import annotations

import math

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.observations import occupancy_grid_from_lidar
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains.config.rough import DISCRETE_OBSTACLES_ONLY
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.navigation.mdp.vis_utils import acquire_debug_draw, draw_occupancy_grid_points

from .locomotion_env_cfg import LocomotionVelEnvCfg, MySceneCfg, ObservationsCfg


@configclass
class OccupancyLidarSceneCfg(MySceneCfg):
    """Locomotion scene augmented with the Unitree L2 LiDAR."""

    l2_lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.28945, 0.0, -0.046825)),
        ray_alignment="yaw",
        max_distance=30.0,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=8,
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=1.0,
        ),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class OccupancyGridObsCfg(ObsGroup):
    """Ego-centric 32×32 occupancy grid at 0.4 m per cell."""

    occupancy_grid = ObsTerm(
        func=occupancy_grid_from_lidar,
        params={"sensor_cfg": SceneEntityCfg("l2_lidar"), "grid_size": 32, "grid_resolution": 0.4},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class OccupancyLocomotionObservationsCfg(ObservationsCfg):
    """Low-level observations plus a separate LiDAR occupancy group."""

    lidar: OccupancyGridObsCfg = OccupancyGridObsCfg()


@configclass
class LocomotionVelOccupancyEnvCfg(LocomotionVelEnvCfg):
    """Low-level locomotion configuration that exposes the L2 occupancy grid."""

    scene: OccupancyLidarSceneCfg = OccupancyLidarSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: OccupancyLocomotionObservationsCfg = OccupancyLocomotionObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.l2_lidar.update_period = self.decimation * self.sim.dt


@configclass
class LocomotionVelOccupancyEnvCfg_PLAY(LocomotionVelOccupancyEnvCfg):
    """Play variant of the occupancy-grid locomotion task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.policy.base_lin_vel.modifiers = None
        self.observations.policy.imu_ang_vel.modifiers = None
        self.observations.policy.imu_lin_acc.modifiers = None
        self.commands.base_velocity.resampling_time_range = (10000.0, 10000.0)


@configclass
class LocomotionVelOccupancyEnvCfg_ROLLOUT(LocomotionVelOccupancyEnvCfg):
    """Rollout variant for collecting occupancy-grid locomotion datasets."""

    def __post_init__(self):
        super().__post_init__()
        rollout_length = 10.0
        self.commands.base_velocity.resampling_time_range = (rollout_length / 4.0, rollout_length)
        self.episode_length_s = rollout_length
        self.observations.policy.enable_corruption = True
        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(rollout_length, rollout_length),
            rel_standing_envs=0.10,
            rel_rotating_standing_envs=0.10,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-2.0, 2.0),
                heading=(-math.pi, math.pi),
            ),
        )


@configclass
class LocomotionVelOccupancyEnvCfg_LIDAR_TEST(LocomotionVelOccupancyEnvCfg_PLAY):
    """Occupancy-grid visualisation task with discrete obstacles."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator = DISCRETE_OBSTACLES_ONLY
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.l2_lidar.debug_vis = True


class LocomotionLidarVizEnv(ManagerBasedRLEnv):
    """Environment that overlays the L2 occupancy grid in the viewport."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._occ_draw = acquire_debug_draw()
        self._occ_sensor_cfg = SceneEntityCfg("l2_lidar")

    def step(self, action: torch.Tensor):
        result = super().step(action)
        if self._occ_draw is not None:
            grid_flat = occupancy_grid_from_lidar(self, self._occ_sensor_cfg, grid_size=32, grid_resolution=0.4)
            sensor_pos = self.scene["l2_lidar"].data.pos_w
            grid_2d = grid_flat[0].reshape(32, 32).cpu().numpy()
            sensor_xy = (float(sensor_pos[0, 0].item()), float(sensor_pos[0, 1].item()))
            self._occ_draw.clear_points()
            draw_occupancy_grid_points(self._occ_draw, grid_2d, sensor_xy, grid_resolution=0.4)
        return result
