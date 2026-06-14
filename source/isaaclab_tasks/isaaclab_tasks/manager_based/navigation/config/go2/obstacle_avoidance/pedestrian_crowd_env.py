"""Go2 navigation env that steps a vectorized social-force pedestrian crowd.

Each env step, :class:`SocialForceCrowdManager` advances all pedestrian agents on
``env.device`` and their poses are written into the ``pedestrians`` ``RigidObjectCollection``
(kinematic capsules). The robot does not participate in the social-force balance — pedestrians
are only physically present for lidar/contact sensing, so per-env crowd dynamics never depend
on other environments' robot state.
"""

from __future__ import annotations

import torch
from pxr import Gf

from isaaclab.assets.rigid_object_collection import RigidObjectCollection
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim.utils import get_current_stage
from isaaclab.terrains import TerrainImporter

from isaaclab_tasks.manager_based.navigation.mdp.events import reset_pedestrian_crowd
from isaaclab_tasks.manager_based.navigation.mdp.social_force_crowd import SocialForceCrowdManager
from isaaclab_tasks.manager_based.navigation.mdp.visual_utils import get_env_color

from .pedestrian_scene import PED_CAPSULE_HEIGHTS, PED_RADII


class PedestrianCrowdNavigationEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv subclass that steps a :class:`SocialForceCrowdManager` each step."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self.crowd_manager = SocialForceCrowdManager(cfg.social_force, self.num_envs, self.device)
        self.crowd_manager.set_radii(
            torch.tensor(PED_RADII, device=self.device),
            torch.tensor(PED_CAPSULE_HEIGHTS, device=self.device),
        )

        self._pedestrians: RigidObjectCollection = self.scene["pedestrians"]

        # Per-env mask, fixed for the whole run: True for envs pinned to the "ped_corridor"
        # terrain column, False for static obstacle/maze columns (mixed env). All-True for
        # the pure-pedestrian PEDESTRIAN_CORRIDOR terrain.
        terrain: TerrainImporter = self.scene["terrain"]
        env_terrain_names = terrain.get_env_terrain_names()
        self.is_pedestrian_env = torch.tensor(
            [name == "ped_corridor" for name in env_terrain_names], dtype=torch.bool, device=self.device
        )

        # Per-env episode scenario, sampled each reset by reset_pedestrian_scenario_robot:
        # 0 = flow (goal up/downstream), 1 = crossing (goal across the flow). Read by
        # CorridorPedestrianPose2dCommand to select the matching goal. Allocated here so it
        # exists before the first reset (events/commands read it during reset).
        self.pedestrian_scenario_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Persistent scratch buffer for _write_pedestrians_to_sim. Indices 4:6 (qx, qy)
        # are never written and stay zero for the lifetime of the env.
        self._pose_buf = torch.zeros(self.num_envs, self.crowd_manager.max_pedestrians, 7, device=self.device)

        # Seed the initial active count / speed range (the pedestrian curriculum only
        # updates these on subsequent episode resets), then place the crowd for all envs.
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        init_count = torch.where(
            self.is_pedestrian_env,
            torch.full((self.num_envs,), cfg.pedestrian_init_count, device=self.device, dtype=torch.long),
            torch.zeros(self.num_envs, device=self.device, dtype=torch.long),
        )
        init_speed_range = torch.tensor(cfg.pedestrian_init_speed_range, device=self.device).expand(
            self.num_envs, 2
        )
        slot_idx = torch.arange(self.crowd_manager.max_pedestrians, device=self.device).unsqueeze(0)
        self.crowd_manager.active_mask[:] = slot_idx < init_count.unsqueeze(1)
        self.crowd_manager._speed_range[:] = init_speed_range

        reset_pedestrian_crowd(self, all_env_ids, flow_dir=cfg.pedestrian_flow_dir)
        self._write_pedestrians_to_sim()
        self._randomize_per_env_colors()

    def _randomize_per_env_colors(self):
        """Give every pedestrian capsule and the robot in an env the same color, distinct across envs.

        ``clone()`` (used to replicate the ``Pedestrian_i``/``Robot`` prims across envs) copies
        each env's prims independently (``copy_from_source=True``), so each env's materials can
        be recolored/replaced without affecting other envs.
        """
        stage = get_current_stage()
        for env_id, env_prim_path in enumerate(self.scene.env_prim_paths):
            color = Gf.Vec3f(*get_env_color(env_id))
            for ped_idx in range(self.crowd_manager.max_pedestrians):
                shader_prim = stage.GetPrimAtPath(f"{env_prim_path}/Pedestrian_{ped_idx}/geometry/material/Shader")
                if shader_prim.IsValid():
                    shader_prim.GetAttribute("inputs:diffuseColor").Set(color)

    def _write_pedestrians_to_sim(self):
        pos_xy = self.crowd_manager.get_world_positions()  # (N, P, 2)
        z = self.crowd_manager.get_heights()  # (N, P)
        vel = self.crowd_manager.get_velocities()  # (N, P, 2)

        pose = self._pose_buf
        pose[..., 0:2] = pos_xy
        pose[..., 2] = z

        # Orient capsules toward their direction of travel (identity if stationary).
        yaw = torch.atan2(vel[..., 1], vel[..., 0])
        half_yaw = 0.5 * yaw
        pose[..., 3] = torch.cos(half_yaw)  # qw
        pose[..., 6] = torch.sin(half_yaw)  # qz

        self._pedestrians.write_object_pose_to_sim(pose, env_ids=None, object_ids=None)

    def step(self, action: torch.Tensor):
        result = super().step(action)
        robot_pos = self.scene["robot"].data.root_pos_w[:, :2]
        self.crowd_manager.step(dt=self.cfg.sim.dt * self.cfg.decimation, robot_pos=robot_pos)
        self._write_pedestrians_to_sim()
        return result

    def _reset_idx(self, env_ids):
        # event_manager.apply(mode="reset") inside super()._reset_idx runs
        # reset_pedestrian_crowd, which (re)spawns the crowd for env_ids.
        super()._reset_idx(env_ids)
        self._write_pedestrians_to_sim()
