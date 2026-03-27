"""RVO2 crowd simulation navigation environment for Unitree Go2.

Extends the end-to-end navigation env with simulated persons modelled as
kinematic capsule rigid bodies that move via the RVO2 collision-avoidance
algorithm.  The robot is fed into RVO2 as a dynamic obstacle so persons
naturally steer around it.
"""

from __future__ import annotations

import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass

from .e2e_navigation_env_cfg import (
    MySceneCfg,
    NavigationEnd2EndNoEncoderEnvCfg,
)

from isaaclab_tasks.manager_based.navigation.mdp.rvo2_crowd import RVO2CrowdManager

##
# Constants
##

NUM_PERSONS = 5
PERSON_RADIUS = 0.3          # capsule radius [m]
PERSON_HEIGHT = 1.2          # capsule cylinder height [m] → total ≈ 1.8 m
PERSON_Z = PERSON_RADIUS + PERSON_HEIGHT / 2.0   # spawn / standing height
PERSON_SPAWN_RADIUS = 4.0    # circle radius for initial placement [m]
PERSON_SPEED = 1.2           # max RVO2 speed [m/s]

# Distinct colours for each person (RGB 0-1)
_PERSON_COLORS = [
    (0.85, 0.20, 0.20),  # red
    (0.20, 0.75, 0.20),  # green
    (0.20, 0.40, 0.90),  # blue
    (0.90, 0.80, 0.10),  # yellow
    (0.90, 0.45, 0.10),  # orange
]


def _capsule_cfg(color: tuple[float, float, float]) -> sim_utils.CapsuleCfg:
    return sim_utils.CapsuleCfg(
        radius=PERSON_RADIUS,
        height=PERSON_HEIGHT,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=70.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


##
# Scene configuration
##

@configclass
class RVO2SceneCfg(MySceneCfg):
    """Extends the base Go2 navigation scene with 5 person capsules."""

    person_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Person_0",
        spawn=_capsule_cfg(_PERSON_COLORS[0]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, PERSON_Z)),
    )
    person_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Person_1",
        spawn=_capsule_cfg(_PERSON_COLORS[1]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.0, 0.0, PERSON_Z)),
    )
    person_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Person_2",
        spawn=_capsule_cfg(_PERSON_COLORS[2]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 2.0, PERSON_Z)),
    )
    person_3: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Person_3",
        spawn=_capsule_cfg(_PERSON_COLORS[3]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -2.0, PERSON_Z)),
    )
    person_4: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Person_4",
        spawn=_capsule_cfg(_PERSON_COLORS[4]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 1.5, PERSON_Z)),
    )


##
# Environment configurations
##

@configclass
class RVO2NavigationEnvCfg(NavigationEnd2EndNoEncoderEnvCfg):
    """Training/play config for the RVO2 crowd navigation environment."""

    scene: RVO2SceneCfg = RVO2SceneCfg(num_envs=1, env_spacing=10.0)

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
        # Keep terrain flat so persons walk on level ground
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class RVO2NavigationEnvCfg_PLAY(RVO2NavigationEnvCfg):
    """Play (visualisation) variant — single env, long episode, no contact terminations."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 10.0
        self.episode_length_s = 300.0  # 5-minute episodes so persons have time to move
        # Disable contact/velocity terminations — only time_out remains.
        # This prevents constant resets from a random (untrained) policy.
        self.terminations.base_contact = None
        self.terminations.base_contact_discrete_obstacles = None
        self.terminations.base_vel_out_of_limit = None


##
# Custom environment class
##

class RVO2NavigationEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv subclass that steps RVO2 crowd simulation each env step.

    Person capsules are kinematic rigid bodies; their world-space root states
    are overwritten each step from the RVO2 simulator output.
    """

    cfg: RVO2NavigationEnvCfg

    def __init__(self, cfg: RVO2NavigationEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._rvo2_manager: RVO2CrowdManager | None = None
        self._person_objects: list[RigidObject] = []
        self._person_goals: list[tuple[float, float]] = []
        self._setup_rvo2()

    # ------------------------------------------------------------------
    # RVO2 setup helpers
    # ------------------------------------------------------------------

    def _setup_rvo2(self):
        """Gather person rigid-objects from scene and initialise RVO2."""
        self._person_objects = []
        for i in range(NUM_PERSONS):
            name = f"person_{i}"
            try:
                self._person_objects.append(self.scene[name])
            except KeyError:
                pass
        if not self._person_objects:
            return

        positions, goals = [], []
        for i in range(len(self._person_objects)):
            angle = 2.0 * math.pi * i / len(self._person_objects)
            x = PERSON_SPAWN_RADIUS * math.cos(angle)
            y = PERSON_SPAWN_RADIUS * math.sin(angle)
            positions.append((x, y))
            opp = angle + math.pi
            goals.append((PERSON_SPAWN_RADIUS * math.cos(opp),
                          PERSON_SPAWN_RADIUS * math.sin(opp)))

        self._person_goals = goals
        self._rvo2_manager = RVO2CrowdManager(
            num_agents=len(self._person_objects),
            sim_dt=self.cfg.sim.dt * self.cfg.decimation,
            radius=PERSON_RADIUS,
            max_speed=PERSON_SPEED,
        )
        self._rvo2_manager.reset(positions, goals)

    # ------------------------------------------------------------------
    # Per-step helpers
    # ------------------------------------------------------------------

    def _get_robot_xy(self) -> tuple[float, float]:
        pos = self.scene["robot"].data.root_pos_w[0]
        env_origin = self.scene.env_origins[0]
        return float(pos[0].item()) - float(env_origin[0].item()), float(pos[1].item()) - float(env_origin[1].item())

    def _update_person_goals(self):
        """Assign a new goal when a person is within 0.5 m of its current goal."""
        if self._rvo2_manager is None:
            return
        positions_2d = self._rvo2_manager.get_positions()
        new_goals = list(self._person_goals)
        changed = False
        for i, (gx, gy) in enumerate(self._person_goals):
            px, py = float(positions_2d[i, 0]), float(positions_2d[i, 1])
            if math.sqrt((px - gx) ** 2 + (py - gy) ** 2) < 0.5:
                opp = math.atan2(py, px) + math.pi
                new_goals[i] = (PERSON_SPAWN_RADIUS * math.cos(opp),
                                PERSON_SPAWN_RADIUS * math.sin(opp))
                changed = True
        if changed:
            self._person_goals = new_goals
            self._rvo2_manager.set_goals(new_goals)

    def _write_persons_to_sim(self):
        """Teleport kinematic capsules to their RVO2 positions."""
        if self._rvo2_manager is None or not self._person_objects:
            return
        positions_2d = self._rvo2_manager.get_positions()
        # Account for env origin offset
        env_origin = self.scene.env_origins[0]  # shape (3,)
        ox, oy = float(env_origin[0].item()), float(env_origin[1].item())
        for i, person_obj in enumerate(self._person_objects):
            if i >= len(positions_2d):
                break
            x = float(positions_2d[i, 0]) + ox
            y = float(positions_2d[i, 1]) + oy
            # Build pose tensor: (1, 7) = [x, y, z, qw, qx, qy, qz]
            pose = person_obj.data.root_state_w[:, :7].clone()
            pose[:, 0] = x
            pose[:, 1] = y
            pose[:, 2] = PERSON_Z
            pose[:, 3] = 1.0   # qw
            pose[:, 4:7] = 0.0  # qx, qy, qz
            person_obj.write_root_pose_to_sim(pose)

    # ------------------------------------------------------------------
    # Overridden ManagerBasedRLEnv methods
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids):
        """Re-initialise RVO2 whenever the scene resets selected envs."""
        super()._reset_idx(env_ids)
        # Re-spawn persons at fresh circle positions after scene reset.
        self._setup_rvo2()
        # Immediately write so Isaac Sim sees the new positions this step.
        self._write_persons_to_sim()

    def step(self, action: torch.Tensor):
        if self._rvo2_manager is not None:
            rx, ry = self._get_robot_xy()
            self._rvo2_manager.update_robot_obstacle((rx, ry), radius=0.5)
            self._update_person_goals()
            self._rvo2_manager.step()
        # Call super FIRST — it handles physics + any internal env resets.
        result = super().step(action)
        # Write person positions AFTER super so they override any reset that
        # happened inside (internal resets snap persons back to init_state).
        self._write_persons_to_sim()
        return result

    def reset(self, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self._setup_rvo2()
        self._write_persons_to_sim()
        return result
