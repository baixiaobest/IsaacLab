"""Dynamic-crowd evaluation overlay for the RVO2/occupancy-map Go2 task.

Mirrors the mixed lidar benchmark (``obstacle_avoidance/mixed_scenario_mixins.py``)
for the occupancy-map policy:

- same corridor terrain (``PEDESTRIAN_CORRIDOR``), same 24 benchmark profiles
  (crossing / with-flow / against-flow x crowd counts up to 16), same relaxed
  goal-reached protocol and evaluation reset hooks;
- the robot senses persons through the occupancy grid -- the same 0.2 m / 50x50
  robot-centred sensor the policy was trained with, extended to 16 person slots;
- persons are driven by the GPU social-force model on the RVO2 scene's person
  capsules, matching the mixed benchmark's crowd dynamics.

The policy observations and architecture are untouched: this module re-registers
the task's ``-Play-v0`` id (see :func:`register_rvo2_eval_task`) with a benchmark
environment whose robot sensor/model are exactly the training ones.  It is
imported by ``scripts/reinforcement_learning/rsl_rl/evaluate.py`` only when the
evaluation task is an RVO2-crowd task, and it imports the base environment
(``rvo2_navigation_env_cfg.py``) from the checkout it runs in -- i.e. the
experiment's own occupancy-grid environment.
"""

from __future__ import annotations

import math

import torch

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp

from .obstacle_avoidance.pedestrian_scenario_mixins import (
    _CROSSING_NORTH_SPAWN_POSE_RANGE,
    _CROSSING_SOUTH_SPAWN_POSE_RANGE,
    _FLOW_SPAWN_POSE_RANGE,
    _ZERO_VELOCITY_RANGE,
)
from .obstacle_avoidance.pedestrian_terrains import PEDESTRIAN_CORRIDOR
from .rvo2_navigation_env_cfg import (
    GRID_CELLS,
    GRID_RESOLUTION,
    GRID_SIZE_M,
    NUM_PERSONS,
    PERSON_HEIGHT,
    PERSON_RADIUS,
    RVO2NavigationEnv,
    RVO2NavigationEnvCfg_PLAY,
    RVO2SceneCfg,
    STATIC_ROBOT_RADIUS,
    _PERSON_COLORS,
    _STATIC_ROBOT_POSITIONS,
    _capsule_cfg,
)
from isaaclab_tasks.manager_based.navigation.mdp.social_force_crowd import SocialForceCrowdCfg

# ---------------------------------------------------------------------------
# Benchmark constants (mirror mixed_scenario_mixins)
# ---------------------------------------------------------------------------

EVALUATION_CROWD_SPEED_RANGE = (0.9, 1.5)
"""Pedestrian desired-speed range used by the standardized dynamic-crowd benchmark."""

EVALUATION_CROWD_LATERAL_HEADING_MAX = math.radians(12.0)
"""Fixed maximum pedestrian heading offset from the corridor flow axis [rad]."""

EVALUATION_GOAL_REACHED_DISTANCE_THRESHOLD = 0.5
"""Maximum goal distance for an evaluation success [m]."""

EVALUATION_GOAL_REACHED_ANGULAR_THRESHOLD = math.radians(45.0)
"""Maximum goal-heading error for an evaluation success [rad]."""

EVALUATION_GOAL_REACHED_VELOCITY_THRESHOLD = 0.3
"""Maximum horizontal robot speed for an evaluation success [m/s]."""

EVALUATION_GOAL_REACHED_STAY_FOR_SECONDS = 0.1
"""Required continuous time satisfying the evaluation goal condition [s]."""

EVALUATION_SCENARIO_CODES = {"crossing": 0, "with_flow": 1, "against_flow": 2}
"""Stable scenario names and codes used by the dynamic-crowd benchmark and its artifacts."""

EVALUATION_EPISODE_LENGTH_S = 15.0
"""Benchmark episode length [s] (mirrors the mixed benchmark)."""

EVALUATION_MAX_PEDESTRIANS = 16
"""Benchmark crowd capacity (the RVO2 scene is extended from 10 to 16 person slots)."""

_EVAL_EXTRA_PERSON_COLORS = [
    (0.45, 0.20, 0.85),  # violet
    (0.90, 0.60, 0.05),  # amber
    (0.10, 0.70, 0.45),  # teal
    (0.85, 0.35, 0.25),  # vermilion
    (0.35, 0.55, 0.90),  # steel blue
    (0.75, 0.15, 0.60),  # magenta
]
"""Extra capsule colours for person slots 10..15 (the first 10 reuse the training colours)."""

_EVAL_PERSON_COLORS = _PERSON_COLORS + _EVAL_EXTRA_PERSON_COLORS


def _eval_person_count(env: ManagerBasedRLEnv) -> int:
    """Number of person capsules the occupancy grid should rasterise.

    The training env has ``NUM_PERSONS`` capsules; the benchmark env exposes a
    social-force crowd manager whose capacity is the slot count (16).
    """
    crowd = getattr(env, "crowd_manager", None)
    if crowd is not None and getattr(crowd, "max_pedestrians", None):
        return int(crowd.max_pedestrians)
    return NUM_PERSONS


def eval_mixed_occupancy_grid(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Rasterize dynamic persons and static robots in each robot's yaw frame.

    Identical sensor to the training ``mixed_occupancy_grid`` (0.2 m / 50x50
    robot-centred grid, flattened ``[num_envs, 2500]``), except the person count
    follows the crowd manager's capacity so the benchmark crowd (up to 16) is
    fully visible.
    """
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w
    robot_yaw = yaw_quat(robot.data.root_quat_w)
    occupant_pos = []
    occupant_radius = []
    for i in range(_eval_person_count(env)):
        occupant_pos.append(env.scene[f"person_{i}"].data.root_pos_w)
        occupant_radius.append(PERSON_RADIUS)
    for i in range(len(_STATIC_ROBOT_POSITIONS)):
        occupant_pos.append(env.scene[f"static_robot_{i}"].data.root_pos_w)
        occupant_radius.append(STATIC_ROBOT_RADIUS)

    positions = torch.stack(occupant_pos, dim=1)
    delta_w = positions - robot_pos.unsqueeze(1)
    quat = robot_yaw.unsqueeze(1).expand(-1, positions.shape[1], -1).reshape(-1, 4)
    delta_b = quat_apply_inverse(quat, delta_w.reshape(-1, 3)).reshape(delta_w.shape)[..., :2]

    grid = torch.zeros((env.num_envs, GRID_CELLS * GRID_CELLS), device=robot_pos.device, dtype=torch.float32)
    half = GRID_SIZE_M / 2.0
    max_radius_cells = math.ceil(max(occupant_radius) / GRID_RESOLUTION)
    offsets = torch.arange(-max_radius_cells, max_radius_cells + 1, device=robot_pos.device)
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    offset_x = offset_x.flatten()
    offset_y = offset_y.flatten()

    cols = torch.floor((delta_b[..., 0] + half) / GRID_RESOLUTION).long()
    rows = torch.floor((delta_b[..., 1] + half) / GRID_RESOLUTION).long()
    for occupant_index, radius in enumerate(occupant_radius):
        radius_cells = radius / GRID_RESOLUTION
        footprint = offset_x.square() + offset_y.square() <= radius_cells**2
        cell_cols = cols[:, occupant_index, None] + offset_x[footprint]
        cell_rows = rows[:, occupant_index, None] + offset_y[footprint]
        valid = (cell_cols >= 0) & (cell_cols < GRID_CELLS) & (cell_rows >= 0) & (cell_rows < GRID_CELLS)
        cell_indices = cell_rows * GRID_CELLS + cell_cols
        env_indices = torch.arange(env.num_envs, device=grid.device)[:, None].expand_as(cell_indices)
        grid[env_indices[valid], cell_indices[valid]] = 1.0
    return grid


# ---------------------------------------------------------------------------
# Benchmark scene: extend the RVO2 scene from 10 to 16 person capsules.
# ---------------------------------------------------------------------------


def _eval_capsule_cfg(index: int) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Person_{index}",
        spawn=_capsule_cfg(_EVAL_PERSON_COLORS[index]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.5, -2.5 + 1.0 * (index - 10), PERSON_HEIGHT / 2.0 + PERSON_RADIUS)),
    )


@configclass
class RVO2EvalSceneCfg(RVO2SceneCfg):
    """RVO2 scene with 16 person capsules (10 training slots + 6 benchmark slots)."""

    person_10: RigidObjectCfg = _eval_capsule_cfg(10)
    person_11: RigidObjectCfg = _eval_capsule_cfg(11)
    person_12: RigidObjectCfg = _eval_capsule_cfg(12)
    person_13: RigidObjectCfg = _eval_capsule_cfg(13)
    person_14: RigidObjectCfg = _eval_capsule_cfg(14)
    person_15: RigidObjectCfg = _eval_capsule_cfg(15)


# ---------------------------------------------------------------------------
# Benchmark environment configuration
# ---------------------------------------------------------------------------


@configclass
class RVO2CrowdEvalEnvCfg(RVO2NavigationEnvCfg_PLAY):
    """Dynamic-crowd benchmark config for the occupancy-map RVO2 task.

    The robot sensor/model are the training ones; only the terrain, crowd model,
    scenario command, and goal protocol follow the mixed benchmark.
    """

    scene: RVO2EvalSceneCfg = RVO2EvalSceneCfg(num_envs=1, env_spacing=10.0)

    # Social-force crowd backend (the benchmark's person model).
    crowd_backend: str = "social_force"
    social_force: SocialForceCrowdCfg = SocialForceCrowdCfg(max_pedestrians=EVALUATION_MAX_PEDESTRIANS)
    pedestrian_init_count: int = 2
    pedestrian_init_speed_range: tuple[float, float] = EVALUATION_CROWD_SPEED_RANGE
    pedestrian_flow_dir: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        # The benchmark env is constructed with the benchmark env class; the base
        # sensor terms are kept, only the crowd rasterisation follows the crowd.
        from isaaclab.managers import ObservationTermCfg as ObsTerm

        self.episode_length_s = EVALUATION_EPISODE_LENGTH_S
        self.observations.policy.osbtacles_scan = ObsTerm(func=eval_mixed_occupancy_grid)

        # Same corridor terrain as the mixed benchmark; spread envs over tiles.
        self.scene.terrain.terrain_generator = PEDESTRIAN_CORRIDOR
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.curriculum = True

        self.social_force.lateral_heading_max = EVALUATION_CROWD_LATERAL_HEADING_MAX

        # Benchmark goal protocol (accepted arrival instead of the stricter training pose).
        self.terminations.goal_reached = DoneTerm(
            func=nav_mdp.pose_2d_command_goal_reached,
            params={
                "distance_threshold": EVALUATION_GOAL_REACHED_DISTANCE_THRESHOLD,
                "angular_threshold": EVALUATION_GOAL_REACHED_ANGULAR_THRESHOLD,
                "velocity_threshold": EVALUATION_GOAL_REACHED_VELOCITY_THRESHOLD,
                "stay_for_seconds": EVALUATION_GOAL_REACHED_STAY_FOR_SECONDS,
            },
        )

        # Benchmark fixes terrain and crowd difficulty rather than advancing curricula.
        self.curriculum.terrain_levels = None
        self.curriculum.random_rough_level = None
        self.curriculum.discrete_obstacles_level = None

        # Corridor flow/crossing goals (same command term as the mixed benchmark).
        self.commands.pose_2d_command = nav_mdp.CorridorPedestrianPose2dCommandCfg(
            asset_name="robot",
            simple_heading=False,
            stationary_prob=0.0,
            ranges=nav_mdp.CorridorPedestrianPose2dCommandCfg.Ranges(
                pos_x=(0.0, 0.0),  # Not used
                pos_y=(0.0, 0.0),  # Not used
                heading=(-math.pi, math.pi),
                pos_z=(0.3, 0.4),
            ),
            resampling_time_range=(EVALUATION_EPISODE_LENGTH_S + 0.1, EVALUATION_EPISODE_LENGTH_S + 0.1),
            goal_distance_range=(4.0, 8.0),
            corridor_half_length=9.0,
            corridor_half_width=2.0,
            goal_y=5.0,
            crossing_x_range=(-1.5, 1.5),
            debug_vis=False,
        )

        # Evaluation resets: scenario robot pose + crowd respawn (same hooks as the
        # mixed benchmark).  The fallback in reset_evaluation_pedestrian_scenario_robot
        # covers the bootstrap reset before profiles are installed.
        self.events.reset_base = EventTerm(
            func=nav_mdp.reset_evaluation_pedestrian_scenario_robot,
            mode="reset",
            params={
                "flow_pose_range": _FLOW_SPAWN_POSE_RANGE,
                "crossing_south_pose_range": _CROSSING_SOUTH_SPAWN_POSE_RANGE,
                "crossing_north_pose_range": _CROSSING_NORTH_SPAWN_POSE_RANGE,
                "velocity_range": _ZERO_VELOCITY_RANGE,
                "speed_range": EVALUATION_CROWD_SPEED_RANGE,
            },
        )
        self.events.reset_pedestrians = EventTerm(
            func=nav_mdp.reset_pedestrian_crowd,
            mode="reset",
            params={"flow_dir": 1.0},
        )


# ---------------------------------------------------------------------------
# Benchmark environment: social-force crowd on the RVO2 scene.
# ---------------------------------------------------------------------------


class RVO2CrowdEvalNavigationEnv(RVO2NavigationEnv):
    """Occupancy-grid Go2 env with a GPU social-force pedestrian crowd.

    The robot sensor/observation layout and static robots are exactly the training
    ones; only the person model is swapped (per-env Python-RVO2 sims -> vectorized
    social-force crowd), and the crowd is written into the scene's person capsules
    each step so the occupancy-grid observation sees it.
    """

    cfg: RVO2CrowdEvalEnvCfg

    def _setup_rvo2(self, env_ids=None):
        """No-op: the benchmark uses the social-force crowd backend."""
        return

    def __init__(self, cfg: RVO2CrowdEvalEnvCfg, **kwargs):
        # Base constructor runs the RVO2 scene + occupancy-grid buffer; the RVO2
        # per-env sims are disabled by the no-op _setup_rvo2 above.
        super().__init__(cfg, **kwargs)

        from isaaclab_tasks.manager_based.navigation.mdp.events import reset_pedestrian_crowd
        from isaaclab_tasks.manager_based.navigation.mdp.social_force_crowd import SocialForceCrowdManager

        if cfg.social_force.max_pedestrians > EVALUATION_MAX_PEDESTRIANS:
            raise ValueError(
                "The benchmark crowd capacity exceeds the scene person slots: "
                f"{cfg.social_force.max_pedestrians} > {EVALUATION_MAX_PEDESTRIANS}."
            )
        self.crowd_manager = SocialForceCrowdManager(cfg.social_force, self.num_envs, self.device)
        self.crowd_manager.set_radii(
            torch.full((cfg.social_force.max_pedestrians,), PERSON_RADIUS, device=self.device),
            torch.full((cfg.social_force.max_pedestrians,), PERSON_HEIGHT, device=self.device),
        )

        # All envs sit on the ped_corridor terrain column in the benchmark.
        terrain = self.scene["terrain"]
        self.is_pedestrian_env = torch.tensor(
            [name == "ped_corridor" for name in terrain.get_env_terrain_names()],
            dtype=torch.bool,
            device=self.device,
        )
        # Per-env episode scenario, sampled each reset by the evaluation reset hook
        # and read by CorridorPedestrianPose2dCommand for the goal placement.
        self.pedestrian_scenario_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Seed the initial active count / speed range, then place the crowd for all envs.
        all_ids = torch.arange(self.num_envs, device=self.device)
        init_count = torch.full(
            (self.num_envs,), cfg.pedestrian_init_count, device=self.device, dtype=torch.long
        )
        init_speed = torch.tensor(cfg.pedestrian_init_speed_range, device=self.device).expand(self.num_envs, 2)
        slot_idx = torch.arange(self.crowd_manager.max_pedestrians, device=self.device).unsqueeze(0)
        self.crowd_manager.active_mask[:] = slot_idx < init_count.unsqueeze(1)
        self.crowd_manager._speed_range[:] = init_speed

        reset_pedestrian_crowd(self, all_ids, flow_dir=cfg.pedestrian_flow_dir)
        self._write_crowd_to_sim()
        self._reset_static_robots()

    def _write_crowd_to_sim(self) -> None:
        """Write the social-force crowd world poses into the person capsules."""
        if self.crowd_manager is None:
            return
        pos_xy = self.crowd_manager.get_world_positions()  # (N, P, 2) world-XY
        heights = self.crowd_manager.get_heights()         # (N, P)
        vel = self.crowd_manager.get_velocities()          # (N, P, 2)
        active = self.crowd_manager.get_active_mask()      # (N, P)
        # Park inactive slots far outside the ±5 m occupancy window: the top-down
        # grid ignores z, so the crowd manager's z=-50 parking alone would still
        # rasterise them into the sensor.
        pos_xy = torch.where(active.unsqueeze(-1), pos_xy, torch.full_like(pos_xy, 1e3))
        yaw = torch.atan2(vel[..., 1], vel[..., 0])
        half_yaw = 0.5 * yaw
        for i in range(self.crowd_manager.max_pedestrians):
            try:
                person = self.scene[f"person_{i}"]
            except KeyError:
                continue
            pose = person.data.root_state_w[:, :7].clone()
            pose[:, 0] = pos_xy[:, i, 0]
            pose[:, 1] = pos_xy[:, i, 1]
            pose[:, 2] = heights[:, i]
            pose[:, 3] = torch.cos(half_yaw[:, i])
            pose[:, 6] = torch.sin(half_yaw[:, i])
            person.write_root_pose_to_sim(pose)

    def _compute_occupancy_grid(self) -> torch.Tensor:
        self._occupancy_grid = eval_mixed_occupancy_grid(self).reshape(
            self.num_envs, GRID_CELLS, GRID_CELLS
        )
        return self._occupancy_grid

    def step(self, action: torch.Tensor):
        # Physics step first (mirrors the mixed benchmark env): the observation
        # computed inside sees the crowd state from the previous crowd advance.
        result = ManagerBasedRLEnv.step(self, action)
        robot_pos = self.scene["robot"].data.root_pos_w[:, :2]
        self.crowd_manager.step(dt=self.cfg.sim.dt * self.cfg.decimation, robot_pos=robot_pos)
        self._write_crowd_to_sim()
        self._compute_occupancy_grid()
        self.extras["occupancy_grid"] = self._occupancy_grid
        return result

    def _reset_idx(self, env_ids):
        # event_manager.apply(mode="reset") inside the base reset runs the
        # evaluation scenario robot reset and the crowd respawn.
        ManagerBasedRLEnv._reset_idx(self, env_ids)
        self._write_crowd_to_sim()
        self._reset_static_robots()
        self._compute_occupancy_grid()


# ---------------------------------------------------------------------------
# Evaluator integration
# ---------------------------------------------------------------------------


def configure_rvo2_dynamic_crowd_evaluation(env_cfg: RVO2CrowdEvalEnvCfg) -> RVO2CrowdEvalEnvCfg:
    """No-op overlay: the benchmark cfg already carries the full setup.

    Kept for symmetry with the mixed benchmark's ``configure_dynamic_crowd_evaluation``
    (evaluate.py calls it before constructing the environment).
    """
    return env_cfg


def install_rvo2_dynamic_crowd_evaluation_profiles(env, pedestrian_counts, scenario_codes) -> None:
    """Install fixed profile tensors consumed by the benchmark reset hooks.

    Scenario codes are ``0=crossing``, ``1=with_flow``, and ``2=against_flow``.
    Inputs must contain one assignment per vector environment and are copied to
    the environment device.
    """
    counts = torch.as_tensor(pedestrian_counts, device=env.device, dtype=torch.long)
    scenarios = torch.as_tensor(scenario_codes, device=env.device, dtype=torch.long)
    if counts.numel() != env.num_envs or scenarios.numel() != env.num_envs:
        raise ValueError("Evaluation profiles must provide exactly one count and scenario per environment.")
    if torch.any((counts < 1) | (counts > env.crowd_manager.max_pedestrians)):
        raise ValueError("Evaluation pedestrian counts must fit the configured crowd capacity.")
    if torch.any((scenarios < 0) | (scenarios > 2)):
        raise ValueError("Evaluation scenario codes must be 0 (crossing), 1 (with), or 2 (against).")

    env.evaluation_pedestrian_count = counts
    env.evaluation_scenario = scenarios
    env.evaluation_flow_goal_direction = torch.where(
        scenarios == 1,
        torch.ones_like(scenarios),
        torch.where(scenarios == 2, -torch.ones_like(scenarios), torch.zeros_like(scenarios)),
    )


def register_rvo2_eval_task() -> None:
    """Re-register the RVO2-Crowd ``-Play-v0`` task with the benchmark environment.

    Called by evaluate.py before the task config is resolved, so the benchmark env
    class/cfg (social-force crowd, 16 person slots, corridor terrain) is used while
    the policy observations/architecture remain the training ones.
    """
    import gymnasium as gym

    from . import agents  # noqa: F401  (imports the policy cfg module for the entry point string)

    gym.register(
        id="Isaac-Navigation-RVO2-Crowd-Unitree-Go2-Play-v0",
        entry_point=__name__ + ":RVO2CrowdEvalNavigationEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": __name__ + ":RVO2CrowdEvalEnvCfg",
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RVO2CrowdPPORunnerCfg_v0"
            ),
        },
    )
