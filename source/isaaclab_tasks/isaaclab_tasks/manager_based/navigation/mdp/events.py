from __future__ import annotations
import inspect
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg
from isaaclab.assets import Articulation
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class activate_event_terrain_level_reached(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.activated = False
        self.matched_env_ids = None
    
    def __call__(self,
                env: ManagerBasedEnv,
                env_ids: torch.Tensor,
                func: callable,
                terrain_names: list[str],
                operator: str = "max", # max or mean
                terrain_level_threshold: float = 0.0,
                callback_params: dict = {}
    ):
        
        terrain: TerrainImporter = env.scene.terrain
        if self.matched_env_ids is None:
            env_terrain_names = terrain.get_env_terrain_names()
            self.matched_env_ids = torch.tensor([i for i, name in enumerate(env_terrain_names) if name in terrain_names], dtype=torch.int, device=env.device) 

        if self.matched_env_ids.size() == 0:
            raise ValueError("No environments match the specified terrain names.")
        
        matched_env_terrain_levels = terrain.terrain_levels[self.matched_env_ids]

        level = 0
        if operator == "max":
            level = torch.max(matched_env_terrain_levels.float()).item()
        elif operator == "mean":
            level = torch.mean(matched_env_terrain_levels.float()).item()
        else:
            raise ValueError("operator must be 'max' or 'mean'")

        if not self.activated and level >= terrain_level_threshold:
            self.activated = True
        
        if self.activated:
            func(env, env_ids, **callback_params)
        else:
            return


def reset_pedestrian_scenario_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    flow_pose_range: dict[str, tuple[float, float]],
    crossing_south_pose_range: dict[str, tuple[float, float]],
    crossing_north_pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    crossing_prob: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Sample the per-env pedestrian scenario and reset the robot root state accordingly.

    For each env in ``env_ids`` this draws a scenario mode and stores it in
    ``env.pedestrian_scenario_mode`` (read later by :class:`CorridorPedestrianPose2dCommand` to
    pick the matching goal):

    - ``0`` = flow: spawn from ``flow_pose_range``, goal up/downstream along local-x.
    - ``1`` = crossing south→north: spawn from ``crossing_south_pose_range`` (facing +y), goal
      across the flow on the north side.
    - ``2`` = crossing north→south: spawn from ``crossing_north_pose_range`` (facing -y), goal
      across the flow on the south side.

    ``crossing_prob`` is the probability of a crossing episode (vs. flow); a crossing episode is
    then south→north or north→south with equal probability, so the robot experiences the crowd
    sweeping across its path from both relative directions. The pedestrian crowd itself is
    scenario-independent (it always flows along local-x), so it is (re)spawned separately by
    :func:`reset_pedestrian_crowd`.

    Must be declared as a ``mode="reset"`` event so it runs before the command manager resamples
    the goal within the same reset, and ``env.pedestrian_scenario_mode`` must already exist (it
    is allocated in :class:`PedestrianCrowdNavigationEnv`).
    """
    n = len(env_ids)

    # -- sample the scenario mode (0 = flow, 1 = cross S->N, 2 = cross N->S) --
    is_crossing = torch.rand(n, device=env.device) < crossing_prob
    is_north_start = torch.rand(n, device=env.device) < 0.5  # only meaningful when crossing
    mode = torch.where(
        is_crossing,
        torch.where(is_north_start, torch.full_like(is_crossing, 2, dtype=torch.long),
                    torch.full_like(is_crossing, 1, dtype=torch.long)),
        torch.zeros_like(is_crossing, dtype=torch.long),
    )
    _reset_robot_from_pedestrian_modes(
        env,
        env_ids,
        mode,
        flow_pose_range,
        crossing_south_pose_range,
        crossing_north_pose_range,
        velocity_range,
        asset_cfg,
    )


def _reset_robot_from_pedestrian_modes(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mode: torch.Tensor,
    flow_pose_range: dict[str, tuple[float, float]],
    crossing_south_pose_range: dict[str, tuple[float, float]],
    crossing_north_pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Reset robots for already-selected flow/crossing modes."""
    asset: Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)
    is_crossing = mode >= 1
    is_north_start = mode == 2
    env.pedestrian_scenario_mode[env_ids] = mode

    root_states = asset.data.default_root_state[env_ids].clone()

    keys = ["x", "y", "z", "roll", "pitch", "yaw"]

    def _sample(pose_range: dict[str, tuple[float, float]]) -> torch.Tensor:
        ranges = torch.tensor([pose_range.get(k, (0.0, 0.0)) for k in keys], device=asset.device)
        return math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=asset.device)

    flow_samples = _sample(flow_pose_range)
    south_samples = _sample(crossing_south_pose_range)
    north_samples = _sample(crossing_north_pose_range)
    crossing_samples = torch.where(is_north_start.unsqueeze(-1), north_samples, south_samples)
    rand_samples = torch.where(is_crossing.unsqueeze(-1), crossing_samples, flow_samples)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    vel_ranges = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in keys], device=asset.device)
    vel_samples = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (n, 6), device=asset.device)
    velocities = root_states[:, 7:13] + vel_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_evaluation_pedestrian_scenario_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    flow_pose_range: dict[str, tuple[float, float]],
    crossing_south_pose_range: dict[str, tuple[float, float]],
    crossing_north_pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    speed_range: tuple[float, float] = (0.9, 1.5),
    slow_speed_range: tuple[float, float] | None = None,
    slow_scenario_codes: tuple[int, ...] = (),
    crossing_scenario_codes: tuple[int, ...] = (0,),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset a benchmark profile's robot pose.

    ``env.evaluation_scenario`` contains 0 (crossing), 1 (with flow), 2 (against flow),
    3 (with flow and a slow leader), 4 (crossing with a slow crowd), or 5 (against flow
    with a slow crowd). Crossing directions remain balanced per reset; flow direction is
    controlled separately by ``env.evaluation_flow_goal_direction`` when the command is
    resampled.  When ``slow_speed_range``/``slow_scenario_codes`` are provided, the
    crowd's desired-speed band is lowered to ``slow_speed_range`` for every env whose
    scenario is in ``slow_scenario_codes`` (the whole crowd is slow — no leader).
    ``crossing_scenario_codes`` lists every scenario that uses the crossing spawn/goal
    geometry (defaults to plain crossing only).
    """
    # The RSL-RL wrapper performs one reset during construction. Profiles are installed by the
    # evaluator immediately afterward, so let that unobserved bootstrap reset use the standard
    # random scenario instead of requiring profile buffers too early.
    if not hasattr(env, "evaluation_scenario"):
        reset_pedestrian_scenario_robot(
            env,
            env_ids,
            flow_pose_range,
            crossing_south_pose_range,
            crossing_north_pose_range,
            velocity_range,
            asset_cfg=asset_cfg,
        )
        return

    configure_evaluation_pedestrian_crowd(
        env,
        env_ids,
        speed_range,
        slow_speed_range=slow_speed_range,
        slow_scenario_codes=slow_scenario_codes,
    )
    scenario = env.evaluation_scenario[env_ids]
    is_crossing = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    for code in crossing_scenario_codes:
        is_crossing |= scenario == code
    is_north_start = torch.rand(len(env_ids), device=env.device) < 0.5
    mode = torch.where(
        is_crossing,
        torch.where(
            is_north_start,
            torch.full_like(scenario, 2),
            torch.full_like(scenario, 1),
        ),
        torch.zeros_like(scenario),
    )
    _reset_robot_from_pedestrian_modes(
        env,
        env_ids,
        mode,
        flow_pose_range,
        crossing_south_pose_range,
        crossing_north_pose_range,
        velocity_range,
        asset_cfg,
    )


def configure_evaluation_pedestrian_crowd(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    speed_range: tuple[float, float] = (0.9, 1.5),
    slow_speed_range: tuple[float, float] | None = None,
    slow_scenario_codes: tuple[int, ...] = (),
):
    """Apply fixed per-environment crowd counts and speeds before a benchmark reset.

    The normal band is ``speed_range`` for every resetting env.  When
    ``slow_speed_range`` and ``slow_scenario_codes`` are given, envs whose
    ``env.evaluation_scenario`` is in ``slow_scenario_codes`` get the slow band
    instead (a whole-crowd speed perturbation, distinct from the slot-0 leader).
    """
    if not hasattr(env, "evaluation_pedestrian_count"):
        raise RuntimeError("Evaluation crowd buffers must be installed before resetting the environment.")
    counts = env.evaluation_pedestrian_count[env_ids]
    speed = torch.tensor(speed_range, device=env.device, dtype=torch.float32).expand(len(env_ids), 2)
    if slow_speed_range is not None and slow_scenario_codes:
        slow = torch.tensor(slow_speed_range, device=env.device, dtype=torch.float32).expand(len(env_ids), 2)
        scenario = env.evaluation_scenario[env_ids]
        is_slow = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
        for code in slow_scenario_codes:
            is_slow |= scenario == code
        speed = torch.where(is_slow.unsqueeze(-1), slow, speed)
    env.crowd_manager.set_active_count(env_ids, counts)
    env.crowd_manager.set_speed_range(env_ids, speed)


def reset_robot_mixed(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    static_pose_range: dict[str, tuple[float, float]],
    static_velocity_range: dict[str, tuple[float, float]],
    flow_pose_range: dict[str, tuple[float, float]],
    crossing_south_pose_range: dict[str, tuple[float, float]],
    crossing_north_pose_range: dict[str, tuple[float, float]],
    pedestrian_velocity_range: dict[str, tuple[float, float]],
    crossing_prob: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot root state, branching per-env on ``env.is_pedestrian_env``.

    Static envs (no pedestrian crowd) are reset with the standard
    :func:`isaaclab.envs.mdp.events.reset_root_state_uniform`. Pedestrian-corridor envs are reset
    via :func:`reset_pedestrian_scenario_robot`, which also samples the per-env flow/crossing
    scenario mode.
    """
    from isaaclab.envs.mdp.events import reset_root_state_uniform

    is_ped = env.is_pedestrian_env[env_ids]
    ped_env_ids = env_ids[is_ped]
    static_env_ids = env_ids[~is_ped]

    if len(static_env_ids) > 0:
        reset_root_state_uniform(env, static_env_ids, static_pose_range, static_velocity_range, asset_cfg)

    if len(ped_env_ids) > 0:
        reset_pedestrian_scenario_robot(
            env,
            ped_env_ids,
            flow_pose_range,
            crossing_south_pose_range,
            crossing_north_pose_range,
            pedestrian_velocity_range,
            crossing_prob,
            asset_cfg,
        )


def reset_pedestrian_crowd(env: ManagerBasedEnv, env_ids: torch.Tensor, flow_dir: float = 1.0):
    """(Re)spawn the social-force pedestrian crowd for ``env_ids``.

    Derives the per-env corridor geometry (origin, length, width) from the env's current
    sub-terrain (``terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]`` plus
    the terrain generator's ``size``), then calls ``env.crowd_manager.reset_idx`` — preserving
    the active pedestrian count and speed range last set by :func:`pedestrian_crowd_curriculum`.

    Must be declared after ``reset_base`` (mode="reset") so ``terrain_origins``/``terrain_levels``
    reflect this episode's terrain assignment before being read here.

    ``env_ids`` is filtered down to ``env.is_pedestrian_env`` envs — a no-op for envs sitting on
    a static (non-"ped_corridor") terrain column.
    """
    env_ids = env_ids[env.is_pedestrian_env[env_ids]]
    if len(env_ids) == 0:
        return

    terrain: TerrainImporter = env.scene.terrain

    levels = terrain.terrain_levels[env_ids]
    types = terrain.terrain_types[env_ids]
    corridor_origin = terrain.terrain_origins[levels, types][:, :2]

    size = terrain.cfg.terrain_generator.size
    corridor_length = torch.full((len(env_ids),), size[0], device=env.device)
    corridor_width = torch.full((len(env_ids),), size[1], device=env.device)
    flow_dir_t = torch.full((len(env_ids),), flow_dir, device=env.device)

    crowd_manager = env.crowd_manager
    num_active = crowd_manager.active_mask[env_ids].sum(dim=1)
    speed_range = crowd_manager._speed_range[env_ids]
    robot_pos = env.scene["robot"].data.root_pos_w[env_ids, :2]

    crowd_manager.reset_idx(
        env_ids,
        corridor_origin,
        flow_dir_t,
        corridor_length,
        corridor_width,
        num_active,
        speed_range,
        robot_pos=robot_pos,
    )


def reset_evaluation_pedestrian_crowd(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    flow_dir: float = 1.0,
    slow_leader_scenario_code: int = 3,
    slow_leader_speed_range_mps: tuple[float, float] = (0.25, 0.45),
    slow_leader_start_ahead_range_m: tuple[float, float] = (1.5, 3.0),
    slow_leader_lateral_offset_range_m: tuple[float, float] = (-0.25, 0.25),
):
    """Reset an evaluation crowd and install its bounded-random slow leader when requested.

    The base reset still supplies the randomized crowd used by all existing benchmark profiles.
    For the dedicated slow-leader profile, pedestrian slot 0 is placed directly ahead of the
    robot's lane, travels in the same ``+x`` direction, and samples its speed and relative
    pose from narrow benchmark ranges.  That makes overtaking deliberate without teaching a
    policy one fixed initial condition.
    """
    reset_pedestrian_crowd(env, env_ids, flow_dir=flow_dir)

    if not hasattr(env, "evaluation_scenario"):
        return

    def _validate_range(name: str, value: tuple[float, float], *, strictly_positive: bool = False) -> None:
        if len(value) != 2 or value[0] > value[1] or (strictly_positive and value[0] <= 0.0):
            qualifier = "positive and ordered" if strictly_positive else "ordered"
            raise ValueError(f"{name} must be a two-value {qualifier} range.")

    _validate_range("Slow-leader speed", slow_leader_speed_range_mps, strictly_positive=True)
    _validate_range("Slow-leader starting distance", slow_leader_start_ahead_range_m, strictly_positive=True)
    _validate_range("Slow-leader lateral offset", slow_leader_lateral_offset_range_m)

    pedestrian_env_ids = env_ids[env.is_pedestrian_env[env_ids]]
    if len(pedestrian_env_ids) == 0:
        return
    leader_env_ids = pedestrian_env_ids[
        env.evaluation_scenario[pedestrian_env_ids] == slow_leader_scenario_code
    ]
    if len(leader_env_ids) == 0:
        return

    crowd_manager = env.crowd_manager
    # The profile contract includes one active pedestrian, but retain this guard if a caller
    # accidentally installs an incompatible profile.
    leader_env_ids = leader_env_ids[crowd_manager.active_mask[leader_env_ids, 0]]
    if len(leader_env_ids) == 0:
        return

    robot_pos = env.scene["robot"].data.root_pos_w[leader_env_ids, :2]
    corridor_origin = crowd_manager.corridor_origin[leader_env_ids]
    local_robot_pos = robot_pos - corridor_origin
    half_length = crowd_manager.corridor_length[leader_env_ids] / 2.0
    half_width = (
        crowd_manager.corridor_width[leader_env_ids] / 2.0 - crowd_manager.cfg.wall_margin
    ).clamp(min=0.0)

    sample_count = len(leader_env_ids)
    leader_speed = torch.empty(sample_count, device=env.device).uniform_(*slow_leader_speed_range_mps)
    leader_start_ahead = torch.empty(sample_count, device=env.device).uniform_(
        *slow_leader_start_ahead_range_m
    )
    leader_lateral_offset = torch.empty(sample_count, device=env.device).uniform_(
        *slow_leader_lateral_offset_range_m
    )

    # Keep the leader approximately in the robot's lane, but inside the corridor and before
    # the downstream boundary. In this evaluation profile the robot's goal and crowd both
    # travel in +x.
    leader_x = (local_robot_pos[:, 0] + leader_start_ahead).clamp(
        min=-half_length + crowd_manager.cfg.wall_margin,
        max=half_length - crowd_manager.cfg.wall_margin,
    )
    leader_y = (local_robot_pos[:, 1] + leader_lateral_offset).clamp(min=-half_width, max=half_width)
    leader_local_pos = torch.stack((leader_x, leader_y), dim=-1)
    leader_pos = corridor_origin + leader_local_pos
    leader_goal_local = torch.stack((half_length - crowd_manager.cfg.wall_margin, leader_y), dim=-1)
    leader_goal = corridor_origin + leader_goal_local
    leader_direction = leader_goal - leader_pos
    leader_direction = leader_direction / torch.linalg.vector_norm(
        leader_direction, dim=-1, keepdim=True
    ).clamp(min=1e-6)

    crowd_manager.pos[leader_env_ids, 0] = leader_pos
    crowd_manager.goal[leader_env_ids, 0] = leader_goal
    crowd_manager.desired_speed[leader_env_ids, 0] = leader_speed
    crowd_manager.vel[leader_env_ids, 0] = leader_direction * leader_speed.unsqueeze(-1)
    # Retain the exact reset samples for per-episode evaluation artifacts.
    if hasattr(env, "evaluation_slow_leader_speed_mps"):
        env.evaluation_slow_leader_speed_mps[leader_env_ids] = leader_speed
        env.evaluation_slow_leader_start_ahead_m[leader_env_ids] = leader_start_ahead
        env.evaluation_slow_leader_lateral_offset_m[leader_env_ids] = leader_lateral_offset
