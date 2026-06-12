"""Vectorized GPU social-force-model crowd simulation for pedestrian-flow scenarios.

Unlike :mod:`rvo2_crowd`, which runs a single CPU/numpy RVO2 simulation, this module
simulates ``num_envs x max_pedestrians`` agents in parallel on ``env.device`` using a
Helbing-Molnar style social force model. The robot does not participate in the force
balance (no robot->pedestrian coupling) so that per-env pedestrian dynamics never depend
on other environments' robot state.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass


# Per-pedestrian-slot size randomization range (capsule collision proxy).
PED_RADIUS_RANGE = (0.18, 0.30)
PED_HEIGHT_RANGE = (1.45, 1.95)


@configclass
class SocialForceCrowdCfg:
    """Configuration for :class:`SocialForceCrowdManager`."""

    max_pedestrians: int = 12
    """Maximum number of pedestrian slots simulated per environment."""

    tau: float = 0.5
    """Relaxation time of the goal-attraction force [s]."""

    a_ped: float = 2.1
    """Pedestrian-pedestrian repulsion strength."""

    b_ped: float = 0.3
    """Pedestrian-pedestrian repulsion range [m]."""

    a_wall: float = 5.0
    """Corridor-wall repulsion strength."""

    b_wall: float = 0.2
    """Corridor-wall repulsion range [m]."""

    max_force: float = 8.0
    """Clamp applied to the total social force [m/s^2]."""

    max_speed_factor: float = 1.3
    """Pedestrian speed is clamped to ``max_speed_factor * desired_speed``."""

    a_robot: float = 2.1
    """Robot-pedestrian repulsion strength (one-way: pedestrians avoid the robot, the
    robot itself is not pushed — its motion is governed entirely by the RL policy)."""

    b_robot: float = 0.3
    """Robot-pedestrian repulsion range [m]."""

    robot_radius: float = 0.4
    """Effective collision radius of the robot used for the repulsion force [m]."""

    recycle_margin: float = 0.5
    """Distance past the corridor end (along the flow axis) that triggers recycling [m]."""

    wall_margin: float = 0.3
    """Lateral margin kept clear of the corridor walls when (re)spawning pedestrians [m]."""


class SocialForceCrowdManager:
    """Vectorized social-force pedestrian simulation.

    All state tensors have a leading ``(num_envs, max_pedestrians)`` shape and live on
    ``device``. Positions/velocities are expressed in world XY coordinates. Each
    environment owns a single straight corridor, described by ``corridor_origin``
    (world-frame position of the corridor-local origin), ``flow_dir`` (sign of the
    pedestrian-flow axis in corridor-local x), and ``corridor_length``/``corridor_width``.
    """

    def __init__(self, cfg: SocialForceCrowdCfg, num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.max_pedestrians = cfg.max_pedestrians
        self.device = device

        n, p = num_envs, self.max_pedestrians

        self.pos = torch.zeros(n, p, 2, device=device)
        self.vel = torch.zeros(n, p, 2, device=device)
        self.goal = torch.zeros(n, p, 2, device=device)
        self.desired_speed = torch.ones(n, p, device=device)
        self.radius = torch.full((n, p), 0.25, device=device)
        self.height = torch.full((n, p), 1.7, device=device)
        self.active_mask = torch.zeros(n, p, dtype=torch.bool, device=device)

        self.flow_dir = torch.ones(n, device=device)
        self.corridor_origin = torch.zeros(n, 2, device=device)
        self.corridor_length = torch.ones(n, device=device)
        self.corridor_width = torch.ones(n, device=device)

        # Cached curriculum-controlled speed range, used when (re)spawning/recycling.
        self._speed_range = torch.zeros(n, 2, device=device)
        self._speed_range[:, 1] = 1.0

        # Parking spot for inactive pedestrian slots (far below ground, out of the way).
        self._park_z = -50.0

        # Eye mask to exclude self-interaction in pairwise repulsion (1, P, P).
        self._self_mask = ~torch.eye(p, dtype=torch.bool, device=device).unsqueeze(0)

        # Cached per-step constants, recomputed only when the quantities they depend on
        # change (radii: set_radii; active_mask: reset_idx/set_active_count; corridor_width:
        # reset_idx) rather than every step.
        self._radius_sum = self.radius.unsqueeze(2) + self.radius.unsqueeze(1)  # (N, P, P)
        self._radius_sum_robot = self.radius + cfg.robot_radius  # (N, P)
        self._pair_mask = torch.zeros(n, p, p, dtype=torch.bool, device=device)  # (N, P, P)
        self._half_width = self.corridor_width.unsqueeze(1) / 2.0  # (N, 1)

    # ------------------------------------------------------------------
    # Per-instance geometry (capsule radius/height) — set once at startup.
    # ------------------------------------------------------------------

    def set_radii(self, radius: torch.Tensor, height: torch.Tensor) -> None:
        """Set per-slot capsule radius/height.

        Args:
            radius: Shape ``(max_pedestrians,)`` or ``(num_envs, max_pedestrians)``.
            height: Same shape as ``radius``.
        """
        self.radius[:] = radius
        self.height[:] = height
        self._radius_sum = self.radius.unsqueeze(2) + self.radius.unsqueeze(1)
        self._radius_sum_robot = self.radius + self.cfg.robot_radius

    # ------------------------------------------------------------------
    # Reset / recycle
    # ------------------------------------------------------------------

    def reset_idx(
        self,
        env_ids: torch.Tensor,
        corridor_origin: torch.Tensor,
        flow_dir: torch.Tensor,
        corridor_length: torch.Tensor,
        corridor_width: torch.Tensor,
        num_active: torch.Tensor,
        speed_range: torch.Tensor,
    ) -> None:
        """(Re)initialize all pedestrian slots for ``env_ids``.

        Args:
            env_ids: Environment indices to reset, shape ``(E,)``.
            corridor_origin: World-XY corridor-local origin, shape ``(E, 2)``.
            flow_dir: Sign (+1/-1) of the pedestrian-flow axis along corridor-local x, shape ``(E,)``.
            corridor_length: Corridor extent along the flow axis [m], shape ``(E,)``.
            corridor_width: Corridor extent across the flow axis [m], shape ``(E,)``.
            num_active: Number of active pedestrian slots, shape ``(E,)``.
            speed_range: ``[min, max]`` preferred-speed range, shape ``(E, 2)``.
        """
        e = env_ids
        p = self.max_pedestrians

        self.corridor_origin[e] = corridor_origin
        self.flow_dir[e] = flow_dir
        self.corridor_length[e] = corridor_length
        self.corridor_width[e] = corridor_width
        self._speed_range[e] = speed_range

        num_active = num_active.to(self.device).long()
        slot_idx = torch.arange(p, device=self.device).unsqueeze(0)  # (1, P)
        self.active_mask[e] = slot_idx < num_active.unsqueeze(1)
        self._update_pair_mask()
        self._half_width[e] = corridor_width.unsqueeze(1) / 2.0

        local_pos, vel, goal, speed = self._spawn_pedestrians(e, corridor_length, corridor_width, flow_dir, speed_range)

        self.pos[e] = corridor_origin.unsqueeze(1) + local_pos
        self.vel[e] = vel
        self.goal[e] = corridor_origin.unsqueeze(1) + goal
        self.desired_speed[e] = speed

        # Park inactive slots out of the way so they neither collide nor render visibly.
        self._park_inactive(e)

    def set_active_count(self, env_ids: torch.Tensor, num_active: torch.Tensor) -> None:
        """Update the number of active pedestrian slots for ``env_ids``.

        Newly activated slots are (re)spawned at the start line; newly deactivated slots
        are parked out of the way.
        """
        e = env_ids
        p = self.max_pedestrians
        num_active = num_active.to(self.device).long()

        slot_idx = torch.arange(p, device=self.device).unsqueeze(0)  # (1, P)
        new_mask = slot_idx < num_active.unsqueeze(1)
        newly_active = new_mask & (~self.active_mask[e])

        self.active_mask[e] = new_mask
        self._update_pair_mask()

        if newly_active.any():
            flow_dir = self.flow_dir[e]
            corridor_length = self.corridor_length[e]
            corridor_width = self.corridor_width[e]
            speed_range = self._speed_range[e]

            local_pos, vel, goal, speed = self._spawn_pedestrians(e, corridor_length, corridor_width, flow_dir, speed_range)
            world_pos = self.corridor_origin[e].unsqueeze(1) + local_pos
            world_goal = self.corridor_origin[e].unsqueeze(1) + goal

            self.pos[e] = torch.where(newly_active.unsqueeze(-1), world_pos, self.pos[e])
            self.vel[e] = torch.where(newly_active.unsqueeze(-1), vel, self.vel[e])
            self.goal[e] = torch.where(newly_active.unsqueeze(-1), world_goal, self.goal[e])
            self.desired_speed[e] = torch.where(newly_active, speed, self.desired_speed[e])

        self._park_inactive(e)

    def set_speed_range(self, env_ids: torch.Tensor, speed_range: torch.Tensor) -> None:
        """Update the cached preferred-speed range used for future spawns/recycles.

        Does not affect the ``desired_speed`` of currently in-flight pedestrians.
        """
        self._speed_range[env_ids] = speed_range

    def _update_pair_mask(self) -> None:
        """Refresh the cached pairwise active-pair mask from ``active_mask``."""
        active = self.active_mask
        self._pair_mask = active.unsqueeze(2) & active.unsqueeze(1) & self._self_mask

    def _park_inactive(self, env_ids: torch.Tensor) -> None:
        inactive = ~self.active_mask[env_ids]
        if inactive.any():
            self.vel[env_ids] = torch.where(inactive.unsqueeze(-1), torch.zeros_like(self.vel[env_ids]), self.vel[env_ids])

    def _spawn_pedestrians(
        self,
        env_ids: torch.Tensor,
        corridor_length: torch.Tensor,
        corridor_width: torch.Tensor,
        flow_dir: torch.Tensor,
        speed_range: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample fresh corridor-local positions/velocities/goals/speeds for all slots.

        Returns corridor-local ``pos (E,P,2)``, ``vel (E,P,2)``, corridor-local
        ``goal (E,P,2)``, and ``desired_speed (E,P)``.
        """
        e = env_ids
        p = self.max_pedestrians
        n = e.shape[0]
        margin = self.cfg.wall_margin

        # Spread pedestrians uniformly along the corridor length, starting from the
        # upstream end (local x = 0) and walking toward local x = corridor_length.
        frac = torch.rand(n, p, device=self.device)
        local_x = frac * corridor_length.unsqueeze(1)

        half_width = (corridor_width.unsqueeze(1) / 2.0 - margin).clamp(min=0.0)
        local_y = (torch.rand(n, p, device=self.device) * 2.0 - 1.0) * half_width

        local_pos = torch.stack([local_x, local_y], dim=-1)

        speed = speed_range[:, 0].unsqueeze(1) + torch.rand(n, p, device=self.device) * (
            speed_range[:, 1] - speed_range[:, 0]
        ).unsqueeze(1)

        # Goal is the downstream end of the corridor along the flow direction, at the
        # pedestrian's own lateral offset (keeps the goal-attraction force purely axial).
        # flow_dir is +-1; when flow_dir == -1 the downstream end is local x = 0.
        goal_x = torch.where(
            flow_dir.unsqueeze(1) > 0, corridor_length.unsqueeze(1), torch.zeros_like(corridor_length.unsqueeze(1))
        )
        goal = torch.stack([goal_x.expand(-1, p), local_y], dim=-1)

        vel = torch.zeros(n, p, 2, device=self.device)
        vel[..., 0] = flow_dir.unsqueeze(1) * speed

        return local_pos, vel, goal, speed

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self, dt: float, robot_pos: torch.Tensor | None = None) -> None:
        """Advance the social-force simulation by ``dt`` seconds.

        Args:
            robot_pos: Optional world-XY robot position, shape ``(num_envs, 2)``. If given,
                pedestrians are repelled by the robot (one-way — the robot is not pushed
                back, since its motion is governed entirely by the RL policy).
        """
        cfg = self.cfg
        active = self.active_mask

        # --- 1. Goal-attraction force ---------------------------------------------------
        to_goal = self.goal - self.pos
        dist_to_goal = torch.linalg.norm(to_goal, dim=-1, keepdim=True).clamp(min=1e-6)
        desired_dir = to_goal / dist_to_goal
        desired_vel = desired_dir * self.desired_speed.unsqueeze(-1)
        f_goal = (desired_vel - self.vel) / cfg.tau

        # --- 2. Pedestrian-pedestrian repulsion ------------------------------------------
        diff = self.pos.unsqueeze(2) - self.pos.unsqueeze(1)  # (N, P, P, 2) -> pos_i - pos_j
        dist = torch.linalg.norm(diff, dim=-1).clamp(min=1e-6)  # (N, P, P)
        magnitude = cfg.a_ped * torch.exp((self._radius_sum - dist) / cfg.b_ped)
        magnitude = torch.where(self._pair_mask, magnitude, torch.zeros_like(magnitude))

        f_ped = ((magnitude / dist).unsqueeze(-1) * diff).sum(dim=2)  # (N, P, 2)

        # --- 2b. Robot-pedestrian repulsion (one-way: robot is not pushed) ----------------
        if robot_pos is not None:
            diff_robot = self.pos - robot_pos.unsqueeze(1)  # (N, P, 2)
            dist_robot = torch.linalg.norm(diff_robot, dim=-1).clamp(min=1e-6)  # (N, P)
            magnitude_robot = cfg.a_robot * torch.exp((self._radius_sum_robot - dist_robot) / cfg.b_robot)
            magnitude_robot = torch.where(active, magnitude_robot, torch.zeros_like(magnitude_robot))
            f_robot = (magnitude_robot / dist_robot).unsqueeze(-1) * diff_robot
        else:
            f_robot = torch.zeros_like(f_ped)

        # --- 3. Corridor-wall repulsion (analytic, local-y only) -------------------------
        local_y = self.pos[..., 1] - self.corridor_origin[:, 1:2]
        half_width = self._half_width

        dist_to_pos_wall = (half_width - local_y).clamp(min=1e-6)
        dist_to_neg_wall = (half_width + local_y).clamp(min=1e-6)

        # Push away from the +y wall (negative direction) and away from the -y wall
        # (positive direction) — i.e. always back toward the corridor centerline.
        f_wall_y = -cfg.a_wall * torch.exp((self.radius - dist_to_pos_wall) / cfg.b_wall)
        f_wall_y = f_wall_y + cfg.a_wall * torch.exp((self.radius - dist_to_neg_wall) / cfg.b_wall)

        # --- 4. Combine, clamp, integrate (semi-implicit Euler) ---------------------------
        force = f_goal + f_ped + f_robot
        force[..., 1] += f_wall_y
        force_mag = torch.linalg.norm(force, dim=-1, keepdim=True).clamp(min=1e-6)
        force = force * (cfg.max_force / force_mag).clamp(max=1.0)
        force = torch.where(active.unsqueeze(-1), force, torch.zeros_like(force))

        new_vel = self.vel + force * dt
        speed = torch.linalg.norm(new_vel, dim=-1, keepdim=True).clamp(min=1e-6)
        max_speed = (self.desired_speed * cfg.max_speed_factor).unsqueeze(-1)
        new_vel = new_vel * (max_speed / speed).clamp(max=1.0)
        new_vel = torch.where(active.unsqueeze(-1), new_vel, torch.zeros_like(new_vel))

        self.vel = new_vel
        self.pos = self.pos + self.vel * dt

        # --- 5. Recycle pedestrians that reached the downstream end -----------------------
        self._recycle()

    def _recycle(self) -> None:
        local = self.pos - self.corridor_origin.unsqueeze(1)
        local_x = local[..., 0]

        flow_dir = self.flow_dir.unsqueeze(1)
        end_x = torch.where(flow_dir > 0, self.corridor_length.unsqueeze(1), torch.zeros_like(self.corridor_length.unsqueeze(1)))
        start_x = torch.where(flow_dir > 0, torch.zeros_like(end_x), self.corridor_length.unsqueeze(1))

        margin = self.cfg.recycle_margin
        end_x_b = end_x.expand_as(local_x)
        start_x_b = start_x.expand_as(local_x)
        crossed_pos_dir = (flow_dir > 0) & (local_x > end_x_b + margin)
        crossed_neg_dir = (flow_dir < 0) & (local_x < end_x_b - margin)
        crossed = (crossed_pos_dir | crossed_neg_dir) & self.active_mask

        if not crossed.any():
            return

        n, p = self.pos.shape[:2]
        margin_y = self.cfg.wall_margin
        half_width = (self.corridor_width.unsqueeze(1) / 2.0 - margin_y).clamp(min=0.0)
        new_y = (torch.rand(n, p, device=self.device) * 2.0 - 1.0) * half_width

        speed_min = self._speed_range[:, 0:1]
        speed_max = self._speed_range[:, 1:2]
        new_speed = speed_min + torch.rand(n, p, device=self.device) * (speed_max - speed_min)

        new_local_x = start_x_b
        new_local_pos = torch.stack([new_local_x, new_y], dim=-1)
        new_world_pos = self.corridor_origin.unsqueeze(1) + new_local_pos

        new_goal_x = end_x_b
        new_goal = torch.stack([new_goal_x, new_y], dim=-1)
        new_world_goal = self.corridor_origin.unsqueeze(1) + new_goal

        new_vel = torch.zeros_like(self.vel)
        new_vel[..., 0] = self.flow_dir.unsqueeze(1).expand(-1, p) * new_speed

        crossed_pos = crossed.unsqueeze(-1)
        self.pos = torch.where(crossed_pos, new_world_pos, self.pos)
        self.goal = torch.where(crossed_pos, new_world_goal, self.goal)
        self.vel = torch.where(crossed_pos, new_vel, self.vel)
        self.desired_speed = torch.where(crossed, new_speed, self.desired_speed)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_world_positions(self) -> torch.Tensor:
        """Return world-XY positions, shape ``(num_envs, max_pedestrians, 2)``."""
        return self.pos

    def get_velocities(self) -> torch.Tensor:
        """Return world-XY velocities, shape ``(num_envs, max_pedestrians, 2)``."""
        return self.vel

    def get_heights(self) -> torch.Tensor:
        """Return the world-frame z position (capsule center) for each pedestrian slot."""
        return self.radius + self.height / 2.0

    def get_active_mask(self) -> torch.Tensor:
        """Return the active-slot mask, shape ``(num_envs, max_pedestrians)``."""
        return self.active_mask
