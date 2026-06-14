"""Vectorized GPU social-force-model crowd simulation for pedestrian-flow scenarios.

Unlike :mod:`rvo2_crowd`, which runs a single CPU/numpy RVO2 simulation, this module
simulates ``num_envs x max_pedestrians`` agents in parallel on ``env.device`` using a
Helbing-Molnar style social force model. The robot does not participate in the force
balance (no robot->pedestrian coupling) so that per-env pedestrian dynamics never depend
on other environments' robot state.

Each pedestrian ``i`` is driven by a Helbing-Molnar style social force
:math:`f_i = f_i^{goal} + f_i^{ped} + f_i^{robot} + f_i^{wall}`, clamped to
``max_force`` and integrated with semi-implicit Euler:
``v_i <- clamp(v_i + f_i * dt, max_speed_factor * desired_speed_i)``,
``pos_i <- pos_i + v_i * dt``.

**Goal-attraction force** (relaxation toward the desired velocity)::

    f_i^goal = (desired_speed_i * e_i - v_i) / tau

where ``e_i = (goal_i - pos_i) / |goal_i - pos_i|`` is the unit vector
toward the pedestrian's current goal, and ``tau`` is the relaxation time.

**Pedestrian-pedestrian repulsion** (summed over all other pedestrians
``j`` in the same environment, ``d_ij = |pos_i - pos_j|``)::

    f_i^ped = sum_{j != i} a_ped * exp((r_i + r_j - d_ij) / b_ped) * (pos_i - pos_j) / d_ij

where ``r_i``/``r_j`` are the pedestrian capsule radii, ``a_ped`` sets the
repulsion strength and ``b_ped`` its falloff range.

**Robot-pedestrian repulsion** (one-way; ``d_i = |pos_i - pos_robot|``)::

    f_i^robot = a_robot * exp((r_i + robot_radius - d_i) / b_robot_i) * (pos_i - pos_robot) / d_i

with strength ``a_robot`` and per-pedestrian falloff length ``b_robot_i``, drawn from
``b_robot_range`` to represent varying pedestrian attentiveness to the robot. The robot
itself receives no reaction force.

**Corridor-wall repulsion** (only the lateral/local-y component is
affected; ``y`` is the pedestrian's offset from the corridor centerline,
``w`` is ``corridor_width``)::

    d_pos = w/2 - y      # distance to the +y wall
    d_neg = w/2 + y      # distance to the -y wall
    f_i^wall_y = -a_wall * exp((r_i - d_pos) / b_wall) + a_wall * exp((r_i - d_neg) / b_wall)

with strength ``a_wall`` and falloff range ``b_wall``. ``f_i^wall_x = 0``.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass


# Per-pedestrian-slot size randomization range (capsule collision proxy).
PED_RADIUS_RANGE = (0.18, 0.30)
PED_HEIGHT_RANGE = (1.45, 1.95)


@configclass
class SocialForceCrowdCfg:
    """Configuration for :class:`SocialForceCrowdManager`.
    """

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

    b_robot_range: tuple[float, float] = (0.15, 1.0)
    """Range from which each pedestrian's robot-repulsion decay length ``b_robot`` [m] is
    sampled (domain randomization for pedestrian "attentiveness": low values react late/
    sharply to the robot, high values react early/gently). Resampled per slot on spawn and
    recycle."""

    robot_radius: float = 0.4
    """Effective collision radius of the robot used for the repulsion force [m]."""

    recycle_margin: float = 0.5
    """Distance from the corridor end (along the flow axis) at which a pedestrian is recycled
    back to the start, before it actually reaches the end [m]."""

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

        # Per-pedestrian robot-repulsion decay length (attentiveness), resampled on
        # spawn/recycle from cfg.b_robot_range.
        self.b_robot = torch.full((n, p), sum(cfg.b_robot_range) / 2.0, device=device)

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

        local_pos, vel, goal, speed, b_robot = self._spawn_pedestrians(
            e, corridor_length, corridor_width, flow_dir, speed_range
        )

        self.pos[e] = corridor_origin.unsqueeze(1) + local_pos
        self.vel[e] = vel
        self.goal[e] = corridor_origin.unsqueeze(1) + goal
        self.desired_speed[e] = speed
        self.b_robot[e] = b_robot

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

            local_pos, vel, goal, speed, b_robot = self._spawn_pedestrians(
                e, corridor_length, corridor_width, flow_dir, speed_range
            )
            world_pos = self.corridor_origin[e].unsqueeze(1) + local_pos
            world_goal = self.corridor_origin[e].unsqueeze(1) + goal

            self.pos[e] = torch.where(newly_active.unsqueeze(-1), world_pos, self.pos[e])
            self.vel[e] = torch.where(newly_active.unsqueeze(-1), vel, self.vel[e])
            self.goal[e] = torch.where(newly_active.unsqueeze(-1), world_goal, self.goal[e])
            self.desired_speed[e] = torch.where(newly_active, speed, self.desired_speed[e])
            self.b_robot[e] = torch.where(newly_active, b_robot, self.b_robot[e])

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample fresh corridor-local positions/velocities/goals/speeds/b_robot for all slots.

        Returns corridor-local ``pos (E,P,2)``, ``vel (E,P,2)``, corridor-local
        ``goal (E,P,2)``, ``desired_speed (E,P)``, and ``b_robot (E,P)``.
        """
        e = env_ids
        p = self.max_pedestrians
        n = e.shape[0]
        margin = self.cfg.wall_margin

        # corridor_origin is the CENTER of the corridor (see CorridorPedestrianPose2dCommand,
        # which samples robot spawn/goals relative to this same center). Spread pedestrians
        # uniformly along the corridor length in local x = [-corridor_length/2, +corridor_length/2],
        # i.e. centered on the origin, walking toward the downstream end (+half_length if
        # flow_dir > 0, else -half_length).
        half_length = corridor_length.unsqueeze(1) / 2.0
        frac = torch.rand(n, p, device=self.device)
        local_x = frac * corridor_length.unsqueeze(1) - half_length

        half_width = (corridor_width.unsqueeze(1) / 2.0 - margin).clamp(min=0.0)
        local_y = (torch.rand(n, p, device=self.device) * 2.0 - 1.0) * half_width

        local_pos = torch.stack([local_x, local_y], dim=-1)

        speed = speed_range[:, 0].unsqueeze(1) + torch.rand(n, p, device=self.device) * (
            speed_range[:, 1] - speed_range[:, 0]
        ).unsqueeze(1)

        # Goal is the downstream end of the corridor along the flow direction, at the
        # pedestrian's own lateral offset (keeps the goal-attraction force purely axial).
        # flow_dir is +-1; when flow_dir == -1 the downstream end is local x = -half_length.
        goal_x = torch.where(flow_dir.unsqueeze(1) > 0, half_length, -half_length)
        goal = torch.stack([goal_x.expand(-1, p), local_y], dim=-1)

        vel = torch.zeros(n, p, 2, device=self.device)
        vel[..., 0] = flow_dir.unsqueeze(1) * speed

        b_lo, b_hi = self.cfg.b_robot_range
        b_robot = b_lo + torch.rand(n, p, device=self.device) * (b_hi - b_lo)

        return local_pos, vel, goal, speed, b_robot

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
            magnitude_robot = cfg.a_robot * torch.exp((self._radius_sum_robot - dist_robot) / self.b_robot)
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
        half_length = self.corridor_length.unsqueeze(1) / 2.0
        end_x = torch.where(flow_dir > 0, half_length, -half_length)
        start_x = torch.where(flow_dir > 0, -half_length, half_length)

        # The goal-attraction force has constant magnitude (desired_speed) regardless of
        # distance to the goal, so pedestrians oscillate around the corridor end rather than
        # reliably overshooting it. Recycle as soon as a pedestrian gets within `margin` of
        # the end (on its approach side) instead of waiting for it to cross past the end.
        margin = self.cfg.recycle_margin
        end_x_b = end_x.expand_as(local_x)
        start_x_b = start_x.expand_as(local_x)
        crossed_pos_dir = (flow_dir > 0) & (local_x > end_x_b - margin)
        crossed_neg_dir = (flow_dir < 0) & (local_x < end_x_b + margin)
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

        b_lo, b_hi = self.cfg.b_robot_range
        new_b_robot = b_lo + torch.rand(n, p, device=self.device) * (b_hi - b_lo)

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
        self.b_robot = torch.where(crossed, new_b_robot, self.b_robot)

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
        """Return the world-frame z position (capsule center) for each pedestrian slot.

        Inactive slots are parked at ``self._park_z`` (far below ground) so they neither
        collide with the robot nor register as lidar hits.
        """
        active_height = self.radius + self.height / 2.0
        return torch.where(self.active_mask, active_height, torch.full_like(active_height, self._park_z))

    def get_active_mask(self) -> torch.Tensor:
        """Return the active-slot mask, shape ``(num_envs, max_pedestrians)``."""
        return self.active_mask

    def get_robot_collision(self, robot_pos_w: torch.Tensor) -> torch.Tensor:
        """Return a per-env collision mask against active pedestrian capsules.

        Args:
            robot_pos_w: World-XY robot positions, shape ``(num_envs, 2)``.

        Returns:
            Boolean tensor of shape ``(num_envs,)``, ``True`` where ``robot_pos_w`` lies within
            ``robot_radius + pedestrian_radius`` of any active pedestrian (XY distance only).
        """
        dist = torch.norm(robot_pos_w.unsqueeze(1) - self.pos, dim=-1)
        return torch.any((dist < self._radius_sum_robot) & self.active_mask, dim=1)
