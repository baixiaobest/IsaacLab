"""Unit tests for robot-awareness randomization in the social-force crowd."""

from __future__ import annotations

import pytest
import torch

from isaaclab_tasks.manager_based.navigation.mdp.social_force_crowd import (
    SocialForceCrowdCfg,
    SocialForceCrowdManager,
)


def _reset(manager: SocialForceCrowdManager, num_active: int) -> None:
    env_ids = torch.arange(manager.num_envs, device=manager.device)
    manager.reset_idx(
        env_ids=env_ids,
        corridor_origin=torch.zeros(manager.num_envs, 2, device=manager.device),
        flow_dir=torch.ones(manager.num_envs, device=manager.device),
        corridor_length=torch.full((manager.num_envs,), 20.0, device=manager.device),
        corridor_width=torch.full((manager.num_envs,), 6.0, device=manager.device),
        num_active=torch.full((manager.num_envs,), num_active, device=manager.device),
        speed_range=torch.tensor([[1.0, 1.0]], device=manager.device).expand(manager.num_envs, -1),
        robot_pos=torch.full((manager.num_envs, 2), 100.0, device=manager.device),
    )


def _manager(robot_ignore_probability: float, max_pedestrians: int = 3) -> SocialForceCrowdManager:
    return SocialForceCrowdManager(
        SocialForceCrowdCfg(
            max_pedestrians=max_pedestrians,
            robot_ignore_probability=robot_ignore_probability,
            a_ped=0.0,
            a_wall=0.0,
        ),
        num_envs=1,
        device="cpu",
    )


def test_robot_ignore_probability_default_and_validation():
    assert SocialForceCrowdCfg().robot_ignore_probability == pytest.approx(0.20)

    with pytest.raises(ValueError, match="robot_ignore_probability"):
        SocialForceCrowdCfg(robot_ignore_probability=-0.01)
    with pytest.raises(ValueError, match="robot_ignore_probability"):
        SocialForceCrowdCfg(robot_ignore_probability=1.01)

    cfg = SocialForceCrowdCfg(robot_ignore_probability=0.20)
    cfg.robot_ignore_probability = -0.01
    with pytest.raises(ValueError, match="robot_ignore_probability"):
        SocialForceCrowdManager(cfg, num_envs=1, device="cpu")


@pytest.mark.parametrize(("probability", "expected"), [(0.0, False), (1.0, True)])
def test_reset_assigns_only_active_slots(probability: float, expected: bool):
    manager = _manager(probability)
    _reset(manager, num_active=2)

    assert torch.equal(manager.ignores_robot[0, :2], torch.full((2,), expected, dtype=torch.bool))
    assert not torch.any(manager.ignores_robot[0, 2:])

    manager.vel[0, 2] = torch.tensor([3.0, 4.0])
    manager.step(dt=0.1, robot_pos=torch.zeros(1, 2))
    assert torch.equal(manager.vel[0, 2], torch.zeros(2))


def test_ignoring_pedestrian_receives_no_robot_repulsion():
    responsive = _manager(0.0, max_pedestrians=1)
    ignoring = _manager(1.0, max_pedestrians=1)
    no_robot = _manager(1.0, max_pedestrians=1)
    for manager in (responsive, ignoring, no_robot):
        _reset(manager, num_active=1)
        manager.pos[:] = torch.tensor([[[1.0, 0.0]]])
        manager.goal[:] = torch.tensor([[[4.0, 0.0]]])
        manager.vel.zero_()
        manager.desired_speed.fill_(1.0)
        manager.b_robot.fill_(0.5)

    responsive.step(dt=0.1, robot_pos=torch.zeros(1, 2))
    ignoring.step(dt=0.1, robot_pos=torch.zeros(1, 2))
    no_robot.step(dt=0.1, robot_pos=None)

    assert responsive.vel[0, 0, 0] > ignoring.vel[0, 0, 0]
    assert torch.allclose(ignoring.vel, no_robot.vel)


def test_reset_redraws_assignments_and_recycle_preserves_them(monkeypatch):
    manager = _manager(0.5, max_pedestrians=2)
    samples = iter(
        (
            torch.tensor([[0.1, 0.9]]),  # first reset: [ignore, responsive]
            torch.tensor([[0.9, 0.1]]),  # second reset: [responsive, ignore]
        )
    )
    original_rand = torch.rand

    def fake_rand(*shape, **kwargs):
        if shape == ((1, 2),):
            return next(samples).to(**kwargs)
        return original_rand(*shape, **kwargs)

    monkeypatch.setattr(torch, "rand", fake_rand)
    _reset(manager, num_active=2)
    assert torch.equal(manager.ignores_robot, torch.tensor([[True, False]]))

    # Crossing the downstream recycle margin must retain the episode assignment.
    manager.pos[:, :, 0] = 9.6
    manager._recycle(robot_pos=torch.full((1, 2), 100.0))
    assert torch.equal(manager.ignores_robot, torch.tensor([[True, False]]))

    _reset(manager, num_active=2)
    assert torch.equal(manager.ignores_robot, torch.tensor([[False, True]]))
