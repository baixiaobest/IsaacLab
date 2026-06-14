"""Shared per-env color scheme for pedestrian/goal-marker visuals."""

from __future__ import annotations

import colorsys


def get_env_color(env_id: int) -> tuple[float, float, float]:
    """Distinct per-env RGB color via golden-ratio hue stepping.

    Shared by the pedestrian capsule materials (``PedestrianCrowdNavigationEnv``) and the
    goal-pose markers (:class:`isaaclab_tasks.manager_based.navigation.mdp.mixed_commands.MixedTerrainPose2dCommand`)
    so each env's pedestrians and goal arrow render in the same color. Hues cycle but rarely
    land close together for neighboring env ids, even when ``num_envs`` is large.
    """
    hue = (env_id * 0.6180339887) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.55, 0.85)
