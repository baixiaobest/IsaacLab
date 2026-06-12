"""Pedestrian rigid-object collections for social-force crowd scenarios.

Defines a fixed-size pool of ``SocialForceCrowdCfg.max_pedestrians`` kinematic capsule rigid
objects (``scene.pedestrians``) — the lidar/contact-visible proxy for the pedestrians simulated
by :class:`isaaclab_tasks.manager_based.navigation.mdp.social_force_crowd.SocialForceCrowdManager`.
All slots default-spawn parked below ground; :class:`PedestrianCrowdNavigationEnv` overwrites
their poses every reset/step from the crowd simulation.

Per-slot capsule radius/total-height are drawn once at *config-definition time* from a fixed
seed (``PED_RADII``/``PED_TOTAL_HEIGHTS``, ``PED_CAPSULE_HEIGHTS``) so the 12 pedestrian slots
have reproducible, visually-distinct human-scale sizes. True per-(env, pedestrian)
randomization isn't supported by ``RigidObjectCollection``'s single-spawn-per-name model — this
is the documented v1 compromise.

Optional human-mesh visuals (``scene.pedestrian_visuals``) are gated behind
:data:`ENABLE_PEDESTRIAN_VISUAL_MESHES`, which defaults to ``False`` until the Nucleus "People"
USD paths are verified on the target Isaac Sim install; capsules get a skin-tone material
instead.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.rigid_object_collection import RigidObjectCollectionCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.navigation.mdp.social_force_crowd import (
    PED_HEIGHT_RANGE,
    PED_RADIUS_RANGE,
    SocialForceCrowdCfg,
)

# Set to True once the Nucleus "People" character USD path below has been verified on the
# target Isaac Sim install.
ENABLE_PEDESTRIAN_VISUAL_MESHES = False

_PEOPLE_USD_PATH = f"{ISAAC_NUCLEUS_DIR}People/Characters/original_male_adult_police_04/male_adult_police_04.usd"

# Pedestrians are parked here (below ground) until the crowd manager places them.
PED_PARK_Z = -50.0

MAX_PEDESTRIANS = SocialForceCrowdCfg().max_pedestrians

# Fixed, reproducible per-slot sizes (does not affect global RNG state).
_size_rng = torch.Generator().manual_seed(42)
PED_RADII: list[float] = torch.empty(MAX_PEDESTRIANS).uniform_(*PED_RADIUS_RANGE, generator=_size_rng).tolist()
PED_TOTAL_HEIGHTS: list[float] = torch.empty(MAX_PEDESTRIANS).uniform_(*PED_HEIGHT_RANGE, generator=_size_rng).tolist()
# Capsule "height" is the cylinder length (excluding the two hemispherical caps); the total
# standing height is `height + 2 * radius`, matching SocialForceCrowdManager.get_heights().
PED_CAPSULE_HEIGHTS: list[float] = [max(0.1, h - 2.0 * r) for h, r in zip(PED_TOTAL_HEIGHTS, PED_RADII)]


def _pedestrian_capsule_cfg(radius: float, height: float) -> sim_utils.CapsuleCfg:
    return sim_utils.CapsuleCfg(
        radius=radius,
        height=height,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=70.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.65, 0.55)),
    )


def _pedestrian_rigid_object_cfg(index: int) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Pedestrian_{index}",
        spawn=_pedestrian_capsule_cfg(PED_RADII[index], PED_CAPSULE_HEIGHTS[index]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, PED_PARK_Z)),
    )


def _pedestrian_visual_cfg(index: int) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/PedestrianVisual_{index}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_PEOPLE_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, PED_PARK_Z)),
    )


@configclass
class PedestrianCollectionCfg(RigidObjectCollectionCfg):
    """Pool of ``MAX_PEDESTRIANS`` kinematic capsule pedestrians, parked below ground."""

    rigid_objects: dict[str, RigidObjectCfg] = {
        f"ped_{i}": _pedestrian_rigid_object_cfg(i) for i in range(MAX_PEDESTRIANS)
    }


@configclass
class PedestrianVisualCollectionCfg(RigidObjectCollectionCfg):
    """Pool of human-mesh visuals, one per pedestrian capsule slot."""

    rigid_objects: dict[str, RigidObjectCfg] = {
        f"ped_visual_{i}": _pedestrian_visual_cfg(i) for i in range(MAX_PEDESTRIANS)
    }
