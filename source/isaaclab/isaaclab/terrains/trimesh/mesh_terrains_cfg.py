# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrains
import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.utils import configclass

from ..terrain_generator_cfg import SubTerrainBaseCfg
from ..test_terrain_generator_cfg import SubTerrainTestCfg

"""
Different trimesh terrain configurations.
"""


@configclass
class MeshPlaneTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a plane mesh terrain."""

    function = mesh_terrains.flat_terrain


@configclass
class MeshPyramidStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a pyramid stair mesh terrain."""

    function = mesh_terrains.pyramid_stairs_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """
    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the steps (in m)."""
    step_width: float = MISSING
    """The width of the steps (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    holes: bool = False
    """If True, the terrain will have holes in the steps. Defaults to False.

    If :obj:`holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.
    """

@configclass
class MeshWalledLinearStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a linear stair mesh terrain with vertical walls."""

    # bind to your generator
    function = mesh_terrains.walled_linear_stairs_terrain

    # --- base ground plane ---
    size: tuple[float, float] = (6.0, 6.0)
    """The size of the flat base plane (x, y) in meters."""

    # --- stairs geometry ---
    stairs_length: float = 3.0
    """Total length of the stair corridor (y direction) in meters."""

    stairs_width: float = 1.0
    """Nominal stairs width (used if no width range/min is set)."""

    step_width: float = MISSING
    """The depth of each step (y direction shrink per side) in meters."""

    num_steps: int = MISSING
    """The number of risers (steps)."""

    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum riser height in meters (difficulty interpolates between them)."""

    # --- difficulty-driven width shrink options ---
    stairs_width_range: tuple[float, float] | None = None
    """Optional. (start_width, end_width) for difficulty interpolation."""

    min_stairs_width: float | None = None
    """Optional. If provided, width shrinks to this value at difficulty=1.0."""

    width_shrink_ratio: float | None = None
    """Optional. Fractional shrink (e.g., 0.5 → shrink to 50%) at difficulty=1.0."""

    # --- placement offsets ---
    stairs_center_y_offset: float = 0.0
    """Offset along +y to shift stairs center from terrain center."""

    origin_offset_y: float = 0.0
    """Origin y-offset relative to terrain center (used to place robot spawn)."""

    # --- walls ---
    wall_thickness: float = 0.06
    """Thickness of the side walls in meters."""

    wall_clearance: float = 0.02
    """Gap between stair usable edge and inside of wall."""

    wall_height_extra: float = 0.05
    """Extra wall height beyond total stairs height."""

@configclass
class MeshInvertedPyramidStairsTerrainCfg(MeshPyramidStairsTerrainCfg):
    """Configuration for an inverted pyramid stair mesh terrain.

    Note:
        This is the same as :class:`MeshPyramidStairsTerrainCfg` except that the steps are inverted.
    """

    function = mesh_terrains.inverted_pyramid_stairs_terrain

@configclass
class MeshLinearStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a linear stairs mesh terrain."""

    function = mesh_terrains.linear_stairs_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """
    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the steps (in m)."""
    num_steps: int = MISSING
    """The number of steps in the terrain."""
    step_width: float = MISSING
    """
        The width of the steps (in m). 
        From edge to edge of the stairs, not the width of the step itself.
    """
    stairs_width: float = MISSING
    """ Width of the stairs (in m)."""
    stairs_center_y_offset: float = 0.0
    """The offset of the center of the stairs along the y-axis (in m)."""
    stairs_length: float = 6.0
    """The width of the stairs (in m)."""
    origin_offset_y: float = 0.0
    """The offset of the origin of the terrain (in m). """
    origin_offset_z: float = 0.0

@configclass
class MeshTurningStairs90TerrainCfg(SubTerrainBaseCfg):
    """L-shaped stairs: run-1 along +y, landing, run-2 along +x (or -x)."""
    function = mesh_terrains.turning_stairs_90_terrain

    # base plane
    size: tuple[float, float] = (8.0, 8.0)

    # steps
    num_steps_run1: int = MISSING
    num_steps_run2: int = MISSING
    step_width: float = MISSING                 # tread depth
    step_height_range: tuple[float, float] = MISSING

    # width controls
    stairs_width: float = 1.2
    stairs_width_range: tuple[float, float] | None = None
    min_stairs_width: float | None = None
    width_shrink_ratio: float | None = None

    # lengths (centerline run lengths, without landing)
    run1_length: float = 3.0
    run2_length: float = 3.0

    # landing
    landing_length: float = 1.2                 # y length of landing after run1
    landing_width: float | None = None          # defaults to stairs_width at runtime

    # orientation & offsets
    turn_right: bool = True                     # True: run2 along +x; False: along -x
    stairs_center_y_offset: float = 0.0
    stairs_center_x_offset: float = 0.0
    origin_offset_y: float = 0.0

    # walls
    wall_thickness: float = 0.06
    wall_clearance: float = 0.02
    wall_height_extra: float = 0.05


@configclass
class MeshTurningStairs180TerrainCfg(SubTerrainBaseCfg):
    """U-shaped stairs: run-1 (+y), landing, run-2 (-y) offset in +x or -x."""
    function = mesh_terrains.turning_stairs_180_terrain

    size: tuple[float, float] = (8.0, 8.0)

    # steps
    num_steps_run1: int = MISSING
    num_steps_run2: int = MISSING
    step_width: float = MISSING
    step_height_range: tuple[float, float] = MISSING

    # width controls
    stairs_width: float = 1.2
    stairs_width_range: tuple[float, float] | None = None
    min_stairs_width: float | None = None
    width_shrink_ratio: float | None = None

    # lengths (centerline run lengths, without landing)
    run1_length: float = 3.0
    run2_length: float = 3.0

    # landing
    landing_length: float = 1.2                 # y length of landing between runs
    landing_offset_x: float = 1.4               # corridor offset to place run2 parallel to run1
    landing_width: float | None = None

    # orientation & offsets
    run2_on_positive_x: bool = True             # True: run2 shifted +x; False: -x
    stairs_center_y_offset: float = 0.0
    stairs_center_x_offset: float = 0.0
    origin_offset_y: float = 0.0

    # walls
    wall_thickness: float = 0.06
    wall_clearance: float = 0.02
    wall_height_extra: float = 0.05


@configclass
class MeshRandomGridTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a random grid mesh terrain."""

    function = mesh_terrains.random_grid_terrain

    grid_width: float = MISSING
    """The width of the grid cells (in m)."""
    grid_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the grid cells (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    holes: bool = False
    """If True, the terrain will have holes in the steps. Defaults to False.

    If :obj:`holes` is True, the terrain will have randomized grid cells only along the plane extending
    from the platform (like a plus sign). The remaining area remains empty and no border will be added.
    """


@configclass
class MeshRailsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with box rails as extrusions."""

    function = mesh_terrains.rails_terrain

    rail_thickness_range: tuple[float, float] = MISSING
    """The thickness of the inner and outer rails (in m)."""
    rail_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the rails (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

@configclass
class MeshTwosidedRailsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with one-sided box rails as extrusions."""

    function = mesh_terrains.two_sided_rails_terrain

    rail_thickness: float = MISSING
    """The thickness of the inner and outer rails (in m)."""
    rail_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the rails (in m)."""
    rail_width: float = MISSING
    """The width of the rails (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class MeshPitTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a pit that leads out of the pit."""

    function = mesh_terrains.pit_terrain

    pit_depth_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the pit (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    double_pit: bool = False
    """If True, the pit contains two levels of stairs. Defaults to False."""


@configclass
class MeshBoxTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with boxes (similar to a pyramid)."""

    function = mesh_terrains.box_terrain

    box_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the box (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    double_box: bool = False
    """If True, the pit contains two levels of stairs/boxes. Defaults to False."""


@configclass
class MeshGapTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a gap around the platform."""

    function = mesh_terrains.gap_terrain

    gap_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the gap (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class MeshFloatingRingTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a floating ring around the center."""

    function = mesh_terrains.floating_ring_terrain

    ring_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the ring (in m)."""
    ring_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the ring (in m)."""
    ring_thickness: float = MISSING
    """The thickness (along z) of the ring (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class MeshStarTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a star pattern."""

    function = mesh_terrains.star_terrain

    num_bars: int = MISSING
    """The number of bars per-side the star. Must be greater than 2."""
    bar_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the bars in the star (in m)."""
    bar_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the bars in the star (in m)."""
    platform_width: float = 1.0
    """The width of the cylindrical platform at the center of the terrain. Defaults to 1.0."""


@configclass
class MeshRepeatedObjectsTerrainCfg(SubTerrainBaseCfg):
    """Base configuration for a terrain with repeated objects."""

    @configclass
    class ObjectCfg:
        """Configuration of repeated objects."""

        num_objects: int = MISSING
        """The number of objects to add to the terrain."""
        height: float = MISSING
        """The height (along z) of the object (in m)."""

    function = mesh_terrains.repeated_objects_terrain

    object_type: Literal["cylinder", "box", "cone"] | callable = MISSING
    """The type of object to generate.

    The type can be a string or a callable. If it is a string, the function will look for a function called
    ``make_{object_type}`` in the current module scope. If it is a callable, the function will
    use the callable to generate the object.
    """
    object_params_start: ObjectCfg = MISSING
    """The object curriculum parameters at the start of the curriculum."""
    object_params_end: ObjectCfg = MISSING
    """The object curriculum parameters at the end of the curriculum."""

    max_height_noise: float = 0.0
    """The maximum amount of noise to add to the height of the objects (in m). Defaults to 0.0."""
    platform_width: float = 1.0
    """The width of the cylindrical platform at the center of the terrain. Defaults to 1.0."""


@configclass
class MeshRepeatedPyramidsTerrainCfg(MeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated pyramids."""

    @configclass
    class ObjectCfg(MeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for a curriculum of repeated pyramids."""

        radius: float = MISSING
        """The radius of the pyramids (in m)."""
        max_yx_angle: float = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""

    object_type = mesh_utils_terrains.make_cone

    object_params_start: ObjectCfg = MISSING
    """The object curriculum parameters at the start of the curriculum."""
    object_params_end: ObjectCfg = MISSING
    """The object curriculum parameters at the end of the curriculum."""


@configclass
class MeshRepeatedBoxesTerrainCfg(MeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated boxes."""

    @configclass
    class ObjectCfg(MeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for repeated boxes."""

        size: tuple[float, float] = MISSING
        """The width (along x) and length (along y) of the box (in m)."""
        max_yx_angle: float = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""

    object_type = mesh_utils_terrains.make_box

    object_params_start: ObjectCfg = MISSING
    """The box curriculum parameters at the start of the curriculum."""
    object_params_end: ObjectCfg = MISSING
    """The box curriculum parameters at the end of the curriculum."""


@configclass
class MeshRepeatedCylindersTerrainCfg(MeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated cylinders."""

    @configclass
    class ObjectCfg(MeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for repeated cylinder."""

        radius: float = MISSING
        """The radius of the pyramids (in m)."""
        max_yx_angle: float = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""

    object_type = mesh_utils_terrains.make_cylinder

    object_params_start: ObjectCfg = MISSING
    """The box curriculum parameters at the start of the curriculum."""
    object_params_end: ObjectCfg = MISSING
    """The box curriculum parameters at the end of the curriculum."""


@configclass
class MeshRoomTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a room-like structure."""

    function = mesh_terrains.room_terrain

    wall_thickness: float = MISSING
    """The thickness of the walls (in m)."""
    wall_height: float = MISSING
    """The height of the walls (in m)."""
    door_width_range: tuple[float, float] = MISSING
    """The width of the door (in m)."""
    door_height: float = MISSING
    """The height of the door (in m)."""
    room_size: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""