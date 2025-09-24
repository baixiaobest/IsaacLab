import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg, FlatPatchSamplingCfg


FLAT_PATCH_STAIRS = FlatPatchSamplingCfg(
    num_patches=1000,
    patch_radius=0.35,
    x_range=(-8.0, 8.0),
    y_range=(-8.0, 8.0),
    z_range=(0.5, 5.0), # setpoint can only be set on stairs
    max_height_diff=0.2,
    min_distance=0.0
)

FLAT_PATCH_PYRAMIDS = FlatPatchSamplingCfg(
    num_patches=1000,
    patch_radius=0.4,
    x_range=(-8, 8.0),
    y_range=(-8.0, 8.0),
    z_range=(-5.0, 5.0), # setpoint can only be set on stairs
    max_height_diff=0.1,
    min_distance=0.0
)

PYRAMIDS_ONLY = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.5,
            holes=False,
            flat_patch_sampling={"target": FLAT_PATCH_PYRAMIDS}
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0, 0.15),        
            step_width=0.3,
            platform_width=3.0,
            border_width=1.5,
            holes=False,
            flat_patch_sampling={"target": FLAT_PATCH_PYRAMIDS}
        ),
    },
)

DIVERSE_STAIRS = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "linear_stairs_ground": terrain_gen.MeshLinearStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.15),
            num_steps=10,
            step_width=0.2,
            stairs_width=2.0,
            stairs_length=6.0,
            origin_offset_y=4.0,
            flat_patch_sampling={"target": FLAT_PATCH_STAIRS}
        ),
        "linear_stairs_walled": terrain_gen.MeshWalledLinearStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.15),
            num_steps=10,
            step_width=0.2,
            stairs_width_range=(2.0, 0.8),   # easy→hard width shrink
            stairs_length=6.0,
            origin_offset_y=4.5,
            wall_thickness=0.08,
            wall_clearance=0.03,
            wall_height_extra=0.5,
            flat_patch_sampling={"target": FLAT_PATCH_STAIRS}
        ),
        "turning_stairs_90_right": terrain_gen.MeshTurningStairs90TerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.15),
            step_width=0.20,
            num_steps_run1=10,
            num_steps_run2=10,
            run1_length=3.0,
            run2_length=3.0,
            stairs_width=1.4,
            stairs_width_range=(1.4, 0.8),  # easy→hard
            landing_length=1.2,
            landing_width=None,             # None → equals usable width
            turn_right=True,                # second run along +x
            origin_offset_y=2.5,
            wall_thickness=0.08,
            wall_clearance=0.03,
            wall_height_extra=0.10,
            flat_patch_sampling={"target": FLAT_PATCH_STAIRS}
        ),

        "turning_stairs_90_left": terrain_gen.MeshTurningStairs90TerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.15),
            step_width=0.20,
            num_steps_run1=10,
            num_steps_run2=10,
            run1_length=3.0,
            run2_length=3.0,
            stairs_width=1.4,
            stairs_width_range=(1.4, 0.8),  # easy→hard
            landing_length=1.2,
            landing_width=None,             # None → equals usable width
            turn_right=False,                # second run along +x
            origin_offset_y=2.5,
            wall_thickness=0.08,
            wall_clearance=0.03,
            wall_height_extra=0.10,
            flat_patch_sampling={"target": FLAT_PATCH_STAIRS}
        ),

        "turning_stairs_180_right": terrain_gen.MeshTurningStairs180TerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.15),
            step_width=0.20,
            num_steps_run1=10,
            num_steps_run2=10,
            run1_length=3.0,
            run2_length=3.0,
            stairs_width=1.4,
            stairs_width_range=(1.4, 0.8),
            landing_length=1.2,
            landing_offset_x=1.6,           # corridor spacing
            landing_width=None,
            run2_on_positive_x=True,        # place second run at +x
            origin_offset_y=2.5,
            wall_thickness=0.08,
            wall_clearance=0.03,
            wall_height_extra=0.10,
            flat_patch_sampling={"target": FLAT_PATCH_STAIRS}
        ),

        "turning_stairs_180_left": terrain_gen.MeshTurningStairs180TerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.15),
            step_width=0.20,
            num_steps_run1=10,
            num_steps_run2=10,
            run1_length=3.0,
            run2_length=3.0,
            stairs_width=1.4,
            stairs_width_range=(1.4, 0.8),
            landing_length=1.2,
            landing_offset_x=1.6,           # corridor spacing
            landing_width=None,
            run2_on_positive_x=False,        # place second run at +x
            origin_offset_y=2.5,
            wall_thickness=0.08,
            wall_clearance=0.03,
            wall_height_extra=0.10,
            flat_patch_sampling={"target": FLAT_PATCH_STAIRS}
        ),
    },
)