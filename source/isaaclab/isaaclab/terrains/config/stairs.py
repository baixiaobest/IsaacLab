import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg, FlatPatchSamplingCfg
import copy


FLAT_PATCH_STAIRS = FlatPatchSamplingCfg(
    num_patches=1000,
    patch_radius=0.35,
    x_range=(-8.0, 8.0),
    y_range=(-8.0, 8.0),
    z_range=(0.2, 10.0), # setpoint can only be set on stairs
    max_height_diff=0.7,
    min_distance=0.0
)

FLAT_PATCH_SPIRAL_STAIRS = FlatPatchSamplingCfg(
    num_patches=300,
    patch_radius=0.2,
    x_range=(-8.0, 8.0),
    y_range=(-8.0, 8.0),
    z_range=(0.1, 3.0), # setpoint can only be set on stairs
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

PYRAMIDS_CLIMB_UP = TerrainGeneratorCfg(
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

PYRAMIDS_CLIMB_DOWN = TerrainGeneratorCfg(
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
        )
    },
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

SPIRAL_STAIRS = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "spiral_stairs_cw": terrain_gen.MeshSpiralStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.02, 0.12),
            step_width=0.26,                # tangential tread length
            stairs_width=1.4,
            stairs_width_range=(2.5, 2.0),  # easy → hard radial width
            inner_radius=0.75,
            revolutions=(0.2, 0.5),
            num_steps=None,                 # infer from revolutions & step_width
            clockwise=True,
            start_angle=0.0,
            # landing & pole (None → defaults inside generator)
            landing_thickness=None,
            landing_radius=None,
            add_center_pole=True,
            center_pole_radius=None,
            # placement
            origin_offset_y=0.5,
            origin_offset_x=1.5,
            # sampling targets only on stairs/landing
            flat_patch_sampling={"target": FLAT_PATCH_SPIRAL_STAIRS},
            has_guide_lines=True
        ),
        "spiral_stairs_ccw": terrain_gen.MeshSpiralStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.02, 0.12),
            step_width=0.26,
            stairs_width=1.4,
            stairs_width_range=(2.5, 2.0),
            inner_radius=0.75,
            revolutions=(0.2, 0.5),
            num_steps=None,
            clockwise=False,
            start_angle=0.0,
            landing_thickness=None,
            landing_radius=None,
            add_center_pole=True,
            center_pole_radius=None,
            origin_offset_y=-0.5,
            origin_offset_x=1.5,
            flat_patch_sampling={"target": FLAT_PATCH_SPIRAL_STAIRS},
            has_guide_lines=True
        ),
    },
)

TURN_90_STAIRS = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "turning_stairs_90_right": terrain_gen.MeshTurningStairs90TerrainCfg(
            proportion=1.0,
            second_run_curriculum=True,
            step_height_range=(0.02, 0.15),
            step_width=0.26,
            num_steps_run1=10,
            num_steps_run2=10,
            run1_length=3.0,
            run2_length=3.0,
            stairs_width=1.4,
            stairs_width_range=(2.0, 1.2),  # easy→hard
            landing_length=1.2,
            landing_width=None,             # None → equals usable width
            turn_right=True,                # second run along +x
            origin_offset_y=-0.5,
            wall_thickness=0.08,
            wall_clearance=0.03,
            wall_height_extra=0.10,
            flat_patch_sampling={"target": FLAT_PATCH_STAIRS},
            has_guide_lines=True
        ),

        "turning_stairs_90_left": terrain_gen.MeshTurningStairs90TerrainCfg(
            proportion=1.0,
            second_run_curriculum=True,
            step_height_range=(0.02, 0.12),
            step_width=0.26,
            num_steps_run1=10,
            num_steps_run2=10,
            run1_length=3.0,
            run2_length=3.0,
            stairs_width=1.4,
            stairs_width_range=(2.0, 1.4),  # easy→hard
            landing_length=1.2,
            landing_width=None,             # None → equals usable width
            turn_right=False,                # second run along +x
            origin_offset_y=-0.5,
            wall_thickness=0.08,
            wall_clearance=0.03,
            wall_height_extra=0.10,
            flat_patch_sampling={"target": FLAT_PATCH_STAIRS},
            has_guide_lines=True
        ),
    },
)

FLAT_PATCH_STAIRS_TEST = FlatPatchSamplingCfg(
    num_patches=300,
    patch_radius=0.35,
    x_range=(-8.0, 8.0),
    y_range=(-8.0, 8.0),
    z_range=(0.1, 10.0), # setpoint can only be set on stairs
    max_height_diff=0.3,
    min_distance=0.0
)

FLAT_PATCH_STAIRS_TEST_HIGH_DIFFICULTY = FlatPatchSamplingCfg(
    num_patches=300,
    patch_radius=0.35,
    x_range=(-8.0, 8.0),
    y_range=(-8.0, 8.0),
    z_range=(0.1, 10.0), # setpoint can only be set on stairs
    max_height_diff=0.2,
    min_distance=0.0
)

TURN_90_STAIRS_TEST_LEVEL_1 = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "turning_stairs_90_right": terrain_gen.MeshTurningStairs90TerrainCfg(
            proportion=1.0,
            second_run_curriculum=False,
            step_height_range=(0.04, 0.04),
            step_width=0.26,
            num_steps_run1=10,
            num_steps_run2=10,
            run1_length=3.0,
            run2_length=3.0,
            stairs_width=1.4,
            stairs_width_range=(2.0, 2.0), 
            landing_length=1.2,
            landing_width=None,             # None → equals usable width
            turn_right=True,                # second run along +x
            origin_offset_y=-0.0,
            wall_thickness=0.08,
            wall_clearance=0.03,
            wall_height_extra=0.10,
            flat_patch_sampling={"target": FLAT_PATCH_STAIRS_TEST},
            has_guide_lines=True
        ),

        "turning_stairs_90_left": terrain_gen.MeshTurningStairs90TerrainCfg(
            proportion=1.0,
            second_run_curriculum=False,
            step_height_range=(0.04, 0.04),
            step_width=0.26,
            num_steps_run1=10,
            num_steps_run2=10,
            run1_length=3.0,
            run2_length=3.0,
            stairs_width=1.4,
            stairs_width_range=(2.0, 2.0),
            landing_length=1.2,
            landing_width=None,             # None → equals usable width
            turn_right=False,                # second run along +x
            origin_offset_y=-0.0,
            wall_thickness=0.08,
            wall_clearance=0.03,
            wall_height_extra=0.10,
            flat_patch_sampling={"target": FLAT_PATCH_STAIRS_TEST},
            has_guide_lines=True
        ),
    },
)

TURN_90_STAIRS_TEST_LEVEL_0 = copy.deepcopy(TURN_90_STAIRS_TEST_LEVEL_1)
TURN_90_STAIRS_TEST_LEVEL_0.sub_terrains["turning_stairs_90_right"].step_height_range = (0.02, 0.02)
TURN_90_STAIRS_TEST_LEVEL_0.sub_terrains["turning_stairs_90_left"].step_height_range = (0.02, 0.02)

TURN_90_STAIRS_TEST_LEVEL_1.sub_terrains["turning_stairs_90_right"].flat_patch_sampling = {"target": FLAT_PATCH_STAIRS_TEST}
TURN_90_STAIRS_TEST_LEVEL_1.sub_terrains["turning_stairs_90_left"].flat_patch_sampling = {"target": FLAT_PATCH_STAIRS_TEST}

TURN_90_STAIRS_TEST_LEVEL_2 = copy.deepcopy(TURN_90_STAIRS_TEST_LEVEL_1)
TURN_90_STAIRS_TEST_LEVEL_2.sub_terrains["turning_stairs_90_right"].step_height_range = (0.06, 0.06)
TURN_90_STAIRS_TEST_LEVEL_2.sub_terrains["turning_stairs_90_left"].step_height_range = (0.06, 0.06)

TURN_90_STAIRS_TEST_LEVEL_3 = copy.deepcopy(TURN_90_STAIRS_TEST_LEVEL_1)
TURN_90_STAIRS_TEST_LEVEL_3.sub_terrains["turning_stairs_90_right"].step_height_range = (0.08, 0.08)
TURN_90_STAIRS_TEST_LEVEL_3.sub_terrains["turning_stairs_90_left"].step_height_range = (0.08, 0.08)

TURN_90_STAIRS_TEST_LEVEL_4 = copy.deepcopy(TURN_90_STAIRS_TEST_LEVEL_1)
TURN_90_STAIRS_TEST_LEVEL_4.sub_terrains["turning_stairs_90_right"].step_height_range = (0.10, 0.10)
TURN_90_STAIRS_TEST_LEVEL_4.sub_terrains["turning_stairs_90_left"].step_height_range = (0.10, 0.10)

TURN_90_STAIRS_TEST_LEVEL_5 = copy.deepcopy(TURN_90_STAIRS_TEST_LEVEL_1)
TURN_90_STAIRS_TEST_LEVEL_5.sub_terrains["turning_stairs_90_right"].step_height_range = (0.12, 0.12)
TURN_90_STAIRS_TEST_LEVEL_5.sub_terrains["turning_stairs_90_left"].step_height_range = (0.12, 0.12)
TURN_90_STAIRS_TEST_LEVEL_5.sub_terrains["turning_stairs_90_right"].flat_patch_sampling = {"target": FLAT_PATCH_STAIRS_TEST_HIGH_DIFFICULTY}
TURN_90_STAIRS_TEST_LEVEL_5.sub_terrains["turning_stairs_90_left"].flat_patch_sampling = {"target": FLAT_PATCH_STAIRS_TEST_HIGH_DIFFICULTY}

TURN_90_STAIRS_TEST_LEVEL_6 = copy.deepcopy(TURN_90_STAIRS_TEST_LEVEL_1)
TURN_90_STAIRS_TEST_LEVEL_6.sub_terrains["turning_stairs_90_right"].step_height_range = (0.14, 0.14)
TURN_90_STAIRS_TEST_LEVEL_6.sub_terrains["turning_stairs_90_left"].step_height_range = (0.14, 0.14)
TURN_90_STAIRS_TEST_LEVEL_6.sub_terrains["turning_stairs_90_right"].flat_patch_sampling = {"target": FLAT_PATCH_STAIRS_TEST_HIGH_DIFFICULTY}
TURN_90_STAIRS_TEST_LEVEL_6.sub_terrains["turning_stairs_90_left"].flat_patch_sampling = {"target": FLAT_PATCH_STAIRS_TEST_HIGH_DIFFICULTY}

TURN_180_STAIRS = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "turning_stairs_180_right": terrain_gen.MeshTurningStairs180TerrainCfg(
            proportion=1.0,
            second_run_curriculum=True,
            step_height_range=(0.02, 0.12),
            step_width=0.26,
            num_steps_run1=10,
            num_steps_run2=10,
            run1_length=3.0,
            run2_length=3.0,
            stairs_width_range=(2.0, 1.4),
            landing_length=1.2,
            landing_width=None,
            run2_on_positive_x=True,        # place second run at +x
            origin_offset_y=-0.5,
            wall_thickness=0.08,
            wall_clearance=0.03,
            wall_height_extra=0.10,
            flat_patch_sampling={"target": FLAT_PATCH_STAIRS},
            has_guide_lines=True
        ),

        "turning_stairs_180_left": terrain_gen.MeshTurningStairs180TerrainCfg(
            proportion=1.0,
            second_run_curriculum=True,
            step_height_range=(0.02, 0.12),
            step_width=0.26,
            num_steps_run1=10,
            num_steps_run2=10,
            run1_length=3.0,
            run2_length=3.0,
            stairs_width_range=(2.0, 1.4),
            landing_length=1.2,
            landing_width=None,
            run2_on_positive_x=False,        # place second run at -x
            origin_offset_y=-0.5,
            wall_thickness=0.08,
            wall_clearance=0.03,
            wall_height_extra=0.10,
            flat_patch_sampling={"target": FLAT_PATCH_STAIRS},
            has_guide_lines=True
        ),
    },
)

TURN_180_STAIRS_NO_SECOND_RUN_CURRICULUM = copy.deepcopy(TURN_180_STAIRS)
TURN_180_STAIRS_NO_SECOND_RUN_CURRICULUM.sub_terrains["turning_stairs_180_right"].second_run_curriculum = False
TURN_180_STAIRS_NO_SECOND_RUN_CURRICULUM.sub_terrains["turning_stairs_180_left"].second_run_curriculum = False

TURN_180_STAIRS_TEST_LEVEL_1 = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "turning_stairs_180_right": terrain_gen.MeshTurningStairs180TerrainCfg(
            proportion=1.0,
            second_run_curriculum=False,
            step_height_range=(0.04, 0.04),
            step_width=0.26,
            num_steps_run1=10,
            num_steps_run2=10,
            run1_length=3.0,
            run2_length=3.0,
            stairs_width_range=(2.0, 2.0),
            landing_length=1.4,
            landing_width=None,
            run2_on_positive_x=True,
            origin_offset_y=-0.5,
            wall_thickness=0.08,
            wall_clearance=0.03,
            wall_height_extra=0.10,
            flat_patch_sampling={"target": FLAT_PATCH_STAIRS_TEST},
            has_guide_lines=True
        ),

        "turning_stairs_180_left": terrain_gen.MeshTurningStairs180TerrainCfg(
            proportion=1.0,
            second_run_curriculum=False,
            step_height_range=(0.04, 0.04),
            step_width=0.26,
            num_steps_run1=10,
            num_steps_run2=10,
            run1_length=3.0,
            run2_length=3.0,
            stairs_width_range=(2.0, 2.0),
            landing_length=1.4,
            landing_width=None,
            run2_on_positive_x=False,
            origin_offset_y=-0.5,
            wall_thickness=0.08,
            wall_clearance=0.03,
            wall_height_extra=0.10,
            flat_patch_sampling={"target": FLAT_PATCH_STAIRS_TEST},
            has_guide_lines=True
        ),
    },
)

TURN_180_STAIRS_TEST_LEVEL_1.sub_terrains["turning_stairs_180_right"].flat_patch_sampling = {"target": FLAT_PATCH_STAIRS_TEST}
TURN_180_STAIRS_TEST_LEVEL_1.sub_terrains["turning_stairs_180_left"].flat_patch_sampling = {"target": FLAT_PATCH_STAIRS_TEST}

TURN_180_STAIRS_TEST_LEVEL_2 = copy.deepcopy(TURN_180_STAIRS_TEST_LEVEL_1)
TURN_180_STAIRS_TEST_LEVEL_2.sub_terrains["turning_stairs_180_right"].step_height_range = (0.06, 0.06)
TURN_180_STAIRS_TEST_LEVEL_2.sub_terrains["turning_stairs_180_left"].step_height_range = (0.06, 0.06)

TURN_180_STAIRS_TEST_LEVEL_3 = copy.deepcopy(TURN_180_STAIRS_TEST_LEVEL_1)
TURN_180_STAIRS_TEST_LEVEL_3.sub_terrains["turning_stairs_180_right"].step_height_range = (0.08, 0.08)
TURN_180_STAIRS_TEST_LEVEL_3.sub_terrains["turning_stairs_180_left"].step_height_range = (0.08, 0.08)

TURN_180_STAIRS_TEST_LEVEL_4 = copy.deepcopy(TURN_180_STAIRS_TEST_LEVEL_1)
TURN_180_STAIRS_TEST_LEVEL_4.sub_terrains["turning_stairs_180_right"].step_height_range = (0.10, 0.10)
TURN_180_STAIRS_TEST_LEVEL_4.sub_terrains["turning_stairs_180_left"].step_height_range = (0.10, 0.10)

TURN_180_STAIRS_TEST_LEVEL_5 = copy.deepcopy(TURN_180_STAIRS_TEST_LEVEL_1)
TURN_180_STAIRS_TEST_LEVEL_5.sub_terrains["turning_stairs_180_right"].step_height_range = (0.12, 0.12)
TURN_180_STAIRS_TEST_LEVEL_5.sub_terrains["turning_stairs_180_left"].step_height_range = (0.12, 0.12)

TURN_180_STAIRS_TEST_LEVEL_6 = copy.deepcopy(TURN_180_STAIRS_TEST_LEVEL_1)
TURN_180_STAIRS_TEST_LEVEL_6.sub_terrains["turning_stairs_180_right"].step_height_range = (0.14, 0.14)
TURN_180_STAIRS_TEST_LEVEL_6.sub_terrains["turning_stairs_180_left"].step_height_range = (0.14, 0.14)

SPIRAL_STAIRS_TEST_LEVEL_1 = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "spiral_stairs_cw": terrain_gen.MeshSpiralStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.04, 0.04),
            step_width=0.3,
            stairs_width=1.4,
            stairs_width_range=(2.5, 2.5),
            inner_radius=0.5,
            revolutions=(0.5, 0.5),
            num_steps=None,
            clockwise=True,
            start_angle=0.0,
            landing_thickness=None,
            landing_radius=None,
            add_center_pole=True,
            center_pole_radius=None,
            origin_offset_y=1.5,
            origin_offset_x=2.5,
            flat_patch_sampling={"target": FLAT_PATCH_SPIRAL_STAIRS},
            has_guide_lines=True
        ),
        "spiral_stairs_ccw": terrain_gen.MeshSpiralStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.04, 0.04),
            step_width=0.3,
            stairs_width=1.4,
            stairs_width_range=(2.5, 2.5),
            inner_radius=0.5,
            revolutions=(0.5, 0.5),
            num_steps=None,
            clockwise=False,
            start_angle=0.0,
            landing_thickness=None,
            landing_radius=None,
            add_center_pole=True,
            center_pole_radius=None,
            origin_offset_y=-1.5,
            origin_offset_x=2.5,
            flat_patch_sampling={"target": FLAT_PATCH_SPIRAL_STAIRS},
            has_guide_lines=True
        ),
    },
)

SPIRAL_STAIRS_TEST_LEVEL_1.sub_terrains["spiral_stairs_cw"].flat_patch_sampling = {"target": FLAT_PATCH_SPIRAL_STAIRS}
SPIRAL_STAIRS_TEST_LEVEL_1.sub_terrains["spiral_stairs_ccw"].flat_patch_sampling = {"target": FLAT_PATCH_SPIRAL_STAIRS}

SPIRAL_STAIRS_TEST_LEVEL_2 = copy.deepcopy(SPIRAL_STAIRS_TEST_LEVEL_1)
SPIRAL_STAIRS_TEST_LEVEL_2.sub_terrains["spiral_stairs_cw"].step_height_range = (0.06, 0.06)
SPIRAL_STAIRS_TEST_LEVEL_2.sub_terrains["spiral_stairs_ccw"].step_height_range = (0.06, 0.06)


SPIRAL_STAIRS_TEST_LEVEL_3 = copy.deepcopy(SPIRAL_STAIRS_TEST_LEVEL_1)
SPIRAL_STAIRS_TEST_LEVEL_3.sub_terrains["spiral_stairs_cw"].step_height_range = (0.08, 0.08)
SPIRAL_STAIRS_TEST_LEVEL_3.sub_terrains["spiral_stairs_ccw"].step_height_range = (0.08, 0.08)


SPIRAL_STAIRS_TEST_LEVEL_4 = copy.deepcopy(SPIRAL_STAIRS_TEST_LEVEL_1)
SPIRAL_STAIRS_TEST_LEVEL_4.sub_terrains["spiral_stairs_cw"].step_height_range = (0.10, 0.10)
SPIRAL_STAIRS_TEST_LEVEL_4.sub_terrains["spiral_stairs_ccw"].step_height_range = (0.10, 0.10)


SPIRAL_STAIRS_TEST_LEVEL_5 = copy.deepcopy(SPIRAL_STAIRS_TEST_LEVEL_1)
SPIRAL_STAIRS_TEST_LEVEL_5.sub_terrains["spiral_stairs_cw"].step_height_range = (0.12, 0.12)
SPIRAL_STAIRS_TEST_LEVEL_5.sub_terrains["spiral_stairs_ccw"].step_height_range = (0.12, 0.12)


SPIRAL_STAIRS_TEST_LEVEL_6 = copy.deepcopy(SPIRAL_STAIRS_TEST_LEVEL_1)
SPIRAL_STAIRS_TEST_LEVEL_6.sub_terrains["spiral_stairs_cw"].step_height_range = (0.14, 0.14)
SPIRAL_STAIRS_TEST_LEVEL_6.sub_terrains["spiral_stairs_ccw"].step_height_range = (0.14, 0.14)


LINEAR_STAIRS_TEST_LEVEL_1 = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "linear_stairs_ground": terrain_gen.MeshOneSidedLinearStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.04, 0.04),
            num_steps=10,
            step_width=0.3,
            stairs_width_range=(3.0, 1.5),
            stairs_length=6.0,
            origin_offset_y=-4.0,
            landing_length=2.0,
            flat_patch_sampling={"target": FLAT_PATCH_STAIRS_TEST},
            has_guide_lines=True
        ),
    },
)

LINEAR_STAIRS_TEST_LEVEL_2 = copy.deepcopy(LINEAR_STAIRS_TEST_LEVEL_1)
LINEAR_STAIRS_TEST_LEVEL_2.sub_terrains["linear_stairs_ground"].step_height_range = (0.06, 0.06)

LINEAR_STAIRS_TEST_LEVEL_3 = copy.deepcopy(LINEAR_STAIRS_TEST_LEVEL_1)
LINEAR_STAIRS_TEST_LEVEL_3.sub_terrains["linear_stairs_ground"].step_height_range = (0.08, 0.08)

LINEAR_STAIRS_TEST_LEVEL_4 = copy.deepcopy(LINEAR_STAIRS_TEST_LEVEL_1)
LINEAR_STAIRS_TEST_LEVEL_4.sub_terrains["linear_stairs_ground"].step_height_range = (0.10, 0.10)

LINEAR_STAIRS_TEST_LEVEL_5 = copy.deepcopy(LINEAR_STAIRS_TEST_LEVEL_1)
LINEAR_STAIRS_TEST_LEVEL_5.sub_terrains["linear_stairs_ground"].step_height_range = (0.12, 0.12)

LINEAR_STAIRS_TEST_LEVEL_6 = copy.deepcopy(LINEAR_STAIRS_TEST_LEVEL_1)
LINEAR_STAIRS_TEST_LEVEL_6.sub_terrains["linear_stairs_ground"].step_height_range = (0.14, 0.14)

LINEAR_STAIRS = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "linear_stairs_ground": terrain_gen.MeshOneSidedLinearStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.04, 0.14),
            num_steps=10,
            step_width=0.3,
            stairs_width_range=(3.0, 1.5),
            stairs_length=6.0,
            origin_offset_y=-4.0,
            landing_length=2.0,
            flat_patch_sampling={"target": FLAT_PATCH_STAIRS_TEST},
            has_guide_lines=True
        ),
    },
)