from isaaclab.terrains.test_terrain_generator import TestTerrainGenerator
from isaaclab.terrains.test_terrain_generator_cfg import (
    TestTerrainGeneratorCfg, SubTerrainTestCfg, CubeCfg, CylinderCfg, SphereCfg
)

# Create a simple test terrain with a cube, cylinder, and sphere
sub_terrain1 = SubTerrainTestCfg(
    cubes=[
        CubeCfg(
            position=[2, 2, 0.5],
            dimensions=[1.2, 1.2, 2.0]
        ),
        CubeCfg(
            position=[-2, 2, 0.5],
            dimensions=[1.2, 1.2, 2.0]
        ),
        CubeCfg(
            position=[2, -2, 0.5],
            dimensions=[1.2, 1.2, 2.0]
        ),
        CubeCfg(
            position=[-2, -2, 0.5],
            dimensions=[1.2, 1.2, 2.0]
        ),
    ],
    start_position=[-2.0, -5.0, 0.0],
    goal_position=[6.0, 6.0, 0.0]
)

# Create a second sub-terrain with different objects
sub_terrain2 = SubTerrainTestCfg(
    cubes=[
        CubeCfg(
            position=[3.0, 3.0, 0.5],
            dimensions=[2.0, 2.0, 1.0]
        ),
        CubeCfg(
            position=[3.0, -3.0, 0.5],
            dimensions=[1.0, 1.0, 1.0]
        ),
        CubeCfg(
            position=[-3.0, -3.0, 0.5],
            dimensions=[0.5, 0.5, 1.0]
        ),
        CubeCfg(
            position=[-1.5, 1.5, 0.5],
            dimensions=[0.5, 0.5, 1.0]
        )
    ],
    start_position=[0.0, 0.0, 0.0],
    goal_position=[9.0, 9.0, 0.0]
)

# Create the terrain generator configuration
TEST_TERRAIN_CFG = TestTerrainGeneratorCfg(
    size=(50.0, 50.0),
    num_rows=1,
    num_cols=1,
    sub_terrains=[sub_terrain1],
    subterrain_spacing=4.0
)