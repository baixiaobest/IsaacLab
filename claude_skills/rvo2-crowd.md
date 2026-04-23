# RVO2 Crowd Simulation in IsaacLab

Use this skill when asked to add crowd simulation (moving persons/pedestrians/agents) to an IsaacLab environment using Python-RVO2.

---

## 1. Install Python-RVO2

Python-RVO2 requires building from source (no pre-built wheel for IsaacSim Python):

```bash
# Clone and build C++ library first
cd /tmp && git clone https://github.com/sybrenstuvel/Python-RVO2
mkdir -p /tmp/Python-RVO2/build/RVO2
cd /tmp/Python-RVO2/build/RVO2 && cmake ../.. -DCMAKE_CXX_FLAGS=-fPIC
cmake --build .

# Install Cython into IsaacSim's Python, then build the extension
/opt/IsaacSim/python.sh -m pip install cython
cd /tmp/Python-RVO2 && /opt/IsaacSim/python.sh setup.py build_ext --inplace

# Copy .so to user site-packages so it's importable everywhere
cp /tmp/Python-RVO2/rvo2.cpython-310-x86_64-linux-gnu.so \
   ~/.local/lib/python3.10/site-packages/

# Verify
/opt/IsaacSim/python.sh -c "import rvo2; print('OK')"
```

---

## 2. RVO2CrowdManager (`mdp/rvo2_crowd.py`)

Minimal wrapper around `rvo2.PyRVOSimulator`:

```python
import math, numpy as np
try:
    import rvo2
except ImportError:
    raise ImportError("Install Python-RVO2: https://github.com/sybrenstuvel/Python-RVO2")

class RVO2CrowdManager:
    def __init__(self, num_agents, sim_dt, radius=0.3, max_speed=1.4,
                 neighbor_dist=5.0, max_neighbors=10, time_horizon=5.0):
        self.num_agents = num_agents
        self.sim_dt = sim_dt
        self.radius = radius
        self.max_speed = max_speed
        self._neighbor_dist = neighbor_dist
        self._max_neighbors = max_neighbors
        self._time_horizon = time_horizon
        self._sim = None
        self._goals = [(0.0, 0.0)] * num_agents
        self._robot_agent_id = None

    def reset(self, positions, goals):
        self._goals = list(goals)
        self._robot_agent_id = None
        self._sim = rvo2.PyRVOSimulator(
            self.sim_dt, self._neighbor_dist, self._max_neighbors,
            self._time_horizon, self._time_horizon, self.radius, self.max_speed)
        for pos in positions:
            self._sim.addAgent(pos)

    def step(self):
        if self._sim is None:
            return
        for i in range(self.num_agents):
            pos = self._sim.getAgentPosition(i)
            gx, gy = self._goals[i]
            dx, dy = gx - pos[0], gy - pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 0.1:
                self._sim.setAgentPrefVelocity(i, (0.0, 0.0))
            else:
                s = min(self.max_speed, dist) / dist
                self._sim.setAgentPrefVelocity(i, (dx*s, dy*s))
        if self._robot_agent_id is not None:
            self._sim.setAgentPrefVelocity(self._robot_agent_id, (0.0, 0.0))
        self._sim.doStep()

    def get_positions(self):
        if self._sim is None:
            return np.zeros((self.num_agents, 2))
        return np.array([self._sim.getAgentPosition(i) for i in range(self.num_agents)])

    def set_goals(self, goals):
        self._goals = list(goals)

    def update_robot_obstacle(self, position, radius=None):
        if self._sim is None:
            return
        r = radius if radius is not None else self.radius * 1.5
        if self._robot_agent_id is None:
            self._robot_agent_id = self._sim.addAgent(
                position, self._neighbor_dist, self._max_neighbors,
                self._time_horizon, self._time_horizon, r, 0.0, (0.0, 0.0))
        else:
            self._sim.setAgentPosition(self._robot_agent_id, position)
```

Export it from `mdp/__init__.py`:
```python
from .rvo2_crowd import RVO2CrowdManager
```

---

## 3. Person Capsules in Scene Config

Add kinematic capsule rigid bodies to your scene config:

```python
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

PERSON_RADIUS = 0.3
PERSON_HEIGHT = 1.2
PERSON_Z = PERSON_RADIUS + PERSON_HEIGHT / 2.0   # = 0.9 m

@configclass
class RVO2SceneCfg(MySceneCfg):
    person_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Person_0",
        spawn=sim_utils.CapsuleCfg(
            radius=PERSON_RADIUS,
            height=PERSON_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=70.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, PERSON_Z)),
    )
    # Repeat for person_1 ... person_N with different positions/colors
```

**Key**: `kinematic_enabled=True` — PhysX ignores gravity/forces; you control position each step.

---

## 4. Custom Environment Class

Subclass `ManagerBasedRLEnv` to bridge RVO2 ↔ Isaac Sim:

```python
from isaaclab.envs import ManagerBasedRLEnv

class RVO2NavigationEnv(ManagerBasedRLEnv):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._rvo2_manager = None
        self._person_objects = []
        self._person_goals = []
        self._step_count = 0
        self._setup_rvo2()

    def _setup_rvo2(self):
        # CRITICAL: use scene["name"] not getattr/hasattr — InteractiveScene is dict-like
        self._person_objects = []
        for i in range(NUM_PERSONS):
            try:
                self._person_objects.append(self.scene[f"person_{i}"])
            except KeyError:
                break

        if not self._person_objects:
            return

        n = len(self._person_objects)
        positions, goals = [], []
        for i in range(n):
            angle = 2.0 * math.pi * i / n
            x = SPAWN_RADIUS * math.cos(angle)
            y = SPAWN_RADIUS * math.sin(angle)
            positions.append((x, y))
            goals.append((SPAWN_RADIUS * math.cos(angle + math.pi),
                          SPAWN_RADIUS * math.sin(angle + math.pi)))

        self._person_goals = goals
        self._rvo2_manager = RVO2CrowdManager(
            num_agents=n,
            sim_dt=self.cfg.sim.dt * self.cfg.decimation,
            radius=PERSON_RADIUS,
        )
        self._rvo2_manager.reset(positions, goals)

    def _get_robot_xy(self):
        # CRITICAL: subtract env_origin to get local coords for RVO2
        pos = self.scene["robot"].data.root_pos_w[0]
        origin = self.scene.env_origins[0]
        return float(pos[0] - origin[0]), float(pos[1] - origin[1])

    def _write_persons_to_sim(self):
        if self._rvo2_manager is None:
            return
        positions_2d = self._rvo2_manager.get_positions()
        # CRITICAL: account for env origin offset (world vs local frame)
        origin = self.scene.env_origins[0]
        ox, oy = float(origin[0]), float(origin[1])

        for i, person_obj in enumerate(self._person_objects):
            pose = person_obj.data.root_state_w[:, :7].clone()
            pose[:, 0] = float(positions_2d[i, 0]) + ox
            pose[:, 1] = float(positions_2d[i, 1]) + oy
            pose[:, 2] = PERSON_Z
            pose[:, 3] = 1.0   # qw
            pose[:, 4:7] = 0.0 # qx qy qz
            # CRITICAL: use write_root_pose_to_sim, not write_root_state_to_sim
            person_obj.write_root_pose_to_sim(pose)

    def _reset_idx(self, env_ids):
        # CRITICAL: hook into internal resets to re-init RVO2 positions
        super()._reset_idx(env_ids)
        self._setup_rvo2()
        self._write_persons_to_sim()

    def step(self, action):
        self._step_count += 1
        if self._rvo2_manager is not None:
            rx, ry = self._get_robot_xy()
            self._rvo2_manager.update_robot_obstacle((rx, ry), radius=0.5)
            self._rvo2_manager.step()
        # CRITICAL: call super() FIRST, write positions AFTER
        # super().step() may trigger internal resets (snapping persons to init_state)
        # writing after ensures RVO2 positions always win
        result = super().step(action)
        self._write_persons_to_sim()
        return result

    def reset(self, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self._setup_rvo2()
        self._write_persons_to_sim()
        return result
```

---

## 5. Environment & PPO Config

```python
@configclass
class RVO2NavigationEnvCfg(NavigationEnd2EndNoEncoderEnvCfg):
    scene: RVO2SceneCfg = RVO2SceneCfg(num_envs=1, env_spacing=10.0)

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
        self.scene.terrain.max_init_terrain_level = 0  # flat terrain for persons

@configclass
class RVO2NavigationEnvCfg_PLAY(RVO2NavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 300.0
        # Disable contact terminations so random/untrained policy doesn't reset every second
        self.terminations.base_contact = None
        self.terminations.base_contact_discrete_obstacles = None
        self.terminations.base_vel_out_of_limit = None

# Give RVO2 env its own experiment name so it never loads incompatible checkpoints
@configclass
class UnitreeGo2RVO2CrowdPPORunnerCfg_v0(UnitreeGo2NavigationEnd2EndNoEncoderEnvCfgPPORunnerCfg_v0):
    experiment_name = "unitree_go2_rvo2_crowd_v0"
```

---

## 6. Register in `__init__.py`

Use the custom env class as `entry_point`, not `ManagerBasedRLEnv`:

```python
gym.register(
    id="Isaac-Navigation-RVO2-Crowd-Unitree-Go2-v0",
    entry_point="...:RVO2NavigationEnv",   # custom subclass, not ManagerBasedRLEnv
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "...:RVO2NavigationEnvCfg",
        "rsl_rl_cfg_entry_point": "...:UnitreeGo2RVO2CrowdPPORunnerCfg_v0",
    },
)
gym.register(
    id="Isaac-Navigation-RVO2-Crowd-Unitree-Go2-Play-v0",
    entry_point="...:RVO2NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "...:RVO2NavigationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "...:UnitreeGo2RVO2CrowdPPORunnerCfg_v0",
    },
)
```

---

## 7. Run

```bash
source ~/env_isaac/bin/activate

# Play (visualise, no checkpoint needed)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Navigation-RVO2-Crowd-Unitree-Go2-Play-v0 \
  --num_envs 1 --headless

# Train
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Navigation-RVO2-Crowd-Unitree-Go2-v0 \
  --num_envs 1024 --headless
```

---

## Critical Gotchas (learn from these)

| Mistake | Correct approach |
|---------|-----------------|
| `hasattr(self.scene, "person_0")` | `self.scene["person_0"]` — `InteractiveScene` is dict-like |
| `write_root_state_to_sim(13-elem)` | `write_root_pose_to_sim(7-elem)` for kinematic bodies |
| RVO2 local coords written as world coords | Add `scene.env_origins[0]` offset to x/y when writing; subtract when reading robot |
| Write positions before `super().step()` | Write AFTER `super().step()` — internal resets inside super snap persons back to `init_state` |
| Override only `reset()` for re-init | Also override `_reset_idx()` — internal per-env resets bypass the outer `reset()` |
| PLAY config inherits contact terminations | Disable them in PLAY config — random policy falls immediately causing constant resets |
| Single `experiment_name` for all envs | Give each env variant its own name to avoid loading incompatible checkpoints |
| JIT `.pt` files picked by checkpoint loader | Filter out `*_jit.pt` in `get_checkpoint_path()`; add `--no_load` flag to `play.py` for fresh envs |
