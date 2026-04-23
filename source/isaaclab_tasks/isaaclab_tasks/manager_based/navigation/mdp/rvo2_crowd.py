"""RVO2-based crowd simulation manager for IsaacLab navigation environments."""

from __future__ import annotations
import numpy as np

try:
    import rvo2
except ImportError:
    raise ImportError(
        "Python-RVO2 is not installed. Install it from: "
        "https://github.com/sybrenstuvel/Python-RVO2"
    )


class RVO2CrowdManager:
    """Manages a crowd of simulated persons using the RVO2 algorithm.

    Each person is represented as a circular agent in 2D space. The manager
    handles collision avoidance between persons and can incorporate the robot
    as a dynamic obstacle.
    """

    def __init__(
        self,
        num_agents: int,
        sim_dt: float,
        neighbor_dist: float = 5.0,
        max_neighbors: int = 10,
        time_horizon: float = 5.0,
        time_horizon_obst: float = 5.0,
        radius: float = 0.3,
        max_speed: float = 1.4,
    ):
        """Initialize the RVO2 crowd manager.

        Args:
            num_agents: Number of person agents to simulate.
            sim_dt: Simulation timestep in seconds.
            neighbor_dist: Maximum distance to consider neighbors.
            max_neighbors: Maximum number of neighbors per agent.
            time_horizon: Time horizon for agent-agent collision avoidance.
            time_horizon_obst: Time horizon for agent-obstacle collision avoidance.
            radius: Radius of each person agent in meters.
            max_speed: Maximum speed of each person agent in m/s.
        """
        self.num_agents = num_agents
        self.sim_dt = sim_dt
        self.radius = radius
        self.max_speed = max_speed
        self._neighbor_dist = neighbor_dist
        self._max_neighbors = max_neighbors
        self._time_horizon = time_horizon
        self._time_horizon_obst = time_horizon_obst

        self._sim: rvo2.PyRVOSimulator | None = None
        self._goals: list[tuple[float, float]] = [(0.0, 0.0)] * num_agents
        self._robot_agent_id: int | None = None

    def reset(
        self,
        positions: list[tuple[float, float]],
        goals: list[tuple[float, float]],
    ) -> None:
        """Reset the simulation with new agent positions and goals.

        Args:
            positions: Initial (x, y) positions for each agent.
            goals: Goal (x, y) positions for each agent.
        """
        assert len(positions) == self.num_agents, f"Expected {self.num_agents} positions, got {len(positions)}"
        assert len(goals) == self.num_agents, f"Expected {self.num_agents} goals, got {len(goals)}"

        self._goals = list(goals)

        # Create new RVO2 simulator
        self._sim = rvo2.PyRVOSimulator(
            self.sim_dt,
            self._neighbor_dist,
            self._max_neighbors,
            self._time_horizon,
            self._time_horizon_obst,
            self.radius,
            self.max_speed,
        )

        # Add person agents
        for pos in positions:
            self._sim.addAgent(pos)

        self._robot_agent_id = None

    def step(self) -> None:
        """Advance the RVO2 simulation by one timestep.

        Sets preferred velocities toward goals, then runs the RVO2 step.
        """
        if self._sim is None:
            return

        num_rvo_agents = self._sim.getNumAgents()
        person_ids = range(self.num_agents)

        for i in person_ids:
            pos = self._sim.getAgentPosition(i)
            goal = self._goals[i]
            dx = goal[0] - pos[0]
            dy = goal[1] - pos[1]
            dist = np.sqrt(dx * dx + dy * dy)

            if dist < 0.1:
                # Close to goal — stop or pick new random direction
                self._sim.setAgentPrefVelocity(i, (0.0, 0.0))
            else:
                scale = min(self.max_speed, dist) / dist
                self._sim.setAgentPrefVelocity(i, (dx * scale, dy * scale))

        # Robot agent has zero preferred velocity (it's just an obstacle)
        if self._robot_agent_id is not None:
            self._sim.setAgentPrefVelocity(self._robot_agent_id, (0.0, 0.0))

        self._sim.doStep()

    def get_positions(self) -> np.ndarray:
        """Get current (x, y) positions of all person agents.

        Returns:
            Array of shape (num_agents, 2).
        """
        if self._sim is None:
            return np.zeros((self.num_agents, 2))
        return np.array([self._sim.getAgentPosition(i) for i in range(self.num_agents)])

    def get_velocities(self) -> np.ndarray:
        """Get current (vx, vy) velocities of all person agents.

        Returns:
            Array of shape (num_agents, 2).
        """
        if self._sim is None:
            return np.zeros((self.num_agents, 2))
        return np.array([self._sim.getAgentVelocity(i) for i in range(self.num_agents)])

    def set_goals(self, goals: list[tuple[float, float]]) -> None:
        """Update the goal positions for all agents.

        Args:
            goals: New goal (x, y) positions for each agent.
        """
        assert len(goals) == self.num_agents
        self._goals = list(goals)

    def update_robot_obstacle(self, position: tuple[float, float], radius: float | None = None) -> None:
        """Update or add the robot as a dynamic obstacle agent.

        The robot is added as an extra RVO2 agent with zero preferred velocity.
        Its position is updated each step so other agents avoid it.

        Args:
            position: Current (x, y) position of the robot.
            radius: Radius of the robot obstacle (uses agent radius if None).
        """
        if self._sim is None:
            return

        if self._robot_agent_id is None:
            # Add robot as a new agent
            r = radius if radius is not None else self.radius * 1.5  # slightly larger for safety
            self._robot_agent_id = self._sim.addAgent(
                position,
                self._neighbor_dist,
                self._max_neighbors,
                self._time_horizon,
                self._time_horizon_obst,
                r,
                0.0,  # max speed 0 — robot is "placed" not navigating via RVO2
                (0.0, 0.0),
            )
        else:
            self._sim.setAgentPosition(self._robot_agent_id, position)
