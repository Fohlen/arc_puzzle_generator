from collections import defaultdict
from itertools import chain
from typing import cast, Iterator, Iterable, Callable, Literal, Sequence

import numpy as np

import logging

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.geometry import PointSet, Point
from arc_puzzle_generator.neighbourhood import resolve_point_set_neighbourhood, Neighbourhood, zero_neighbours
from arc_puzzle_generator.state import AgentState
from arc_puzzle_generator.topology import Topology, identity_topology

logger = logging.getLogger(__name__)

ExecutionMode = Literal["sequential", "parallel"]
CollisionMode = Literal["current", "history"]


class Model(Iterator[np.ndarray], Iterable[np.ndarray]):
    def __init__(
            self,
            output_grid: np.ndarray,
            agents: Sequence[Agent],
            neighbourhood: Neighbourhood = zero_neighbours,
            topology: Topology = identity_topology,
            execution_mode: ExecutionMode = "parallel",
            collision_mode: CollisionMode = "current",
    ):
        self.output_grid = output_grid
        self.agents = agents
        self.neighbourhood = neighbourhood
        self.topology = topology
        self.execution_mode = execution_mode
        self.collision_mode = collision_mode
        self.current_agent_idx = 0
        self.agents_by_label = defaultdict(list)
        self.labels = set(agent.label for agent in agents)
        self.steps = [output_grid.copy()]
        self.step_iterator = iter(self.steps)

        for agent in agents:
            self.agents_by_label[agent.label].append(agent)

            if agent.active:
                position = np.array(list(agent.position))
                self.output_grid[position[:, 0], position[:, 1]] = agent.color

        # Note: It is not clear if this is the optimal way to handle the step iterator.
        while self.active:
            self.step()

    @property
    def active(self) -> bool:
        """Check if any agent is active."""
        return any(agent.active for agent in self.agents)

    def __iter__(self) -> 'Model':
        return self

    def __next__(self) -> np.ndarray:
        return next(self.step_iterator)

    def _process_agent(self, agent: Agent) -> None:
        # calculate the neighbourhood for the agent's position
        neighbourhood = resolve_point_set_neighbourhood(agent.position, self.neighbourhood)

        # determine the eligible agents based on the agent's label and topology
        topology_labels = self.topology(agent.label, self.labels)
        eligible_agents = set(chain.from_iterable(self.agents_by_label[label] for label in topology_labels))

        # create a mapping of agent positions to their states
        agent_position_mapping = {}
        for eligible_agent in eligible_agents:
            for point in eligible_agent.position:
                agent_position_mapping[point] = eligible_agent.state

                if self.collision_mode == "history":
                    for history in eligible_agent.history:
                        for history_point in history.position:
                            agent_position_mapping[history_point] = history

        # determine possible collisions and their states
        eligible_positions = set(agent_position_mapping.keys())
        position_intersect = cast(PointSet, eligible_positions & neighbourhood)
        position_intersect_mapping = {
            point: AgentState(
                position=agent_position_mapping[point].position,
                direction=agent_position_mapping[point].direction,
                color=self.output_grid[point[0], point[1]].item(),
                charge=agent_position_mapping[point].charge
            )
            for point in position_intersect
        }

        for step in agent.steps(position_intersect, position_intersect_mapping):
            pos, _, color, charge = step
            if charge > 0 or charge == -1:
                position = np.array(list(pos))
                logger.debug("Position: %s, Color: %s", pos, color)
                self.output_grid[position[:, 0], position[:, 1]] = color
                self.steps.append(self.output_grid.copy())

    def step(self) -> None:
        if self.execution_mode == "sequential":
            # Sequential: process one agent at a time until it is inactive
            while self.current_agent_idx < len(self.agents):
                agent = self.agents[self.current_agent_idx]
                if agent.active:
                    self._process_agent(agent)
                    if not agent.active:
                        self.current_agent_idx += 1
                    break  # Only process one agent per step
                else:
                    self.current_agent_idx += 1
        elif self.execution_mode == "parallel":
            # Parallel: process all active agents in one step
            for agent in self.agents:
                if agent.active:
                    self._process_agent(agent)
        else:
            raise NotImplementedError


ModelSetup = Callable[[np.ndarray], Model]
