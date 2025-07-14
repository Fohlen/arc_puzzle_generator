from collections import defaultdict
from itertools import chain
from typing import cast, Iterator, Iterable

import numpy as np

from abm.agent import Agent
from abm.geometry import PointSet
from abm.neighbourhood import directional_neighbours


class Model(Iterator[np.ndarray], Iterable[np.ndarray]):
    def __init__(
            self,
            output_grid: np.ndarray,
            agents: Iterable[Agent]
    ):
        self.output_grid = output_grid
        self.agents = agents
        self.agents_by_label = defaultdict(list)
        self.labels = set(agent.label for agent in agents)
        self.steps = [output_grid.copy()]
        self.step_iterator = iter(self.steps)

        for agent in agents:
            self.agents_by_label[agent.label].append(agent)

            if agent.active:
                position = np.array(list(agent.position))
                self.output_grid[position[:, 0], position[:, 1]] = next(agent.colors)

    @property
    def active(self) -> bool:
        """Check if any agent is active."""
        return any(agent.active for agent in self.agents)

    def __iter__(self) -> 'Model':
        return self

    def __next__(self) -> np.ndarray:
        return next(self.step_iterator)

    def step(self) -> None:
        for agent in self.agents:
            # Select active agents
            if agent.active:
                # Calculate the neighbourhood of the agent
                if agent.neighbourhood is directional_neighbours:
                    neighbourhood = directional_neighbours(agent.position, agent.direction)
                else:
                    neighbourhood = agent.neighbourhood(agent.position)

                # Filter eligible agents based on the agent's topology
                topology_labels = agent.topology(agent.label, self.labels)
                eligible_agents = set(chain.from_iterable(self.agents_by_label[label] for label in topology_labels))
                agent_position_mapping = {
                    point: agent.state
                    for agent in eligible_agents
                    for point in agent.position
                }
                eligible_positions = set(agent_position_mapping.keys())

                # Calculate the collision positions
                position_intersect = cast(PointSet, eligible_positions & neighbourhood)
                position_intersect_mapping = {
                    point: (agent_position_mapping[point], self.output_grid[point[0], point[1]].item())
                    for point in position_intersect
                }

                # Update the agent's state based on collision positions
                for step in agent.steps(position_intersect, position_intersect_mapping):
                    pos, _, colors, charge = step

                    # If the agent is still active after the step, update the output grid
                    if charge > 0 or charge == -1:
                        position = np.array(list(pos))
                        color = next(colors)
                        self.output_grid[position[:, 0], position[:, 1]] = color
                        self.steps.append(self.output_grid.copy())
