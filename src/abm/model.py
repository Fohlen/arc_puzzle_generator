from collections import defaultdict
from itertools import chain
from typing import cast, Iterator, Iterable

import numpy as np

from abm.agent import Agent
from abm.geometry import PointSet
from abm.neighbourhood import resolve_point_set_neighbours
from abm.selection import resolve_point_set_selectors


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
                neighbourhood = resolve_point_set_neighbours(agent.position, agent.neighbourhood)

                # Select interesting positions based on the agent's selector
                selection = resolve_point_set_selectors(
                    agent.position,
                    neighbourhood,
                    agent.selector
                )

                # Filter eligible agents based on the agent's topology
                topology_labels = agent.topology(agent.label, self.labels)
                eligible_agents = set(chain.from_iterable(self.agents_by_label[label] for label in topology_labels))
                eligible_positions = set.union(*[agent.position for agent in eligible_agents])

                # Calculate the collision positions
                position_intersect = cast(PointSet, selection & eligible_positions)

                # Update the agent's state based on collision positions
                for step in agent.steps(position_intersect):
                    pos, _, colors, _ = step

                    # Update the grid
                    position = np.array(list(pos))
                    self.output_grid[position[:, 0], position[:, 1]] = next(colors)
                    self.steps.append(self.output_grid.copy())
