from collections import defaultdict
from itertools import chain
from typing import cast

import numpy as np

from abm.agent import Agent
from abm.geometry import PointSet
from abm.neighbourhood import resolve_point_set_neighbours
from abm.physics import direction_to_unit_vector
from abm.selection import resolve_point_set_selectors


class Model:
    def __init__(
            self,
            output_grid: np.ndarray,
            agent_set: set[Agent]
    ):
        self.output_grid = output_grid
        self.agent_set = agent_set
        self.agents_by_label = defaultdict(list)
        self.labels = set(agent.label for agent in agent_set)
        self.steps = [output_grid.copy()]

        for agent in agent_set:
            self.agents_by_label[agent.label].append(agent)

    def step(self) -> None:
        for agent in self.agent_set:
            # Select active agents
            if agent.active():
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
                future_direction = agent.direction_rule(agent.direction, agent.position)
                future_step = agent.position + direction_to_unit_vector(future_direction)
                position_intersect = cast(PointSet, selection & eligible_positions & future_step)

                # Update the agent's state based on collision positions
                pos, _, colors, _ = agent.step(position_intersect)

                # Update the grid
                position = np.array(list(pos))
                self.output_grid[position[:, 0], position[:, 1]] = next(colors)
