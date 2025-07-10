from collections import defaultdict
from itertools import chain
from typing import Literal

import numpy as np

from abm.agent import Agent
from abm.neighbourhood import resolve_point_set_neighbours
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

    def step(self):
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
                position_intersect = selection.intersection(eligible_positions)

