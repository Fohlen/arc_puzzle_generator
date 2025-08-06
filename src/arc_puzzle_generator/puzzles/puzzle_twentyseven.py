from itertools import cycle
from typing import cast

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.neighbourhood import moore_neighbours
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import identity_rule, RuleNode, OutOfGridRule, TrappedCollisionRule, GravityRule, Rule, \
    AgentSpawnRule
from arc_puzzle_generator.topology import all_topology
from arc_puzzle_generator.utils.entities import find_connected_objects
from arc_puzzle_generator.utils.grid import unmask


def puzzle_twentyseven(input_grid: np.ndarray) -> Playground:
    """
    Generates a playground for puzzle 27 based on the input grid.
    Puzzle 27 is a simple puzzle where agents drop down, and new agents are spawned left and right of the existing agents.
    :param input_grid: The input grid representing the puzzle layout.
    :return: The generated playground for puzzle 27.
    """

    border_color = 2
    agent_color = 6

    agents: list[Agent] = []
    labels, bboxes, num_objects = find_connected_objects(input_grid == border_color)
    for i in range(1, num_objects + 1):
        agents.append(Agent(
            position=unmask(labels == i),
            direction="none",
            label="border",
            node=RuleNode(cast(Rule, identity_rule)),
            colors=iter([border_color]),
            charge=0,
        ))

    labels_agent, bboxes_agent, num_objects_agent = find_connected_objects(input_grid == agent_color)
    for i in range(1, num_objects_agent + 1):
        indices = np.argwhere(labels_agent == i)
        start = (indices[-1][0].item(), indices[-1][1].item())

        agents.append(Agent(
            position=PointSet([start]),
            direction="down",
            label="agent",
            node=RuleNode(
                OutOfGridRule(grid_size=input_grid.shape),
                alternative_node=RuleNode(
                    TrappedCollisionRule(select_direction=True, direction_rule=identity_direction),
                    next_node=RuleNode(
                        AgentSpawnRule(select_direction=True, directions=["left", "right"])
                    ),
                    alternative_node=RuleNode(
                        GravityRule(grid_size=input_grid.shape),
                    )
                )
            ),
            colors=cycle([agent_color]),
            charge=-1,
        ))

    return Playground(
        output_grid=input_grid,
        agents=agents,
        neighbourhood=moore_neighbours,
        topology=all_topology,
        collision_mode="history",
    )
