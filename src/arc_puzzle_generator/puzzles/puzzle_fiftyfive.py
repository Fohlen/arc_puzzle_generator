from itertools import cycle

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.neighbourhood import moore_neighbours
from arc_puzzle_generator.physics import direction_to_unit_vector
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, TrappedCollisionRule, DirectionRule
from arc_puzzle_generator.topology import all_topology
from arc_puzzle_generator.utils.entities import colour_count, find_connected_objects


def puzzle_fiftyfive(input_grid: np.ndarray) -> Playground:
    """
    Implementation of puzzle fiftyfive, a grid filling agent-based puzzle.
    :param input_grid: The input grid for the puzzle.
    :return: A Playground instance representing the puzzle.
    """

    foreground_color = 2

    sorted_colors = colour_count(input_grid)
    border_color = sorted_colors[1][0]

    agents: list[Agent] = []
    labels, bboxes, num_objects = find_connected_objects(input_grid == border_color, neighbourhood=moore_neighbours)
    for i in range(1, num_objects + 1):
        if not i in (3, 6):
            positions = PointSet.from_numpy(np.argwhere(labels == i))
            agents.append(Agent(
                position=positions.shift(direction_to_unit_vector("bottom_right")),
                direction="bottom_right",
                label=f"border_{i}",
                node=RuleNode(
                    TrappedCollisionRule(select_direction=True, direction_rule=identity_direction),
                    alternative_node=RuleNode(
                        DirectionRule(select_direction=True, direction_rule=identity_direction),
                    )
                ),
                colors=cycle([foreground_color]),
                charge=-1,
            ))

    return Playground(
        output_grid=input_grid,
        agents=agents,
        neighbourhood=moore_neighbours,
        topology=all_topology,
    )
