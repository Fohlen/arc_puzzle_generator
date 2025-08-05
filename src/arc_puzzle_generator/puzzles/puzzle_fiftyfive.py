from itertools import cycle
from typing import Optional, cast

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.neighbourhood import moore_neighbours
from arc_puzzle_generator.physics import direction_to_unit_vector
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, TrappedCollisionRule, DirectionRule, Rule, identity_rule
from arc_puzzle_generator.topology import all_topology, FixedGroupTopology
from arc_puzzle_generator.utils.entities import colour_count, find_connected_objects


def puzzle_fiftyfive(
        input_grid: np.ndarray,
        excluded_lines: Optional[list[int]] = None,
) -> Playground:
    """
    Implementation of puzzle fiftyfive, a grid filling agent-based puzzle.
    :param input_grid: The input grid for the puzzle.
    :param excluded_lines: The index of lines that will not become agents.
    :return: A Playground instance representing the puzzle.
    """

    if excluded_lines is None:
        excluded_lines = [3, 6]

    foreground_color = 2

    sorted_colors = colour_count(input_grid)
    border_color = sorted_colors[1][0]

    agents: list[Agent] = []
    labels, bboxes, num_objects = find_connected_objects(input_grid == border_color, neighbourhood=moore_neighbours)
    for i in range(1, num_objects + 1):
        if i not in excluded_lines:
            positions = PointSet.from_numpy(np.argwhere(labels == i))
            agents.extend([
                Agent(
                    position=positions,
                    direction="none",
                    label="border",
                    node=RuleNode(cast(Rule, identity_rule)),
                    colors=cycle([foreground_color]),
                    charge=0,
                ),
                Agent(
                    position=positions.shift(direction_to_unit_vector("bottom_right")),
                    direction="bottom_right",
                    label=f"agent_{i}",
                    node=RuleNode(
                        TrappedCollisionRule(select_direction=True, direction_rule=identity_direction),
                        alternative_node=RuleNode(
                            DirectionRule(select_direction=True, direction_rule=identity_direction),
                        )
                    ),
                    colors=cycle([foreground_color]),
                    charge=-1,
                )
            ])

    return Playground(
        output_grid=input_grid,
        agents=agents,
        neighbourhood=moore_neighbours,
        topology=FixedGroupTopology({"border"}),
    )
