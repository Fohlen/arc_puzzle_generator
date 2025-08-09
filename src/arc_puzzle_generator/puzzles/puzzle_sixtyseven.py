from itertools import cycle
from typing import Iterator

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.geometry import Direction
from arc_puzzle_generator.neighbourhood import MooreNeighbourhood
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, TrappedCollisionRule, DirectionRule
from arc_puzzle_generator.topology import all_topology
from arc_puzzle_generator.utils.entities import colour_count, find_connected_objects
from arc_puzzle_generator.utils.grid import unmask


def puzzle_sixtyseven(
        input_grid: np.ndarray,
        directions: Iterator[Direction] = cycle(["left"])
) -> Playground:
    """
    Generates a playground for puzzle sixty-seven based on the provided input grid.
    Puzzle 67 lets a bunch of agents move inside of boxes and the direction is based on an instruction column.
    The instructions are supplied as a separate argument to this generator function.
    :param input_grid: The input grid representing the initial state of the puzzle.
    :param directions: Sequence of directions for the agents
    :return: A Playground instance configured for puzzle sixty-seven.
    """

    sorted_colors = colour_count(input_grid)
    box_colors = [color for color, count in sorted_colors[1:] if count >= 8]

    agents: list[Agent] = []
    for box_color in box_colors:
        box_labels, box_bboxes, num_boxes = find_connected_objects(input_grid == box_color, neighbourhood=MooreNeighbourhood())

        for i in range(1, num_boxes + 1):
            row_min, row_max = box_bboxes[i - 1, 1, 0], box_bboxes[i - 1, 3, 0]
            col_min, col_max = box_bboxes[i - 1, 1, 1], box_bboxes[i - 1, 3, 1]

            # we check if two sides of the box are filled with the same color, which in puzzle 67 means that the box is closed
            if np.all(box_labels[row_min:row_max + 1, col_min]) and np.all(box_labels[row_max, col_min:col_max + 1]):
                pass  # NOTE: In theory one can determine the direction of agents within the box but this is better left for the ARC solver
            else:
                agents.append(Agent(
                    position=unmask(box_labels == i),
                    direction="none",
                    label=f"box_{box_color}_{i}",
                    colors=cycle([box_color]),
                ))

    agent_color = 4
    agent_labels, agent_bboxes, num_agents = find_connected_objects(input_grid == agent_color)

    for i, direction in zip(range(1, num_agents + 1), directions):
        agents.append(Agent(
            position=unmask(agent_labels == i),
            label=f"agent_{agent_color}_{i}",
            direction=direction,
            colors=cycle([agent_color]),
            node=RuleNode(
                TrappedCollisionRule(direction_rule=identity_direction, select_direction=True),
                alternative_node=RuleNode(
                    DirectionRule(direction_rule=identity_direction, select_direction=True),
                )
            ),
            charge=-1,
        ))


    return Playground(
        output_grid=input_grid.copy(),
        agents=agents,
        neighbourhood=MooreNeighbourhood(),
        topology=all_topology,
        backfill_color=0,
    )
