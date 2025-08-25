from itertools import cycle

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.neighbourhood import von_neumann_neighbours
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, OutOfGridRule, TrappedCollisionRule, CollisionConditionDirectionRule
from arc_puzzle_generator.topology import identity_topology
from arc_puzzle_generator.utils.entities import colour_count, find_5x5_grids_with_border


def puzzle_ninety(input_grid: np.ndarray) -> Playground:
    """
    Puzzle ninety is a puzzle in which all red grids move to the right and all blue grids move left.
    :param input_grid: The input grid for the puzzle, represented as a 2D numpy array.
    :return: A Playground instance that simulates the puzzle.
    """

    sorted_colors = colour_count(input_grid)
    background_color = sorted_colors[0][0]

    node = RuleNode(
        OutOfGridRule(grid_size=input_grid.shape),
        alternative_node=RuleNode(
            TrappedCollisionRule(direction_rule=identity_direction, select_direction=True),
            alternative_node=RuleNode(
                CollisionConditionDirectionRule(
                    direction_rule=identity_direction,
                    conditions=[(False, "none")]
                ),
            )
        )
    )

    agents: list[Agent] = []
    bboxes_red = find_5x5_grids_with_border(input_grid, border_color=2)
    # sort red to the right by the bottom_right's Y coordinate in descending order
    for bbox in sorted(bboxes_red, key=lambda b: b[3][1], reverse=True):
        position = PointSet([
            (x, y)
            for x in range(bbox[1, 0], bbox[3, 0] + 1)
            for y in range(bbox[1, 1], bbox[3, 1] + 1)
        ])

        colors = input_grid[
            bbox[1, 0]:bbox[3, 0] + 1,
            bbox[1, 1]:bbox[3, 1] + 1,
        ]
        color_iterator = cycle([colors.flatten().tolist(), ])

        agents.append(Agent(
            position=position,
            direction="right",
            label="red",
            colors=color_iterator,
            node=node,
            charge=-1,
        ))

    bboxes_light_blue = find_5x5_grids_with_border(input_grid, border_color=8)
    # sort the light blue bboxes in ascending order of the bottom left's Y coordinate
    for bbox in sorted(bboxes_light_blue, key=lambda b: b[0][1], reverse=False):
        position = PointSet([
            (x, y)
            for x in range(bbox[1, 0], bbox[3, 0] + 1)
            for y in range(bbox[1, 1], bbox[3, 1] + 1)
        ])

        colors = input_grid[
            bbox[1, 0]:bbox[3, 0] + 1,
            bbox[1, 1]:bbox[3, 1] + 1,
        ]
        color_iterator = cycle([colors.flatten().tolist(), ])

        agents.append(Agent(
            position=position,
            direction="left",
            label="light_blue",
            colors=color_iterator,
            node=node,
            charge=-1,
        ))

    return Playground(
        output_grid=input_grid,
        agents=agents,
        backfill_color=background_color,
        topology=identity_topology,
        neighbourhood=von_neumann_neighbours,
    )
