from itertools import cycle
from typing import cast

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.neighbourhood import MooreNeighbourhood, moore_neighbours
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, OutOfGridRule, DirectionRule, Rule, collision_entity_redirect_rule, \
    resize_entity_to_exit_rule
from arc_puzzle_generator.topology import all_topology
from arc_puzzle_generator.utils.entities import find_connected_objects, relative_box_direction, mask_to_bbox
from arc_puzzle_generator.utils.grid import unmask


def puzzle_fourtyfour(input_grid: np.ndarray) -> Playground:
    """
    Implements the 44th puzzle in the ARC dataset.
    In this puzzle a shooter follows a path of balloons.
    :param input_grid: The input grid for the puzzle.
    :return: A Playground object representing the puzzle.
    """

    output_grid = input_grid.copy()
    agent_tip_color = 4
    agent_color = 2
    background_color = 8

    agents: list[Agent] = []
    blue_grid = (input_grid == 1)
    green_grid = (input_grid == 3)

    labels, bbox, num_objects = find_connected_objects(blue_grid | green_grid, neighbourhood=moore_neighbours)
    for i in range(1, num_objects + 1):
        labels_green = (labels == i) & green_grid
        labels_blue = (labels == i) & blue_grid
        bbox_green = mask_to_bbox(labels_green)
        bbox_blue = mask_to_bbox(labels_blue)

        direction = relative_box_direction(
            bbox_blue, bbox_green,
        )

        colors = input_grid[labels == i]

        color_iterator = iter([colors.flatten().tolist(), ])

        agents.append(Agent(
            position=unmask(labels == i),
            direction=direction,
            colors=color_iterator,
            charge=0,
            label=f"balloon_{i}",
        ))
        output_grid[labels == i] = background_color

    agents.append(Agent(
        position=unmask(input_grid == agent_tip_color),
        direction="up",
        label="shooter",
        colors=cycle([agent_color]),
        node=RuleNode(
            OutOfGridRule(grid_size=input_grid.shape),
            alternative_node=RuleNode(
                cast(Rule, collision_entity_redirect_rule),
                next_node=RuleNode(
                    cast(Rule, resize_entity_to_exit_rule)
                ),
                alternative_node=RuleNode(
                    DirectionRule(direction_rule=identity_direction, select_direction=True)
                )
            )
        ),
        charge=-1
    ))

    return Playground(
        output_grid=output_grid,
        agents=agents,
        neighbourhood=moore_neighbours,
        topology=all_topology,
        collision_mode="current",
    )
