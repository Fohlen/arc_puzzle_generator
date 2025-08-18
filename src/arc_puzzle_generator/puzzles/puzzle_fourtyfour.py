from itertools import cycle
from typing import cast

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.neighbourhood import MooreNeighbourhood
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, OutOfGridRule, DirectionRule, CollisionFillRule, \
    collision_entity_color_rule, Rule, collision_direction_change
from arc_puzzle_generator.topology import all_topology
from arc_puzzle_generator.utils.entities import find_connected_objects, relative_box_direction, box_distance
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
    labels_blue, bbox_blue, num_blocks_blue = find_connected_objects(input_grid == 1)
    blocks: list[np.ndarray] = []

    for i in range(1, num_blocks_blue + 1):
        agents.append(Agent(
            position=unmask(labels_blue == i),
            direction="none",
            colors=iter([agent_color]),
            charge=0,
            label=f"ball_{i}",
        ))
        blocks.append(bbox_blue[i - 1])
        output_grid[labels_blue == i] = background_color

    labels_green, bbox_green, num_blocks_green = find_connected_objects(input_grid == 3)
    for i in range(1, num_blocks_green + 1):
        block_directions = [
            relative_box_direction(
                block,
                bbox_green[i - 1]
            )
            for block in blocks
        ]

        block_distances = [
            box_distance(
                bbox_green[i - 1],
                block,
                direction,
            )
            for block, direction in zip(blocks, block_directions)
        ]

        _, direction = min(zip(block_distances, block_directions), key=lambda x: x[0])
        agents.append(Agent(
            position=unmask(labels_green == i),
            direction=direction,
            colors=iter([agent_color]),
            charge=0,
            label=f"ball_top_{i}",
        ))
        output_grid[labels_green == i] = background_color

    agents.append(Agent(
        position=unmask(input_grid == agent_tip_color),
        direction="up",
        label="shooter",
        colors=cycle([agent_color]),
        node=RuleNode(
            OutOfGridRule(grid_size=input_grid.shape),
            alternative_node=RuleNode(
                cast(Rule, collision_entity_color_rule),
                next_node=RuleNode(
                    cast(Rule, collision_direction_change),
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
        neighbourhood=MooreNeighbourhood(),
        topology=all_topology,
        collision_mode="current",
    )
