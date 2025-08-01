from itertools import cycle
from typing import cast

import numpy as np

from arc_puzzle_generator.rule import RuleNode, OutOfGridRule, CollisionFillRule, backtrack_rule, Rule, \
    identity_rule, \
    DirectionRule
from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.geometry import Direction
from arc_puzzle_generator.utils.grid import unmask
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.neighbourhood import zero_neighbours, AxisNeighbourhood
from arc_puzzle_generator.topology import identity_topology, FixedGroupTopology
from arc_puzzle_generator.utils.entities import find_connected_objects


def puzzle_fourteen(input_grid: np.ndarray) -> Playground:
    """
    Puzzle 14: The cloud shooter puzzle.
    :param input_grid: A 2D numpy array representing the input grid.
    :return: A Model object containing the simulation setup for the cloud shooter puzzle.
    """

    background_color = input_grid[0, 0]
    start_color = 4
    fill_color = 3
    foreground_color = next(
        color for color in np.unique(input_grid).tolist() if color not in [background_color, start_color, fill_color]
    )

    target_mask = input_grid == start_color
    labeled_grid, bounding_box, num_objects = find_connected_objects(target_mask)

    direction: Direction

    if bounding_box[0, 0, 0] == (input_grid.shape[0] - 1):
        direction = "up"
    elif bounding_box[0, 0, 0] == 0:
        direction = "down"
    elif bounding_box[0, 0, 1] == 0:
        direction = "right"
    elif bounding_box[0, 0, 1] == (input_grid.shape[1] - 1):
        direction = "left"
    else:
        raise ValueError("Puzzle scenario is not supported")

    agents = [Agent(
        position=unmask(input_grid == foreground_color),
        direction="none",
        label="foreground",
        node=RuleNode(cast(Rule, identity_rule)),
        colors=cycle([foreground_color]),
        charge=0,
    ), Agent(
        unmask(labeled_grid),
        direction=direction,
        label="cloud_shooter",
        node=RuleNode(
            OutOfGridRule(grid_size=input_grid.shape),
            alternative_node=RuleNode(
                CollisionFillRule(fill_color=fill_color),
                next_node=RuleNode(cast(Rule, backtrack_rule)),
                alternative_node=RuleNode(
                    DirectionRule(direction_rule=identity_direction, select_direction=True),
                )
            )
        ),
        colors=cycle([
            start_color, background_color, start_color, background_color, fill_color, background_color
        ]),
        charge=input_grid.shape[0] if direction in ["up", "down"] else input_grid.shape[1],
    )]

    return Playground(
        output_grid=input_grid,
        agents=agents,
        neighbourhood=AxisNeighbourhood(
            grid_size=input_grid.shape,
            axis="vertical" if direction in ["up", "down"] else "horizontal"
        ),
        topology=FixedGroupTopology(group={"foreground"}),
    )
