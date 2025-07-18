from itertools import cycle
from typing import cast

import numpy as np

from arc_puzzle_generator.rule import OutOfGridRule, CollisionDirectionRule, DirectionRule, \
    collision_color_mapping_rule, Rule, \
    RuleNode, identity_rule
from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction, orthogonal_direction
from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.model import Model
from arc_puzzle_generator.neighbourhood import zero_neighbours, moore_neighbours
from arc_puzzle_generator.physics import Direction, starting_point
from arc_puzzle_generator.topology import FixedGroupTopology, identity_topology
from arc_puzzle_generator.utils.entities import find_colors, find_connected_objects, is_l_shape
from arc_puzzle_generator.utils.grid import make_smallest_square_from_mask


def puzzle_four(input_grid: np.ndarray) -> Model:
    """
    The laser shooter puzzle.
    :param input_grid: The input grid for the puzzle.
    :return: A Model object containing the simulation setup for the laser shooter puzzle.
    """

    colors = find_colors(input_grid, background=0)
    l_shapes: list[tuple[int, np.ndarray, Direction]] = []
    blocks: list[tuple[int, np.ndarray]] = []

    for target_color in colors:
        target_mask = input_grid == target_color
        labeled_grid, bounding_box, num_objects = find_connected_objects(target_mask)

        for label in range(1, num_objects + 1):
            box = make_smallest_square_from_mask(input_grid, labeled_grid == label)

            if box is not None:
                direction = is_l_shape(box)
                if direction is not None:
                    l_shapes.append((target_color, bounding_box[(label - 1), :], direction))
                else:
                    blocks.append((target_color, bounding_box[(label - 1), :]))

    agents = [Agent(
        position=PointSet.from_numpy(bbox),
        direction="none",
        label="bbox",
        node=RuleNode(cast(Rule, identity_rule)),
        colors=cycle([target_color]),
        charge=0
    ) for target_color, bbox in blocks]

    node = RuleNode(
        OutOfGridRule(grid_size=(input_grid.shape[0], input_grid.shape[1])),
        alternative_node=RuleNode(
            cast(Rule, collision_color_mapping_rule),
            next_node=RuleNode(
                CollisionDirectionRule(orthogonal_direction),
            ),
            alternative_node=RuleNode(
                DirectionRule(identity_direction),
            )
        )
    )

    agents += [Agent(
        position=PointSet.from_numpy(
            starting_point(
                bounding_box=bbox,
                direction=direction,
                point_width=1
            )
        ),
        direction=direction,
        label="puzzle_four_agent",
        node=node,
        colors=cycle([color]),
        charge=-1,
    ) for color, bbox, direction in l_shapes]

    return Model(
        output_grid=input_grid.copy(),
        agents=agents,
        neighbourhood=moore_neighbours,
        topology=FixedGroupTopology(group={"bbox"})
    )
