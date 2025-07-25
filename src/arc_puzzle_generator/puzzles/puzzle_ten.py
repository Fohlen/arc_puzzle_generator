from collections import defaultdict
from itertools import cycle
from typing import Mapping, cast

import numpy as np

from arc_puzzle_generator.rule import OutOfGridRule, TrappedCollisionRule, CollisionDirectionRule, \
    DirectionRule, \
    CollisionBorderRule, backtrack_rule, Rule, RuleNode, identity_rule
from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction, snake_direction
from arc_puzzle_generator.geometry import PointSet, unmask
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.neighbourhood import zero_neighbours, von_neumann_neighbours
from arc_puzzle_generator.topology import FixedGroupTopology, identity_topology
from arc_puzzle_generator.utils.entities import colour_count, find_colors, find_connected_objects


def puzzle_ten(input_grid: np.ndarray) -> Playground:
    """
    Puzzle 10: The Snake
    :param input_grid: The input grid for the puzzle.
    :return: A Model object containing the simulation setup for the snake puzzle.
    """

    start_rows: list[int] = []
    start_col = input_grid[:, 0]
    start_col_colors = colour_count(start_col)
    grid_colors = [start_col_colors[0][0], start_col_colors[1][0]]

    for index, row in enumerate(start_col):
        if row == start_col_colors[-1][0]:
            start_rows.append(index)

    # this is primitive but works
    outside_color = input_grid[0, 0].item()
    inside_color: int = input_grid[start_rows[0], 1]  # type: ignore

    color_boxes: Mapping[int, list[tuple[int, int]]] = defaultdict(list)  # { row: [(col, color)] }
    colors = find_colors(input_grid)

    for color in [color for color in colors if color not in grid_colors]:
        labels, bboxes, num_objects = find_connected_objects(input_grid == color)
        for bbox in bboxes.tolist():
            if bbox[0] != bbox[3]:
                color_boxes[bbox[0][0]].append((bbox[0][1] + 1, color))
                input_grid[bbox[1][0]:(bbox[3][0] + 1), bbox[0][1]:(bbox[3][1] + 1)] = outside_color

    sequence_row = min(color_boxes.keys())
    border_row = max(color_boxes.keys())
    color_sequence = [color for col, color in sorted(color_boxes[sequence_row], key=lambda x: x[0])]
    border_color = color_boxes[border_row][0][1]

    topology = FixedGroupTopology(group={"foreground"})

    foreground_position = unmask(input_grid == outside_color)
    agents = [Agent(
        position=foreground_position,
        direction="none",
        label="foreground",
        node=RuleNode(cast(Rule, identity_rule)),
        colors=cycle([outside_color]),
        charge=0,
    )]

    node = RuleNode(
        OutOfGridRule(grid_size=(input_grid.shape[0], input_grid.shape[1])),
        alternative_node=RuleNode(
            CollisionBorderRule(
                border_color=border_color,
                direction_rule=snake_direction,
                select_direction=True
            ),
            next_node=RuleNode(
                cast(Rule, backtrack_rule),
            ),
            alternative_node=RuleNode(
                TrappedCollisionRule(direction_rule=snake_direction, select_direction=True),
                alternative_node=RuleNode(
                    CollisionDirectionRule(direction_rule=snake_direction, select_direction=True),
                    alternative_node=RuleNode(
                        DirectionRule(direction_rule=identity_direction, select_direction=True)
                    )
                )
            )
        )
    )

    agents += [Agent(
        position=PointSet([(row, 0)]),
        direction="right",
        label="snake",
        node=node,
        colors=cycle(color_sequence),
        charge=-1,
    ) for row in start_rows]

    return Playground(
        input_grid.copy(),
        agents=agents,
        neighbourhood=von_neumann_neighbours,
        topology=topology,
    )
