from itertools import permutations, cycle
from typing import Iterator

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.geometry import PointSet, Direction
from arc_puzzle_generator.neighbourhood import MooreNeighbourhood
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, OutOfGridRule, DirectionRule
from arc_puzzle_generator.utils.entities import colour_count, find_connected_objects, box_contained, extreme_point


def puzzle_thirty(
        input_grid: np.ndarray,
        directions: Iterator[Direction] = cycle(["left"])
) -> Playground:
    """
    Generates a playground for puzzle thirty based on the provided input grid.
    :param input_grid: The input grid representing the initial state of the puzzle.
    :param directions: The directions of the agents in the playground, defaulting to a cycle of "left".
    :return: A Playground instance configured for puzzle thirty.
    """

    sorted_colors = colour_count(input_grid)
    background_color = sorted_colors[0][0]

    # find all connected objects in the input grid
    # check which bboxes are nested
    # the outer box is the agent's starting position

    labels: list[np.ndarray] = []  # [mask]
    boxes: list[np.ndarray] = []  # [bbox]
    box_colors: list[int] = []  # [color]

    for target_color, _ in sorted_colors[1:]:
        label, bboxes, num_objects = find_connected_objects(
            mask=input_grid == target_color,
            neighbourhood=MooreNeighbourhood()
        )

        for i in range(1, num_objects + 1):
            labels.append(label == i)
            boxes.append(bboxes[i - 1, :])
            box_colors.append(target_color)

    target_boxes: list[tuple[int, int]] = []  # (outer_box, inner_box)

    for box1, box2 in permutations(range(len(boxes)), 2):
        if box_contained(boxes[box1], boxes[box2]):
            target_boxes.append((box2, box1))

    output_grid = input_grid.copy()
    agents = []

    # finally one can initialise the playground with agents, where the charge of each agent is the number of objects in the box
    for outer_box, inner_box in target_boxes:
        output_grid[labels[inner_box]] = background_color
        direction: Direction = next(directions)
        charge = (np.count_nonzero(labels[inner_box]) + 1)   # +1 for the extreme point (the agent itself)
        colors = [box_colors[outer_box]] + [box_colors[inner_box]] * charge

        agents.append(Agent(
            position=PointSet([extreme_point(labels[outer_box], direction)]),
            direction=direction,
            label=f"agent_{outer_box}",
            node=RuleNode(
                OutOfGridRule(grid_size=input_grid.shape),
                alternative_node=RuleNode(
                    DirectionRule(direction_rule=identity_direction)
                )
            ),
            colors=iter(colors),
            charge=charge,
        ))

    return Playground(
        output_grid=output_grid,
        agents=agents,
    )
