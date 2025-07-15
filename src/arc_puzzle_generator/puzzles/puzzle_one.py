from collections import OrderedDict

import numpy as np

from arc_puzzle_generator.action import DirectionAction, ActionNode
from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.utils.color_sequence_iterator import ColorSequenceIterator
from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.model import Model
from arc_puzzle_generator.neighbourhood import zero_neighbours
from arc_puzzle_generator.physics import Direction
from arc_puzzle_generator.direction import identity_direction_rule
from arc_puzzle_generator.topology import identity_topology
from arc_puzzle_generator.utils.entities import colour_count, find_connected_objects


def puzzle_one(input_grid: np.ndarray) -> Model:
    """
    The color iteration puzzle.
    :param input_grid: The input grid for the puzzle.
    :return: A Model object containing the simulation setup for the color iteration puzzle.
    """
    output_grid = input_grid.copy()
    sorted_colors = colour_count(input_grid)
    color_sequences: list[tuple[int, list[tuple[int, int]]]] = []  # [(row, [color, count]]
    background_color = sorted_colors[0][0]
    charge: int
    direction: Direction = "right"

    line_color = sorted_colors[1][0]

    separator_labels, separator_bboxes, separator_count = find_connected_objects(input_grid == line_color)
    # right-to-left
    if np.all(input_grid[:, :separator_bboxes[0][0, 1]] == background_color):
        direction = "left"
        input_grid = input_grid[:, (separator_bboxes[0][0, 1] + 1):]
        start_col = (separator_bboxes[0][1, 1] - 1).item()
        charge = separator_bboxes[0][3][1].item()
    # left-to-right
    else:
        input_grid = input_grid[:, :separator_bboxes[0][0, 1]]
        start_col = (separator_bboxes[0][2, 1] + 1).item()
        charge = (output_grid.shape[1] - separator_bboxes[0][3][1] - 1).item()

    for index, row in enumerate(input_grid):
        color_order = OrderedDict()  # { color: count }

        # empty row
        if np.all(row == background_color):
            continue

        colors = row if direction == "right" else row[::-1]
        for color in colors:
            if color != background_color:

                if color not in color_order:
                    color_order[color] = 0
                color_order[color] += 1

        color_sequence = color_order.items() if direction == "left" else reversed(color_order.items())
        color_sequences.append((index, list(color_sequence)))

    return Model(
        output_grid=output_grid,
        agents=[Agent(
            position=PointSet([(row, start_col)]),
            direction=direction,
            label="puzzle_one_agent",
            topology=identity_topology,
            neighbourhood=zero_neighbours,
            node=ActionNode(DirectionAction(identity_direction_rule)),
            charge=charge,
            colors=ColorSequenceIterator(color_sequence, background_color),
        ) for row, color_sequence in color_sequences]
    )
