from collections import OrderedDict

import numpy as np

from abm.action import DirectionAction, ActionNode
from abm.agent import Agent
from abm.utils.color_sequence_iterator import ColorSequenceIterator
from abm.geometry import PointSet
from abm.model import Model
from abm.neighbourhood import zero_neighbours
from abm.physics import Direction
from abm.direction import identity_direction_rule
from abm.topology import identity_topology
from abm.utils.entities import colour_count, find_connected_objects


def puzzle_one(input_grid: np.ndarray) -> Model:
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
