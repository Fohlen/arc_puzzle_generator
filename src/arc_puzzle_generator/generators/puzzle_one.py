from collections import OrderedDict
from typing import Iterable

import numpy as np

from arc_puzzle_generator.color_iterator import ColorIterator
from arc_puzzle_generator.entities import colour_count, find_connected_objects
from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.puzzle_generator import PuzzleGenerator
from arc_puzzle_generator.physics import Direction


class PuzzleOnePuzzleGenerator(PuzzleGenerator):
    def __init__(self, input_grid: np.ndarray):
        super().__init__(input_grid)

    def setup(self) -> Iterable[Agent]:
        sorted_colors = colour_count(self.input_grid)
        color_sequences: list[tuple[int, list[tuple[int, int]]]] = []  # [(row, [color, count]]
        background_color = sorted_colors[0][0]
        charge: int
        direction: Direction = "right"

        line_color = sorted_colors[1][0]

        separator_labels, separator_bboxes, separator_count = find_connected_objects(self.input_grid == line_color)
        # right-to-left
        if np.all(self.input_grid[:, :separator_bboxes[0][0, 1]] == background_color):
            direction = "left"
            input_grid = self.input_grid[:, (separator_bboxes[0][0, 1] + 1):]
            start_col = separator_bboxes[0][1, 1] - 1
            charge = separator_bboxes[0][3][1]
        # left-to-right
        else:
            input_grid = self.input_grid[:, :separator_bboxes[0][0, 1]]
            start_col = separator_bboxes[0][2, 1] + 1
            charge = self.input_grid.shape[1] - separator_bboxes[0][3][1] - 1

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

        return [Agent(
            output_grid=self.output_grid,
            step=np.array([[row, start_col]]),
            charge=charge,
            direction=direction,
            colors=ColorIterator(color_sequence, background_color),
        ) for row, color_sequence in color_sequences]
