from typing import Iterable
from collections import OrderedDict

import numpy as np

from arc_puzzle_generator.entities import colour_count, find_connected_objects
from arc_puzzle_generator.generators.generator import Generator


class PuzzleOneGenerator(Generator):
    def __init__(self, input_grid: np.ndarray):
        super().__init__(input_grid)
        self.color_sequence: list[tuple[int, int, int]] = []  # [(row, color, counter)]
        self.left_to_right = True
        self.background_color = 0
        self.start_col = 0

    def setup(self) -> None:
        sorted_colors = colour_count(self.input_grid)
        self.background_color = sorted_colors[0][0]

        line_color = sorted_colors[1][0]

        separator_labels, separator_bboxes, separator_count = find_connected_objects(self.input_grid == line_color)
        # right-to-left
        if np.all(self.input_grid[:, :separator_bboxes[0][0, 1]] == self.background_color):
            self.left_to_right = False
            input_grid = self.input_grid[:, (separator_bboxes[0][0, 1] + 1):]
            self.start_col = separator_bboxes[0][1, 1] - 1
        # left-to-right
        else:
            input_grid = self.input_grid[:, :separator_bboxes[0][0, 1]]
            self.start_col = separator_bboxes[0][2, 1] + 1

        for index, row in enumerate(input_grid):
            color_order = OrderedDict()

            # empty row
            if np.all(row == self.background_color):
                continue

            items = row if self.left_to_right else row[::-1]
            for item in items:
                if item != self.background_color:
                    if item not in color_order:
                        color_order[item] = 0
                    color_order[item] += 1

            insert_items = reversed(color_order.items()) if self.left_to_right else color_order.items()
            for color, count in insert_items:
                self.color_sequence.append((index, color, count))

    def __iter__(self, *args, **kwargs) -> Iterable[np.ndarray]:
        for sequence in self.color_sequence:
            current_col = self.start_col
            end = self.input_grid.shape[1] if self.left_to_right else -1
            step = sequence[2] if self.left_to_right else -sequence[2]

            for col in range(current_col, end, step):
                if self.output_grid[sequence[0], col] == self.background_color:
                    self.output_grid[sequence[0], col] = sequence[1]

                yield self.output_grid.copy()
