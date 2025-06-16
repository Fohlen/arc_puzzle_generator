from collections import OrderedDict
from typing import Iterable

import numpy as np

from arc_puzzle_generator.entities import colour_count, find_connected_objects
from arc_puzzle_generator.generators.agent import Agent
from arc_puzzle_generator.generators.generator_new import GeneratorNew
from arc_puzzle_generator.physics import Direction


class PuzzleOneGeneratorNew(GeneratorNew):
    def __init__(self, input_grid: np.ndarray):
        super().__init__(input_grid)

    def setup(self) -> Iterable[Agent]:
        sorted_colors = colour_count(self.input_grid)
        color_sequences: list[tuple[int, list[int]]] = []  # [(row, [color, color, ...]]
        background_color = sorted_colors[0][0]
        start_col = 0
        charge: int = 0
        direction: Direction = "right"

        line_color = sorted_colors[1][0]

        separator_labels, separator_bboxes, separator_count = find_connected_objects(self.input_grid == line_color)
        # right-to-left
        if np.all(self.input_grid[:, :separator_bboxes[0][0, 1]] == background_color):
            direction = "left"
            input_grid = self.input_grid[:, (separator_bboxes[0][0, 1] + 1):]
            start_col = separator_bboxes[0][1, 1] - 1
            charge = separator_bboxes[0][3][1] - 1
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

            items = row if direction == "right" else row[::-1]
            for item in items:
                if item != background_color:

                    if item not in color_order:
                        color_order[item] = 0
                    color_order[item] += 1

            total_count = sum(color_order.values())
            color_sequence = [background_color] * total_count

            insert_items = color_order.items() if direction == "right" else reversed(color_order.items())
            for item in insert_items:
                for i in range(0, total_count, item[1]):
                    color_sequence[i] = item[0]

            color_sequences.append((index, color_sequence))

        return [Agent(
            output_grid=self.output_grid,
            bounding_box=np.array([[row, start_col], [row, start_col], [row, start_col], [row, start_col]]),
            charge=charge,
            direction=direction,
            colors=color_sequence,
        ) for row, color_sequence in color_sequences]
