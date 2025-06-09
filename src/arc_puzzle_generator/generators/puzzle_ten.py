from itertools import cycle
from typing import Iterable, Mapping

import numpy as np
from mypy.checkexpr import defaultdict

from arc_puzzle_generator.entities import find_connected_objects, colour_count, find_colors
from arc_puzzle_generator.generators.generator import Generator
from arc_puzzle_generator.physics import Direction, direction_to_unit_vector


class PuzzleTenGenerator(Generator):
    def __init__(self, input_grid: np.ndarray):
        super().__init__(input_grid)
        self.color_sequence: list[int] = []
        self.start_rows: list[int] = []
        self.border_color = 0
        self.inside_color = 0
        self.outside_color = 0

    def setup(self) -> None:
        start_col = self.input_grid[:, 0]
        start_col_colors = colour_count(start_col)
        grid_colors = [start_col_colors[0][0], start_col_colors[1][0]]

        for index, row in enumerate(start_col):
            if row == start_col_colors[-1][0]:
                self.start_rows.append(index)

        # this is primitive but works
        self.outside_color = self.input_grid[0, 0]
        self.inside_color = self.input_grid[self.start_rows[0], 1]

        color_boxes: Mapping[int, list[tuple[int, int]]] = defaultdict(list)  # { row: [(col, color)] }
        colors = find_colors(self.input_grid)

        for color in [color for color in colors if color not in grid_colors]:
            labels, bboxes, num_objects = find_connected_objects(self.input_grid == color)
            for bbox in bboxes.tolist():
                if bbox[0] != bbox[3]:
                    color_boxes[bbox[0][0]].append((bbox[0][1] + 1, color))
                    self.output_grid[bbox[1][0]:(bbox[3][0] + 1), bbox[0][1]:(bbox[3][1] + 1)] = self.outside_color

        sequence_row = min(color_boxes.keys())
        border_row = max(color_boxes.keys())
        self.color_sequence = [color for col, color in sorted(color_boxes[sequence_row], key=lambda x: x[0])]
        self.border_color = color_boxes[border_row][0][1]

    @staticmethod
    def snake_opposite(direction: Direction) -> Direction:
        """
        Returns the opposite direction of the given direction, moving in a snake pattern.
        :param direction: The input direction
        :return: the opposite direction
        """
        if direction == "right":
            return "up"

        return "right"

    def __iter__(self, *args, **kwargs) -> Iterable[np.ndarray]:
        for row in self.start_rows:
            color = cycle(self.color_sequence)
            direction: Direction = "right"

            step = (row, 0)
            lookahead = step + direction_to_unit_vector(direction)

            while step[0] > -1 and step[1] < self.output_grid.shape[1]:
                self.output_grid[step[0], step[1]] = next(color)

                if -1 < lookahead[0] < self.output_grid.shape[0] and lookahead[1] < self.output_grid.shape[1]:
                    if self.output_grid[lookahead[0], lookahead[1]] == self.inside_color:
                        step = lookahead
                    else:
                        opposite = self.snake_opposite(direction)
                        opposite_step = step + direction_to_unit_vector(opposite)

                        if self.output_grid[opposite_step[0], opposite_step[1]] == self.inside_color:
                            direction = opposite
                            step = opposite_step
                            self.output_grid[lookahead[0], lookahead[1]] = self.border_color
                        else:
                            self.output_grid[lookahead[0], lookahead[1]] = self.border_color

                            # the outer yield is never reached
                            yield self.output_grid.copy()

                            break

                    lookahead = step + direction_to_unit_vector(direction)
                else:
                    # move the step outside our output grid
                    step = step + direction_to_unit_vector(direction)

                yield self.output_grid.copy()
