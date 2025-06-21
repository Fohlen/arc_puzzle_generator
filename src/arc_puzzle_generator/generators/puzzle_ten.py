from collections import defaultdict
from itertools import cycle
from typing import Iterable, Mapping

import numpy as np

from arc_puzzle_generator.collision_rules.BackgroundColorRule import BackgroundColorRule
from arc_puzzle_generator.collisions import snake_direction
from arc_puzzle_generator.entities import colour_count, find_colors, find_connected_objects
from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.puzzle_generator import PuzzleGenerator


class PuzzleTenPuzzleGenerator(PuzzleGenerator):
    def setup(self) -> Iterable[Agent]:
        start_rows: list[int] = []
        start_col = self.input_grid[:, 0]
        start_col_colors = colour_count(start_col)
        grid_colors = [start_col_colors[0][0], start_col_colors[1][0]]

        for index, row in enumerate(start_col):
            if row == start_col_colors[-1][0]:
                start_rows.append(index)

        # this is primitive but works
        outside_color = self.input_grid[0, 0]
        inside_color: int = self.input_grid[start_rows[0], 1]  # type: ignore

        color_boxes: Mapping[int, list[tuple[int, int]]] = defaultdict(list)  # { row: [(col, color)] }
        colors = find_colors(self.input_grid)

        for color in [color for color in colors if color not in grid_colors]:
            labels, bboxes, num_objects = find_connected_objects(self.input_grid == color)
            for bbox in bboxes.tolist():
                if bbox[0] != bbox[3]:
                    color_boxes[bbox[0][0]].append((bbox[0][1] + 1, color))
                    self.output_grid[bbox[1][0]:(bbox[3][0] + 1), bbox[0][1]:(bbox[3][1] + 1)] = outside_color

        sequence_row = min(color_boxes.keys())
        border_row = max(color_boxes.keys())
        color_sequence = [color for col, color in sorted(color_boxes[sequence_row], key=lambda x: x[0])]
        border_color = color_boxes[border_row][0][1]

        return [
            Agent(
                output_grid=self.output_grid,
                bounding_box=np.array([[row, 0], [row, 0], [row, 0], [row, 0]]),
                direction="right",
                colors=cycle(color_sequence),
                charge=-1,
                collision_rule=BackgroundColorRule(
                    background_color=inside_color,
                    direction_rule=snake_direction,
                    border_color=border_color,
                )
            )
            for row in start_rows
        ]
