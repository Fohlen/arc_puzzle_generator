from itertools import cycle
from typing import Iterable

import numpy as np

from arc_puzzle_generator.collision_rules.BoundingBoxRule import BoundingBoxRule
from arc_puzzle_generator.collisions import orthogonal_direction
from abm.entities import find_colors, find_connected_objects, is_l_shape
from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.puzzle_generator import PuzzleGenerator
from arc_puzzle_generator.grid_utils import make_smallest_square_from_mask
from arc_puzzle_generator.physics import Direction, starting_point


class PuzzleFourPuzzleGenerator(PuzzleGenerator):
    def setup(self) -> Iterable[Agent]:
        colors = find_colors(self.input_grid, background=0)
        l_shapes: list[tuple[int, np.ndarray, Direction]] = []
        blocks: list[tuple[int, np.ndarray]] = []

        for target_color in colors:
            target_mask = self.input_grid == target_color
            labeled_grid, bounding_box, num_objects = find_connected_objects(target_mask)

            for label in range(1, num_objects + 1):
                box = make_smallest_square_from_mask(self.output_grid, labeled_grid == label)

                if box is not None:
                    direction = is_l_shape(box)
                    if direction is not None:
                        l_shapes.append((target_color, bounding_box[(label - 1), :], direction))
                    else:
                        blocks.append((target_color, bounding_box[(label - 1), :]))

        bboxes = np.array([bbox for _, bbox in blocks])

        return [Agent(
            output_grid=self.output_grid,
            step=starting_point(
                bounding_box=bbox,
                direction=direction,
                point_width=1
            ),
            direction=direction,
            colors=cycle([color]),
            charge=-1,
            collision_rule=BoundingBoxRule(bboxes, direction_rule=orthogonal_direction)
        ) for color, bbox, direction in l_shapes]
