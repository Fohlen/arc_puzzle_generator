from typing import Iterable, Optional

import numpy as np

from arc_puzzle_generator.entities import find_colors, find_connected_objects, is_l_shape, is_point_adjacent
from arc_puzzle_generator.generators.generator import Generator
from arc_puzzle_generator.grid_utils import make_smallest_square_from_mask
from arc_puzzle_generator.physics import orientation_to_unit_vector, starting_point, orthogonal_orientation, Orientation


class PuzzleFourGenerator(Generator):
    def __init__(self, input_grid: np.ndarray):
        super().__init__(input_grid)
        self.l_shapes: list[tuple[int, np.ndarray, Orientation]] = []
        self.blocks: list[tuple[int, np.ndarray]] = []
        self.bboxes: np.ndarray = np.empty((4, 2))

    def setup(self) -> None:
        colors = find_colors(self.input_grid)

        self.l_shapes = []
        self.blocks = []

        for target_color in colors:
            target_mask = self.input_grid == target_color
            labeled_grid, bounding_box, num_objects = find_connected_objects(target_mask)

            for label in range(1, num_objects + 1):
                box = make_smallest_square_from_mask(self.output_grid, labeled_grid == label)

                if box is not None:
                    orientation = is_l_shape(box)
                    if orientation is not None:
                        self.l_shapes.append((target_color, bounding_box[(label - 1), :], orientation))
                    else:
                        self.blocks.append((target_color, bounding_box[(label - 1), :]))

        self.bboxes = np.array([bbox for _, bbox in self.blocks])

    def __iter__(self, *args, **kwargs) -> Iterable[np.ndarray]:
        for color, bbox, orientation in self.l_shapes:
            current_color = color
            step = orientation_to_unit_vector(orientation) + starting_point(bbox, orientation)

            while self.input_grid.shape[0] > step[0] > -1 < step[1] < self.input_grid.shape[1]:
                colliding_blocks = is_point_adjacent(step, self.bboxes)

                if colliding_blocks is not None:
                    # Determine if collision is horizontal by checking if the beam's x-coordinate
                    # is adjacent to the block's vertical sides
                    block_bbox = self.bboxes[colliding_blocks[0]]
                    is_horizontal = (np.min(block_bbox[:, 0]) <= step[0] <= np.max(block_bbox[:, 0]))

                    current_color = self.blocks[colliding_blocks[0]][0]
                    orientation = orthogonal_orientation(orientation, is_horizontal)

                self.output_grid[step[0], step[1]] = current_color
                step += orientation_to_unit_vector(orientation)
                yield self.output_grid.copy()
