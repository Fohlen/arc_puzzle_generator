from itertools import cycle
from typing import Iterable, Iterator, Optional

import numpy as np

from arc_puzzle_generator.physics import Direction, starting_point, direction_to_unit_vector, orthogonal_direction, \
    collision_axis, is_point_adjacent


class Agent(Iterator[np.ndarray], Iterable[np.ndarray]):
    def __init__(
            self,
            output_grid: np.ndarray,
            bounding_box: np.ndarray,
            direction: Direction,
            colors: Iterable[int],
            charge: int = -1,
            beam_width: int = 1,
            collision_bounding_boxes: Optional[np.ndarray] = None,
    ) -> None:
        self.output_grid = output_grid
        self.bounding_box = bounding_box
        self.direction = direction
        self.colors = iter(colors)
        self.charge = charge
        self.step = starting_point(bounding_box, direction, point_width=beam_width)
        self.collision_bounding_boxes = collision_bounding_boxes

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        if self.charge == -1:
            # compute the next step
            step = self.step + direction_to_unit_vector(self.direction)

            # if the current step runs out of bounds, terminate the agent
            if (self.step[:, 0].min() < 0 or self.step[:, 0].max() >= self.output_grid.shape[0]
                    or self.step[:, 1].min() < 0 or self.step[:, 1].max() >= self.output_grid.shape[1]):
                raise StopIteration

            if self.collision_bounding_boxes is not None:
                colliding_blocks: Optional[np.ndarray] = is_point_adjacent(
                    self.step, self.collision_bounding_boxes
                )

                if colliding_blocks is not None:
                    block_bbox = self.collision_bounding_boxes[colliding_blocks[0]]
                    current_color = self.output_grid[block_bbox[0][0], block_bbox[0][1]]
                    # Determine if collision is horizontal by checking if the beam's x-coordinate
                    # is adjacent to the block's vertical sides
                    axis = collision_axis(self.step, block_bbox, self.direction)

                    self.colors = cycle([current_color])
                    self.output_grid[self.step[:, 0], self.step[:, 1]] = next(self.colors)
                    self.direction = orthogonal_direction(self.direction, axis)
                    self.step = self.step + direction_to_unit_vector(self.direction)

                    return self.output_grid.copy()

            # continue loop
            self.output_grid[self.step[:, 0], self.step[:, 1]] = next(self.colors)
            self.step = step
            return self.output_grid.copy()
        elif self.charge > 0:
            self.output_grid[self.step[:, 0], self.step[:, 1]] = next(self.colors)
            self.step = self.step + direction_to_unit_vector(self.direction)
            self.charge -= 1

            return self.output_grid.copy()
        else:
            raise StopIteration
