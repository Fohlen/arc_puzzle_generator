from itertools import cycle
from typing import Optional, Iterable

import numpy as np

from arc_puzzle_generator.collisions import CollisionResult, orthogonal_direction
from arc_puzzle_generator.physics import Direction, contained, line_axis


class BoundingBoxRule:
    """
    Uses a bounding box to determine collisions.
    """

    def __init__(self, bounding_box: np.ndarray) -> None:
        self.bounding_box = bounding_box

    def __call__(
            self,
            step: np.ndarray,
            neighbourhood: np.ndarray,
            colors: Iterable[int],
            direction: Direction,
            output_grid: np.ndarray,
    ) -> Optional[CollisionResult]:
        colliding_blocks = contained(neighbourhood, self.bounding_box)

        if np.any(colliding_blocks):
            block = neighbourhood[np.where(colliding_blocks)][0]
            current_color: int = output_grid[block[0], block[1]]
            axis = line_axis(neighbourhood[np.where(colliding_blocks)])

            return cycle([current_color]), orthogonal_direction(direction=direction, axis=axis), None
        else:
            return None
