from itertools import chain
from typing import Optional, Iterable

import numpy as np

from arc_puzzle_generator.collisions import CollisionResult, snake_direction
from arc_puzzle_generator.physics import Direction, direction_to_unit_vector


class BackgroundColorRule:
    """
    Uses a given background color to determine collisions.
    If the neighbourhood contains colors other than the background color, a collision is detected and the rule is executed.
    """

    def __init__(self, background_color: int, border_color: int):
        self.background_color = background_color
        self.border_color = border_color

    def __call__(
            self,
            step: np.ndarray,
            neighbourhood: np.ndarray,
            colors: Iterable[int],
            direction: Direction,
            output_grid: np.ndarray,
    ) -> Optional[CollisionResult]:
        if np.any(output_grid[neighbourhood[:, 0], neighbourhood[:, 1]] != self.background_color):
            opposite = snake_direction(direction)
            opposite_step = step + direction_to_unit_vector(opposite)
            collision_border = step + direction_to_unit_vector(direction)

            if np.any(output_grid[opposite_step[:, 0], opposite_step[:, 1]] != self.background_color):
                return True, chain([self.border_color], colors), opposite, collision_border
            else:

                return False, chain([self.border_color], colors), opposite, collision_border
        else:
            return None
