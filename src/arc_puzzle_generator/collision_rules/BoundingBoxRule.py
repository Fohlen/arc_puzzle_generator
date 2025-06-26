from itertools import cycle, chain
from typing import Optional, Iterator

import numpy as np

from arc_puzzle_generator.collisions import CollisionResult, CollisionRule
from arc_puzzle_generator.physics import Direction, contained, line_axis, DirectionRule, direction_to_unit_vector


class BoundingBoxRule(CollisionRule):
    """
    Uses a bounding box to determine collisions.
    """

    def __init__(
            self,
            bounding_box: np.ndarray,
            direction_rule: DirectionRule,
            border_color: Optional[int] = None,
    ) -> None:
        self.bounding_box = bounding_box
        self.direction_rule = direction_rule
        self.border_color = border_color

    def __call__(
            self,
            step: np.ndarray,
            neighbourhood: np.ndarray,
            colors: Iterator[int],
            direction: Direction,
            output_grid: np.ndarray,
    ) -> Optional[CollisionResult]:
        colliding_blocks = contained(neighbourhood, self.bounding_box)

        if np.any(colliding_blocks):
            block = neighbourhood[np.where(colliding_blocks)][0]
            current_color: int = output_grid[block[0], block[1]]
            axis = line_axis(neighbourhood[np.where(colliding_blocks)])

            if self.border_color is not None:
                border_step = step + direction_to_unit_vector(direction)
                return False, chain([self.border_color], colors), self.direction_rule(direction, axis), border_step

            return False, cycle([current_color]), self.direction_rule(direction, axis), None
        else:
            return None
