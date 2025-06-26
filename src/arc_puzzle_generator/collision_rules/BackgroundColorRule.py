from itertools import chain
from typing import Optional, Iterable, Iterator

import numpy as np

from arc_puzzle_generator.collisions import CollisionResult, CollisionRule
from arc_puzzle_generator.physics import Direction, direction_to_unit_vector, DirectionRule


class BackgroundColorRule(CollisionRule):
    """
    Uses a given background color to determine collisions.
    If the neighbourhood contains colors other than the background color, a collision is detected and the rule is executed.
    """

    def __init__(self, background_color: int, direction_rule: DirectionRule,
                 border_color: Optional[int] = None) -> None:
        self.background_color = background_color
        self.border_color = border_color
        self.direction_rule = direction_rule

    def __call__(
            self,
            step: np.ndarray,
            neighbourhood: np.ndarray,
            colors: Iterator[int],
            direction: Direction,
            output_grid: np.ndarray,
    ) -> Optional[CollisionResult]:
        if np.any(output_grid[neighbourhood[:, 0], neighbourhood[:, 1]] != self.background_color):
            opposite: Direction = self.direction_rule(direction)
            opposite_step = step + direction_to_unit_vector(opposite)
            collision_border = step + direction_to_unit_vector(direction)

            if self.border_color is not None:
                if np.any(output_grid[opposite_step[:, 0], opposite_step[:, 1]] != self.background_color):
                    return True, chain([self.border_color], colors), opposite, collision_border
                else:

                    return False, chain([self.border_color], colors), opposite, collision_border
            else:
                if np.any(output_grid[opposite_step[:, 0], opposite_step[:, 1]] != self.background_color):
                    return True, colors, opposite, None  # type: ignore
                else:
                    return False, colors, opposite, None  # type: ignore
        else:
            return None
