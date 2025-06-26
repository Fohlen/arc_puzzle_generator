from itertools import chain, repeat
from typing import Optional, Iterator

import numpy as np

from arc_puzzle_generator.collisions import CollisionRule, CollisionResult
from arc_puzzle_generator.physics import Direction, DirectionRule


class FillColorRule(CollisionRule):
    def __init__(
            self,
            background_color: int,
            direction_rule: DirectionRule,
    ) -> None:
        self.background_color = background_color
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
            color = next(colors)

            return False, chain(repeat(color, len(neighbourhood) + 1), colors), self.direction_rule(
                direction), neighbourhood
        else:
            return None
