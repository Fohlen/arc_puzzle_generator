import abc
from itertools import cycle
from typing import Iterable, Iterator

import numpy as np

from arc_puzzle_generator.physics import Direction, starting_point, direction_to_unit_vector


class Agent(Iterator[np.ndarray], Iterable[np.ndarray]):
    def __init__(
            self,
            output_grid: np.ndarray,
            bounding_box: np.ndarray,
            direction: Direction,
            colors: Iterable[int],
            charge: int = -1,
    ) -> None:
        self.output_grid = output_grid
        self.bounding_box = bounding_box
        self.direction = direction
        self.colors = iter(colors)
        self.charge = charge
        self.step = starting_point(bounding_box, direction)

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        if self.charge == -1:
            step = self.step + direction_to_unit_vector(self.direction)

            if step[0] < 0 or step[0] > self.output_grid.shape[0] or step[1] < 0 or step[1] > self.output_grid.shape[1]:
                raise StopIteration
            else:
                self.output_grid[self.step[0], self.step[1]] = next(self.colors)
                self.step = step
                return self.output_grid.copy()
        elif self.charge > 0:
            self.output_grid[self.step[0], self.step[1]] = next(self.colors)
            self.step = self.step + direction_to_unit_vector(self.direction)
            self.charge -= 1

            return self.output_grid.copy()
        else:
            raise StopIteration
