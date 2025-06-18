from itertools import cycle
from typing import Iterable, Iterator, Optional

import numpy as np

from arc_puzzle_generator.collisions import collision_neighbourhood, orthogonal_direction
from arc_puzzle_generator.physics import Direction, starting_point, direction_to_unit_vector, line_axis


class Agent(Iterator[np.ndarray], Iterable[np.ndarray]):
    def __init__(
            self,
            output_grid: np.ndarray,
            bounding_box: np.ndarray,
            direction: Direction,
            colors: Iterable[int],
            charge: int = -1,
            beam_width: int = 1,
            background_color: Optional[int] = None,
    ) -> None:
        self.output_grid = output_grid
        self.bounding_box = bounding_box
        self.direction = direction
        self.colors = iter(colors)
        self.charge = charge
        self.step = starting_point(bounding_box, direction, point_width=beam_width)
        self.background_color = background_color

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

            if self.background_color is not None:
                # calculate the neighbourhood of the step dependant on the direction
                neighbourhood = collision_neighbourhood(self.step, self.direction)
                # remove collisions which are out of grid
                neighbourhood = neighbourhood[
                    (neighbourhood[:, 0] < self.output_grid.shape[0]) &
                    (neighbourhood[:, 1] < self.output_grid.shape[1])
                    ]
                # mark possible collisions
                colliding_blocks = self.output_grid[neighbourhood[:, 0], neighbourhood[:, 1]] != self.background_color

                if np.any(colliding_blocks):
                    block = neighbourhood[np.where(colliding_blocks)][0]
                    current_color = self.output_grid[block[0], block[1]]
                    axis = line_axis(neighbourhood[np.where(colliding_blocks)])

                    self.colors = cycle([current_color])
                    self.output_grid[self.step[:, 0], self.step[:, 1]] = next(self.colors)
                    self.direction = orthogonal_direction(direction=self.direction, axis=axis)
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
