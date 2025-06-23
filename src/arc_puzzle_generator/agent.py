from typing import Iterable, Iterator, Optional

import numpy as np

from arc_puzzle_generator.collisions import directional_neighbourhood, CollisionRule
from arc_puzzle_generator.physics import Direction, starting_point, direction_to_unit_vector


class Agent(Iterator[np.ndarray], Iterable[np.ndarray]):
    def __init__(
            self,
            output_grid: np.ndarray,
            bounding_box: np.ndarray,
            direction: Direction,
            colors: Iterable[int],
            charge: int = -1,
            beam_width: int = 1,
            collision_rule: Optional[CollisionRule] = None,
    ) -> None:
        self.output_grid = output_grid
        self.bounding_box = bounding_box
        self.direction = direction
        self.colors = iter(colors)
        self.charge = charge
        self.step = starting_point(bounding_box, direction, point_width=beam_width)
        self.collision_rule = collision_rule
        self.terminated = False

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        while self.charge == -1 or self.charge > 0:
            # compute the next step
            step = self.step + direction_to_unit_vector(self.direction)

            # if the agent has previously been terminated
            if self.terminated:
                raise StopIteration

            # if the current step runs out of bounds, terminate the agent
            if (self.step[:, 0].min() < 0 or self.step[:, 0].max() >= self.output_grid.shape[0]
                    or self.step[:, 1].min() < 0 or self.step[:, 1].max() >= self.output_grid.shape[1]):
                raise StopIteration

            if self.collision_rule is not None:
                # calculate the neighborhood of the step dependant on the direction
                neighbourhood = directional_neighbourhood(self.step, self.direction)
                # remove neighbors which are out of grid
                neighbourhood = neighbourhood[
                    (neighbourhood[:, 0] > -1) &
                    (neighbourhood[:, 1] > -1) &
                    (neighbourhood[:, 0] < self.output_grid.shape[0]) &
                    (neighbourhood[:, 1] < self.output_grid.shape[1])
                    ]

                # mark possible collisions
                result = self.collision_rule(self.step, neighbourhood, self.colors, self.direction, self.output_grid)

                # if the collision rule found a collision, apply in order
                # 1) update colors
                # 2) execute extra steps, if any
                # 3) run current step
                # 4) update direction
                if result is not None:
                    terminated, colors, direction, extra_steps = result
                    self.terminated = terminated
                    self.colors = colors

                    if extra_steps is not None:
                        for extra_step in extra_steps:
                            self.output_grid[extra_step[0], extra_step[1]] = next(self.colors)

                    self.output_grid[self.step[:, 0], self.step[:, 1]] = next(self.colors)
                    self.direction = direction
                    self.step = self.step + direction_to_unit_vector(self.direction)

                    if self.charge > 0:
                        self.charge -= 1

                    return self.output_grid.copy()

            # continue loop
            self.output_grid[self.step[:, 0], self.step[:, 1]] = next(self.colors)
            self.step = step

            if self.charge > 0:
                self.charge -= 1

            return self.output_grid.copy()
        else:
            raise StopIteration
