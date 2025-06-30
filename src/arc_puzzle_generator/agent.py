from typing import Iterable, Iterator, Optional

import numpy as np

from arc_puzzle_generator.collisions import CollisionRule, NeighbourhoodRule, \
    directional_neighbourhood, moore_neighbourhood, axis_neighbourhood
from arc_puzzle_generator.physics import Direction, starting_point, direction_to_unit_vector


class Agent(Iterator[np.ndarray], Iterable[np.ndarray]):
    """
    An agent is a generator that iterates over a grid and updates the grid according to the given direction and colors.

    Agents can be bounded by a charge or unbounded, in which case they will continue to run until they reach the end of the grid.
    A neighbourhood rule determines how the agent calculates collisions, and a collision rule determines how the agent reacts to collisions.

    :param output_grid: The grid to update.
    :param step: The initial coordinates of the agent.
    :param direction: The initial direction of the agent.
    :param colors: The colors of the agent.
    :param charge: The charge of the agent (-1 for unbounded).
    :param neighbourhood_rule: The neighbourhood rule of the agent.
    :param collision_rule: The collision rule of the agent.
    """

    def __init__(
            self,
            output_grid: np.ndarray,
            step: np.ndarray,
            direction: Direction,
            colors: Iterable[int],
            charge: int = -1,
            step_size: int = 1,
            neighbourhood_rule: NeighbourhoodRule = directional_neighbourhood,
            collision_rule: Optional[CollisionRule] = None,
    ) -> None:
        self.output_grid = output_grid
        self.direction = direction
        self.colors = iter(colors)
        self.charge = charge
        self.step = step
        self.step_size = step_size
        self.collision_rule = collision_rule
        self.neighbourhood_rule = neighbourhood_rule
        self.terminated = False

    def _neighbours(self) -> np.ndarray:
        if self.neighbourhood_rule == directional_neighbourhood:
            # calculate the neighborhood of the step dependent on the direction
            return directional_neighbourhood(self.step, self.direction)
        elif self.neighbourhood_rule == axis_neighbourhood:
            return axis_neighbourhood(self.step, self.direction, self.output_grid.shape)
        elif self.neighbourhood_rule == moore_neighbourhood:
            return moore_neighbourhood(self.step)
        else:
            return np.empty((0, 2))

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        while self.charge == -1 or self.charge > 0:
            # compute the next step
            step = self.step + direction_to_unit_vector(self.direction) * self.step_size

            # if the agent has previously been terminated
            if self.terminated:
                raise StopIteration

            # if the current step runs out of bounds, terminate the agent
            if (self.step[:, 0].min() < 0 or self.step[:, 0].max() >= self.output_grid.shape[0]
                    or self.step[:, 1].min() < 0 or self.step[:, 1].max() >= self.output_grid.shape[1]):
                raise StopIteration

            if self.collision_rule is not None:
                # calculate the neighbourhood
                neighbourhood = self._neighbours()

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
                    self.step = self.step + direction_to_unit_vector(self.direction) * self.step_size

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
