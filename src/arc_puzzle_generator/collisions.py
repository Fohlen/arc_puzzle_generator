from typing import Optional, Iterator, Protocol

import numpy as np

from arc_puzzle_generator.physics import Direction


class NeighbourhoodRule(Protocol):
    def __call__(self, point: np.ndarray, direction: Direction, *args, **kwargs) -> np.ndarray:
        """
        A collision rule regulates which neighbours will be considered in a collision.
        :param point: The point to determine the neighbourhood for.
        :param direction: The direction to determine the neighbourhood for.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        :return: A 2D array of neighbourhood coordinates.
        """
        pass


def moore_neighbourhood(point: np.ndarray, *arg, **kwargs) -> np.ndarray:
    """
    Determines the neighborhood of a point using the Moore neighborhood.
    :param point: A point to determine the neighborhood for.
    :return: The moore neighborhood of the point.
    """

    if point.ndim != 1:
        return np.concat([moore_neighbourhood(p) for p in point])

    x, y = point

    return np.array([
        (x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
        (x, y - 1), (x, y + 1),
        (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)
    ])


class AxisNeighbourHood(NeighbourhoodRule):
    def __init__(self, grid_size: tuple[int, int]) -> None:
        self.grid_size = grid_size

    def __call__(self, point: np.ndarray, direction: Direction, *args, **kwargs) -> np.ndarray:
        points = set(map(tuple, point.tolist()))
        neighbours = set()

        if direction in ["up", "down"]:
            xs = point[:, 0].tolist()

            for x in xs:
                neighbours.update([(x, i) for i in range(0, self.grid_size[0])])
        else:
            ys = point[:, 1].tolist()

            for y in ys:
                neighbours.update([(i, y) for i in range(0, self.grid_size[1])])

        return np.array(sorted(neighbours - points))


CollisionResult = tuple[bool, Iterator[int], Direction, Optional[np.ndarray]]
"""
A tuple containing:
- A boolean indicating whether the collision terminates the agent.
- A color iterable to replace the current color sequence
- The future direction of the agent.
- A set of additional steps, if any.
"""


class CollisionRule(Protocol):
    def __call__(
            self,
            step: np.ndarray,
            neighbourhood: np.ndarray,
            colors: Iterator[int],
            direction: Direction,
            grid: np.ndarray
    ) -> Optional[CollisionResult]:
        """
        A collision rule regulates the detection of collisions and determines the future behavior of an agent.

        :param step: the current step.
        :param neighbourhood: the neighbourhood of the agent in the current direction.
        :param colors: the current color iterator.
        :param direction: the direction of the agent.
        :param grid: the grid of the collision.
        :return: A collision result.
        """
        pass
