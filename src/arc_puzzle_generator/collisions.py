from typing import Optional, Iterator, Protocol

import numpy as np

from arc_puzzle_generator.physics import Direction, Axis


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


def directional_neighbourhood(point: np.ndarray, direction: Direction) -> np.ndarray:
    """
    Determines the neighborhood of a point based on a direction (Berger neighborhood).
    :param point: The point to determine the neighborhood for.
    :param direction: The direction to determine the neighborhood into.
    :return: A 2D array of neighborhood coordinates.
    """

    # NOTE: Diagonal points will validate the tip of the step

    y_min = point[:, 1].min()
    y_max = point[:, 1].max()
    x_min = point[:, 0].min()
    x_max = point[:, 0].max()

    match direction:
        case "right":
            return np.array([
                [x, y_max + 1]
                for x in range(x_min, x_max + 1)
            ])
        case "left":
            return np.array([
                [x, y_min - 1]
                for x in range(x_min, x_max + 1)
            ])
        case "up":
            return np.array([
                [x_min - 1, y]
                for y in range(y_min, y_max + 1)
            ])
        case "down":
            return np.array([
                [x_max + 1, y]
                for y in range(y_min, y_max + 1)
            ])
        case "top_left":
            return np.array([
                (x_min - 1, y_min - 1), (x_min - 1, y_min), (x_min, y_min - 1)
            ])
        case "top_right":
            return np.array([
                (x_min - 1, y_max), (x_min - 1, y_max + 1), (x_min, y_max + 1)
            ])
        case "bottom_left":
            return np.array([
                (x_min, y_min - 1), (x_min + 1, y_min - 1), (x_min + 1, y_min)
            ])
        case "bottom_right":
            return np.array([
                (x_max, y_max + 1), (x_max + 1, y_max), (x_max + 1, y_max + 1)
            ])

    raise ValueError(f"Invalid direction: {direction}")


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


def orthogonal_direction(direction: Direction, axis: Axis = "horizontal") -> Direction:
    """
    Returns the orthogonal direction of the given direction based on a collision axis.
    :param direction: The direction to convert.
    :param axis: The collision axis.
    :return: The orthogonal direction of the given direction.
    """

    match axis:
        case "vertical":
            # For vertical collisions (hitting vertical walls)
            match direction:
                case "bottom_left":
                    return "bottom_right"
                case "bottom_right":
                    return "bottom_left"
                case "top_left":
                    return "top_right"
                case "top_right":
                    return "top_left"
        case "horizontal":
            # For horizontal collisions (hitting horizontal walls)
            match direction:
                case "bottom_left":
                    return "top_left"
                case "bottom_right":
                    return "top_right"
                case "top_left":
                    return "bottom_left"
                case "top_right":
                    return "bottom_right"

    raise ValueError("Unknown axis {}".format(axis))


def snake_direction(direction: Direction, *args, **kwargs) -> Direction:
    """
    Returns the opposite direction of the given direction, moving in a snake pattern.
    :param direction: The input direction
    :return: the opposite direction
    """
    if direction == "right":
        return "up"

    return "right"


def identity_direction(direction: Direction, *args, **kwargs) -> Direction:
    """
    Continues eternally in the same direction.
    :param direction: The input direction.
    :return: the same direction.
    """

    return direction


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
