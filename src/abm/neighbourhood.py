from itertools import chain
from typing import Protocol, cast

import numpy as np

from abm.geometry import Point, PointSet
from abm.physics import Direction


class Neighbourhood(Protocol):
    """Protocol for neighbourhoods."""

    def __call__(self, point: Point) -> PointSet:
        """Return the neighbourhood of a point."""
        pass


def von_neumann_neighbours(point: Point) -> PointSet:
    """
    Return the Von Neumann neighbourhood of a point.

    :param point: A tuple representing the point (x, y).
    :return: The Von Neumann neighbourhood of the point as a set of points.
    """

    x, y = point

    return PointSet([
        (x - 1, y),  # Left
        (x + 1, y),  # Right
        (x, y - 1),  # Up
        (x, y + 1),  # Down
    ])


def moore_neighbours(point: Point) -> PointSet:
    """
    Return the Moore neighbourhood of a point.
    :param point: A tuple representing the point (x, y).
    :return: The moore neighbourhood of the point as a set of points.
    """
    x, y = point

    return PointSet([
        (x - 1, y - 1),  # Top-left
        (x - 1, y),  # Left
        (x - 1, y + 1),  # Bottom-left
        (x, y - 1),  # Up
        (x, y + 1),  # Down
        (x + 1, y - 1),  # Top-right
        (x + 1, y),  # Right
        (x + 1, y + 1),  # Bottom-right
    ])


def zero_neighbourhood(point: Point) -> PointSet:
    """
    Return an empty neighbourhood for any given point.
    :param point: The input point for which to return an empty neighbourhood.
    :return: An empty PointSet.
    """
    return PointSet()


def resolve_point_set_neighbours(point_set: PointSet, neighbourhood: Neighbourhood) -> PointSet:
    """
    Return a neighbourhood that returns the points in the point set.

    :param point_set: The point set for which to resolve neighbours.
    :param neighbourhood: The neighbourhood function to use.
    :return: All neighbours of all given points in a point set.
    """

    point_neighbours = set(chain.from_iterable([neighbourhood(point) for point in point_set]))
    return cast(PointSet, point_neighbours - point_set)


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
