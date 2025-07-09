from itertools import chain
from typing import Protocol

from abm.geometry import Point, PointSet


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

    return {
        (x - 1, y),  # Left
        (x + 1, y),  # Right
        (x, y - 1),  # Up
        (x, y + 1),  # Down
    }


def moore_neighbours(point: Point) -> PointSet:
    """
    Return the Moore neighbourhood of a point.
    :param point: A tuple representing the point (x, y).
    :return: The moore neighbourhood of the point as a set of points.
    """
    x, y = point

    return {
        (x - 1, y - 1),  # Top-left
        (x - 1, y),  # Left
        (x - 1, y + 1),  # Bottom-left
        (x, y - 1),  # Up
        (x, y + 1),  # Down
        (x + 1, y - 1),  # Top-right
        (x + 1, y),  # Right
        (x + 1, y + 1),  # Bottom-right
    }


def resolve_point_set_neighbours(point_set: PointSet, neighbourhood: Neighbourhood) -> PointSet:
    """
    Return a neighbourhood that returns the points in the point set.

    :param point_set: The point set for which to resolve neighbours.
    :param neighbourhood: The neighbourhood function to use.
    :return: All neighbours of all given points in a point set.
    """

    point_neighbours = set(chain.from_iterable([neighbourhood(point) for point in point_set]))
    return point_neighbours - point_set
