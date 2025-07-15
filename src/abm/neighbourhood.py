from itertools import chain
from typing import Protocol, cast

from abm.geometry import Point, PointSet
from abm.physics import Axis


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


def zero_neighbours(point: Point) -> PointSet:
    """
    Return an empty neighbourhood for any given point.
    :param point: The input point for which to return an empty neighbourhood.
    :return: An empty PointSet.
    """
    return PointSet()


class AxisNeighbourhood(Neighbourhood):
    """
    A neighbourhood that returns all points in the same row and column as the given point.
    :param grid_size: The size of the grid as a tuple (rows, columns).
    :return: A set of points in the same row and column as the given point.
    """

    def __init__(self, grid_size: tuple[int, int], axis: Axis) -> None:
        self.grid_size = grid_size
        self.axis = axis

    def __call__(self, point: Point, *args, **kwargs) -> PointSet:
        neighbours = set()

        x, y = point

        if self.axis == "horizontal":
            neighbours.update([(i, y) for i in range(0, self.grid_size[0])])
        elif self.axis == "vertical":
            neighbours.update([(x, i) for i in range(0, self.grid_size[0])])

        return PointSet(neighbours)


def resolve_point_set_neighbourhood(point_set: PointSet, neighbourhood: Neighbourhood) -> PointSet:
    """
    Return a neighbourhood that returns the points in the point set.

    :param point_set: The point set for which to resolve neighbours.
    :param neighbourhood: The neighbourhood to return.
    :return: All neighbours of all given points in a point set.
    """

    point_neighbours = set(chain.from_iterable([neighbourhood(point) for point in point_set]))
    return cast(PointSet, point_neighbours - point_set)
