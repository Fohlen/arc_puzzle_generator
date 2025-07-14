from itertools import chain
from typing import Protocol, cast

from abm.geometry import Point, PointSet
from abm.physics import Direction


class Neighbourhood(Protocol):
    """Protocol for neighbourhoods."""

    def __call__(self, point: Point, direction: Direction) -> PointSet:
        """Return the neighbourhood of a point."""
        pass


def von_neumann_neighbours(point: Point, *args, **kwargs) -> PointSet:
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


def moore_neighbours(point: Point, *args, **kwargs) -> PointSet:
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


def zero_neighbours(point: Point, *args, **kwargs) -> PointSet:
    """
    Return an empty neighbourhood for any given point.
    :param point: The input point for which to return an empty neighbourhood.
    :return: An empty PointSet.
    """
    return PointSet()


def resolve_point_set_neighbours(point_set: PointSet, direction: Direction, neighbourhood: Neighbourhood) -> PointSet:
    """
    Return a neighbourhood that returns the points in the point set.

    :param point_set: The point set for which to resolve neighbours.
    :param direction: The direction to resolve the neighbourhood in.
    :param neighbourhood: The neighbourhood function to use.
    :return: All neighbours of all given points in a point set.
    """

    point_neighbours = set(chain.from_iterable([neighbourhood(point, direction) for point in point_set]))
    return cast(PointSet, point_neighbours - point_set)


def directional_neighbours(point: PointSet, direction: Direction) -> PointSet:
    """
    Determines the neighborhood of a point based on a direction (Berger neighborhood).
    :param point: The point to determine the neighborhood for.
    :param direction: The direction to determine the neighborhood into.
    :return: A 2D array of neighborhood coordinates.
    """

    # NOTE: Diagonal points will validate the tip of the step

    xs = set(p[0] for p in point)
    ys = set(p[1] for p in point)
    y_min = min(ys)
    y_max = max(ys)
    x_min = min(xs)
    x_max = max(xs)

    match direction:
        case "right":
            return PointSet([
                (x, y_max + 1)
                for x in range(x_min, x_max + 1)
            ])
        case "left":
            return PointSet([
                (x, y_min - 1)
                for x in range(x_min, x_max + 1)
            ])
        case "up":
            return PointSet([
                (x_min - 1, y)
                for y in range(y_min, y_max + 1)
            ])
        case "down":
            return PointSet([
                (x_max + 1, y)
                for y in range(y_min, y_max + 1)
            ])
        case "top_left":
            return PointSet([
                (x_min - 1, y_min - 1), (x_min - 1, y_min), (x_min, y_min - 1)
            ])
        case "top_right":
            return PointSet([
                (x_min - 1, y_max), (x_min - 1, y_max + 1), (x_min, y_max + 1)
            ])
        case "bottom_left":
            return PointSet([
                (x_min, y_min - 1), (x_min + 1, y_min - 1), (x_min + 1, y_min)
            ])
        case "bottom_right":
            return PointSet([
                (x_max, y_max + 1), (x_max + 1, y_max), (x_max + 1, y_max + 1)
            ])

    raise ValueError(f"Invalid direction: {direction}")
