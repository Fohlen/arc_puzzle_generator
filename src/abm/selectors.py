from typing import Protocol

from abm.geometry import Point, PointSet


class Selector(Protocol):
    """
    A selector is a callable that takes a point and a point set and returns a set of points.
    """

    def __call__(self, point: Point, point_set: PointSet) -> PointSet:
        pass


def top_selector(point: Point, point_set: PointSet) -> PointSet:
    """
    Selects points that are above the given point in the point set.

    :param point: The reference point.
    :param point_set: The set of points to select from.
    :return: A set of points that are above the reference point.
    """

    return {(x, y) for (x, y) in point_set if x < point[0]}


def bottom_selector(point: Point, point_set: PointSet) -> PointSet:
    """
    Selects points that are below the given point in the point set.

    :param point: The reference point.
    :param point_set: The set of points to select from.
    :return: A set of points that are below the reference point.
    """

    return {(x, y) for (x, y) in point_set if x > point[0]}


def left_selector(point: Point, point_set: PointSet) -> PointSet:
    """
    Selects points that are to the left of the given point in the point set.
    :param point: The reference point.
    :param point_set: The set of points to select from.
    :return: A set of points that are to the left of the reference point.
    """

    return {(x, y) for (x, y) in point_set if y < point[1]}


def right_selector(point: Point, point_set: PointSet) -> PointSet:
    """
    Selects points that are to the right of the given point in the point set.
    :param point: The reference point.
    :param point_set: The set of points to select from.
    :return: A set of points that are to the right of the reference point.
    """

    return {(x, y) for (x, y) in point_set if y > point[1]}
