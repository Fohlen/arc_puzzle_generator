from typing import cast

from arc_puzzle_generator.geometry import Point, PointSet, Direction
from arc_puzzle_generator.physics import direction_to_unit_vector


def direction_selector(point: Point, point_set: PointSet, direction: Direction) -> PointSet:
    """
    Selects points in a given direction from the point set.
    :param point: The reference point.
    :param point_set: The set of points to select from.
    :param direction: The direction to select points in.
    :return: A set of points in the specified direction from the reference point.
    """

    return cast(PointSet, PointSet([point]).shift(direction_to_unit_vector(direction)) & point_set)


def resolve_point_set_selectors_with_direction(
        point_set: PointSet, neighbourhood: PointSet, direction: Direction
) -> PointSet:
    """
    Resolves a point set using a direction.
    :param point_set: The point set to resolve for.
    :param neighbourhood: The neighbourhood of the point set to use for selection.
    :param direction: The direction to use for selection.
    :return: A set of points selected from the point set based on the direction.
    """

    return PointSet(set.union(*[direction_selector(point, neighbourhood, direction) for point in point_set]))
