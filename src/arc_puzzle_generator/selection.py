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


def resolve_cell_selection(
        point_set: PointSet, direction: Direction,
):
    """
    Selects the maximum or minimum cells in a given direction from the point set.
    E.g. for direction up, the minimum row will be selected.
    :param point_set: The point set to select from.
    :param direction: The direction to select.
    :return: A set of cells selected.
    """

    min_row = min((x for x, y in point_set))
    max_row = max((x for x, y in point_set))
    min_col = min((y for x, y in point_set))
    max_col = max((y for x, y in point_set))

    match direction:
        case "up":
            return PointSet([
                (x, y) for (x, y) in point_set
                if x == min_row
            ])
        case "down":
            return PointSet([
                (x, y) for (x, y) in point_set
                if x == max_row
            ])
        case "left":
            return PointSet([
                (x, y) for (x, y) in point_set
                if y == min_col
            ])
        case "right":
            return PointSet([
                (x, y) for (x, y) in point_set
                if y == max_col
            ])
        case "bottom_left":
            return PointSet([
                (x, y) for (x, y) in point_set
                if x == max_row and y == min_col
            ])
        case "top_left":
            return PointSet([
                (x, y) for (x, y) in point_set
                if x == min_row and y == min_col
            ])
        case "top_right":
            return PointSet([
                (x, y) for (x, y) in point_set
                if x == min_row and y == max_col
            ])
        case "bottom_right":
            return PointSet([
                (x, y) for (x, y) in point_set
                if x == max_row and y == max_col
            ])

    raise ValueError(f"Unknown direction: {direction}")
