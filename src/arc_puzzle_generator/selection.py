from typing import Protocol

from arc_puzzle_generator.geometry import Point, PointSet
from arc_puzzle_generator.physics import Direction


class Selector(Protocol):
    """
    A selector is a callable that takes a point and a point set and returns a set of points.
    Selectors are chainable operators, e.g:
    ```
    p = (1, 1)
    neighbourhood = moore_neighbours(p)
    top_left = left_selector(p, top_selector(p, neighbourhood))
    ```
    """

    def __call__(self, point: Point, point_set: PointSet) -> PointSet:
        pass


def up_selector(point: Point, point_set: PointSet) -> PointSet:
    """
    Selects points that are above the given point in the point set.

    :param point: The reference point.
    :param point_set: The set of points to select from.
    :return: A set of points that are above the reference point.
    """

    return PointSet((x, y) for (x, y) in point_set if x < point[0] and y == point[1])


def down_selector(point: Point, point_set: PointSet) -> PointSet:
    """
    Selects points that are below the given point in the point set.

    :param point: The reference point.
    :param point_set: The set of points to select from.
    :return: A set of points that are below the reference point.
    """

    return PointSet((x, y) for (x, y) in point_set if x > point[0] and y == point[1])


def left_selector(point: Point, point_set: PointSet) -> PointSet:
    """
    Selects points that are to the left of the given point in the point set.
    :param point: The reference point.
    :param point_set: The set of points to select from.
    :return: A set of points that are to the left of the reference point.
    """

    return PointSet((x, y) for (x, y) in point_set if y < point[1] and x == point[0])


def right_selector(point: Point, point_set: PointSet) -> PointSet:
    """
    Selects points that are to the right of the given point in the point set.
    :param point: The reference point.
    :param point_set: The set of points to select from.
    :return: A set of points that are to the right of the reference point.
    """

    return PointSet((x, y) for (x, y) in point_set if y > point[1] and x == point[0])


def resolve_point_set_selectors(point_set: PointSet, neighbourhood: PointSet, selector: Selector) -> PointSet:
    """
    Resolves a point set using a selector function.
    :param point_set: The point set to resolve for.
    :param neighbourhood: The neighbourhood of the point set to use for selection.
    :param selector: The selector function to use.
    :return: A set of points selected from the point set based on the selector.
    """

    return PointSet(set.union(*(selector(point, neighbourhood) for point in point_set)) - point_set)


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

    match direction:
        case "up":
            return resolve_point_set_selectors(point_set, neighbourhood, up_selector)
        case "down":
            return resolve_point_set_selectors(point_set, neighbourhood, down_selector)
        case "left":
            return resolve_point_set_selectors(point_set, neighbourhood, left_selector)
        case "right":
            return resolve_point_set_selectors(point_set, neighbourhood, right_selector)
        case _:
            raise ValueError(f"Unknown direction: {direction}")


def bottom_left_selector(point: Point, direction: Direction) -> Point:
    """
    Selects the bottom left point based on the given point as a center and direction.
    :param point: The center point from which to select.
    :param direction: The direction to select the bottom left point.
    :return: A new point that is the bottom left of the given point in the specified direction.
    """
    match direction:
        case "left":
            return point[0] + 1, point[1] + 1
        case "up":
            return point[0] + 1, point[1] - 1
        case "right":
            return point[0] - 1, point[1] - 1
        case "down":
            return point[0] - 1, point[1] + 1
        case _:
            raise ValueError("Unsupported direction for bottom_left: {}".format(direction))


def bottom_right_selector(point: Point, direction: Direction) -> Point:
    """
    Selects the bottom right point based on the given point as a center and direction.
    :param point: The center point from which to select.
    :param direction: The direction to select the bottom right point.
    :return: A new point that is the bottom right of the given point in the specified direction.
    """
    match direction:
        case "left":
            return point[0] - 1, point[1] + 1
        case "up":
            return point[0] + 1, point[1] + 1
        case "right":
            return point[0] + 1, point[1] - 1
        case "down":
            return point[0] - 1, point[1] - 1
        case _:
            raise ValueError("Unsupported direction for bottom_right: {}".format(direction))
