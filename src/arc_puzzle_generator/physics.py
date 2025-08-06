"""
The physics module contains world *physics*, for instance, calculating direction vectors and other physical properties.
"""

from arc_puzzle_generator.geometry import Point, PointSet, Axis, Direction


def direction_to_unit_vector(direction: Direction) -> Point:
    """
    Returns the unit vector corresponding to the given direction.
    :param direction: The direction to convert.
    :return: A unit vector for the given direction.
    """

    match direction:
        case "none":
            return 0, 0
        case "left":
            return 0, -1
        case "right":
            return 0, 1
        case "up":
            return -1, 0
        case "down":
            return 1, 0
        case "bottom_left":
            return 1, -1
        case "top_left":
            return -1, -1
        case "top_right":
            return -1, 1
        case "bottom_right":
            return 1, 1


def shift(point: Point, vector: Point) -> Point:
    """
    Shifts a point by a given vector.
    :param point: The point to shift.
    :param vector: The vector to shift the point by.
    :return: A new point shifted by the vector.
    """
    return point[0] + vector[0], point[1] + vector[1]


def collision_axis(collision_position: PointSet) -> Axis:
    """
    Determines the axis of collision between an agent and a collision point.
    :param agent_position: The position of the agent.
    :param collision_position: The collision position.
    :return: A string representing the axis of collision.
    """

    xs = set(a[0] for a in collision_position)
    ys = set(a[1] for a in collision_position)

    if len(xs) == len(ys):
        return "horizontal"
    elif len(xs) == 1:
        return "horizontal"
    elif len(ys) == 1:
        return "vertical"
    else:
        raise ValueError("Collision point cannot be the same as agent position.")


def combine_directions(directions: tuple[bool, bool, bool, bool]) -> Direction:
    """
    Combines four boolean cardinal directions into a single direction.
    :param directions: The cardinal directions as a tuple of booleans (left, right, up, down).
    :return: A string representing the combined direction.
    """
    left, right, up, down = directions

    match (left, right, up, down):
        case (True, False, False, False):
            return "left"
        case (False, True, False, False):
            return "right"
        case (False, False, True, False):
            return "up"
        case (False, False, False, True):
            return "down"
        case (True, False, True, False):
            return "top_left"
        case (False, True, True, False):
            return "top_right"
        case (True, False, False, True):
            return "bottom_left"
        case (False, True, False, True):
            return "bottom_right"
        case _:
            raise ValueError("Unknown direction")


def relative_point_direction(
        point1: Point,
        point2: Point,
) -> Direction:
    """
    Returns the relative direction between two points.
    :param point1: The point to determine a relative direction for.
    :param point2: The point to determine a relative direction to.
    :return: The relative direction between the two points.
    """

    left = point2[1] < point1[1]
    right = point1[1] < point2[1]
    up = point2[0] < point1[0]
    down = point1[0] < point2[0]

    return combine_directions((left, right, up, down))
