from typing import Literal

from abm.geometry import Point, PointSet

Axis = Literal["horizontal", "vertical", "diagonal"]
"""
The axis of a line.
"""

Direction = Literal["left", "right", "up", "down", "top_left", "top_right", "bottom_left", "bottom_right"]
"""
The possible directions we can go in our universe.
"""


def direction_to_unit_vector(direction: Direction) -> Point:
    """
    Returns the unit vector corresponding to the given direction.
    :param direction: The direction to convert.
    :return: A unit vector for the given direction.
    """

    match direction:
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

    raise ValueError("Unknown direction {}".format(direction))


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
