from typing import Literal, Protocol

from abm.geometry import Point

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


class DirectionRule(Protocol):
    """
    A direction rule determines the future direction of an agent based on the current direction and additional parameters.
    """

    def __call__(self, direction: Direction, *args, **kwargs) -> Direction:
        pass


def identity_direction_rule(direction: Direction) -> Direction:
    """
    A direction rule that returns the same direction.
    :param direction: The direction to follow.
    :return: A direction rule that returns the same direction.
    """

    return direction
