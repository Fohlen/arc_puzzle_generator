"""
The physics module contains world *physics*, for instance, calculating direction vectors and other physical properties.
"""

import math

import numpy as np

from arc_puzzle_generator.geometry import Point, PointSet, Axis, Direction

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


def direction_to_numpy_unit_vector(direction: Direction) -> np.ndarray:
    """
    Returns the unit vector corresponding to the given direction.
    :param direction: The direction to convert.
    :return: A unit vector for the given direction.
    """

    match direction:
        case "left":
            return np.array([0, -1])
        case "right":
            return np.array([0, 1])
        case "up":
            return np.array([-1, 0])
        case "down":
            return np.array([1, 0])
        case "bottom_left":
            return np.array([1, -1])
        case "top_left":
            return np.array([-1, -1])
        case "top_right":
            return np.array([-1, 1])
        case "bottom_right":
            return np.array([1, 1])

    raise ValueError("Unknown direction {}".format(direction))


def box_distance(box1: np.ndarray, box2: np.ndarray, direction: Direction) -> int:
    """
    Returns the distance between two bounding boxes given a direction.
    :param box1: The first box.
    :param box2: The second box.
    :param direction: The direction between the two boxes.
    :return: The distance between the two points.
    """

    if direction == "left":
        return int(math.dist(box1[0], box2[3]))
    elif direction == "right":
        return int(math.dist(box1[3], box2[0]))
    elif direction == "up":
        return int(math.dist(box1[1], box2[0]))
    elif direction == "down":
        return int(math.dist(box1[0], box2[1]))
    elif direction == "top_left":
        return int(math.dist(box1[1], box2[3]))
    elif direction == "top_right":
        return int(math.dist(box1[2], box2[0]))
    elif direction == "bottom_left":
        return int(math.dist(box1[0], box2[2]))
    elif direction == "bottom_right":
        return int(math.dist(box1[3], box2[1]))

    raise ValueError("Unknown direction {}".format(direction))


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


def relative_box_direction(box1: np.ndarray, box2: np.ndarray) -> Direction:
    """
    Returns the relative direction between two boxes.
    :param box1: The box to determine a relative direction for.
    :param box2: The box to determine a relative direction to.
    :return: The relative direction between the two boxes.
    """

    left = (box2[3, 1] < box1[0, 1]).item()
    right = (box1[3, 1] < box2[0, 1]).item()
    up = (box2[0, 0] < box1[1, 0]).item()
    down = (box1[0, 0] < box2[1, 0]).item()

    return combine_directions((left, right, up, down))
