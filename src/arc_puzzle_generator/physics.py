"""
The physics module contains world *physics*, for instance, calculating direction vectors and other physical properties.
"""
import math

import numpy as np

from abm.physics import Direction


def direction_to_unit_vector(direction: Direction):
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


def starting_point(
        bounding_box: np.ndarray,
        direction: Direction,
        point_width: int = 1,
) -> np.ndarray:
    """
    Returns the starting point of a structure with a given bounding box and direction.
    :param bounding_box: The bounding box of the structure.
    :param direction: The direction of the structure.
    :param point_width: The width of the starting point.
    :return: Starting point of the structure.
    """

    match direction:
        case "left":
            start_pos = ((bounding_box[0] + bounding_box[1]) // 2)
            return np.array([start_pos + [i, 0] for i in range(point_width)])
        case "right":
            start_pos = ((bounding_box[2] + bounding_box[3]) // 2)
            return np.array([start_pos + [i, 0] for i in range(point_width)])
        case "up":
            start_pos = ((bounding_box[1] + bounding_box[2]) // 2)
            return np.array([start_pos + [0, i] for i in range(point_width)])
        case "down":
            start_pos = ((bounding_box[0] + bounding_box[3]) // 2)
            return np.array([start_pos + [0, i] for i in range(point_width)])
        case "bottom_left":
            start_pos = bounding_box[0]
            return np.array([start_pos + (direction_to_unit_vector("bottom_right") * i) for i in range(point_width)])
        case "top_left":
            start_pos = bounding_box[1]
            return np.array([start_pos + (direction_to_unit_vector("top_right") * i) for i in range(point_width)])
        case "top_right":
            start_pos = bounding_box[2]
            return np.array([start_pos + (direction_to_unit_vector("bottom_right") * i) for i in range(point_width)])
        case "bottom_right":
            start_pos = bounding_box[3]
            return np.array([start_pos + (direction_to_unit_vector("top_right") * i) for i in range(point_width)])

    raise ValueError("Unknown direction {}".format(direction))

