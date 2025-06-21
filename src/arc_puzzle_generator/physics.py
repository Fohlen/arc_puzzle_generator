"""
The physics module contains world *physics*, for instance, calculating direction vectors and other physical properties.
"""
import math
from typing import Literal, Protocol

import numpy as np

Axis = Literal["horizontal", "vertical", "diagonal"]
"""
The axis of a line.
"""

Direction = Literal["left", "right", "up", "down", "top_left", "top_right", "bottom_left", "bottom_right"]
"""
The possible directions we can go in our universe.
"""


class DirectionRule(Protocol):
    """
    A direction rule determines the future direction of an agent based on the current direction and additional parameters.
    """

    def __call__(self, direction: Direction, *args, **kwargs) -> Direction:
        pass


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


def contained(point: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    Check if any of the points is contained in any of the bounding boxes.
    :param point: The points to check.
    :param bbox: The bounding boxes to check.
    :return: A numpy array of booleans indicating whether the points are contained in the bounding boxes.
    """

    indexes = np.zeros(point.shape[0], dtype=bool)

    for index, p in enumerate(point):
        for box in bbox:
            if box[1][0] <= p[0] <= box[3][0] and box[0][1] <= p[1] <= box[2][1]:
                indexes[index] = True
                break

    return indexes


def line_axis(box: np.ndarray) -> Axis:
    """
    Determines the axis of the line between two N.
    :param box: The box to determine the axis for.
    :return: The determined axis.
    """

    xs = np.unique(box[:, 0])
    ys = np.unique(box[:, 1])

    if len(xs) == 1 and np.all(xs == box[:, 0]):
        return "horizontal"
    elif len(ys) == 1 and np.all(ys == box[:, 1]):
        return "vertical"
    else:
        return "diagonal"


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
