"""
The physics module contains world *physics*, for instance, calculating direction vectors and other physical properties.
"""
from typing import Literal, Optional

import numpy as np

Axis = Literal["horizontal", "vertical", "diagonal"]
Direction = Literal["left", "right", "up", "down", "top_left", "top_right", "bottom_left", "bottom_right"]


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

    # TODO: Support diagonal directions

    if direction == "left":
        return box1[0][1] - box2[3][1]
    elif direction == "right":
        return box2[0][1] - box1[3][1]
    elif direction == "up":
        return box1[2][0] - box2[0][0]
    else:
        return box2[2][0] - box1[0][0]


def relative_box_direction(box1: np.ndarray, box2: np.ndarray) -> Direction:
    """
    Returns the relative direction between two boxes.
    :param box1: The box to determine a relative direction for.
    :param box2: The box to determine a relative direction to.
    :return: The relative direction between the two boxes.
    """

    # TODO: Support diagonal directions

    if box1[0, 0] < box2[0, 0]:
        return "down"
    elif box1[0, 0] > box2[0, 0]:
        return "up"
    elif box1[0, 1] < box2[0, 1]:
        return "right"
    else:
        return "left"


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

    # TODO: Properly support diagonal directions

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
            return np.array([bounding_box[0]])
        case "top_left":
            return np.array([bounding_box[1]])
        case "top_right":
            return np.array([bounding_box[2]])
        case "bottom_right":
            return np.array([bounding_box[3]])

    raise ValueError("Unknown direction {}".format(direction))
