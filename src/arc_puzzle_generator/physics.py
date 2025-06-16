"""
The physics module contains world *physics*, for instance, calculating direction vectors and other physical properties.
"""
from typing import Literal

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


def orthogonal_direction(direction: Direction, axis: Axis = "horizontal") -> Direction:
    """
    Returns the orthogonal direction of the given direction based on a collision axis.
    :param direction: The direction to convert.
    :param axis: The collision axis.
    :return: The orthogonal direction of the given direction.
    """

    match axis:
        case "horizontal":
            # For horizontal collisions (hitting vertical walls)
            match direction:
                case "bottom_left":
                    return "bottom_right"
                case "bottom_right":
                    return "bottom_left"
                case "top_left":
                    return "top_right"
                case "top_right":
                    return "top_left"
        case "vertical":
            # For vertical collisions (hitting horizontal walls)
            match direction:
                case "bottom_left":
                    return "top_left"
                case "bottom_right":
                    return "top_right"
                case "top_left":
                    return "bottom_left"
                case "top_right":
                    return "bottom_right"

    raise ValueError("Unknown axis {}".format(axis))


def starting_point(bounding_box: np.ndarray, direction: Direction) -> np.ndarray:
    """
    Returns the starting point of a structure with a given bounding box and direction.
    :param bounding_box: The bounding box of the structure.
    :param direction: The direction of the structure.
    :return: Starting point of the structure.
    """

    match direction:
        case "left":
            return (bounding_box[0] + bounding_box[1]) // 2
        case "right":
            return (bounding_box[2] + bounding_box[3]) // 2
        case "up":
            return (bounding_box[1] + bounding_box[2]) // 2
        case "down":
            return (bounding_box[0] + bounding_box[3]) // 2
        case "bottom_left":
            return bounding_box[0]
        case "top_left":
            return bounding_box[1]
        case "top_right":
            return bounding_box[2]
        case "bottom_right":
            return bounding_box[3]

    raise ValueError("Unknown direction {}".format(direction))
