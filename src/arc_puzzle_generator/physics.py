"""
The physics module contains world *physics*, for instance calculating orientation vectors and other physical properties.
"""
from typing import Literal

import numpy as np

Orientation = Literal["top_left", "top_right", "bottom_left", "bottom_right"]
Direction = Literal["left", "right", "up", "down"]


def direction_to_unit_vector(direction: Direction):
    """
    Returns the unit vector corresponding to the given direction.
    :param direction: The direction to convert.
    :return: A unit vector for the given direction.
    """

    if direction == "left":
        return np.array([0, -1])
    elif direction == "right":
        return np.array([0, 1])
    elif direction == "up":
        return np.array([-1, 0])
    return np.array([1, 0])


def orientation_to_unit_vector(orientation: Orientation) -> np.ndarray:
    """
    Returns the unit vector corresponding to the given orientation.
    :param orientation: The orientation to convert.
    :return: A unit vector for the given orientation.
    """
    if orientation == "bottom_left":
        return np.array([1, -1])
    elif orientation == "top_left":
        return np.array([-1, -1])
    elif orientation == "top_right":
        return np.array([-1, 1])

    return np.array([1, 1])


def orthogonal_orientation(orientation: Orientation, is_horizontal_collision: bool) -> Orientation:
    """
    Returns the orthogonal orientation of the given orientation based on collision direction.
    :param orientation: The orientation to convert.
    :param is_horizontal_collision: True if the beam hits horizontally, False if vertically.
    :return: The orthogonal orientation of the given orientation.
    """
    if is_horizontal_collision:
        # For horizontal collisions (hitting vertical walls)
        if orientation == "bottom_left":
            return "bottom_right"
        elif orientation == "bottom_right":
            return "bottom_left"
        elif orientation == "top_left":
            return "top_right"
        return "top_left"
    else:
        # For vertical collisions (hitting horizontal walls)
        if orientation == "bottom_left":
            return "top_left"
        elif orientation == "bottom_right":
            return "top_right"
        elif orientation == "top_left":
            return "bottom_left"
        return "bottom_right"


def starting_point(bounding_box: np.ndarray, orientation: Orientation) -> np.ndarray:
    """
    Returns the starting point of a structure with a given bounding box and orientation.
    :param bounding_box: The bounding box of the structure.
    :param orientation: The orientation of the structure.
    :return: Starting point of the structure.
    """

    if orientation == "bottom_left":
        return bounding_box[0]
    elif orientation == "top_left":
        return bounding_box[1]
    elif orientation == "top_right":
        return bounding_box[2]

    return bounding_box[3]
