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


def collision_axis(point: np.ndarray, box: np.ndarray, direction: Direction) -> Axis:
    """
    Returns the axis of the collision between two points based on the given direction.
    :param point: The point to go from.
    :param box:  The box to go to.
    :param direction: The direction between the two.
    :return: The axis of the collision between the two points.
    """

    match direction:
        case "up":
            return "horizontal"
        case "down":
            return "horizontal"
        case "left":
            return "vertical"
        case "right":
            return "vertical"
        case "top_left" | "bottom_left":
            return "vertical" if box[3][1] < point[:, 1].min() else "horizontal"
        case "top_right" | "bottom_right":
            return "vertical" if point[:, 1].max() < box[0][1] else "horizontal"

    raise ValueError("Unknown collision direction {}".format(direction))


def orthogonal_direction(direction: Direction, axis: Axis = "horizontal") -> Direction:
    """
    Returns the orthogonal direction of the given direction based on a collision axis.
    :param direction: The direction to convert.
    :param axis: The collision axis.
    :return: The orthogonal direction of the given direction.
    """

    match axis:
        case "vertical":
            # For vertical collisions (hitting vertical walls)
            match direction:
                case "bottom_left":
                    return "bottom_right"
                case "bottom_right":
                    return "bottom_left"
                case "top_left":
                    return "top_right"
                case "top_right":
                    return "top_left"
        case "horizontal":
            # For horizontal collisions (hitting horizontal walls)
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


def is_point_adjacent(point: np.ndarray, bboxes: np.ndarray) -> Optional[np.ndarray]:
    """
    Check if a point is adjacent to any of the bounding boxes

    :param point: containing integer coordinates [x, y]
    :param bboxes: containing integer coordinates of N bounding boxes, each with 4 corners in order [bottom_left, top_left, top_right, bottom_right]
    :returns: numpy array of indices where adjacency was found, or None if no adjacency found
    """

    # Ignore empty boxes
    if bboxes.size == 0:
        return None

    # Get min and max coordinates of bounding boxes
    bbox_min_x = np.min(bboxes[:, :, 0], axis=1)
    bbox_max_x = np.max(bboxes[:, :, 0], axis=1)
    bbox_min_y = np.min(bboxes[:, :, 1], axis=1)
    bbox_max_y = np.max(bboxes[:, :, 1], axis=1)

    x = point[:, 0].min()
    y = point[:, 1].max()

    # Check x-adjacency (point is one unit away horizontally and within vertical bounds)
    x_adjacent = ((x == bbox_max_x + 1) | (x == bbox_min_x - 1)) & \
                 (y >= bbox_min_y) & (y <= bbox_max_y)

    # Check y-adjacency (point is one unit away vertically and within horizontal bounds)
    y_adjacent = ((y == bbox_max_y + 1) | (y == bbox_min_y - 1)) & \
                 (x >= bbox_min_x) & (x <= bbox_max_x)

    adjacent = x_adjacent | y_adjacent
    matching_indices = np.where(adjacent)[0]

    return matching_indices if matching_indices.size > 0 else None
