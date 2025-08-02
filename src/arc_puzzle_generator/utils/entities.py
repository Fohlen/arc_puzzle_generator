"""
The entities module contains functions for extracting entities (singular color entities, shapes etc) from a grid.
"""
import math
from typing import Optional

import numpy as np

from arc_puzzle_generator.geometry import Direction, Point
from arc_puzzle_generator.neighbourhood import Neighbourhood, VonNeumannNeighbourhood
from arc_puzzle_generator.physics import combine_directions


def find_colors(grid: np.ndarray, background: Optional[int] = None) -> set[int]:
    """
    Find all colors in the grid.
    :param grid: The grid to search.
    :param background: An optional background color.
    :return: Colors used in the grid except the background color.
    """

    if background is not None:
        return set(color for color in np.unique(grid).tolist() if color != background)
    else:
        return set(np.unique(grid).tolist())


def colour_count(grid: np.ndarray) -> list[tuple[int, int]]:
    """
    Return color count in the grid.
    :param grid: The grid to search.
    :return: The count of every color in the grid as a tuple of (color, frequency) in descending order of frequency.
    """

    values, counts = np.unique(grid, return_counts=True)
    sorted_counts = np.argsort(counts)[::-1]
    return [(values[idx], counts[idx]) for idx in sorted_counts]


def find_connected_objects(
        mask: np.ndarray,
        neighbourhood: Neighbourhood = VonNeumannNeighbourhood(),
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Find connected objects in a binary mask.
    :param mask: The input binary mask.
    :param neighbourhood: The neighbourhood to use.
    :return: A tuple containing:
             - The labeled mask
             - A 2D array of bounding boxes (one per object) with coordinates in order:
               bottom-left, top-left, top-right, bottom-right
             - The number of objects
    """
    rows, cols = mask.shape
    labeled_mask = np.zeros_like(mask, dtype=int)
    object_count = 0
    # Dictionary to store min/max coordinates for each object
    object_bounds = {}

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols and mask[r, c] and labeled_mask[r, c] == 0

    def bfs(r, c, label):
        queue = [(r, c)]
        labeled_mask[r, c] = label
        # Initialize min/max coordinates for this object
        object_bounds[label] = {'min_row': r, 'max_row': r, 'min_col': c, 'max_col': c}

        while queue:
            row, col = queue.pop(0)
            # Update min/max coordinates
            object_bounds[label]['min_row'] = min(object_bounds[label]['min_row'], row)
            object_bounds[label]['max_row'] = max(object_bounds[label]['max_row'], row)
            object_bounds[label]['min_col'] = min(object_bounds[label]['min_col'], col)
            object_bounds[label]['max_col'] = max(object_bounds[label]['max_col'], col)

            neighbors = neighbourhood((row, col))
            for nr, nc in neighbors:
                if is_valid(nr, nc):
                    labeled_mask[nr, nc] = label
                    queue.append((nr, nc))

    for i in range(rows):
        for j in range(cols):
            if mask[i, j] and labeled_mask[i, j] == 0:
                object_count += 1
                bfs(i, j, object_count)

    # Create bounding boxes array
    if object_count > 0:
        bounding_boxes = np.zeros((object_count, 4, 2), dtype=int)
        for label in range(1, object_count + 1):
            bounds = object_bounds[label]
            min_row, max_row = bounds['min_row'], bounds['max_row']
            min_col, max_col = bounds['min_col'], bounds['max_col']

            # Order: bottom-left, top-left, top-right, bottom-right
            bounding_boxes[label - 1] = np.array([
                [max_row, min_col],  # bottom-left
                [min_row, min_col],  # top-left
                [min_row, max_col],  # top-right
                [max_row, max_col]  # bottom-right
            ])
    else:
        bounding_boxes = np.zeros((0, 4, 2), dtype=int)

    return labeled_mask, bounding_boxes, object_count


def is_l_shape(arr: np.ndarray) -> Optional[Direction]:
    """
    Checks if a 2D NumPy array represents an L-shape.

    :param arr: The 2D NumPy array to check.
    :return: The direction of the L-shape (e.g., "bottom right", "top left", etc. if it's an L-shape, otherwise None.
    """

    # L-shape's cannot be straight lines
    if arr.shape[0] == 1:
        return None

    non_zero_indices = np.argwhere(arr != 0)
    num_non_zero = len(non_zero_indices)

    if num_non_zero == 0:
        return None  # Empty array is not an L-shape

    # Find the bounding box of the non-zero elements
    min_row, min_col = np.min(non_zero_indices, axis=0)
    max_row, max_col = np.max(non_zero_indices, axis=0)

    # Check for the two possible arms of the 'L'
    arm1_len = 0
    arm2_len = 0
    corner_row, corner_col = -1, -1

    # Check for a vertical arm along the left edge of the bounding box
    vertical_left_arm_len = np.sum(arr[min_row:max_row + 1, min_col] != 0)
    if vertical_left_arm_len > 0:
        arm1_len = vertical_left_arm_len
        corner_col = min_col

    # Check for a vertical arm along the right edge of the bounding box
    vertical_right_arm_len = np.sum(arr[min_row:max_row + 1, max_col] != 0)
    if vertical_right_arm_len > 0 and vertical_right_arm_len > arm1_len:
        arm1_len = vertical_right_arm_len
        corner_col = max_col

    # Check for a horizontal arm along the top edge of the bounding box
    horizontal_top_arm_len = np.sum(arr[min_row, min_col:max_col + 1] != 0)
    if horizontal_top_arm_len > 0 and horizontal_top_arm_len > arm2_len:
        arm2_len = horizontal_top_arm_len
        corner_row = min_row

    # Check for a horizontal arm along the bottom edge of the bounding box
    horizontal_bottom_arm_len = np.sum(arr[max_row, min_col:max_col + 1] != 0)
    if horizontal_bottom_arm_len > 0 and horizontal_bottom_arm_len > arm2_len:
        arm2_len = horizontal_bottom_arm_len
        corner_row = max_row

    if arm1_len > 1 and arm2_len > 1 and arm1_len + arm2_len - 1 == num_non_zero:
        # Determine direction based on the corner
        if corner_row == max_row and corner_col == max_col:
            return "bottom_right"
        elif corner_row == max_row and corner_col == min_col:
            return "bottom_left"
        elif corner_row == min_row and corner_col == min_col:
            return "top_left"
        elif corner_row == min_row and corner_col == max_col:
            return "top_right"

    return None


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
            return np.array(
                [start_pos + (direction_to_numpy_unit_vector("bottom_right") * i) for i in range(point_width)])
        case "top_left":
            start_pos = bounding_box[1]
            return np.array([start_pos + (direction_to_numpy_unit_vector("top_right") * i) for i in range(point_width)])
        case "top_right":
            start_pos = bounding_box[2]
            return np.array(
                [start_pos + (direction_to_numpy_unit_vector("bottom_right") * i) for i in range(point_width)])
        case "bottom_right":
            start_pos = bounding_box[3]
            return np.array([start_pos + (direction_to_numpy_unit_vector("top_right") * i) for i in range(point_width)])

    raise ValueError("Unknown direction {}".format(direction))


def box_contained(box_a: np.ndarray, box_b: np.ndarray) -> bool:
    """
    Compares two bounding boxes and checks if box_a is contained within box_b.
    :param box_a: The bounding box to check for containment.
    :param box_b: The bounding box to check against.
    :return: True if box_a is contained within box_b, False otherwise.
    """

    min_x, min_y = np.min(box_b, axis=0)
    max_x, max_y = np.max(box_b, axis=0)
    return np.all(
        (box_a[:, 0] >= min_x) &
        (box_a[:, 0] <= max_x) &
        (box_a[:, 1] >= min_y) &
        (box_a[:, 1] <= max_y)
    ).item()


def extreme_point(mask: np.ndarray, direction: Direction) -> Point:
    """
    Get the extreme point (leftmost, rightmost, topmost, bottommost) of a numpy mask based on the direction.
    :param mask: A 2D numpy array where True represents the region of interest.
    :param direction: The direction to find the extreme point ('left', 'right', 'top', 'bottom').
    :return: A tuple (row, col) representing the coordinates of the extreme point.
    """

    match direction:
        case "left":
            col = np.min(np.where(mask)[1]).item()
            row = np.where(mask[:, col])[0][0].item()
            return row, col
        case "right":
            col = np.max(np.where(mask)[1]).item()
            row = np.where(mask[:, col])[0][0].item()
            return row, col
        case "up":
            row = np.min(np.where(mask)[0]).item()
            col = np.where(mask[row, :])[0][0].item()
            return row, col
        case "down":
            row = np.max(np.where(mask)[0]).item()
            col = np.where(mask[row, :])[0][0].item()
            return row, col

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

    return combine_directions((left, right, up, down))


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


def get_bounding_box(points: np.ndarray) -> np.ndarray:
    """
    Calculates the bounding box for a set of 2D points and returns the corners
    in the order: left bottom, top left, top right, right bottom.

    :param points: A numpy array of 2D points (x, y).
    :return: A numpy array of points representing the bounding box corners in the specified order.
    """
    # Find the minimum and maximum x and y coordinates
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)

    left_bottom = (min_x, min_y)
    top_left = (min_x, max_y)
    top_right = (max_x, max_y)
    right_bottom = (max_x, min_y)

    # Return the corners as a numpy array
    return np.array([left_bottom, top_left, top_right, right_bottom])
