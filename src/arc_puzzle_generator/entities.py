"""
The entities module contains functions for extracting entities (singular color entities, shapes etc) from a grid.
"""

from typing import Optional

import numpy as np

from arc_puzzle_generator.physics import Orientation


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


def find_connected_objects(mask) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Find connected objects in a binary mask.
    :param mask: The input binary mask.
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

            # Explore neighbors (4-connectivity: up, down, left, right)
            neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
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


def is_l_shape(arr: np.ndarray) -> Optional[Orientation]:
    """
    Checks if a 2D NumPy array represents an L-shape.

    :param arr: The 2D NumPy array to check.
    :return: The orientation of the L-shape (e.g., "bottom right", "top left", etc. if it's an L-shape, otherwise None.
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
        # Determine orientation based on the corner
        if corner_row == max_row and corner_col == max_col:
            return "bottom_right"
        elif corner_row == max_row and corner_col == min_col:
            return "bottom_left"
        elif corner_row == min_row and corner_col == min_col:
            return "top_left"
        elif corner_row == min_row and corner_col == max_col:
            return "top_right"

    return None


def is_point_adjacent(point: np.ndarray, bboxes: np.ndarray) -> Optional[np.ndarray] | None:
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

    x, y = point

    # Check x-adjacency (point is one unit away horizontally and within vertical bounds)
    x_adjacent = ((x == bbox_max_x + 1) | (x == bbox_min_x - 1)) & \
                 (y >= bbox_min_y) & (y <= bbox_max_y)

    # Check y-adjacency (point is one unit away vertically and within horizontal bounds)
    y_adjacent = ((y == bbox_max_y + 1) | (y == bbox_min_y - 1)) & \
                 (x >= bbox_min_x) & (x <= bbox_max_x)

    adjacent = x_adjacent | y_adjacent
    matching_indices = np.where(adjacent)[0]

    return matching_indices if matching_indices.size > 0 else None
