from typing import NamedTuple

import numpy as np


# Entity = NamedTuple("Entity", [("color", int), ("orientation", np.ndarray), ("bounding_box", np.ndarray)])


def find_num_colors(grid: np.ndarray):
    """
    Returns the number of colors in the grid.
    :param grid: The input grid
    :return: integer number of colors
    """
    return np.unique(grid).size


def find_connected_objects(mask):
    """
    Find connected objects in a binary mask.
    :param mask: The input binary mask.
    :return: The labeled mask and the number of objects.
    """
    rows, cols = mask.shape
    labeled_mask = np.zeros_like(mask, dtype=int)
    object_count = 0

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols and mask[r, c] and labeled_mask[r, c] == 0

    def bfs(r, c, label):
        queue = [(r, c)]
        labeled_mask[r, c] = label
        while queue:
            row, col = queue.pop(0)
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

    return labeled_mask, object_count


def is_l_shape(arr):
    """
    Checks if a 2D NumPy array represents an L-shape.

    Args:
        arr (np.ndarray): The 2D NumPy array to check.

    Returns:
        str or None: The orientation of the L-shape (e.g., "bottom right", "top left", etc.)
                     if it's an L-shape, otherwise None.
    """
    rows, cols = arr.shape
    non_zero_indices = np.argwhere(arr != 0)
    num_non_zero = len(non_zero_indices)

    if num_non_zero == 0:
        return None  # Empty array is not an L-shape

    # Find the bounding box of the non-zero elements
    min_row, min_col = np.min(non_zero_indices, axis=0)
    max_row, max_col = np.max(non_zero_indices, axis=0)
    bbox_height = max_row - min_row + 1
    bbox_width = max_col - min_col + 1

    # An L-shape's bounding box will have an area that is one less
    # than the total number of cells in the bounding box if it were a rectangle.
    if bbox_height * bbox_width - 1 != num_non_zero:
        return None

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

    if arm1_len > 0 and arm2_len > 0 and arm1_len + arm2_len - 1 == num_non_zero:
        # Determine orientation based on the corner
        if corner_row == max_row and corner_col == max_col:
            return "bottom right"
        elif corner_row == max_row and corner_col == min_col:
            return "bottom left"
        elif corner_row == min_row and corner_col == min_col:
            return "top left"
        elif corner_row == min_row and corner_col == max_col:
            return "top right"

    return None
