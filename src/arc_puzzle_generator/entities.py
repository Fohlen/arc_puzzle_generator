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
