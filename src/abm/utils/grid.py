"""
The grid_utils module contains functionality of useful grid operations, such as sub-selection from a grid.
"""
import math

import numpy as np


def make_smallest_square_from_mask(original_matrix: np.ndarray, binary_mask: np.typing.ArrayLike) -> np.ndarray | None:
    """
    Extracts a region from an original matrix based on a binary mask and
    pads it with zeros to make it the smallest possible square array
    based on the total number of elements in the extracted region.

    :param original_matrix: The original 2D NumPy array.
    :param binary_mask: A 2D NumPy array of the same shape as original_matrix, with True (or 1) values indicating the region of interest.
    :return: The smallest square NumPy array containing the masked region, padded with zeros if necessary. Returns None if the mask is all False.
    """

    # Find the indices where the mask is True
    rows, cols = np.where(binary_mask)

    if rows.size == 0 or cols.size == 0:
        return None

    # Determine the bounding box
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # Extract the region of interest
    cropped_array = original_matrix[min_row:max_row + 1, min_col:max_col + 1]

    # Square the matrix
    rows_cropped, cols_cropped = cropped_array.shape
    num_elements = max(rows_cropped, cols_cropped) * max(rows_cropped, cols_cropped)
    side = math.ceil(math.sqrt(num_elements))

    squared_array = np.zeros((side, side), dtype=cropped_array.dtype)
    squared_array[:rows_cropped, :cols_cropped] = cropped_array
    return squared_array
