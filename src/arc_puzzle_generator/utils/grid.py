"""
The grid_utils module contains functionality of useful grid operations, such as sub-selection from a grid.
"""
import math

import numpy as np

from arc_puzzle_generator.geometry import PointSet, Point


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


def unmask(input_mask: np.ndarray) -> PointSet:
    """
    Convert a 2D mask of booleans to a set of points.

    :param input_mask: A 2D list where True indicates a point is present.
    :return: A set of points (x, y) where the mask is True.
    """

    indices = np.where(input_mask)

    return PointSet(list(zip(indices[0].tolist(), indices[1].tolist())))


def in_grid(point: Point, grid_size: Point) -> bool:
    """
    Checks if a point is within the bounds of a grid defined by its size.
    :param point: The point to check, represented as a tuple (x, y).
    :param grid_size: The size of the grid, represented as a tuple (width, height).
    :return: A boolean indicating whether the point is within the grid bounds.
    """
    return 0 <= point[0] < grid_size[0] and 0 <= point[1] < grid_size[1]
