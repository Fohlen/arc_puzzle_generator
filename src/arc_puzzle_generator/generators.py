import math

import numpy as np

from src.arc_puzzle_generator.entities import find_colors, find_connected_objects, is_l_shape, Orientation, \
    is_point_adjacent


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
    Returns the starting point of a structure with given bounding box and orientation.
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


def generate_48d8fb45(input_grid: np.ndarray) -> np.ndarray:
    output_grid = input_grid.copy()
    colors = find_colors(input_grid)

    l_shapes = []
    blocks = []

    for target_color in colors:
        target_mask = input_grid == target_color
        labeled_grid, bounding_box, num_objects = find_connected_objects(target_mask)

        for label in range(1, num_objects + 1):
            box = make_smallest_square_from_mask(output_grid, labeled_grid == label)

            orientation = is_l_shape(box)
            if orientation is not None:
                l_shapes.append((target_color, bounding_box[(label - 1), :], orientation))
            else:
                blocks.append((target_color, bounding_box[(label - 1), :]))

    bboxes = np.array([bbox for _, bbox in blocks])

    for color, bbox, orientation in l_shapes:
        current_color = color
        step = orientation_to_unit_vector(orientation) + starting_point(bbox, orientation)

        while input_grid.shape[0] > step[0] > -1 < step[1] < input_grid.shape[1]:
            colliding_blocks = is_point_adjacent(step, bboxes)

            if colliding_blocks is not None:
                # Determine if collision is horizontal by checking if the beam's x-coordinate
                # is adjacent to the block's vertical sides
                block_bbox = bboxes[colliding_blocks[0]]
                is_horizontal = (np.min(block_bbox[:, 0]) <= step[0] <= np.max(block_bbox[:, 0]))

                current_color = blocks[colliding_blocks[0]][0]
                orientation = orthogonal_orientation(orientation, is_horizontal)

            output_grid[step[0], step[1]] = current_color
            step += orientation_to_unit_vector(orientation)

    return output_grid
