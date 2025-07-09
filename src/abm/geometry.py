import numpy as np

Point = tuple[int, int]
PointSet = set[Point]


def unmask(input_mask: np.ndarray) -> PointSet:
    """
    Convert a 2D mask of booleans to a set of points.

    :param input_mask: A 2D list where True indicates a point is present.
    :return: A set of points (x, y) where the mask is True.
    """

    return {(x, y) for x, y in zip(*np.where(input_mask))}


def mask(point_set: PointSet, grid_size: tuple[int, int]) -> np.ndarray:
    """
    Convert a set of points to a mask of booleans.
    :param point_set:
    :param grid_size:
    :return:
    """

    mask_array = np.full(grid_size, False)

    for x, y in point_set:
        if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
            mask_array[x, y] = True

    return mask_array
