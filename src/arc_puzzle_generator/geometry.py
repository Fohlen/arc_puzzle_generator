from typing import cast, Mapping

import numpy as np

Point = tuple[int, int]
ColorMapping = Mapping[Point, int]


class PointSet(set[tuple[int, int]]):
    """
    A set of points represented as tuples of (x, y) coordinates.
    This class provides methods to move the points by adding or subtracting another point.
    """

    def shift(self, other: 'Point') -> 'PointSet':
        positions = {
            (position[0] + other[0], position[1] + other[1])
            for position in self
        }
        return PointSet(positions)

    @staticmethod
    def from_numpy(array: np.ndarray) -> 'PointSet':
        """
        Convert a numpy array of points to a PointSet.

        :return: A PointSet containing the points from the numpy array.
        """
        return PointSet((x.item(), y.item()) for x, y in array)


def unmask(input_mask: np.ndarray) -> PointSet:
    """
    Convert a 2D mask of booleans to a set of points.

    :param input_mask: A 2D list where True indicates a point is present.
    :return: A set of points (x, y) where the mask is True.
    """

    indices = np.where(input_mask)

    return PointSet(list(zip(indices[0].tolist(), indices[1].tolist())))
