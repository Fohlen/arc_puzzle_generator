from typing import Mapping, Literal

import numpy as np

Point = tuple[int, int]
"""
A point in 2D space represented as a tuple of (x, y) coordinates.
"""

ColorMapping = Mapping[Point, int]
"""
A mapping from points to colors, where each point is represented as a tuple of (x, y) coordinates and each color is represented as an integer.
"""

Axis = Literal["horizontal", "vertical", "diagonal"]
"""
The axis of a line.
"""


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


Direction = Literal["left", "right", "up", "down", "top_left", "top_right", "bottom_left", "bottom_right", "none"]
