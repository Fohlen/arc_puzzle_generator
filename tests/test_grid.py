from unittest import TestCase

import numpy as np

from arc_puzzle_generator.utils.grid import unmask
from arc_puzzle_generator.geometry import in_grid


class GridTestCase(TestCase):
    def test_unmask(self):
        test_grid = np.array([
            [False, True, False],
            [True, False, True],
            [False, True, False]
        ])
        expected = {(0, 1), (1, 0), (1, 2), (2, 1)}
        self.assertEqual(expected, unmask(test_grid))

    def test_in_grid(self):
        grid_size = (3, 3)
        point_inside = (1, 1)
        point_outside = (5, 5)

        self.assertTrue(in_grid(point_inside, grid_size))
        self.assertFalse(in_grid(point_outside, grid_size))
