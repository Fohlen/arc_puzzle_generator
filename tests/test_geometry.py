from unittest import TestCase

import numpy as np

from abm.geometry import unmask, mask


class GeometryTestCase(TestCase):
    def test_unmask(self):
        test_grid = np.array([
            [False, True, False],
            [True, False, True],
            [False, True, False]
        ])
        expected = {(0, 1), (1, 0), (1, 2), (2, 1)}
        self.assertEqual(expected, unmask(test_grid))

    def test_mask(self):
        grid_size = (3, 3)
        point_set = {(0, 0)}

        masked = mask(point_set, grid_size)
        self.assertEqual(grid_size, masked.shape)
        self.assertEqual(1, np.sum(masked))
        self.assertTrue(masked[0, 0])
