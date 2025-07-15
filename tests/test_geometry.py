from unittest import TestCase

import numpy as np

from arc_puzzle_generator.geometry import unmask


class GeometryTestCase(TestCase):
    def test_unmask(self):
        test_grid = np.array([
            [False, True, False],
            [True, False, True],
            [False, True, False]
        ])
        expected = {(0, 1), (1, 0), (1, 2), (2, 1)}
        self.assertEqual(expected, unmask(test_grid))

