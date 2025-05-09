import unittest

import numpy as np

from src.arc_puzzle_generator.physics import orientation_to_unit_vector


class PhysicsTestCase(unittest.TestCase):
    def test_orientation_to_unit_vector(self):
        arr = np.array([4, 4])

        top_left = orientation_to_unit_vector("top_left") + arr
        self.assertTrue(np.array_equal(top_left, np.array([3, 3])))

        top_right = orientation_to_unit_vector("top_right") + arr
        self.assertTrue(np.array_equal(top_right, np.array([3, 5])))

        bottom_left = orientation_to_unit_vector("bottom_left") + arr
        self.assertTrue(np.array_equal(bottom_left, np.array([5, 3])))

        bottom_right = orientation_to_unit_vector("bottom_right") + arr
        self.assertTrue(np.array_equal(bottom_right, np.array([5, 5])))
    pass
