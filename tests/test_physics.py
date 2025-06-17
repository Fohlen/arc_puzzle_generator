import unittest

import numpy as np

from arc_puzzle_generator.physics import direction_to_unit_vector, is_point_adjacent


class PhysicsTestCase(unittest.TestCase):
    def test_direction_to_unit_vector(self):
        arr = np.array([4, 4])

        top_left = direction_to_unit_vector("top_left") + arr
        self.assertTrue(np.array_equal(top_left, np.array([3, 3])))

        top_right = direction_to_unit_vector("top_right") + arr
        self.assertTrue(np.array_equal(top_right, np.array([3, 5])))

        bottom_left = direction_to_unit_vector("bottom_left") + arr
        self.assertTrue(np.array_equal(bottom_left, np.array([5, 3])))

        bottom_right = direction_to_unit_vector("bottom_right") + arr
        self.assertTrue(np.array_equal(bottom_right, np.array([5, 5])))

    def test_collisions(self):
        bboxes = np.array([[
            [4, 4], [4, 4], [4, 6], [4, 6]
        ]])

        # handle collisions on the left side
        self.assertIsNotNone(is_point_adjacent(np.array([[4, 3]]), bboxes))
        self.assertIsNone(is_point_adjacent(np.array([[4, 2]]), bboxes))

        # handle collision on the top
        self.assertIsNotNone(is_point_adjacent(np.array([[3, 4]]), bboxes))
        self.assertIsNotNone(is_point_adjacent(np.array([[3, 5]]), bboxes))
        self.assertIsNotNone(is_point_adjacent(np.array([[3, 6]]), bboxes))
        self.assertIsNone(is_point_adjacent(np.array([[2, 5]]), bboxes))

        # handle collisions on the right side
        self.assertIsNotNone(is_point_adjacent(np.array([[4, 7]]), bboxes))
        self.assertIsNone(is_point_adjacent(np.array([[4, 8]]), bboxes))

        # handle collisions on the bottom side
        self.assertIsNotNone(is_point_adjacent(np.array([[5, 4]]), bboxes))
        self.assertIsNotNone(is_point_adjacent(np.array([[5, 5]]), bboxes))
        self.assertIsNotNone(is_point_adjacent(np.array([[5, 6]]), bboxes))
        self.assertIsNone(is_point_adjacent(np.array([[6, 4]]), bboxes))

    def test_collisions_left_right(self):
        arr = np.array([[
            [4, 4], [3, 4], [3, 4], [4, 4]
        ]])

        self.assertIsNotNone(is_point_adjacent(np.array([[3, 3]]), arr))
        self.assertIsNone(is_point_adjacent(np.array([[4, 2]]), arr))
        self.assertIsNotNone(is_point_adjacent(np.array([[3, 5]]), arr))
        self.assertIsNone(is_point_adjacent(np.array([[4, 6]]), arr))
