import unittest
from unittest import TestCase

import numpy as np

from arc_puzzle_generator.physics import direction_to_unit_vector, contained, box_distance


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

    def test_contained(self):
        points = np.array([
            [0, 0],
            [1, 1],
            [10, 0]
        ])

        box = np.array([[
            [5, 0], [0, 0],
            [0, 5], [5, 5]
        ]])

        collisions = contained(points, box)
        self.assertTrue(np.any(collisions))
        self.assertFalse(np.all(collisions))

    def test_not_contained(self):
        point = np.array([[15, 20]])

        box = np.array([[
            [5, 0], [0, 0],
            [0, 5], [5, 5]
        ]])

        collisions = contained(point, box)
        self.assertFalse(np.any(collisions))

    def test_box_distance(self):
        point_a = np.array([[5, 3], [5, 3], [5, 3], [5, 3]])
        point_b = np.array([[5, 2], [5, 2], [5, 2], [5, 2]])
        point_c = np.array([[5, 4], [5, 4], [5, 4], [5, 4]])
        point_d = np.array([[4, 4], [4, 4], [4, 4], [4, 4]])
        point_e = np.array([[4, 2], [4, 2], [4, 2], [4, 2]])

        self.assertEqual(box_distance(point_a, point_b, "left"), 1)
        self.assertEqual(box_distance(point_b, point_c, "right"), 2)
        self.assertEqual(box_distance(point_c, point_d, "down"), 1)
        self.assertEqual(box_distance(point_d, point_c, "up"), 1)
        self.assertEqual(box_distance(point_a, point_d, "top_right"), 1)
        self.assertEqual(box_distance(point_d, point_a, "bottom_left"), 1)
        self.assertEqual(box_distance(point_a, point_e, "top_left"), 1)
        self.assertEqual(box_distance(point_e, point_a, "bottom_right"), 1)
