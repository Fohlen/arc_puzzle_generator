import unittest

import numpy as np

from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.physics import collision_axis, direction_to_numpy_unit_vector, box_distance, \
    relative_box_direction, \
    starting_point, box_contained, extreme_point


class PhysicsTestCase(unittest.TestCase):
    def test_direction_to_unit_vector(self):
        arr = np.array([4, 4])

        top_left = direction_to_numpy_unit_vector("top_left") + arr
        self.assertTrue(np.array_equal(top_left, np.array([3, 3])))

        top_right = direction_to_numpy_unit_vector("top_right") + arr
        self.assertTrue(np.array_equal(top_right, np.array([3, 5])))

        bottom_left = direction_to_numpy_unit_vector("bottom_left") + arr
        self.assertTrue(np.array_equal(bottom_left, np.array([5, 3])))

        bottom_right = direction_to_numpy_unit_vector("bottom_right") + arr
        self.assertTrue(np.array_equal(bottom_right, np.array([5, 5])))

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

    def test_relative_box_direction(self):
        point_a = np.array([[5, 3], [5, 3], [5, 3], [5, 3]])
        point_b = np.array([[5, 2], [5, 2], [5, 2], [5, 2]])
        point_c = np.array([[4, 3], [4, 3], [4, 3], [4, 3]])
        point_d = np.array([[4, 2], [4, 2], [4, 2], [4, 2]])

        self.assertEqual(relative_box_direction(point_a, point_b), "left")
        self.assertEqual(relative_box_direction(point_b, point_a), "right")
        self.assertEqual(relative_box_direction(point_a, point_c), "up")
        self.assertEqual(relative_box_direction(point_c, point_a), "down")
        self.assertEqual(relative_box_direction(point_c, point_b), "bottom_left")
        self.assertEqual(relative_box_direction(point_b, point_c), "top_right")
        self.assertEqual(relative_box_direction(point_a, point_d), "top_left")
        self.assertEqual(relative_box_direction(point_d, point_a), "bottom_right")

    def test_starting_point(self):
        point_a = np.array([[5, 3], [5, 3], [5, 3], [5, 3]])

        step_a = np.array([[5, 3], [6, 3]])
        self.assertTrue(np.array_equal(
            starting_point(point_a, "left", 2),
            step_a
        ))

        self.assertTrue(np.array_equal(
            starting_point(point_a, "right", 2),
            step_a
        ))

        step_b = np.array([[5, 3], [5, 4]])
        self.assertTrue(np.array_equal(
            starting_point(point_a, "up", 2),
            step_b
        ))

        self.assertTrue(np.array_equal(
            starting_point(point_a, "down", 2),
            step_b
        ))

    def test_starting_point_diagonal(self):
        point_a = np.array([[5, 3], [5, 3], [5, 3], [5, 3]])

        step_c = np.array([[5, 3], [6, 4]])
        self.assertTrue(np.array_equal(
            starting_point(point_a, "top_right", 2),
            step_c
        ))

        step_d = np.array([[5, 3], [4, 4]])
        self.assertTrue(np.array_equal(
            starting_point(point_a, "bottom_right", 2),
            step_d
        ))

        step_e = np.array([[5, 3], [6, 4]])
        self.assertTrue(np.array_equal(
            starting_point(point_a, "bottom_left", 2),
            step_e
        ))

        step_f = np.array([[5, 3], [4, 4]])
        self.assertTrue(np.array_equal(
            starting_point(point_a, "top_left", 2),
            step_f
        ))

    def test_collision_axis(self):
        agent = PointSet({(0, 0)})
        orientation = collision_axis(agent)

        self.assertEqual("horizontal", orientation)

    def test_box_contained(self):
        box_a = np.array([
            [4, 0], [0, 0], [0, 4], [4, 4]
        ])

        box_b = np.array([
            [2, 2], [1, 2], [1, 3], [2, 3]
        ])

        box_c = np.array([
            [5, 5], [4, 5], [4, 6], [5, 6]
        ])

        self.assertTrue(box_contained(box_b, box_a))
        self.assertFalse(box_contained(box_a, box_b))
        self.assertFalse(box_contained(box_c, box_a))

    def test_extreme_point(self):
        mask = np.array([
            [False, False, True, False],
            [False, True, True, True],
            [False, False, True, False],
            [True, True, True, False],
        ])

        self.assertEqual((3, 0), extreme_point(mask, "left"))
        self.assertEqual((1, 3), extreme_point(mask, "right"))
        self.assertEqual((0, 2), extreme_point(mask, "up"))
        self.assertEqual((3, 0), extreme_point(mask, "down"))
