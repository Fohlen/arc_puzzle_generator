import unittest

import numpy as np

from arc_puzzle_generator.data_loader import load_puzzle
from arc_puzzle_generator.entities import find_connected_objects, is_l_shape, find_colors, is_point_adjacent
from tests.utils import test_dir


class EntityTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "142ca369.json"
        self.puzzle = load_puzzle(file_path)

    def test_colors(self):
        colors = find_colors(self.puzzle.train[0].input)
        self.assertEqual(len(colors), 4)

    def test_find_connected_objects(self):
        target_mask = self.puzzle.train[0].input == 2
        label_mask, bounding_boxes, object_count = find_connected_objects(target_mask)
        self.assertEqual(object_count, 2)

        # Check that we have the correct number of bounding boxes
        self.assertEqual(bounding_boxes.shape, (2, 4, 2))

        # Check that each bounding box has 4 points (bottom-left, top-left, top-right, bottom-right)
        for i in range(object_count):
            # Verify bottom-left and top-left have same column
            self.assertEqual(bounding_boxes[i, 0, 1], bounding_boxes[i, 1, 1])

            # Verify top-left and top-right have same row
            self.assertEqual(bounding_boxes[i, 1, 0], bounding_boxes[i, 2, 0])

            # Verify top-right and bottom-right have same column
            self.assertEqual(bounding_boxes[i, 2, 1], bounding_boxes[i, 3, 1])

            # Verify bottom-right and bottom-left have same row
            self.assertEqual(bounding_boxes[i, 3, 0], bounding_boxes[i, 0, 0])

        # Manually verify red L-shape in example puzzle
        bbox = np.array([[4, 4], [3, 4], [3, 5], [4, 5]])
        self.assertTrue(np.array_equal(bounding_boxes[0], bbox))

    def test_is_l_shape(self):
        # Example usage:
        array1 = np.array([[0, 1], [1, 1]])
        array2 = np.array([[1, 0], [1, 1]])
        array3 = np.array([[1, 1], [0, 1]])
        array4 = np.array([[1, 1], [1, 0]])
        array5 = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0]])
        array6 = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 1]])
        array7 = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 1]])
        array8 = np.array([[1, 1, 0], [1, 0, 0], [1, 0, 0]])
        array9 = np.array([[1, 1], [1, 1]])  # Not an L-shape
        array10 = np.array([[1, 0, 1], [1, 1, 1]])  # Not an L-shape
        array11 = np.array([[1, 1, 1]])  # Not an L-shape
        array12 = np.array([[1]])  # Not an L-shape
        array13 = np.array([[4, 4, 4], [0, 0, 0], [0, 0, 0]])  # Not an L-shape

        self.assertEqual(is_l_shape(array1), "bottom_right")
        self.assertEqual(is_l_shape(array2), "bottom_left")
        self.assertEqual(is_l_shape(array3), "top_right")
        self.assertEqual(is_l_shape(array4), "top_left")
        self.assertEqual(is_l_shape(array5), "bottom_left")
        self.assertEqual(is_l_shape(array6), "bottom_right")
        self.assertEqual(is_l_shape(array7), "top_right")
        self.assertEqual(is_l_shape(array8), "top_left")
        self.assertEqual(is_l_shape(array9), None)
        self.assertEqual(is_l_shape(array10), None)
        self.assertEqual(is_l_shape(array11), None)
        self.assertEqual(is_l_shape(array12), None)
        self.assertEqual(is_l_shape(array13), None)

    def test_collisions(self):
        bboxes = np.array([[
            [4, 4], [4, 4], [4, 6], [4, 6]
        ]])

        # handle collisions on the left side
        self.assertIsNotNone(is_point_adjacent(np.array([4, 3]), bboxes))
        self.assertIsNone(is_point_adjacent(np.array([4, 2]), bboxes))

        # handle collision on the top
        self.assertIsNotNone(is_point_adjacent(np.array([3, 4]), bboxes))
        self.assertIsNotNone(is_point_adjacent(np.array([3, 5]), bboxes))
        self.assertIsNotNone(is_point_adjacent(np.array([3, 6]), bboxes))
        self.assertIsNone(is_point_adjacent(np.array([2, 5]), bboxes))

        # handle collisions on the right side
        self.assertIsNotNone(is_point_adjacent(np.array([4, 7]), bboxes))
        self.assertIsNone(is_point_adjacent(np.array([4, 8]), bboxes))

        # handle collisions on the bottom side
        self.assertIsNotNone(is_point_adjacent(np.array([5, 4]), bboxes))
        self.assertIsNotNone(is_point_adjacent(np.array([5, 5]), bboxes))
        self.assertIsNotNone(is_point_adjacent(np.array([5, 6]), bboxes))
        self.assertIsNone(is_point_adjacent(np.array([6, 4]), bboxes))

    def test_collisions_left_right(self):
        arr = np.array([[
            [4, 4], [3, 4], [3, 4], [4, 4]
        ]])

        self.assertIsNotNone(is_point_adjacent(np.array([3, 3]), arr))
        self.assertIsNone(is_point_adjacent(np.array([4, 2]), arr))
        self.assertIsNotNone(is_point_adjacent(np.array([3, 5]), arr))
        self.assertIsNone(is_point_adjacent(np.array([4, 6]), arr))
