import unittest

import numpy as np

from src.arc_puzzle_generator.data_loader import load_puzzle
from src.arc_puzzle_generator.entities import find_num_colors, find_connected_objects, is_l_shape
from tests.utils import test_dir


class EntityTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "142ca369.json"
        self.puzzle = load_puzzle(file_path)

    def test_colors(self):
        num_colors = find_num_colors(self.puzzle.train[0].input)
        self.assertEqual(num_colors, 5)

    def test_find_connected_objects(self):
        target_mask = self.puzzle.train[0].input == 2
        label_mask, object_count = find_connected_objects(target_mask)
        self.assertEqual(object_count, 2)

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
