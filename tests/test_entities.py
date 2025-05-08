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

        print(f"Array 1: {array1}, Orientation: {is_l_shape(array1)}")
        print(f"Array 2: {array2}, Orientation: {is_l_shape(array2)}")
        print(f"Array 3: {array3}, Orientation: {is_l_shape(array3)}")
        print(f"Array 4: {array4}, Orientation: {is_l_shape(array4)}")
        print(f"Array 5: {array5}, Orientation: {is_l_shape(array5)}")
        print(f"Array 6: {array6}, Orientation: {is_l_shape(array6)}")
        print(f"Array 7: {array7}, Orientation: {is_l_shape(array7)}")
        print(f"Array 8: {array8}, Orientation: {is_l_shape(array8)}")
        print(f"Array 9: {array9}, Orientation: {is_l_shape(array9)}")
        print(f"Array 10: {array10}, Orientation: {is_l_shape(array10)}")
