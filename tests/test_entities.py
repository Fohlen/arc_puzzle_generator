import unittest

from src.arc_puzzle_generator.data_loader import load_puzzle
from src.arc_puzzle_generator.entities import find_num_colors, find_connected_objects
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
