import unittest

from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir


class DataLoaderTestCase(unittest.TestCase):
    def test_load_puzzle(self):
        file_path = test_dir / "data" / "48d8fb45.json"
        puzzle = load_puzzle(file_path)

        self.assertEqual(len(puzzle.train), 3)
        self.assertEqual(len(puzzle.test), 1)
        self.assertEqual(puzzle.train[0].input.shape, (10, 10))
        self.assertEqual(puzzle.train[0].output.shape, (3, 3))
