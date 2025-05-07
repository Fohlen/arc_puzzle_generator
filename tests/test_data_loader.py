import os
import unittest
from pathlib import Path

from src.arc_puzzle_generator.data_loader import load_puzzle


class DataLoaderTestCase(unittest.TestCase):
    def test_load_puzzle(self):
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        file_path = script_dir / "data" / "48d8fb45.json"
        puzzle = load_puzzle(file_path)

        self.assertEqual(len(puzzle.train), 3)
        self.assertEqual(len(puzzle.test), 1)
        self.assertEqual(puzzle.train[0].input.shape, (10, 10))
        self.assertEqual(puzzle.train[0].output.shape, (3, 3))
