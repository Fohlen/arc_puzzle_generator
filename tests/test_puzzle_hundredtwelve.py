import unittest

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_hundredtwelve import puzzle_hundredtwelve
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir


class PuzzleHundredTwelveTest(unittest.TestCase):
    def setUp(self):
        puzzle_path = test_dir / "data" / "e376de54.json"
        self.puzzle = load_puzzle(puzzle_path)

    def test_generate_e376de54(self):
        playground = puzzle_hundredtwelve(self.puzzle.train[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))
