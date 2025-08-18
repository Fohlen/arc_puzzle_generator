from unittest import TestCase

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_ninetyeight import puzzle_ninetyeight
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir


class PuzzleEightyNineTestCase(TestCase):
    def setUp(self):
        puzzle_path = test_dir / "data" / "cb2d8a2c.json"
        self.puzzle = load_puzzle(puzzle_path)

    def test_generate_cb2d8a2c(self):
        playground = puzzle_ninetyeight(self.puzzle.train[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))
