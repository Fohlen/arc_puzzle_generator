import unittest

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_ninetytwo import puzzle_ninetytwo
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir


class PuzzleNinetyTwoTest(unittest.TestCase):
    def setUp(self):
        puzzle_path = test_dir / "data" / "16de56c4.json"
        self.puzzle = load_puzzle(puzzle_path)

    def test_generate_16de56c4(self):
        playground = puzzle_ninetytwo(
            self.puzzle.train[0].input,
            orientation="right",
        )
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_16de56c4_second(self):
        playground = puzzle_ninetytwo(
            self.puzzle.train[1].input,
            orientation="up",
        )
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))
