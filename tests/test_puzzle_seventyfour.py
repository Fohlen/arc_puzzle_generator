import logging
from unittest import TestCase

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_seventyfour import puzzle_seventyfour
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir

logging.basicConfig(level=logging.DEBUG)


class PuzzleSeventyFourTestCase(TestCase):
    def setUp(self):
        puzzle_path = test_dir / "data" / "8f3a5a89.json"
        self.puzzle = load_puzzle(puzzle_path)

    def test_generate_8f3a5a89(self):
        input_grid = self.puzzle.train[0].input.copy()
        input_grid[:, 7:] = 8
        playground = puzzle_seventyfour(input_grid)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_8f3a5a89_second(self):
        playground = puzzle_seventyfour(self.puzzle.train[1].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))
