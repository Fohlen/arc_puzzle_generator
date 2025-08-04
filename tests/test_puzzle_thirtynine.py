import unittest

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_thirtynine import puzzle_thirtynine
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir


class PuzzleThirtyNineTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "53fb4810.json"
        self.puzzle = load_puzzle(file_path)

    def test_generate_53fb4810(self):
        playground = puzzle_thirtynine(input_grid=self.puzzle.train[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_53fb4810_second(self):
        playground = puzzle_thirtynine(
            input_grid=self.puzzle.train[1].input,
        )
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_53fb4810_prompt(self):
        playground = puzzle_thirtynine(
            input_grid=self.puzzle.test[0].input,
        )
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))
