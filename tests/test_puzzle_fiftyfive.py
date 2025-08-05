import unittest

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_fiftyfive import puzzle_fiftyfive
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir


class PuzzleFiftyFiveTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "7666fa5d.json"
        self.puzzle = load_puzzle(file_path)

    def test_generate_7666fa5d(self):
        playground = puzzle_fiftyfive(input_grid=self.puzzle.train[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_7666fa5d_second(self):
        playground = puzzle_fiftyfive(
            input_grid=self.puzzle.train[1].input,
        )
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_7666fa5d_prompt(self):
        playground = puzzle_fiftyfive(
            input_grid=self.puzzle.test[0].input,
        )
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))
