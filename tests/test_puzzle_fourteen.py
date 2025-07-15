import unittest

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_fourteen import puzzle_fourteen
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir


class PuzzleFourteenTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "221dfab4.json"
        self.puzzle = load_puzzle(file_path)

    def test_generate_221dfab4(self):
        model = puzzle_fourteen(self.puzzle.train[0].input)
        *_, output_grid = model
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_221dfab4_second(self):
        model = puzzle_fourteen(self.puzzle.train[1].input)
        *_, output_grid = model
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_221dfab4_prompt(self):
        model = puzzle_fourteen(self.puzzle.test[0].input)
        *_, output_grid = model
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))

    def test_generate_221dfab4_prompt_second(self):
        model = puzzle_fourteen(self.puzzle.test[1].input)
        *_, output_grid = model
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[1].output))
