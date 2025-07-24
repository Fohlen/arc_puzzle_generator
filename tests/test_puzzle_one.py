import unittest

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_one import puzzle_one
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir


class PuzzleOneTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "1ae2feb7.json"
        self.puzzle = load_puzzle(file_path)

    def test_generate_1ae2feb7(self):
        playground = puzzle_one(self.puzzle.train[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_1ae2feb7_second(self):
        playground = puzzle_one(self.puzzle.train[1].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_1ae2feb7_third(self):
        playground = puzzle_one(self.puzzle.train[2].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[2].output))

    def test_generate_1ae2feb7_prompt(self):
        playground = puzzle_one(self.puzzle.test[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))

    def test_generate_1ae2feb7_prompt_second(self):
        playground = puzzle_one(self.puzzle.test[1].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[1].output))

    def test_generate_1ae2feb7_prompt_third(self):
        playground = puzzle_one(self.puzzle.test[2].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[2].output))
