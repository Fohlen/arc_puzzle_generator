import logging
from unittest import TestCase

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_ninetyeight import puzzle_ninetyeight
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir

logging.basicConfig(level=logging.DEBUG)


class PuzzleEightyNineTestCase(TestCase):
    def setUp(self):
        puzzle_path = test_dir / "data" / "cb2d8a2c.json"
        self.puzzle = load_puzzle(puzzle_path)

    def test_generate_cb2d8a2c(self):
        playground = puzzle_ninetyeight(self.puzzle.train[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_cb2d8a2c_second(self):
        playground = puzzle_ninetyeight(self.puzzle.train[1].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_cb2d8a2c_third(self):
        playground = puzzle_ninetyeight(self.puzzle.train[2].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[2].output))

    def test_generate_cb2d8a2c_fourth(self):
        playground = puzzle_ninetyeight(self.puzzle.train[3].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[3].output))

    def test_generate_cb2d8a2c_prompt(self):
        playground = puzzle_ninetyeight(self.puzzle.test[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))
