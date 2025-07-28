import logging
import unittest

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_twentyfive import puzzle_twentyfive
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir

logging.basicConfig(level=logging.DEBUG)


class PuzzleTwentyFiveTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "332f06d7.json"
        self.puzzle = load_puzzle(file_path)

    def test_generate_332f06d7(self):
        playground = puzzle_twentyfive(self.puzzle.train[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_332f06d7_second(self):
        playground = puzzle_twentyfive(self.puzzle.train[1].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_332f06d7_third(self):
        playground = puzzle_twentyfive(self.puzzle.train[2].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[2].output))

    def test_generate_332f06d7_fourth(self):
        playground = puzzle_twentyfive(self.puzzle.train[3].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[3].output))

    def test_generate_332f06d7_prompt(self):
        playground = puzzle_twentyfive(self.puzzle.test[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))
