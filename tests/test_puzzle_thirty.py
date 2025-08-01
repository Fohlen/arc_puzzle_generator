import logging
import unittest

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_thirty import puzzle_thirty
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir

logging.basicConfig(level=logging.DEBUG)


class PuzzleThirtyTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "3dc255db.json"
        self.puzzle = load_puzzle(file_path)

    def test_generate_3dc255db(self):
        playground = puzzle_thirty(
            self.puzzle.train[0].input, directions=iter(["right", "left"])
        )
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_3dc255db_second(self):
        playground = puzzle_thirty(self.puzzle.train[1].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_3dc255db_third(self):
        playground = puzzle_thirty(self.puzzle.train[2].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[2].output))

    def test_generate_3dc255db_prompt(self):
        playground = puzzle_thirty(self.puzzle.test[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))
