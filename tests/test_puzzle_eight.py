import unittest
from unittest import TestCase

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_eight import puzzle_eight
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir


class PuzzleEightTestCase(TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "13e47133.json"
        self.puzzle = load_puzzle(file_path)

    #@unittest.skip("Skipping test_generate_13e47133 as it is not implemented yet.")
    def test_generate_13e47133(self):
        model = puzzle_eight(self.puzzle.train[0].input)
        *_, output_grid = model
        # output_grid = output_grid[1:-1, 1:-1]
        self.assertTrue(np.array_equal(self.puzzle.train[0].output, output_grid))
