from unittest import TestCase

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_hundredfive import puzzle_hundredfive
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir


class PuzzleHundredFiveTestCase(TestCase):
    def setUp(self):
        puzzle_path = test_dir / "data" / "db695cfb.json"
        self.puzzle = load_puzzle(puzzle_path)

    def test_generate_db695cfb(self):
        playground = puzzle_hundredfive(self.puzzle.train[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))
