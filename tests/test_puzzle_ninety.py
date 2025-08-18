from unittest import TestCase

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_ninety import puzzle_ninety
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir


class PuzzleNinetyTestCase(TestCase):
    def setUp(self):
        puzzle_path = test_dir / "data" / "b5ca7ac4.json"
        self.puzzle = load_puzzle(puzzle_path)

    def test_generate_b5ca7ac4(self):
        playground = puzzle_ninety(self.puzzle.train[0].input)
        output_grid, *_ = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_b5ca7ac4_second(self):
        playground = puzzle_ninety(self.puzzle.train[1].input)
        output_grid, *_ = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_b5ca7ac4_third(self):
        playground = puzzle_ninety(self.puzzle.train[2].input)
        output_grid, *_ = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[2].output))

    def test_generate_b5ca7ac4_prompt(self):
        playground = puzzle_ninety(self.puzzle.test[0].input)
        output_grid, *_ = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))
