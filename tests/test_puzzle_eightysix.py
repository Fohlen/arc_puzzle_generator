import logging
from unittest import TestCase

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_eightysix import puzzle_eightysix
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir

logging.basicConfig(level=logging.DEBUG)


class PuzzleEightySixTestCase(TestCase):
    def setUp(self):
        puzzle_path = test_dir / "data" / "aa4ec2a5.json"
        self.puzzle = load_puzzle(puzzle_path)

    def test_generate_aa4ec2a5(self):
        playground = puzzle_eightysix(self.puzzle.train[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_aa4ec2a5_second(self):
        playground = puzzle_eightysix(self.puzzle.train[1].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_aa4ec2a5_third(self):
        playground = puzzle_eightysix(self.puzzle.train[2].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[2].output))

    def test_generate_aa4ec2a5_prompt(self):
        playground = puzzle_eightysix(self.puzzle.test[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))

