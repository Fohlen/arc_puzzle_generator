import logging
from unittest import TestCase

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_twentyseven import puzzle_twentyseven
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir

logging.basicConfig(level=logging.DEBUG)


class PuzzleTwentySevenTestCase(TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "36a08778.json"
        self.puzzle = load_puzzle(file_path)

    def test_gest_generate_36a08778(self):
        playground = puzzle_twentyseven(self.puzzle.train[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_36a08778_second(self):
        playground = puzzle_twentyseven(self.puzzle.train[1].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_36a08778_third(self):
        playground = puzzle_twentyseven(self.puzzle.train[2].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[2].output))

    def test_generate_36a08778_fourth(self):
        playground = puzzle_twentyseven(self.puzzle.train[3].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[3].output))

    def test_generate_36a08778_fifth(self):
        playground = puzzle_twentyseven(self.puzzle.train[4].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[4].output))

    def test_generate_36a08778_sixth(self):
        playground = puzzle_twentyseven(self.puzzle.train[5].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[5].output))

    def test_generate_36a08778_prompt(self):
        playground = puzzle_twentyseven(self.puzzle.test[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))

    def test_generate_36a08778_prompt_second(self):
        playground = puzzle_twentyseven(self.puzzle.test[1].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[1].output))
