from unittest import TestCase

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_sixtyfour import puzzle_sixtyfour
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir


class PuzzleSixtyFourTestCase(TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "80a900e0.json"
        self.puzzle = load_puzzle(file_path)

    def test_generate_80a900e0(self):
        playground = puzzle_sixtyfour(self.puzzle.train[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_80a900e0_second(self):
        playground = puzzle_sixtyfour(self.puzzle.train[1].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_80a900e1_prompt(self):
        playground = puzzle_sixtyfour(self.puzzle.test[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))
