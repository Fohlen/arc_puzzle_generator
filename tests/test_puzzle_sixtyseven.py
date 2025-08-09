from unittest import TestCase

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_sixtyseven import puzzle_sixtyseven
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir


class PuzzleSixtySevenTestCase(TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "88e364bc.json"
        self.puzzle = load_puzzle(file_path)

    def test_generate_88e364bc(self):
        playground = puzzle_sixtyseven(self.puzzle.train[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_88e364bc_second(self):
        playground = puzzle_sixtyseven(self.puzzle.train[1].input, directions=("bottom_right",))
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_88e364bc_third(self):
        playground = puzzle_sixtyseven(self.puzzle.train[2].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[2].output))

    def test_generate_88e364bc_prompt(self):
        playground = puzzle_sixtyseven(self.puzzle.test[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))

    def test_generate_88e364bc_prompt_second(self):
        playground = puzzle_sixtyseven(self.puzzle.test[1].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[1].output))
