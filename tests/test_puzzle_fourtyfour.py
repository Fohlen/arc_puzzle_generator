from unittest import TestCase

import numpy as np

from arc_puzzle_generator.puzzles.puzzle_fourtyfour import puzzle_fourtyfour
from arc_puzzle_generator.utils.data_loader import load_puzzle
from tests.utils import test_dir


class PuzzleFourtyFourTestCase(TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "5961cc34.json"
        self.puzzle = load_puzzle(file_path)

    def test_generate_5961cc34(self):
        playground = puzzle_fourtyfour(self.puzzle.train[0].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_5961cc34_second(self):
        playground = puzzle_fourtyfour(self.puzzle.train[1].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_5961cc34_third(self):
        playground = puzzle_fourtyfour(self.puzzle.train[2].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[2].output))

    def test_generate_5961cc34_fourth(self):
        playground = puzzle_fourtyfour(self.puzzle.train[3].input)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[3].output))

    def test_generate_5961cc34_prompt(self):
        # NOTE: in the test puzzle an abstraction is introduced where direction instructions can be contained inside balloons, we will circumvent this
        input_grid = self.puzzle.test[0].input.copy()
        input_grid[(5, 8)] = 1
        input_grid[(6, 8)] = 1
        input_grid[(5, 9)] = 3
        input_grid[(6, 9)] = 3

        playground = puzzle_fourtyfour(input_grid)
        *_, output_grid = playground
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))
