import unittest

import numpy as np

from abm.puzzles.puzzle_ten import puzzle_ten
from abm.utils.data_loader import load_puzzle
from tests.utils import test_dir


class PuzzleTenTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "195c6913.json"
        self.puzzle = load_puzzle(file_path)

    def test_generate_195c6913(self):
        simulation = puzzle_ten(self.puzzle.train[0].input)
        *_, output_grid = simulation.run()
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_195c6913_second(self):
        simulation = puzzle_ten(self.puzzle.train[1].input)
        *_, output_grid = simulation.run()
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_195c6913_third(self):
        simulation = puzzle_ten(self.puzzle.train[2].input)
        *_, output_grid = simulation.run()
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[2].output))

    def test_generate_195c6913_prompt(self):
        simulation = puzzle_ten(self.puzzle.test[0].input)
        *_, output_grid = simulation.run()
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))

    def test_generate_195c6913_prompt_second(self):
        simulation = puzzle_ten(self.puzzle.test[1].input)
        *_, output_grid = simulation.run()
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[1].output))
