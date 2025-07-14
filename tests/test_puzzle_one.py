import unittest

import numpy as np

from abm.puzzles.puzzle_one import puzzle_one
from abm.utils.data_loader import load_puzzle
from tests.utils import test_dir


class PuzzleOneTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "1ae2feb7.json"
        self.puzzle = load_puzzle(file_path)

    def test_generate_1ae2feb7(self):
        simulation = puzzle_one(self.puzzle.train[0].input)
        *_, output_grid = simulation.run()
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_1ae2feb7_second(self):
        simulation = puzzle_one(self.puzzle.train[1].input)
        *_, output_grid = simulation.run()
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_1ae2feb7_third(self):
        simulation = puzzle_one(self.puzzle.train[2].input)
        *_, output_grid = simulation.run()
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[2].output))

    def test_generate_1ae2feb7_prompt(self):
        simulation = puzzle_one(self.puzzle.test[0].input)
        *_, output_grid = simulation.run()
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))

    def test_generate_1ae2feb7_prompt_second(self):
        simulation = puzzle_one(self.puzzle.test[1].input)
        *_, output_grid = simulation.run()
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[1].output))

    def test_generate_1ae2feb7_prompt_third(self):
        simulation = puzzle_one(self.puzzle.test[2].input)
        *_, output_grid = simulation.run()
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[2].output))
