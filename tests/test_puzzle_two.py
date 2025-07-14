import unittest

import numpy as np

from abm.puzzles.puzzle_two import puzzle_two
from abm.utils.data_loader import load_puzzle
from tests.utils import test_dir


class PuzzleTwoTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "3e6067c3.json"
        self.puzzle = load_puzzle(file_path)

    def test_generate_3e6067c3(self):
        simulator = puzzle_two(self.puzzle.train[0].input)
        *_, output_grid = simulator.run()
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_3e6067c3_second(self):
        simulator = puzzle_two(self.puzzle.train[1].input)
        *_, output_grid = simulator.run()
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_3e6067c3_third(self):
        simulator = puzzle_two(self.puzzle.train[2].input)
        *_, output_grid = simulator.run()
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[2].output))

    def test_generate_3e6067c3_prompt(self):
        simulator = puzzle_two(self.puzzle.test[0].input)
        *_, output_grid = simulator.run()
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))

    def test_generate_3e6067c3_prompt_second(self):
        simulator = puzzle_two(self.puzzle.test[1].input)
        *_, output_grid = simulator.run()
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[1].output))
