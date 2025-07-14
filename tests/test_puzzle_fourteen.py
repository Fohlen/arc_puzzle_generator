import unittest

import numpy as np

from abm.data_loader import load_puzzle
from arc_puzzle_generator.generators import PuzzleFourteenPuzzleGenerator
from tests.utils import test_dir


class PuzzleFourteenTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "221dfab4.json"
        self.puzzle = load_puzzle(file_path)

    def test_generate_221dfab4(self):
        generator = PuzzleFourteenPuzzleGenerator(self.puzzle.train[0].input)
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_221dfab4_second(self):
        generator = PuzzleFourteenPuzzleGenerator(self.puzzle.train[1].input)
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_221dfab4_prompt(self):
        generator = PuzzleFourteenPuzzleGenerator(self.puzzle.test[0].input)
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))

    def test_generate_221dfab4_prompt_second(self):
        generator = PuzzleFourteenPuzzleGenerator(self.puzzle.test[1].input)
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[1].output))
