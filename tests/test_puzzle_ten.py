import unittest

import numpy as np

from arc_puzzle_generator.data_loader import load_puzzle
from arc_puzzle_generator.generators.puzzle_ten import PuzzleTenGenerator
from tests.utils import test_dir


class PuzzleTenTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "195c6913.json"
        self.puzzle = load_puzzle(file_path)

    def test_generate_195c6913(self):
        generator = PuzzleTenGenerator(self.puzzle.train[0].input)
        generator.setup()
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_195c6913_second(self):
        generator = PuzzleTenGenerator(self.puzzle.train[1].input)
        generator.setup()
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_195c6913_third(self):
        generator = PuzzleTenGenerator(self.puzzle.train[2].input)
        generator.setup()
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[2].output))

    def test_generate_195c6913_prompt(self):
        generator = PuzzleTenGenerator(self.puzzle.test[0].input)
        generator.setup()
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))

    def test_generate_195c6913_prompt_second(self):
        generator = PuzzleTenGenerator(self.puzzle.test[1].input)
        generator.setup()
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[1].output))
