import unittest

import numpy as np

from arc_puzzle_generator.data_loader import load_puzzle
from arc_puzzle_generator.generators.puzzle_four import PuzzleFourGenerator
from tests.utils import test_dir


class GeneratorTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "142ca369.json"
        self.puzzle = load_puzzle(file_path)

    def test_generate_48d8fb45(self):
        generator = PuzzleFourGenerator(self.puzzle.train[0].input)
        generator.setup()
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_48d8fb45_second(self):
        generator = PuzzleFourGenerator(self.puzzle.train[1].input)
        generator.setup()
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_48d8fb45_third(self):
        generator = PuzzleFourGenerator(self.puzzle.train[2].input)
        generator.setup()
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[2].output))

    def test_generate_48d8fb45_prompt(self):
        generator = PuzzleFourGenerator(self.puzzle.test[0].input)
        generator.setup()
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))
