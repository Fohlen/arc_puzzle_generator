import unittest

import numpy as np

from arc_puzzle_generator.data_loader import load_puzzle
from arc_puzzle_generator.generators.puzzle_one import PuzzleOneGenerator
from tests.utils import test_dir


class PuzzleOneTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "1ae2feb7.json"
        self.puzzle = load_puzzle(file_path)

    def test_generate_1ae2feb7(self):
        generator = PuzzleOneGenerator(self.puzzle.train[0].input)
        generator.setup()
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))


    def test_generate_1ae2feb7_second(self):
        generator = PuzzleOneGenerator(self.puzzle.train[1].input)
        generator.setup()
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_1ae2feb7_third(self):
        generator = PuzzleOneGenerator(self.puzzle.train[2].input)
        generator.setup()
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[2].output))


    def test_generate_1ae2feb7_prompt(self):
        generator = PuzzleOneGenerator(self.puzzle.test[0].input)
        generator.setup()
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))

    def test_generate_1ae2feb7_prompt_second(self):
        generator = PuzzleOneGenerator(self.puzzle.test[1].input)
        generator.setup()
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[1].output))

    def test_generate_1ae2feb7_prompt_third(self):
        generator = PuzzleOneGenerator(self.puzzle.test[2].input)
        generator.setup()
        *_, output_grid = generator
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[2].output))

