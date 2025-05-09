import unittest

import numpy as np

from src.arc_puzzle_generator.data_loader import load_puzzle
from src.arc_puzzle_generator.generators import generate_48d8fb45
from tests.utils import test_dir


class GeneratorTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "142ca369.json"
        self.puzzle = load_puzzle(file_path)

    def test_generate_48d8fb45(self):
        output_grid = generate_48d8fb45(self.puzzle.train[0].input)
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[0].output))

    def test_generate_48d8fb45_second(self):
        output_grid = generate_48d8fb45(self.puzzle.train[1].input)
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[1].output))

    def test_generate_48d8fb45_third(self):
        output_grid = generate_48d8fb45(self.puzzle.train[2].input)
        self.assertTrue(np.array_equal(output_grid, self.puzzle.train[2].output))

    def test_generate_48d8fb45_prompt(self):
        output_grid = generate_48d8fb45(self.puzzle.test[0].input)
        self.assertTrue(np.array_equal(output_grid, self.puzzle.test[0].output))
