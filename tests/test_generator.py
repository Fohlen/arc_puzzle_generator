import unittest

import numpy as np

from src.arc_puzzle_generator.data_loader import load_puzzle
from src.arc_puzzle_generator.generators import generate_48d8fb45, orientation_to_unit_vector
from tests.utils import test_dir


class GeneratorTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "142ca369.json"
        self.puzzle = load_puzzle(file_path)

    def test_orientation_to_unit_vector(self):
        arr = np.array([4, 4])

        top_left = orientation_to_unit_vector("top_left") + arr
        self.assertTrue(np.array_equal(top_left, np.array([3, 3])))

        top_right = orientation_to_unit_vector("top_right") + arr
        self.assertTrue(np.array_equal(top_right, np.array([3, 5])))

        bottom_left = orientation_to_unit_vector("bottom_left") + arr
        self.assertTrue(np.array_equal(bottom_left, np.array([5, 3])))

        bottom_right = orientation_to_unit_vector("bottom_right") + arr
        self.assertTrue(np.array_equal(bottom_right, np.array([5, 5])))

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
