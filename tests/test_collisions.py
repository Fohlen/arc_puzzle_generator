from unittest import TestCase

import numpy as np

from abm.direction import identity_direction_rule
from arc_puzzle_generator.collisions import moore_neighbourhood, \
    AxisNeighbourHood


class CollisionTest(TestCase):
    def test_moore_neighbourhood(self):
        point = np.array([5, 3])
        neighbours = np.array([
            [4, 2], [4, 3], [4, 4],
            [5, 2], [5, 4],
            [6, 2], [6, 3], [6, 4]
        ])
        self.assertTrue(np.array_equal(moore_neighbourhood(point), neighbours))

    def test_identity_direction(self):
        self.assertEqual(identity_direction_rule("right"), "right")

    def test_axis_neighbours(self):
        step = np.array([[5, 3]])
        grid_size = (10, 10)
        axis_neighbourhood = AxisNeighbourHood(grid_size)
        neighbours = axis_neighbourhood(step, "up")
        self.assertEqual(neighbours.shape, (9, 2))
        self.assertEqual(neighbours[0, 0], 5)
        self.assertEqual(neighbours[8, 0], 5)
        self.assertEqual(neighbours[0, 1], 0)
        self.assertEqual(neighbours[8, 1], 9)

        neighbours_vertical = axis_neighbourhood(step, "right")
        self.assertEqual(neighbours_vertical.shape, (9, 2))
        self.assertEqual(neighbours_vertical[0, 1], 3)
        self.assertEqual(neighbours_vertical[8, 1], 3)
        self.assertEqual(neighbours_vertical[0, 0], 0)
        self.assertEqual(neighbours_vertical[8, 0], 9)
