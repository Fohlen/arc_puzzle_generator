from unittest import TestCase

from arc_puzzle_generator.geometry import in_grid


class GeometryTestCase(TestCase):
    def test_in_grid(self):
        grid_size = (3, 3)
        point_inside = (1, 1)
        point_outside = (5, 5)

        self.assertTrue(in_grid(point_inside, grid_size))
        self.assertFalse(in_grid(point_outside, grid_size))
