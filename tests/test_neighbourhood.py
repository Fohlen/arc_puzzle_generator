from unittest import TestCase

from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.neighbourhood import von_neumann_neighbours, moore_neighbours, resolve_point_set_neighbourhood


class NeighbourhoodTestCase(TestCase):
    def setUp(self):
        self.point = (1, 1)

    def test_von_neumann_neighbourhood(self):
        expected = {(0, 1), (1, 0), (1, 2), (2, 1)}

        self.assertEqual(expected, von_neumann_neighbours(self.point))

    def test_moore_neighbourhood(self):
        expected = {
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 2),
            (2, 0), (2, 1), (2, 2)
        }

        self.assertEqual(expected, moore_neighbours(self.point))

    def test_point_set_neighbours(self):
        point_set = PointSet({(1, 1), (1, 2)})
        expected_van_neumann = {
            (0, 1), (0, 2),
            (1, 0), (1, 3),
            (2, 1), (2, 2)
        }


        self.assertEqual(expected_van_neumann, resolve_point_set_neighbourhood(point_set, von_neumann_neighbours))
