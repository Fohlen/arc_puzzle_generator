from unittest import TestCase

from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.neighbourhood import resolve_point_set_neighbourhood, VonNeumannNeighbourhood, \
    MooreNeighbourhood, moore_neighbours, von_neumann_neighbours


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

    def test_neighbourhood_size(self):
        point = (10, 10)
        neighbourhood_a = VonNeumannNeighbourhood()
        self.assertEqual(4, len(neighbourhood_a(point)))

        neighbourhood_b = VonNeumannNeighbourhood(size=2)
        self.assertEqual(8, len(neighbourhood_b(point)))

        neighbourhood_c = MooreNeighbourhood(size=1)
        self.assertEqual(8, len(neighbourhood_c(point)))

        neighbourhood_d = MooreNeighbourhood(size=2)
        self.assertEqual(24, len(neighbourhood_d(point)))

    def test_point_set_neighbours(self):
        point_set = PointSet({(1, 1), (1, 2)})
        expected_van_neumann = {
            (0, 1), (0, 2),
            (1, 0), (1, 3),
            (2, 1), (2, 2)
        }

        self.assertEqual(expected_van_neumann, resolve_point_set_neighbourhood(point_set, von_neumann_neighbours))
