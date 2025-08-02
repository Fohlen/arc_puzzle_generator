from unittest import TestCase

from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.neighbourhood import resolve_point_set_neighbourhood, VonNeumannNeighbourhood, \
    MooreNeighbourhood


class NeighbourhoodTestCase(TestCase):
    def setUp(self):
        self.point = (1, 1)

    def test_von_neumann_neighbourhood(self):
        expected = {(0, 1), (1, 0), (1, 2), (2, 1)}
        neighbourhood = VonNeumannNeighbourhood()

        self.assertEqual(expected, neighbourhood(self.point))

    def test_moore_neighbourhood(self):
        expected = {
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 2),
            (2, 0), (2, 1), (2, 2)
        }
        neighbourhood = MooreNeighbourhood()

        self.assertEqual(expected, neighbourhood(self.point))

    def test_point_set_neighbours(self):
        point_set = PointSet({(1, 1), (1, 2)})
        expected_van_neumann = {
            (0, 1), (0, 2),
            (1, 0), (1, 3),
            (2, 1), (2, 2)
        }

        neighbourhood = VonNeumannNeighbourhood()
        self.assertEqual(expected_van_neumann, resolve_point_set_neighbourhood(point_set, neighbourhood))
