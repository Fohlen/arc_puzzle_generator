from unittest import TestCase

from abm.neighbourhood import von_neumann_neighbours, moore_neighbours, IdentityPointSetNeighbourhood


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
        point_set = {(1, 1), (1, 2)}
        expected_van_moore = {
            (0, 1), (0, 2),
            (1, 0), (1, 3),
            (2, 1), (2, 2)
        }

        multi_point_neighbourhood = IdentityPointSetNeighbourhood(von_neumann_neighbours)

        self.assertEqual(expected_van_moore, multi_point_neighbourhood(point_set))
