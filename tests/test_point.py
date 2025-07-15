from unittest import TestCase

from arc_puzzle_generator.geometry import Point, PointSet


class PointTestCase(TestCase):
    def test_point(self):
        point = (1, 2)
        self.assertEqual(1, point[0])
        self.assertEqual(2, point[1])
        self.assertEqual(2, len(point))

    def test_point_set(self):
        point_set = PointSet([(1, 2), (3, 4)])
        self.assertIn((1, 2), point_set)
        self.assertIn((3, 4), point_set)
        self.assertNotIn((5, 6), point_set)
        self.assertEqual(2, len(point_set))

        updated = point_set.shift((1, 1))
        self.assertIn((2, 3), updated)
        self.assertIn((4, 5), updated)
        self.assertNotIn((5, 6), updated)
        self.assertEqual(2, len(point_set))
