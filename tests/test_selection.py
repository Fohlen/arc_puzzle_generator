from unittest import TestCase

from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.neighbourhood import moore_neighbours
from arc_puzzle_generator.selection import resolve_point_set_selectors_with_direction
from arc_puzzle_generator.neighbourhood import MooreNeighbourhood, VonNeumannNeighbourhood


class SelectorsTestCase(TestCase):
    def setUp(self) -> None:
        self.point = (1, 1)
        neighbourhood = MooreNeighbourhood()
        self.neighbourhood = neighbourhood(self.point)

    def test_top_selector(self):
        expected = {(0, 1)}
        self.assertEqual(
            expected,
            resolve_point_set_selectors_with_direction({self.point}, self.neighbourhood, "up")
        )

    def test_bottom_selector(self):
        expected = {(2, 1)}
        self.assertEqual(
            expected,
            resolve_point_set_selectors_with_direction({self.point}, self.neighbourhood, "down")
        )

    def test_left_selector(self):
        expected = {(1, 0)}
        self.assertEqual(
            expected,
            resolve_point_set_selectors_with_direction({self.point}, self.neighbourhood, "left")
        )

    def test_right_selector(self):
        expected = {(1, 2)}
        self.assertEqual(
            expected,
            resolve_point_set_selectors_with_direction({self.point}, self.neighbourhood, "right")
        )

    def test_bottom_right_selector(self):
        expected_bottom_right = {(2, 2)}
        self.assertEqual(
            expected_bottom_right,
            resolve_point_set_selectors_with_direction({self.point}, self.neighbourhood, "bottom_right")
        )
