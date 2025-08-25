from unittest import TestCase

from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.neighbourhood import moore_neighbours
from arc_puzzle_generator.selection import resolve_point_set_selectors_with_direction, resolve_cell_selection


class SelectorsTestCase(TestCase):
    def setUp(self) -> None:
        self.point = (1, 1)
        neighbourhood = moore_neighbours
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

    def test_cell_selector(self):
        points = PointSet([
            (1, 1), (1, 2),
            (2, 1), (2, 2)
        ])

        points_left = PointSet([(1, 1), (2, 1)])
        self.assertEqual(points_left, resolve_cell_selection(points, "left"))

        points_up = PointSet([(1, 1), (1, 2)])
        self.assertEqual(points_up, resolve_cell_selection(points, "up"))

        points_right = PointSet([(1, 2), (2, 2)])
        self.assertEqual(points_right, resolve_cell_selection(points, "right"))

        points_down = PointSet([(2, 1), (2, 2)])
        self.assertEqual(points_down, resolve_cell_selection(points, "down"))

        points_bottom_left = PointSet([(2, 1)])
        self.assertEqual(points_bottom_left, resolve_cell_selection(points, "bottom_left"))

        points_top_left = PointSet([(1, 1)])
        self.assertEqual(points_top_left, resolve_cell_selection(points, "top_left"))

        points_top_right = PointSet([(1, 2)])
        self.assertEqual(points_top_right, resolve_cell_selection(points, "top_right"))

        points_bottom_right = PointSet([(2, 2)])
        self.assertEqual(points_bottom_right, resolve_cell_selection(points, "bottom_right"))
