from unittest import TestCase

from arc_puzzle_generator.neighbourhood import moore_neighbours, von_neumann_neighbours
from arc_puzzle_generator.selection import up_selector, down_selector, left_selector, right_selector, \
    resolve_point_set_selectors, resolve_point_set_selectors_with_direction


class SelectorsTestCase(TestCase):
    def setUp(self) -> None:
        self.point = (1, 1)
        self.neighbourhood = moore_neighbours(self.point)

    def test_top_selector(self):
        expected = {(0, 1)}
        self.assertEqual(expected, up_selector(self.point, self.neighbourhood))

    def test_bottom_selector(self):
        expected = {(2, 1)}
        self.assertEqual(expected, down_selector(self.point, self.neighbourhood))

    def test_left_selector(self):
        expected = {(1, 0)}
        self.assertEqual(expected, left_selector(self.point, self.neighbourhood))

    def test_right_selector(self):
        expected = {(1, 2)}
        self.assertEqual(expected, right_selector(self.point, self.neighbourhood))

    def test_combined_selectors(self):
        combined = up_selector(self.point, self.neighbourhood) | down_selector(self.point, self.neighbourhood) | \
                   left_selector(self.point, self.neighbourhood) | right_selector(self.point, self.neighbourhood)
        neighbours = von_neumann_neighbours(self.point)
        self.assertEqual(neighbours, combined)

    def test_resolve_point_set_selectors(self):
        expected_top = {(0, 1), }
        self.assertEqual(expected_top, resolve_point_set_selectors({self.point}, self.neighbourhood, up_selector))

        expected_bottom = {(2, 1)}
        self.assertEqual(expected_bottom, resolve_point_set_selectors({self.point}, self.neighbourhood, down_selector))

    def test_direction_selection(self):
        expected_bottom_right = {(2, 2)}
        self.assertEqual(expected_bottom_right,
                         resolve_point_set_selectors_with_direction({self.point}, self.neighbourhood, "bottom_right"))
