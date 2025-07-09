from unittest import TestCase

from abm.neighbourhood import moore_neighbours
from abm.selectors import top_selector, bottom_selector, left_selector, right_selector


class SelectorsTestCase(TestCase):
    def setUp(self) -> None:
        self.point = (1, 1)
        self.neighbourhood = moore_neighbours(self.point)

    def test_top_selector(self):
        expected = {(0, 0), (0, 1), (0, 2)}
        self.assertEqual(expected, top_selector(self.point, self.neighbourhood))

    def test_bottom_selector(self):
        expected = {(2, 0), (2, 1), (2, 2)}
        self.assertEqual(expected, bottom_selector(self.point, self.neighbourhood))

    def test_left_selector(self):
        expected = {(0, 0), (1, 0), (2, 0)}
        self.assertEqual(expected, left_selector(self.point, self.neighbourhood))

    def test_right_selector(self):
        expected = {(0, 2), (1, 2), (2, 2)}
        self.assertEqual(expected, right_selector(self.point, self.neighbourhood))

    def test_combined_selectors(self):
        combined = top_selector(self.point, self.neighbourhood) | bottom_selector(self.point, self.neighbourhood) | \
            left_selector(self.point, self.neighbourhood) | right_selector(self.point, self.neighbourhood)
        self.assertEqual(self.neighbourhood, combined)
