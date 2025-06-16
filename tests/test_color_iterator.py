from unittest import TestCase

from arc_puzzle_generator.color_iterator import ColorIterator


class TestColorIterator(TestCase):
    def test_color_iterator(self):
        color_sequence = [(4, 2), (3, 3)]
        color_iterator = ColorIterator(color_sequence)

        colors = [color_iterator.__next__() for _ in range(10)]
        self.assertEqual(colors, [4, 0, 4, 3, 4, 0, 4, 0, 4, 3])
