from unittest import TestCase

from arc_puzzle_generator.utils.color_sequence_iterator import ColorSequenceIterator

class ColorSequenceIteratorTestCase(TestCase):

    def test_color_iterator(self):
        color_sequence = [(4, 2), (3, 3)]
        color_iterator = ColorSequenceIterator(color_sequence)

        colors = [next(color_iterator) for _ in range(10)]
        self.assertEqual(colors, [4, 0, 4, 3, 4, 0, 4, 0, 4, 3])

    def test_color_iterator_complex(self):
        color_sequence = [(4, 2), (3, 3), (2, 1)]
        color_iterator = ColorSequenceIterator(color_sequence)
        colors = [next(color_iterator) for _ in range(10)]
        self.assertEqual(colors, [4, 2, 4, 3, 4, 2, 4, 2, 4, 3])

    def test_color_iterator_empty(self):
        color_sequence = []
        color_iterator = ColorSequenceIterator(color_sequence, background_color=5)
        color = next(color_iterator)
        self.assertEqual(color, 5)

    def test_color_sequence_iterator(self):
        sequence = [(1, 5)]
        it = ColorSequenceIterator(sequence, background_color=0)

        colors = [next(it) for _ in range(11)]
        self.assertEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], colors)
