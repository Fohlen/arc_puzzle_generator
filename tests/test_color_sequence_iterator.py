from unittest import TestCase

from abm.color_sequence_iterator import ColorSequenceIterator

class ColorSequenceIteratorTestCase(TestCase):

    def test_color_sequence_iterator(self):
        sequence = [(1, 5)]
        it = ColorSequenceIterator(sequence, background_color=0)

        colors = [next(it) for _ in range(11)]
        self.assertEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], colors)
