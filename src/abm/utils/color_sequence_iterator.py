from typing import Iterable, Iterator


class ColorSequenceIterator(Iterator[int], Iterable[int]):
    """
    Iterates over a sequence of colors in order until infinity.
    """

    def __init__(self, color_sequence: list[tuple[int, int]], background_color: int = 0) -> None:
        """
        Constructs a color iterator.
        :param color_sequence: A sequence of colors and their frequencies.
        :param background_color: The background color to fill instead.
        """
        self.background_color = background_color
        self.color_sequence = color_sequence
        self.index = 1

    def __iter__(self):
        return self

    def __next__(self) -> int:
        for color, count in self.color_sequence:
            if (self.index - 1) % count == 0:
                self.index += 1
                return color
        else:
            self.index += 1
            return self.background_color


def duplicate_first(iterable: Iterable[int]) -> Iterator[int]:
    """
    Returns an iterator that yields the first element of the iterable twice, followed by the rest of the elements.
    :param iterable: The iterable to duplicate the first element from.
    :return: A generator that yields the first element twice, then the rest of the elements.
    """
    it = iter(iterable)
    try:
        first = next(it)
        yield first
        yield first
    except StopIteration:
        return
    yield from it
