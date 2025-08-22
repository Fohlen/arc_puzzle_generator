import numpy as np

from arc_puzzle_generator.playground import Playground


def puzzle_hundredtwelve(input_grid: np.ndarray) -> Playground:
    """
    Puzzle 112 is a puzzle in which the middle strand determines the length of all other strands.
    :param input_grid: The input grid.
    :return: A Playground instance.
    """

    output_grid = input_grid.copy()
