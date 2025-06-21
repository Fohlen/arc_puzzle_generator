from abc import ABC, abstractmethod
from itertools import chain
from typing import Iterable, Iterator

import numpy as np

from arc_puzzle_generator.agent import Agent


class PuzzleGenerator(Iterator[np.ndarray], Iterable[np.ndarray], ABC):
    """
    A generator class will receive an input grid and yield one step at a time, until finished.
    """

    def __init__(self, input_grid: np.ndarray) -> None:
        self.input_grid = input_grid
        self.output_grid = input_grid.copy()
        self.iter = None

    @abstractmethod
    def setup(self) -> Iterable[Agent]:
        pass

    def __iter__(self, *args, **kwargs):
        self.iter = chain.from_iterable(self.setup())
        return self.iter

    def __next__(self) -> np.ndarray:
        if self.iter is None:
            raise StopIteration
        return next(self.iter)
