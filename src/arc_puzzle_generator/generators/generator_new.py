from abc import ABC, abstractmethod
from itertools import chain
from typing import Iterable, Iterator

import numpy as np

from arc_puzzle_generator.generators.agent import Agent


class GeneratorNew(ABC):
    """
    A generator class will receive an input grid and yield one step at a time, until finished.
    """

    def __init__(self, input_grid: np.ndarray) -> None:
        self.input_grid = input_grid
        self.output_grid = input_grid.copy()
        self.iter = None

    def setup(self) -> Iterable[Agent]:
        pass

    def __iter__(self, *args, **kwargs):
        self.iter = chain.from_iterable(self.setup())
        return self.iter

    def __next__(self) -> np.ndarray:
        return next(self.iter)
