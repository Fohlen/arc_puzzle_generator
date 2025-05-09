from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class Generator(ABC):
    """
    A generator class will receive an input grid and yield one step at a time, until finished.
    """

    def __init__(self, input_grid: np.ndarray) -> None:
        self.input_grid = input_grid
        self.output_grid = input_grid.copy()

    def setup(self) -> None:
        pass

    @abstractmethod
    def __iter__(self, *args, **kwargs) -> Iterable[np.ndarray]:
        pass
