from typing import Callable, NamedTuple, Iterable

import numpy as np

from abm.model import Model


class Simulation(NamedTuple):
    model: Model
    num_iterations: int

    def run(self) -> Iterable[np.ndarray]:
        if self.num_iterations > 0:
            for _ in range(self.num_iterations):
                self.model.step()
        else:
            while self.model.active:
                self.model.step()

        steps = list(self.model)
        return steps


SimulationSetup = Callable[[np.ndarray], Simulation]
