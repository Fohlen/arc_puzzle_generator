from typing import Protocol

from abm.geometry import PointSet
from abm.state import AgentState


class Action(Protocol):
    """
    An action is a callable that takes the current position and state of an agent and returns a new state.
    """

    def __call__(self, state: AgentState, collision: PointSet) -> AgentState:
        pass
