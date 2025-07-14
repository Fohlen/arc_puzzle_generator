from typing import NamedTuple, Iterator, Mapping

from abm.geometry import PointSet, Point
from abm.physics import Direction

ColorIterator = Iterator[int]


class AgentState(NamedTuple):
    position: PointSet
    direction: Direction
    color: int
    charge: int


AgentStateMapping = Mapping[Point, AgentState]
"""
Maps a point to a tuple containing the agent's state at the current step.
"""
