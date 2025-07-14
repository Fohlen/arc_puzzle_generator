from typing import NamedTuple, Iterator, Mapping

from abm.geometry import PointSet, Point
from abm.physics import Direction

ColorIterator = Iterator[int]


class AgentState(NamedTuple):
    position: PointSet
    direction: Direction
    colors: ColorIterator
    charge: int


AgentStateMapping = Mapping[Point, tuple[AgentState, int]]
"""
Maps a point to a tuple containing the agent's state and an integer representing the color of the grid at the current step.
"""
