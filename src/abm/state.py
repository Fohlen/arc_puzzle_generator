from typing import NamedTuple, Iterator

from abm.geometry import PointSet
from abm.physics import Direction

ColorIterator = Iterator[int]


class AgentState(NamedTuple):
    position: PointSet
    direction: Direction
    colors: ColorIterator
    charge: int
