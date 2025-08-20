from typing import NamedTuple, Iterator, Mapping

from arc_puzzle_generator.geometry import PointSet, Point, Direction

ColorIterator = Iterator[int]


class AgentState(NamedTuple):
    """
    Represents the state of an agent at a specific step in the simulation.

    :param position: The current position of the agent as a set of points.
    :param direction: The current direction of the agent.
    :param color: The current color of the agent.
    :param charge: The current charge of the agent, which is either positive (running), 0 (terminated) or -1 (indefinite).
    """
    position: PointSet
    direction: Direction
    color: int
    charge: int


AgentStateMapping = Mapping[Point, AgentState]
"""
Maps a point to a tuple containing the agent's state at the current step.
"""
