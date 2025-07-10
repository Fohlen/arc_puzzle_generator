from typing import Protocol

from abm.geometry import PointSet
from abm.physics import Direction, direction_to_unit_vector


class Transition(Protocol):
    def __call__(
            self,
            position: PointSet,
            neighbourhood: PointSet,
            *args,
            **kwargs
    ) -> PointSet:
        pass


class DirectionTransition(Transition):
    def __init__(self, direction: Direction) -> None:
        self.direction = direction_to_unit_vector(direction)

    def __call__(
            self,
            position: PointSet,
            neighbourhood: PointSet,
            *args,
            **kwargs
    ) -> PointSet:
        next_position = {
            (x + self.direction[0], y + self.direction[1])
            for x, y in position
        }

        return next_position


def collision_transition(
        position: PointSet,
        neighbourhood: PointSet,
        *args,
        **kwargs
) -> PointSet:
    """
    A transition that stops the agent if a given position collides with the .
    :param position: The current position of the agent.
    :param neighbourhood: The neighbourhood of the agent.
    :return: The same position as input.
    """
    return neighbourhood


def identity_transition(
        position: PointSet,
        neighbourhood: PointSet,
        *args,
        **kwargs
) -> PointSet:
    """
    A transition that does not change the position.
    :param position: The current position of the agent.
    :param neighbourhood: The neighbourhood of the agent.
    :return: The same position as input.
    """
    return position
