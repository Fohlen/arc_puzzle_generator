from typing import Protocol

from abm.geometry import PointSet
from abm.physics import DirectionRule
from abm.state import AgentState
from arc_puzzle_generator.physics import direction_to_unit_vector


class Action(Protocol):
    """
    An action is a callable that takes the current position and state of an agent and returns a new state.
    """

    def __call__(self, state: AgentState, collision: PointSet) -> AgentState:
        pass


def identity_action(state: AgentState, collision: PointSet) -> AgentState:
    """
    An identity action that returns the state unchanged.

    :param state: The current state of the agent.
    :param collision: The set of points that are in collision with the agent.
    :return: The same state as the input.
    """

    return state


class DirectionAction(Action):
    """
    An action that changes the direction of an agent based on the given direction rule.
    """

    def __init__(self, direction_rule: DirectionRule) -> None:
        self.direction_rule = direction_rule

    def __call__(self, state: AgentState, collision: PointSet) -> AgentState:
        """
        Change the direction of the agent based on the direction rule.

        :param state: The current state of the agent.
        :param collision: The set of points that are in collision with the agent.
        :return: A new state with the updated direction.
        """

        new_direction = self.direction_rule(state.direction)
        new_position = state.position + direction_to_unit_vector(new_direction)

        return AgentState(
            position=new_position,
            direction=new_direction,
            colors=state.colors,
            charge=state.charge - 1
        )

