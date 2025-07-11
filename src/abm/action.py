from typing import Protocol, Optional

from abm.geometry import PointSet, Point
from abm.physics import DirectionRule, direction_to_unit_vector
from abm.state import AgentState


class Action(Protocol):
    """
    An action is a callable that takes the current position and state of an agent and returns a new state.
    """

    def __call__(self, state: AgentState, collision: PointSet) -> Optional[AgentState]:
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
            charge=state.charge - 1 if state.charge > 0 else state.charge
        )


class OutOfGridAction(Action):
    """
    An action that removes the agent if the next step is out of grid, by setting its charge to 0.
    """

    def __init__(self, direction_rule: DirectionRule, grid_size: Point) -> None:
        self.direction_rule = direction_rule
        self.grid_size = grid_size

    def __call__(self, state: AgentState, collision: PointSet) -> Optional[AgentState]:
        """
        Remove the agent from the grid.

        :param state: The current state of the agent.
        :param collision: The set of points that are in collision with the agent.
        :return: None, indicating that the agent is removed from the grid.
        """

        new_direction = self.direction_rule(state.direction)
        new_position = state.position + direction_to_unit_vector(new_direction)

        min_x = min(pos[0] for pos in new_position)
        max_x = max(pos[0] for pos in new_position)
        min_y = min(pos[1] for pos in new_position)
        max_y = max(pos[1] for pos in new_position)

        if min_x < 0 or max_x >= self.grid_size[0] or \
                min_y < 0 or max_y >= self.grid_size[1]:
            return AgentState(
                position=state.position,
                direction=state.direction,
                colors=state.colors,
                charge=0  # Set charge to 0 to indicate removal
            )

        return None
