from itertools import chain
from typing import Protocol, Optional

from abm.geometry import PointSet, Point, ColorMapping
from abm.physics import DirectionRule, direction_to_unit_vector
from abm.state import AgentState


class Action(Protocol):
    """
    An action is a callable that takes the current position and state of an agent and returns a new state.
    """

    def __call__(
            self,
            state: AgentState,
            collision: PointSet,
            collision_map: Optional[ColorMapping] = None
    ) -> Optional[AgentState]:
        pass


def identity_action(state: AgentState, collision: PointSet, *args) -> AgentState:
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

    def __call__(self, state: AgentState, collision: PointSet, *args) -> AgentState:
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

    def __init__(self, grid_size: Point) -> None:
        self.grid_size = grid_size

    def __call__(self, state: AgentState, collision: PointSet, *args) -> Optional[AgentState]:
        """
        Remove the agent from the grid.

        :param state: The current state of the agent.
        :param collision: The set of points that are in collision with the agent.
        :return: None, indicating that the agent is removed from the grid.
        """

        next_position = state.position + direction_to_unit_vector(state.direction)

        min_x = min(pos[0] for pos in next_position)
        max_x = max(pos[0] for pos in next_position)
        min_y = min(pos[1] for pos in next_position)
        max_y = max(pos[1] for pos in next_position)

        if min_x < 0 or max_x >= self.grid_size[0] or \
                min_y < 0 or max_y >= self.grid_size[1]:
            return AgentState(
                position=state.position,
                direction=state.direction,
                colors=state.colors,
                charge=0  # Set charge to 0 to indicate removal
            )

        return None


class CollisionDirectionAction(Action):
    """
    An action that handles collisions by applying a direction rule on collision.
    """

    def __init__(self, direction_rule: DirectionRule) -> None:
        self.direction_rule = direction_rule

    def __call__(self, state: AgentState, collision: PointSet, *args) -> Optional[AgentState]:
        """
        Handle the collision by returning the current state unchanged.

        :param state: The current state of the agent.
        :param collision: The set of points that are in collision with the agent.
        :return: The same state as the input.
        """

        if len(collision) > 0:
            new_direction = self.direction_rule(state.direction)
            new_position = state.position + direction_to_unit_vector(new_direction)

            return AgentState(
                position=new_position,
                direction=new_direction,
                colors=state.colors,
                charge=state.charge - 1 if state.charge > 0 else state.charge
            )

        return None


def collision_color_mapping(
        state: AgentState,
        collision: PointSet,
        collision_mapping: ColorMapping
) -> Optional[AgentState]:
    """
    Handle the collision by updating the agent's colors based on the collision points.

    :param state: The current state of the agent.
    :param collision: The set of points that are in collision with the agent.
    :param collision_mapping: The mapping between collision points and the agent's colors.
    :return: A new state with updated colors.
    """

    if len(collision) > 0:
        new_colors = chain([collision_mapping[collision] for collision in collision], state.colors)
        return AgentState(
            position=state.position,
            direction=state.direction,
            colors=new_colors,
            charge=state.charge
        )

    return None

