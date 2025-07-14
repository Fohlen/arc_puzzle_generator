from itertools import chain, cycle
from typing import Protocol, Optional, Mapping

from abm.geometry import PointSet, Point
from abm.physics import direction_to_unit_vector, collision_axis
from abm.direction import DirectionRule
from abm.state import AgentState, AgentStateMapping, ColorIterator

ActionResult = Optional[tuple[AgentState, ColorIterator]]


class Action(Protocol):
    """
    An action is a callable that takes the current position and state of an agent and returns a new state.
    """

    def __call__(
            self,
            state: AgentState,
            colors: ColorIterator,
            collision: PointSet,
            collision_map: AgentStateMapping
    ) -> ActionResult:
        pass


def identity_action(state: AgentState, *args) -> AgentState:
    """
    An identity action that returns the state unchanged.

    :param state: The current state of the agent.
    :return: The same state as the input.
    """

    return state


class DirectionAction(Action):
    """
    An action that changes the direction of an agent based on the given direction rule.
    """

    def __init__(self, direction_rule: DirectionRule) -> None:
        self.direction_rule = direction_rule

    def __call__(self, state: AgentState, colors: ColorIterator, collision: PointSet, *args) -> ActionResult:
        """
        Change the direction of the agent based on the direction rule.

        :param state: The current state of the agent.
        :param collision: The set of points that are in collision with the agent.
        :return: A new state with the updated direction.
        """

        if len(collision) == 0:
            new_direction = self.direction_rule(state.direction)
            new_position = state.position.shift(direction_to_unit_vector(new_direction))

            return AgentState(
                position=new_position,
                direction=new_direction,
                color=next(colors),
                charge=state.charge - 1 if state.charge > 0 else state.charge
            ), colors

        return None


class OutOfGridAction(Action):
    """
    An action that removes the agent if the next step is out of grid, by setting its charge to 0.
    """

    def __init__(self, grid_size: Point) -> None:
        self.grid_size = grid_size

    def __call__(self, state: AgentState, colors: ColorIterator, collision: PointSet, *args) -> ActionResult:
        """
        Remove the agent from the grid.

        :param state: The current state of the agent.
        :param collision: The set of points that are in collision with the agent.
        :return: None, indicating that the agent is removed from the grid.
        """

        next_position = state.position.shift(direction_to_unit_vector(state.direction))

        min_x = min(pos[0] for pos in next_position)
        max_x = max(pos[0] for pos in next_position)
        min_y = min(pos[1] for pos in next_position)
        max_y = max(pos[1] for pos in next_position)

        if min_x < 0 or max_x >= self.grid_size[0] or \
                min_y < 0 or max_y >= self.grid_size[1]:
            return AgentState(
                position=state.position,
                direction=state.direction,
                color=next(colors),
                charge=0  # Set charge to 0 to indicate removal
            ), colors

        return None


class CollisionDirectionAction(Action):
    """
    An action that handles collisions by applying a direction rule on collision.
    """

    def __init__(self, direction_rule: DirectionRule) -> None:
        self.direction_rule = direction_rule

    def __call__(
            self,
            state: AgentState,
            colors: ColorIterator,
            collision: PointSet,
            collision_mapping: AgentStateMapping
    ) -> ActionResult:
        """
        Handle the collision by returning the current state unchanged.

        :param state: The current state of the agent.
        :param collision: The set of points that are in collision with the agent.
        :param collision_mapping: The mapping between collision points and the agent's colors.
        :return: The same state as the input.
        """

        if len(collision) > 0:
            axis = collision_axis(collision)
            new_direction = self.direction_rule(state.direction, axis)
            new_position = state.position.shift(direction_to_unit_vector(new_direction))

            return AgentState(
                position=new_position,
                direction=new_direction,
                color=next(colors),
                charge=state.charge - 1 if state.charge > 0 else state.charge
            ), colors

        return None


def collision_color_mapping(
        state: AgentState,
        colors: ColorIterator,
        collision: PointSet,
        collision_mapping: AgentStateMapping
) -> ActionResult:
    """
    Handle the collision by updating the agent's colors based on the collision points.

    :param state: The current state of the agent.
    :param collision: The set of points that are in collision with the agent.
    :param collision_mapping: The mapping between collision points and the agent's colors.
    :return: A new state with updated colors.
    """

    if len(collision) > 0:
        new_colors = cycle([collision_mapping[collision][1] for collision in collision])
        return AgentState(
            position=state.position,
            direction=state.direction,
            color=next(new_colors),
            charge=state.charge
        ), new_colors

    return None


class TrappedCollisionAction(Action):
    """
    An action that terminates the agent if it is trapped in a collision.
    """

    def __init__(self, direction_rule: DirectionRule) -> None:
        self.direction_rule = direction_rule

    def __call__(
            self,
            state: AgentState,
            colors: ColorIterator,
            collision: PointSet,
            collision_mapping: AgentStateMapping
    ) -> ActionResult:
        """
        Terminate the agent if it is trapped in a collision.

        :param state: The current state of the agent.
        :param collision: The set of points that are in collision with the agent.
        :param collision_mapping: The mapping between collision points and the agent's colors.
        :return: Terminates the agent at the current state if trapped, otherwise returns None.
        """

        if len(collision) > 0:
            next_direction = self.direction_rule(state.direction)
            next_position = state.position.shift(direction_to_unit_vector(next_direction))

            next_collision = next_position & collision
            if len(next_collision) > 0:
                return AgentState(
                    position=state.position,
                    direction=state.direction,
                    color=next(colors),
                    charge=0  # Set charge to 0 to indicate termination
                ), colors
        return None


class CollisionBorderAction(Action):
    def __init__(self, border_color: int) -> None:
        self.border_color = border_color

    def __call__(
            self,
            state: AgentState,
            colors: ColorIterator,
            collision: PointSet,
            collision_mapping: AgentStateMapping,
    ) -> ActionResult:
        if len(collision) == 1:
            # If the agent collides with the border, change its color to the border color
            new_colors = chain([self.border_color], colors)

            return AgentState(
                position=PointSet(collision),
                direction=state.direction,
                color=next(new_colors),
                charge=state.charge
            ), new_colors

        return None


def backtrack_action(
        state: AgentState,
        colors: ColorIterator,
        collision: PointSet,
        collision_mapping: AgentStateMapping
) -> ActionResult:
    """
    Backtrack the agent to its previous position.

    :param state: The current state of the agent.
    :param collision: The set of points that are in collision with the agent.
    :param collision_mapping: The mapping between collision points and the agent's colors.
    :return: A new state with the position set to the previous position.
    """

    if any(point in collision for point in state.position):
        direction_vector = direction_to_unit_vector(state.direction)
        previous_position = state.position.shift((direction_vector[0] * -1, direction_vector[1] * -1))

        return AgentState(
            position=previous_position,
            direction=state.direction,
            color=next(colors),
            charge=state.charge
        ), colors

    return None
