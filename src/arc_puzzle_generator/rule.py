import math
import random
from itertools import chain, cycle
from typing import Protocol, Optional, Sequence

from arc_puzzle_generator.direction import DirectionTransformer
from arc_puzzle_generator.geometry import PointSet, Point, Direction
from arc_puzzle_generator.physics import direction_to_unit_vector, collision_axis, relative_point_direction
from arc_puzzle_generator.selection import resolve_point_set_selectors_with_direction
from arc_puzzle_generator.state import AgentState, AgentStateMapping, ColorIterator

RuleResult = Optional[tuple[AgentState, ColorIterator]]


class Rule(Protocol):
    """
    A rule is a callable that takes the current position and state of an agent and returns a new state.
    """

    def __call__(
            self,
            states: Sequence[AgentState],
            colors: ColorIterator,
            collision: PointSet,
            collision_map: AgentStateMapping
    ) -> RuleResult:
        pass


class RuleNode:
    """
    A node in the rule chain that contains a rule and optional next and alternative nodes.
    """

    def __init__(
            self,
            rule: Rule,
            next_node: Optional['RuleNode'] = None,
            alternative_node: Optional['RuleNode'] = None
    ) -> None:
        """
        Initialize a rule node with a rule and an optional next node.

        :param rule: The rule to be performed.
        :param next_node: The next rule node in the chain.
        :param alternative_node: The alternative next rule node in the chain.
        """
        self.rule = rule
        self.next_node = next_node
        self.alternative_node = alternative_node


def identity_rule(states: Sequence[AgentState], colors: ColorIterator, *args) -> RuleResult:
    """
    An identity rule that returns the state unchanged.

    :param states: The current states of the agent.
    :param colors: An iterator over the agent's colors.
    :return: The same state as the input.
    """

    return states[-1], colors


class DirectionRule(Rule):
    """
    A rule that changes the direction of an agent based on the given direction rule.
    """

    def __init__(
            self,
            direction_rule: DirectionTransformer,
            select_direction: bool = False,
    ) -> None:
        self.direction_rule = direction_rule
        self.select_direction = select_direction

    def __call__(
            self,
            states: Sequence[AgentState],
            colors: ColorIterator,
            collision: PointSet,
            *args
    ) -> RuleResult:
        """
        Change the direction of the agent based on the direction rule.

        :param states: The current states of the agent.
        :param colors: An iterator over the agent's colors.
        :param collision: The set of points that are in collision with the agent.
        :return: A new state with the updated direction.
        """

        sub_collision = resolve_point_set_selectors_with_direction(
            states[-1].position, collision, states[-1].direction
        ) if self.select_direction else collision

        if len(sub_collision) == 0:
            new_direction = self.direction_rule(states[-1].direction)
            new_position = states[-1].position.shift(direction_to_unit_vector(new_direction))

            return AgentState(
                position=new_position,
                direction=new_direction,
                color=next(colors),
                charge=states[-1].charge - 1 if states[-1].charge > 0 else states[-1].charge,
                commit=states[-1].commit
            ), colors

        return None


class OutOfGridRule(Rule):
    """
    A rule that removes the agent if the next step is out of grid, by setting its charge to 0.
    """

    def __init__(self, grid_size: Point) -> None:
        self.grid_size = grid_size

    def __call__(self, states: Sequence[AgentState], colors: ColorIterator, collision: PointSet, *args) -> RuleResult:
        """
        Remove the agent from the grid.

        :param states: The current states of the agent.
        :param colors: An iterator over the agent's colors.
        :param collision: The set of points that are in collision with the agent.
        :return: None, indicating that the agent is removed from the grid.
        """

        next_position = states[-1].position.shift(direction_to_unit_vector(states[-1].direction))

        min_x = min(pos[0] for pos in next_position)
        max_x = max(pos[0] for pos in next_position)
        min_y = min(pos[1] for pos in next_position)
        max_y = max(pos[1] for pos in next_position)

        if min_x < 0 or max_x >= self.grid_size[0] or \
                min_y < 0 or max_y >= self.grid_size[1]:
            return AgentState(
                position=states[-1].position,
                direction=states[-1].direction,
                color=next(colors),
                charge=0,  # Set charge to 0 to indicate removal
                commit=states[-1].commit
            ), colors

        return None


class CollisionDirectionRule(Rule):
    """
    A rule that handles collisions by applying a direction rule on collision.
    """

    def __init__(
            self,
            direction_rule: DirectionTransformer,
            select_direction: bool = False,
    ) -> None:
        self.direction_rule = direction_rule
        self.select_direction = select_direction

    def __call__(
            self,
            states: Sequence[AgentState],
            colors: ColorIterator,
            collision: PointSet,
            collision_mapping: AgentStateMapping
    ) -> RuleResult:
        """
        Handle the collision by returning the current state unchanged.

        :param states: The current states of the agent.
        :param colors: An iterator over the agent's colors.
        :param collision: The set of points that are in collision with the agent.
        :param collision_mapping: The mapping between collision points and the agent's colors.
        :return: The same state as the input.
        """

        sub_collision = resolve_point_set_selectors_with_direction(
            states[-1].position, collision, states[-1].direction
        ) if self.select_direction else collision

        if len(sub_collision) > 0:
            axis = collision_axis(sub_collision)
            new_direction = self.direction_rule(states[-1].direction, axis)
            new_position = states[-1].position.shift(direction_to_unit_vector(new_direction))

            return AgentState(
                position=new_position,
                direction=new_direction,
                color=next(colors),
                charge=states[-1].charge - 1 if states[-1].charge > 0 else states[-1].charge,
                commit=states[-1].commit
            ), colors

        return None


def collision_color_mapping_rule(
        states: Sequence[AgentState],
        colors: ColorIterator,
        collision: PointSet,
        collision_mapping: AgentStateMapping
) -> RuleResult:
    """
    Handle the collision by updating the agent's colors based on the collision points.

    :param states: The current states of the agent.
    :param colors: An iterator over the agent's colors.
    :param collision: The set of points that are in collision with the agent.
    :param collision_mapping: The mapping between collision points and the agent's colors.
    :return: A new state with updated colors.
    """

    if len(collision) > 0:
        new_colors = cycle([collision_mapping[collision].color for collision in collision])

        return AgentState(
            position=states[-1].position,
            direction=states[-1].direction,
            color=next(new_colors),
            charge=states[-1].charge,
            commit=states[-1].commit
        ), new_colors

    return None


class TrappedCollisionRule(Rule):
    """
    A rule that terminates the agent if it is trapped in a collision.
    """

    def __init__(
            self,
            direction_rule: DirectionTransformer,
            select_direction: bool = False,
            num_directions: int = 1,
    ) -> None:
        self.direction_rule = direction_rule
        self.select_direction = select_direction
        self.num_directions = num_directions

    def __call__(
            self,
            states: Sequence[AgentState],
            colors: ColorIterator,
            collision: PointSet,
            collision_mapping: AgentStateMapping
    ) -> RuleResult:
        """
        Terminate the agent if it is trapped in a collision.

        :param states: The current states of the agent.
        :param colors: An iterator over the agent's colors.
        :param collision: The set of points that are in collision with the agent.
        :param collision_mapping: The mapping between collision points and the agent's colors.
        :return: Terminates the agent at the current state if trapped, otherwise returns None.
        """

        sub_collision = resolve_point_set_selectors_with_direction(
            states[-1].position, collision, states[-1].direction
        ) if self.select_direction else collision

        if len(sub_collision) > 0:
            previous_direction = states[-1].direction

            for _ in range(self.num_directions):
                next_direction = self.direction_rule(previous_direction)
                next_position = states[-1].position.shift(direction_to_unit_vector(next_direction))
                next_sub_collision = resolve_point_set_selectors_with_direction(
                    states[-1].position, collision, next_direction
                ) if self.select_direction else collision

                next_collision = next_position & next_sub_collision

                if len(next_collision) == 0:
                    return None
                else:
                    previous_direction = next_direction

            # If all direction changes lead to a collision, terminate the agent
            return AgentState(
                position=states[-1].position,
                direction=states[-1].direction,
                color=next(colors),
                charge=0,  # Set charge to 0 to indicate termination
                commit=states[-1].commit
            ), colors
        return None


class CollisionBorderRule(Rule):
    """
    A rule that changes the collision border's color to the given border color, if applicable.
    """

    def __init__(
            self,
            border_color: int,
            direction_rule: Optional[DirectionTransformer] = None,
            select_direction: bool = False,
    ) -> None:
        self.border_color = border_color
        self.direction_rule = direction_rule
        self.select_direction = select_direction

    def __call__(
            self,
            states: Sequence[AgentState],
            colors: ColorIterator,
            collision: PointSet,
            collision_mapping: AgentStateMapping,
    ) -> RuleResult:
        sub_collision = resolve_point_set_selectors_with_direction(
            states[-1].position, collision, states[-1].direction
        ) if self.direction_rule is not None and self.select_direction else collision

        if len(sub_collision) == 1:
            point = next(iter(sub_collision))
            if not collision_mapping[point].color == self.border_color:
                # If the agent collides with the border, change its color to the border color
                new_colors = chain([self.border_color], colors)

                return AgentState(
                    position=PointSet(sub_collision),
                    direction=states[-1].direction,
                    color=next(new_colors),
                    charge=states[-1].charge,
                    commit=states[-1].commit
                ), new_colors

        return None


class CollisionFillRule(Rule):
    def __init__(
            self,
            fill_color: int,
    ):
        self.fill_color = fill_color

    def __call__(
            self,
            states: Sequence[AgentState],
            colors: ColorIterator,
            collision: PointSet,
            collision_mapping: AgentStateMapping
    ) -> RuleResult:
        """
        Fill the collision area with the fill color.

        :param states: The current states of the agent.
        :param colors: An iterator over the agent's colors.
        :param collision: The set of points that are in collision with the agent.
        :param collision_mapping: The mapping between collision points and the agent's colors.
        :return: A new state with the fill color applied to the collision area.
        """

        if len(collision) > 0 and states[-1].color == self.fill_color:
            collision_colors = [collision_mapping[point].color for point in collision]
            if any(color != self.fill_color for color in collision_colors):
                new_colors = chain([self.fill_color], colors)

                return AgentState(
                    position=PointSet(states[-1].position | collision),
                    direction=states[-1].direction,
                    color=next(new_colors),
                    charge=states[-1].charge,
                    commit=states[-1].commit
                ), new_colors

        return None


def backtrack_rule(
        states: Sequence[AgentState],
        colors: ColorIterator,
        collision: PointSet,
        collision_mapping: AgentStateMapping
) -> RuleResult:
    """
    Backtrack the agent to its previous position

    :param states: The current states of the agent.
    :param colors: An iterator over the agent's colors.
    :param collision: The set of points that are in collision with the agent.
    :param collision_mapping: The mapping between collision points and the agent's colors.
    :return: A new state with the position set to the previous position.
    """

    return states[-2], colors


def uncharge_rule(
        states: Sequence[AgentState],
        colors: ColorIterator,
        collision: PointSet,
        collision_mapping: AgentStateMapping
) -> RuleResult:
    """
    Reduce the agent's charge by 1.

    :param states: The current states of the agent.
    :param colors: An iterator over the agent's colors.
    :param collision: The set of points that are in collision with the agent.
    :param collision_mapping: The mapping between collision points and the agent's colors.
    :return: A new state with the charge set to 0.
    """

    return AgentState(
        position=states[-1].position,
        direction=states[-1].direction,
        color=next(colors),
        charge=max(0, states[-1].charge - 1),
        commit=states[-1].commit
    ), colors


class GravityRule(Rule):
    def __init__(
            self,
            grid_size: Point
    ):
        self.grid_size = grid_size

    def __call__(
            self,
            states: Sequence[AgentState],
            colors: ColorIterator,
            collision: PointSet,
            collision_mapping: AgentStateMapping
    ) -> RuleResult:
        chance = random.uniform(0, 1)

        directions: list[Direction]
        if chance < 0.5:
            directions = ["down", "left", "right"]
        else:
            directions = ["down", "right", "left"]

        for direction in directions:
            sub_collision = resolve_point_set_selectors_with_direction(
                states[-1].position, collision, direction
            )

            if len(sub_collision) == 0:
                next_position = states[-1].position.shift(direction_to_unit_vector(direction))
                min_x = min(pos[0] for pos in next_position)
                max_x = max(pos[0] for pos in next_position)
                min_y = min(pos[1] for pos in next_position)
                max_y = max(pos[1] for pos in next_position)

                # If there is no collision and the next position is within the grid bounds
                if min_x >= 0 and max_x < self.grid_size[0] and \
                        min_y >= 0 and max_y < self.grid_size[1]:
                    return AgentState(
                        position=next_position,
                        direction=states[-1].direction,
                        color=next(colors),
                        charge=states[-1].charge - 1 if states[-1].charge > 0 else states[-1].charge,
                        commit=states[-1].commit
                    ), colors

        return None


class ProximityRule(Rule):
    def __init__(
            self,
            target: PointSet,
            points: PointSet
    ):
        self.target = target
        self.proximity_mapping = {
            point: min(math.dist(point, target_point) for target_point in target)
            for point in points
        }

        for point in target:
            self.proximity_mapping[point] = 0

    def __call__(
            self,
            states: Sequence[AgentState],
            colors: ColorIterator,
            collision: PointSet,
            collision_mapping: AgentStateMapping
    ) -> RuleResult:
        """
        Apply proximity-based logic to the agent's state.

        :param states: The current states of the agent.
        :param colors: An iterator over the agent's colors.
        :param collision: The set of points that are in collision with the agent.
        :param collision_mapping: The mapping between collision points and the agent's colors.
        :return: A new state based on proximity logic.
        """

        current_position = states[-1].position

        if current_position == self.target:
            return AgentState(
                position=current_position,
                direction=states[-1].direction,
                color=next(colors),
                charge=0,
                commit=states[-1].commit
            ), colors

        possible_points = PointSet(self.proximity_mapping.keys()) - collision - current_position
        eligible_points = [
            point for point in possible_points
            if any(math.dist(point, target_point) == 1 for target_point in current_position)
        ]

        if len(eligible_points) > 0:
            sorted_eligible_points = sorted(
                eligible_points,
                key=lambda point: self.proximity_mapping[point]
            )
            for min_point in sorted_eligible_points:
                closest_point = min(current_position, key=lambda point: math.dist(point, min_point))

                relative_direction = relative_point_direction(
                    closest_point, min_point
                )
                next_position = current_position.shift(direction_to_unit_vector(relative_direction))

                if not any(
                        point in collision for point in next_position
                ):
                    return AgentState(
                        position=next_position,
                        direction=relative_direction,
                        color=next(colors),
                        charge=states[-1].charge - 1 if states[-1].charge > 0 else states[-1].charge,
                        commit=states[-1].commit,
                    ), colors

        return None
