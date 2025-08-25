import math
import random
from collections import deque
from itertools import chain, cycle
from typing import Protocol, Optional, Sequence, Literal

from arc_puzzle_generator.direction import DirectionTransformer
from arc_puzzle_generator.direction import absolute_direction
from arc_puzzle_generator.geometry import PointSet, Point, Direction, in_grid
from arc_puzzle_generator.physics import direction_to_unit_vector, collision_axis, relative_point_direction, shift
from arc_puzzle_generator.selection import resolve_point_set_selectors_with_direction, resolve_cell_selection
from arc_puzzle_generator.state import AgentState, AgentStateMapping, ColorIterator

RuleResult = Optional[tuple[AgentState, ColorIterator, list[AgentState]]]


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

    return states[-1], colors, []


class OutOfGridRule(Rule):
    """
    A rule that removes the agent if the next step is out of grid, by setting its charge to 0.
    """

    def __init__(self, grid_size: Point) -> None:
        self.grid_size = grid_size

    def __call__(
            self,
            states: Sequence[AgentState],
            colors: ColorIterator,
            collision: PointSet,
            *args
    ) -> RuleResult:
        """
        Remove the agent from the grid.

        :param states: The current states of the agent.
        :param colors: An iterator over the agent's colors.
        :param collision: The set of points that are in collision with the agent.
        :return: None, indicating that the agent is removed from the grid.
        """

        next_position = states[-1].position.shift(direction_to_unit_vector(states[-1].direction))

        if not all(in_grid(point, self.grid_size) for point in next_position):
            return AgentState(
                position=states[-1].position,
                direction=states[-1].direction,
                color=next(colors),
                charge=0,  # Set charge to 0 to indicate removal
            ), colors, []

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
            ), colors, []

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
        ), new_colors, []

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
                next_sub_collision = resolve_point_set_selectors_with_direction(
                    states[-1].position, collision, next_direction
                ) if self.select_direction else collision

                if len(next_sub_collision) == 0:
                    return None
                else:
                    previous_direction = next_direction

            # If all direction changes lead to a collision, terminate the agent
            return AgentState(
                position=states[-1].position,
                direction=states[-1].direction,
                color=next(colors),
                charge=0,  # Set charge to 0 to indicate termination
            ), colors, []
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
                ), new_colors, []

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
                ), new_colors, []

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

    return states[-2], colors, []


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
    ), colors, []


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
                if all(in_grid(point, self.grid_size) for point in next_position):
                    return AgentState(
                        position=next_position,
                        direction=direction,
                        color=next(colors),
                        charge=states[-1].charge - 1 if states[-1].charge > 0 else states[-1].charge,
                    ), colors, []

        return None


class ProximityRule(Rule):
    def __init__(
            self,
            target: PointSet,
            points: PointSet
    ):
        self.target = target
        distances = {
            (point, target_point): math.dist(point, target_point)
            for point in points for target_point in target
        }
        min_distance = min(distances.values())
        max_distance = max(distances.values())
        self.proximity_mapping = {
            point: (
                (min(distances[(point, target_point)] for target_point in target) - min_distance)
                / (max_distance - min_distance) if max_distance > min_distance else 0.0
            )
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
            ), colors, []

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
                    ), colors, []

        return None


class RewardRule(Rule):
    """
    Implements a reward learning based rule using a Q-Learning table
    """

    def __init__(
            self,
            grid_size: Point,
            directions: Sequence[Direction],
            target: PointSet,
            denylist: Optional[PointSet] = None,
            gamma: float = 0.1,
            positive_reward: float = 1.0,
            negative_reward: float = 0.00001,
    ):
        self.grid_size = grid_size
        self.directions = directions
        self.q_table: dict[Point, float] = {
            point: positive_reward for point in target
        }

        if denylist is None:
            denylist = PointSet()

        visited = PointSet(target)
        points = deque(target)

        while len(points) > 0:
            point = points.popleft()

            for direction in directions:
                next_point = shift(point, direction_to_unit_vector(direction))

                if in_grid(next_point, grid_size) and next_point not in visited:
                    # mark as visited
                    visited.add(next_point)
                    # add as seed for further exploration
                    points.append(next_point)

                    # calculate the reward for the next point, if valid
                    if next_point in denylist:
                        self.q_table[next_point] = negative_reward
                    else:
                        self.q_table[next_point] = self.q_table[point] * (1 - gamma)

    def __call__(
            self,
            states: Sequence[AgentState],
            colors: ColorIterator,
            collision: PointSet,
            collision_mapping: AgentStateMapping
    ) -> RuleResult:
        """
        Apply the reward rule to the agent's state.

        :param states: The current states of the agent.
        :param colors: An iterator over the agent's colors.
        :param collision: The set of points that are in collision with the agent.
        :param collision_mapping: The mapping between collision points and the agent's colors.
        :return: A new state based on the reward logic.
        """

        current_position = states[-1].position
        valid_positions: list[tuple[PointSet, Direction, float]] = []

        for direction in self.directions:
            next_position = current_position.shift(direction_to_unit_vector(direction))

            if all(in_grid(point, self.grid_size) for point in next_position) and len(next_position & collision) == 0:
                reward = sum(self.q_table[point] for point in next_position)
                valid_positions.append((next_position, direction, reward))

        # Select the position with the maximum Q-value
        if len(valid_positions) > 0:
            next_position, next_direction, _ = max(valid_positions, key=lambda x: x[2])
            return AgentState(
                position=next_position,
                direction=next_direction,
                color=next(colors),
                charge=states[-1].charge - 1 if states[-1].charge > 0 else states[-1].charge,
            ), colors, []

        return None


class AgentSpawnRule(Rule):
    """
    Creates new agents of the same type and given direction, if possible.
    """

    def __init__(
            self,
            directions: Sequence[Direction],
            grid_size: Point,
            select_direction: bool = False,
            denylist: Optional[PointSet] = None,
    ):
        """
        Initialize the agent spawn rule with a list of directions.
        :param directions: The directions in which new agents can be spawned.
        :param grid_size: The size of the grid in which agents can be spawned.
        :param select_direction: If true, the agent will select the direction for collision direction.
        :param denylist: The set of points that are in denylist.
        """
        self.directions = directions
        self.grid_size = grid_size
        self.select_direction = select_direction
        self.denylist = denylist if denylist is not None else PointSet()

    def __call__(
            self,
            states: Sequence[AgentState],
            colors: ColorIterator,
            collision: PointSet,
            collision_mapping: AgentStateMapping
    ) -> RuleResult:
        children: list[AgentState] = []

        for direction in self.directions:
            next_position = states[-1].position.shift(direction_to_unit_vector(direction))
            if len(next_position & self.denylist) == 0 and all(
                    in_grid(point, self.grid_size) for point in next_position):
                next_sub_collision = resolve_point_set_selectors_with_direction(
                    states[-1].position, collision, direction
                ) if self.select_direction else collision

                if len(next_sub_collision) == 0:
                    child = AgentState(
                        position=next_position,
                        direction=direction,
                        color=next(colors),
                        charge=states[-2].charge if states[-2].charge > 0 else states[-2].charge,
                    )
                    children.append(child)

        return states[-1], colors, children


class StayInGridRule(Rule):
    """
    A rule that ensures the agent stays within the grid boundaries.
    """

    def __init__(self, grid_size: Point, direction_rule: DirectionTransformer):
        self.grid_size = grid_size
        self.direction_rule = direction_rule

    def __call__(
            self,
            states: Sequence[AgentState],
            colors: ColorIterator,
            collision: PointSet,
            collision_mapping: AgentStateMapping
    ) -> RuleResult:
        next_position = states[-1].position.shift(direction_to_unit_vector(states[-1].direction))

        if not all(in_grid(point, self.grid_size) for point in next_position):
            alternative_direction = self.direction_rule(states[-1].direction)
            alternative_position = states[-1].position.shift(direction_to_unit_vector(alternative_direction))

            if all(in_grid(point, self.grid_size) for point in alternative_position):
                return AgentState(
                    position=alternative_position,
                    direction=alternative_direction,
                    color=next(colors),
                    charge=states[-1].charge - 1 if states[-1].charge > 0 else states[-1].charge,
                ), colors, []

        return None


class TerminateAtPointRule(Rule):
    """
    A rule that terminates the agent once it reaches a specific target point.
    """

    def __init__(self, target: PointSet, direction_rule: DirectionTransformer):
        self.target = target
        self.direction_rule = direction_rule

    def __call__(
            self,
            states: Sequence[AgentState],
            colors: ColorIterator,
            collision: PointSet,
            collision_mapping: AgentStateMapping
    ) -> RuleResult:
        """
        Terminate the agent if it reaches the target point.

        :param states: The current states of the agent.
        :param colors: An iterator over the agent's colors.
        :param collision: The set of points that are in collision with the agent.
        :param collision_mapping: The mapping between collision points and the agent's colors.
        :return: A new state with charge set to 0 if the target is reached, otherwise None.
        """

        next_position = states[-1].position.shift(
            direction_to_unit_vector(self.direction_rule(states[-1].direction))
        )

        if next_position == self.target:
            return AgentState(
                position=states[-1].position,
                direction=states[-1].direction,
                color=next(colors),
                charge=0,  # Set charge to 0 to indicate termination
            ), colors, []

        return None


Condition = tuple[bool, Direction]
"""
A condition is a tuple of a boolean and a Direction, indicating whether a collision is happening in the indicated direction or not.
"""

ConditionMode = Literal["AND", "OR"]


class CollisionConditionDirectionRule(Rule):
    """
    The CollisionConditionDirectionRule applies a direction rule based on a set of conditions.
    Each condition is a tuple of boolean and Direction, where the boolean indicates whether the condition must be met, and the Direction specifies which direction to select to check for a collision.
    If a direction is none it means the current direction of the agent is used.

    If all conditions are met, the direction rule is applied to the agent's current direction.
    """

    def __init__(
            self,
            direction_rule: DirectionTransformer,
            conditions: Sequence[Condition],
            condition_mode: ConditionMode = "AND",
    ):
        self.direction_rule = direction_rule
        self.conditions = conditions
        self.condition_mode = condition_mode

    def __call__(
            self,
            states: Sequence[AgentState],
            colors: ColorIterator,
            collision: PointSet,
            collision_mapping: AgentStateMapping
    ) -> RuleResult:
        """
        Apply the direction rule based on the conditions.
        :param states: The current states of the agent.
        :param colors: The iterator over the agent's colors.
        :param collision: The set of points that are in collision with the agent.
        :param collision_mapping: The mapping between collision points and the agent's colors.
        :return: A new state with the updated direction if all conditions are met, otherwise None.
        """

        conditions_met = []
        for condition, condition_direction in self.conditions:
            if condition_direction == "none":
                direction = states[-1].direction
            else:
                direction = absolute_direction(states[-1].direction, condition_direction)
            sub_collision = resolve_point_set_selectors_with_direction(
                states[-1].position, collision, direction
            )

            if condition and len(sub_collision) > 0:
                conditions_met.append(True)
            elif not condition and len(sub_collision) == 0:
                conditions_met.append(True)
            else:
                conditions_met.append(False)

        if (self.condition_mode == "AND" and all(conditions_met)) or (
                self.condition_mode == "OR" and any(conditions_met)):
            if self.condition_mode == "OR":
                axis = collision_axis(collision)
                new_direction = self.direction_rule(states[-1].direction, axis)
            else:
                new_direction = self.direction_rule(states[-1].direction)
            new_position = states[-1].position.shift(direction_to_unit_vector(new_direction))

            return AgentState(
                position=new_position,
                direction=new_direction,
                color=next(colors),
                charge=states[-1].charge - 1 if states[-1].charge > 0 else states[-1].charge,
            ), colors, []

        return None


def collision_entity_redirect_rule(
        states: Sequence[AgentState],
        colors: ColorIterator,
        collision: PointSet,
        collision_mapping: AgentStateMapping
):
    """
    When hitting a collision entity, the agent will take the entity's shape and current direction, and recolor it.
    :param states: The current states of the agent.
    :param colors: The iterator over the agent's colors.
    :param collision: The set of points that are in collision with the agent.
    :param collision_mapping: The mapping between collision points and the agent's colors.
    :return:
    """
    sub_collision = resolve_point_set_selectors_with_direction(
        states[-1].position, collision, states[-1].direction
    )

    if len(sub_collision) > 0:
        positions = PointSet(
            [entity_point for point in sub_collision for entity_point in collision_mapping[point].position]
        )
        directions = set([
            collision_mapping[col].direction for col in sub_collision
        ])

        return AgentState(
            position=positions,
            direction=next(iter(directions)),
            color=next(colors),
            charge=states[-1].charge if states[-1].charge > 0 else states[-1].charge,
        ), colors, []

    return None


def resize_entity_to_exit_rule(
        states: Sequence[AgentState],
        colors: ColorIterator,
        collision: PointSet,
        collision_mapping: AgentStateMapping
):
    """
    Shrinks back an entity to fit the exit direction (only select the points leaving the entity)
    :param states: The current states of the agent.
    :param colors: The iterator over the agent's colors.
    :param collision: The set of points that are in collision with the agent.
    :param collision_mapping: A mapping between collision points and the agent's colors.
    :return: A new state with the updated positions
    """

    return AgentState(
        position=resolve_cell_selection(states[-1].position, states[-1].direction),
        color=next(colors),
        charge=states[-1].charge if states[-1].charge > 0 else states[-1].charge,
        direction=states[-1].direction,
    ), colors, []
