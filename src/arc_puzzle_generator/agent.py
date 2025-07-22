import logging
from typing import Iterator, Iterable

from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.physics import Direction
from arc_puzzle_generator.rule import RuleNode
from arc_puzzle_generator.state import AgentState, AgentStateMapping
from arc_puzzle_generator.utils.callable import get_callable_name

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
            self,
            position: PointSet,
            direction: Direction,
            label: str,
            node: RuleNode,
            colors: Iterator[int],
            charge: int = 0,
    ):
        self.position = position
        self.direction = direction
        self.label = label
        self.node = node
        self.colors = colors
        self.charge = charge
        self.color = next(colors)
        self.history: list[AgentState] = []

    @property
    def active(self) -> bool:
        return self.charge > 0 or self.charge == -1

    @property
    def state(self) -> AgentState:
        return AgentState(
            position=self.position,
            direction=self.direction,
            color=self.color,
            charge=self.charge
        )

    def steps(
            self,
            collision: PointSet,
            collision_mapping: AgentStateMapping,
    ) -> Iterable[AgentState]:
        states = [self.state]

        if self.node is None:
            return []

        stack = [self.node]

        while stack:
            current = stack.pop()
            result = current.rule(states, self.colors, collision, collision_mapping)

            if result is not None:
                state, colors = result
                logger.debug(f"Rule {get_callable_name(current.rule)} produced state: {state}")

                self.position = state.position
                self.direction = state.direction
                self.color = state.color
                self.charge = state.charge
                self.colors = colors
                self.history.append(state)
                states.append(state)

                if current.next_node is not None:
                    stack.append(current.next_node)
            elif current.alternative_node is not None:
                stack.append(current.alternative_node)

        return states[1:]
