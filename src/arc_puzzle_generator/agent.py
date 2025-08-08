import logging
from typing import Iterator, Iterable, Optional

from arc_puzzle_generator.geometry import PointSet, Direction
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
            colors: Iterator[int],
            node: Optional[RuleNode] = None,
            charge: int = 0,
            commit: bool = True,
    ):
        """
        An agent that can move through a grid and interact with rules defined in the `RuleNode`.
        :param position: A set of points representing the agent's position in the grid.
        :param direction: A direction in which the agent is currently moving, also known as velocity.
        :param label: The label of the agent, which is used to calculate the topology.
        :param colors: The colors that the agent can use during its lifetime, provided as an iterator.
        :param node: A `RuleNode` that defines the rules the agent will follow.
        :param charge: A charge that the agent has, which can be positive (running), 0 (terminated), or -1 (indefinite).
        :param commit: Whether the resulting state will be committed to the output grid or not.
        """

        self.position = position
        self.direction = direction
        self.label = label
        self.node = node
        self.colors = colors
        self.charge = charge
        self.commit = commit
        self.color = next(colors)
        self.history: list[AgentState] = [self.state]

    @property
    def active(self) -> bool:
        return self.charge > 0 or self.charge == -1

    @property
    def state(self) -> AgentState:
        return AgentState(
            position=self.position,
            direction=self.direction,
            color=self.color,
            charge=self.charge,
            commit=self.commit,
        )

    def steps(
            self,
            collision: PointSet,
            collision_mapping: AgentStateMapping,
    ) -> tuple[Iterable[AgentState], list['Agent']]:
        states = [self.state]
        children = []

        if self.node is None:
            return [], children

        stack = [self.node]

        while stack:
            current = stack.pop()
            result = current.rule(states, self.colors, collision, collision_mapping)

            if result is not None:
                state, colors, rule_children = result
                logger.debug(f"Rule {get_callable_name(current.rule)} produced state: {state}")

                self.position = state.position
                self.direction = state.direction
                self.color = state.color
                self.charge = state.charge
                self.commit = state.commit
                self.colors = colors
                self.history.append(state)
                states.append(state)

                # children inherit label and node from the parent agent
                for child in rule_children:
                    children.append(Agent(
                        position=child.position,
                        direction=child.direction,
                        label=self.label,
                        node=self.node,
                        colors=colors,
                        charge=child.charge,
                        commit=child.commit,
                    ))

                if current.next_node is not None:
                    stack.append(current.next_node)
            elif current.alternative_node is not None:
                stack.append(current.alternative_node)

        return states[1:], children
