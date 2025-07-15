from typing import Iterator, Iterable

from abm.action import ActionNode
from abm.geometry import PointSet
from abm.neighbourhood import Neighbourhood
from abm.physics import Direction
from abm.state import AgentState, AgentStateMapping
from abm.topology import Topology


class Agent:
    def __init__(
            self,
            position: PointSet,
            direction: Direction,
            label: str,
            topology: Topology,
            neighbourhood: Neighbourhood,
            node: ActionNode,
            colors: Iterator[int],
            charge: int = 0,
    ):
        self.position = position
        self.direction = direction
        self.label = label
        self.topology = topology
        self.neighbourhood = neighbourhood
        self.node = node
        self.colors = colors
        self.charge = charge
        self.color = next(colors)

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
            curr = stack.pop()
            result = curr.action(states, self.colors, collision, collision_mapping)

            if result is not None:
                state, colors = result

                self.position = state.position
                self.direction = state.direction
                self.color = state.color
                self.charge = state.charge
                self.colors = colors
                states.append(state)

                if curr.next_node is not None:
                    stack.append(curr.next_node)
            elif curr.alternative_node is not None:
                stack.append(curr.alternative_node)

        return states[:-1]
