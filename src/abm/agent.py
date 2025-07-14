from typing import Iterator, Iterable, Mapping

from abm.action import Action
from abm.geometry import PointSet
from abm.neighbourhood import PointSetNeighbourhood
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
            neighbourhood: PointSetNeighbourhood,
            actions: Iterable[Action],
            colors: Iterator[int],
            charge: int = 0,
    ):
        self.position = position
        self.direction = direction
        self.label = label
        self.topology = topology
        self.neighbourhood = neighbourhood
        self.actions = actions
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
        history = [self.state]
        states = []
        action_iter = iter(self.actions)

        for action in action_iter:
            result = action(history, self.colors, collision, collision_mapping)

            if result is not None:
                state, colors = result

                self.position = state.position
                self.direction = state.direction
                self.color = state.color
                self.charge = state.charge
                self.colors = colors

                states.append(state)
                history.append(state)
        return states
