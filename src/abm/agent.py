from typing import Iterator, Iterable

from abm.action import Action
from abm.geometry import PointSet, ColorMapping
from abm.neighbourhood import Neighbourhood
from abm.physics import Direction
from abm.state import AgentState
from abm.topology import Topology


class Agent:
    def __init__(
            self,
            position: PointSet,
            direction: Direction,
            label: str,
            topology: Topology,
            neighbourhood: Neighbourhood,
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

    @property
    def active(self) -> bool:
        return self.charge > 0 or self.charge == -1

    @property
    def state(self) -> AgentState:
        return AgentState(
            position=self.position,
            direction=self.direction,
            colors=self.colors,
            charge=self.charge
        )

    def steps(self, collision: PointSet, collision_mapping: ColorMapping) -> Iterable[AgentState]:
        states = []
        action_iter = iter(self.actions)

        for action in action_iter:
            state = action(self.state, collision, collision_mapping)

            # If the action returns a valid state, update the agent's attributes
            if state is not None:
                self.position = state.position
                self.direction = state.direction
                self.colors = state.colors
                self.charge = state.charge

                states.append(state)
        return states
