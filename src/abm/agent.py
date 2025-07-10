from abm.geometry import PointSet
from abm.neighbourhood import Neighbourhood
from abm.physics import Direction
from abm.selection import Selector
from abm.topology import Topology


class Agent:
    def __init__(
            self,
            position: PointSet,
            direction: Direction,
            label: str,
            topology: Topology,
            neighbourhood: Neighbourhood,
            selector: Selector,
            charge: int = 0,
    ):
        self.position = position
        self.direction = direction
        self.label = label
        self.topology = topology
        self.neighbourhood = neighbourhood
        self.selector = selector
        self.charge = charge

    def active(self) -> bool:
        return self.charge > 0 or self.charge == -1
