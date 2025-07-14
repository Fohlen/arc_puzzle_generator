from itertools import cycle
from typing import cast

import numpy as np

from abm.action import OutOfGridAction, CollisionDirectionAction, DirectionAction, collision_color_mapping, Action
from abm.agent import Agent
from abm.geometry import PointSet
from abm.model import Model
from abm.neighbourhood import zero_neighbourhood, moore_neighbours
from abm.physics import Direction
from abm.direction import identity_direction_rule, orthogonal_direction
from abm.simulation import Simulation
from abm.topology import FixedGroupTopology, identity_topology
from abm.entities import find_colors, find_connected_objects, is_l_shape
from abm.grid_utils import make_smallest_square_from_mask
from arc_puzzle_generator.physics import starting_point


def puzzle_four(input_grid: np.ndarray) -> Simulation:
    """

    :param input_grid:
    :return:
    """

    colors = find_colors(input_grid, background=0)
    l_shapes: list[tuple[int, np.ndarray, Direction]] = []
    blocks: list[tuple[int, np.ndarray]] = []

    for target_color in colors:
        target_mask = input_grid == target_color
        labeled_grid, bounding_box, num_objects = find_connected_objects(target_mask)

        for label in range(1, num_objects + 1):
            box = make_smallest_square_from_mask(input_grid, labeled_grid == label)

            if box is not None:
                direction = is_l_shape(box)
                if direction is not None:
                    l_shapes.append((target_color, bounding_box[(label - 1), :], direction))
                else:
                    blocks.append((target_color, bounding_box[(label - 1), :]))

    agents = [Agent(
        position=PointSet.from_numpy(bbox),
        direction="right",
        label="bbox",
        topology=identity_topology,
        neighbourhood=zero_neighbourhood,
        actions=[],
        colors=cycle([]),
        charge=0
    ) for _, bbox in blocks]

    topology = FixedGroupTopology(group={"bbox"})
    actions = [
        OutOfGridAction(grid_size=(input_grid.shape[0], input_grid.shape[1])),
        cast(Action, collision_color_mapping),
        CollisionDirectionAction(orthogonal_direction),
        DirectionAction(identity_direction_rule)
    ]

    agents += [Agent(
        position=PointSet.from_numpy(
            starting_point(
                bounding_box=bbox,
                direction=direction,
                point_width=1
            )
        ),
        direction=direction,
        label="puzzle_four_agent",
        topology=topology,
        neighbourhood=moore_neighbours,
        actions=actions,
        colors=cycle([color]),
        charge=-1,
    ) for color, bbox, direction in l_shapes]

    return Simulation(
        Model(
            output_grid=input_grid.copy(),
            agents=agents,
        ),
        -1
    )
