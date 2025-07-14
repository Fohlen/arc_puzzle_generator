from collections import defaultdict
from itertools import cycle
from typing import Mapping

import numpy as np

from abm.action import OutOfGridAction, TrappedCollisionAction, CollisionDirectionAction, DirectionAction
from abm.agent import Agent
from abm.geometry import PointSet, unmask
from abm.model import Model
from abm.neighbourhood import von_neumann_neighbours, dummy_neighbourhood
from abm.simulation import Simulation
from abm.topology import FixedGroupTopology, identity_topology
from arc_puzzle_generator.collisions import snake_direction, identity_direction
from arc_puzzle_generator.entities import colour_count, find_colors, find_connected_objects


def puzzle_ten(input_grid: np.ndarray) -> Simulation:
    start_rows: list[int] = []
    start_col = input_grid[:, 0]
    start_col_colors = colour_count(start_col)
    grid_colors = [start_col_colors[0][0], start_col_colors[1][0]]

    for index, row in enumerate(start_col):
        if row == start_col_colors[-1][0]:
            start_rows.append(index)

    # this is primitive but works
    outside_color = input_grid[0, 0]
    inside_color: int = input_grid[start_rows[0], 1]  # type: ignore

    color_boxes: Mapping[int, list[tuple[int, int]]] = defaultdict(list)  # { row: [(col, color)] }
    colors = find_colors(input_grid)

    for color in [color for color in colors if color not in grid_colors]:
        labels, bboxes, num_objects = find_connected_objects(input_grid == color)
        for bbox in bboxes.tolist():
            if bbox[0] != bbox[3]:
                color_boxes[bbox[0][0]].append((bbox[0][1] + 1, color))
                input_grid[bbox[1][0]:(bbox[3][0] + 1), bbox[0][1]:(bbox[3][1] + 1)] = outside_color

    sequence_row = min(color_boxes.keys())
    border_row = max(color_boxes.keys())
    color_sequence = [color for col, color in sorted(color_boxes[sequence_row], key=lambda x: x[0])]
    border_color = color_boxes[border_row][0][1]

    topology = FixedGroupTopology(group={"foreground"})

    foreground_position = unmask(input_grid == outside_color)
    agents = [Agent(
        position=foreground_position,
        direction="right",
        label="foreground",
        topology=identity_topology,
        neighbourhood=dummy_neighbourhood,
        actions=[],
        colors=iter([]),
        charge=0,
    )]

    actions = [
        OutOfGridAction(grid_size=(input_grid.shape[0], input_grid.shape[1])),
        TrappedCollisionAction(direction_rule=snake_direction),
        CollisionDirectionAction(direction_rule=snake_direction),
        DirectionAction(direction_rule=identity_direction),
    ]

    agents += [Agent(
        position=PointSet([(row, 0)]),
        direction="right",
        label="snake",
        topology=topology,
        neighbourhood=von_neumann_neighbours,
        actions=actions,
        colors=cycle(color_sequence),
        charge=-1,
    ) for row in start_rows]

    return Simulation(Model(
        input_grid.copy(),
        agents=agents,
    ), -1)
