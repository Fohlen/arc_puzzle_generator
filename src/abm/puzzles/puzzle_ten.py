from collections import defaultdict
from itertools import cycle
from typing import Mapping, cast

import numpy as np

from abm.action import OutOfGridAction, TrappedCollisionAction, CollisionDirectionAction, DirectionAction, \
    CollisionBorderAction, backtrack_action, Action
from abm.agent import Agent
from abm.direction import identity_direction_rule, snake_direction_rule
from abm.geometry import PointSet, unmask
from abm.model import Model
from abm.neighbourhood import zero_neighbours, IdentityPointSetNeighbourhood, \
    von_neumann_neighbours
from abm.topology import FixedGroupTopology, identity_topology
from abm.utils.entities import colour_count, find_colors, find_connected_objects


def puzzle_ten(input_grid: np.ndarray) -> Model:
    start_rows: list[int] = []
    start_col = input_grid[:, 0]
    start_col_colors = colour_count(start_col)
    grid_colors = [start_col_colors[0][0], start_col_colors[1][0]]

    for index, row in enumerate(start_col):
        if row == start_col_colors[-1][0]:
            start_rows.append(index)

    # this is primitive but works
    outside_color = input_grid[0, 0].item()
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
        neighbourhood=IdentityPointSetNeighbourhood(zero_neighbours),
        actions=[],
        colors=cycle([outside_color]),
        charge=0,
    )]

    actions = [
        OutOfGridAction(grid_size=(input_grid.shape[0], input_grid.shape[1])),
        CollisionBorderAction(border_color=border_color, direction_rule=identity_direction_rule, select_direction=True),
        cast(Action, backtrack_action),
        TrappedCollisionAction(direction_rule=snake_direction_rule),
        CollisionDirectionAction(direction_rule=snake_direction_rule, select_direction=True),
        DirectionAction(direction_rule=identity_direction_rule, select_direction=True),
    ]

    agents += [Agent(
        position=PointSet([(row, 0)]),
        direction="right",
        label="snake",
        topology=topology,
        neighbourhood=IdentityPointSetNeighbourhood(von_neumann_neighbours),
        actions=actions,
        colors=cycle(color_sequence),
        charge=-1,
    ) for row in start_rows]

    return Model(
        input_grid.copy(),
        agents=agents,
    )
