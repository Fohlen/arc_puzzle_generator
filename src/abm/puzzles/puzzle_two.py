from collections import defaultdict
from math import ceil

import numpy as np

from abm.action import DirectionAction
from abm.agent import Agent
from abm.color_sequence_iterator import ColorSequenceIterator
from abm.geometry import PointSet
from abm.model import Model
from abm.neighbourhood import dummy_neighbourhood
from abm.physics import Direction, identity_direction_rule
from abm.simulation import Simulation
from abm.topology import identity_topology
from abm.entities import colour_count, find_connected_objects
from arc_puzzle_generator.physics import relative_box_direction, box_distance, starting_point


def puzzle_two(input_grid: np.ndarray) -> Simulation:
    """
    Solve the second puzzle by creating a simulation based on the input grid.

    :param input_grid: The input grid for the puzzle.
    :return: A Simulation object containing the model and agents.
    """

    boxes: list[tuple[int, int, np.ndarray]] = []  # [(index, target_color, bounding box)]
    box_order: list[tuple[int, Direction, int]] = []  # [(box, direction, distance)]
    box_size: int | None = None
    grid = input_grid[:-2, :]

    sorted_colors = colour_count(grid)
    background_color = sorted_colors[0][0]
    box_color = sorted_colors[1][0]

    mask = grid == box_color
    labels, bboxes, num_objects = find_connected_objects(mask)

    box_colors = defaultdict(list)
    adjacent_boxes = defaultdict(list)
    horizontal_boxes = defaultdict(list)
    vertical_boxes = defaultdict(list)

    for index, bbox in enumerate(bboxes):
        grid_box = grid[bbox[1, 0]:bbox[0, 0] + 1, bbox[0, 1]:bbox[3, 1] + 1]
        target_color = [color for color in np.unique(grid_box) if color != box_color][0]
        if box_size is None:
            box_size = np.sum(grid_box == target_color)

        boxes.append((index, target_color, bbox))
        box_colors[target_color].append(index)
        horizontal_boxes[bbox[0, 0]].append(index)
        vertical_boxes[bbox[0, 1]].append(index)

    for (index, color, bbox) in boxes:
        horizontal_index = horizontal_boxes[bbox[0, 0]].index(index)
        vertical_index = vertical_boxes[bbox[0, 1]].index(index)

        # we select the left and right boxes on the horizontal scale
        adjacent_boxes[index].extend(
            [idx for idx in horizontal_boxes[bbox[0, 0]][(max(horizontal_index - 1, 0)):(horizontal_index + 2)] if
             idx != index]
        )
        # we select the top and bottom boxes on the vertical scale
        adjacent_boxes[index].extend(
            [idx for idx in vertical_boxes[bbox[0, 1]][(max(vertical_index - 1, 0)):(vertical_index + 2)] if
             idx != index]
        )

    legend = [color for color in input_grid[-2, :] if color != background_color]
    color_order = [(legend[i], legend[i + 1]) for i in range(len(legend) - 1)]

    current_box = None
    start_indexes = box_colors[color_order[0][0]]
    for start_index in start_indexes:
        color2_indexes = box_colors[color_order[0][1]]
        reachable_boxes = [index2 for index2 in color2_indexes if index2 in adjacent_boxes[start_index]]
        if len(reachable_boxes) == 1:
            direction = relative_box_direction(boxes[start_index][2], boxes[reachable_boxes[0]][2])
            distance = box_distance(boxes[start_index][2], boxes[reachable_boxes[0]][2], direction)
            box_order.append((start_index, direction, distance))
            current_box = reachable_boxes[0]
            break

    if current_box is not None:
        for _, color2 in color_order[1:]:
            color2_indexes = box_colors[color2]
            reachable_boxes = [index2 for index2 in color2_indexes if index2 in adjacent_boxes[current_box]]
            if len(reachable_boxes) == 1:
                direction = relative_box_direction(boxes[current_box][2], boxes[reachable_boxes[0]][2])
                distance = box_distance(boxes[current_box][2], boxes[reachable_boxes[0]][2], direction)
                box_order.append((current_box, direction, distance))
                current_box = reachable_boxes[0]

    assert box_size is not None
    beam_width = ceil(box_size / 2)
    steps = max([distance for _, _, distance in box_order])

    return Simulation(Model(
        output_grid=input_grid.copy(),
        agents=[
            Agent(
                position=PointSet(
                    (x.item(), y.item())
                    for (x, y) in starting_point(boxes[box1][2], direction, beam_width)
                ),
                direction=direction,
                label="puzzle_two_agent",
                topology=identity_topology,
                neighbourhood=dummy_neighbourhood,
                actions=[
                    DirectionAction(identity_direction_rule)
                ],
                charge=distance,
                colors=ColorSequenceIterator(
                    [(box_color, distance), (boxes[box1][1], 1)],
                    background_color=background_color
                ),
            ) for box1, direction, distance in box_order
        ]
    ), steps)
