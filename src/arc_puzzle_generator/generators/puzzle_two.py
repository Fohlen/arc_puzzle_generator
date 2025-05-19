from typing import Iterable
from math import ceil

import numpy as np
from mypy.checkexpr import defaultdict

from arc_puzzle_generator.entities import find_connected_objects, colour_count
from arc_puzzle_generator.generators.generator import Generator
from arc_puzzle_generator.physics import direction_to_unit_vector, Direction


class PuzzleTwoGenerator(Generator):
    def __init__(self, input_grid: np.ndarray):
        super().__init__(input_grid)
        self.boxes: list[tuple[int, int, int, np.ndarray]] = []  # index, target_color, bounding box
        self.box_order: list[tuple[int, int]] = []

    def setup(self) -> None:
        grid = self.input_grid[:-2, :]

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
            color_count = np.sum(grid_box == target_color)

            self.boxes.append((index, target_color, color_count, bbox))
            box_colors[target_color].append(index)
            horizontal_boxes[bbox[0, 0]].append(index)
            vertical_boxes[bbox[0, 1]].append(index)

        for (index, color, count, bbox) in self.boxes:
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

        legend = [color for color in self.input_grid[-2, :] if color != background_color]
        color_order = [(legend[i], legend[i + 1]) for i in range(len(legend) - 1)]

        current_box = None
        start_indexes = box_colors[color_order[0][0]]
        for start_index in start_indexes:
            color2_indexes = box_colors[color_order[0][1]]
            reachable_boxes = [index2 for index2 in color2_indexes if index2 in adjacent_boxes[start_index]]
            if len(reachable_boxes) == 1:
                self.box_order.append((start_index, reachable_boxes[0]))
                current_box = reachable_boxes[0]
                break

        if current_box is not None:
            for _, color2 in color_order[1:]:
                color2_indexes = box_colors[color2]
                reachable_boxes = [index2 for index2 in color2_indexes if index2 in adjacent_boxes[current_box]]
                if len(reachable_boxes) == 1:
                    self.box_order.append((current_box, reachable_boxes[0]))
                    current_box = reachable_boxes[0]

    def __iter__(self, *args, **kwargs) -> Iterable[np.ndarray]:
        for box1, box2 in self.box_order:
            box1_bbox = self.boxes[box1][3]
            box2_bbox = self.boxes[box2][3]

            # Determine a relative direction based on bounding box corners
            rectangle_size = (box1_bbox[0, 0] - box1_bbox[1, 0]) + 1
            beam_size = ceil(self.boxes[box1][2] / 2)
            border = (rectangle_size - beam_size) // 2

            if box1_bbox[0, 0] < box2_bbox[0, 0]:
                direction: Direction = "down"
                start_pos = box1_bbox[0] + np.array([0, border])
                start_pos = np.array([start_pos + [0, i] for i in range(beam_size)])
                target_pos = box2_bbox[1] + np.array([-1, border])
                target_pos = np.array([target_pos + [0, i] for i in range(beam_size)])
            elif box1_bbox[0, 0] > box2_bbox[0, 0]:
                direction = "up"
                start_pos = box1_bbox[1] + np.array([0, border])
                start_pos = np.array([start_pos + [0, i] for i in range(beam_size)])
                target_pos = box2_bbox[0] + np.array([1, border])
                target_pos = np.array([target_pos + [0, i] for i in range(beam_size)])
            elif box1_bbox[0, 1] < box2_bbox[0, 1]:
                direction = "right"
                start_pos = box1_bbox[2] + np.array([border, 0])
                start_pos = np.array([start_pos + [i, 0] for i in range(beam_size)])
                target_pos = box2_bbox[1] + np.array([border, -1])
                target_pos = np.array([target_pos + [i, 0] for i in range(beam_size)])
            else:  # elif box1_bbox[0, 1] > box2_bbox[0, 1]:
                direction = "left"
                start_pos = box1_bbox[1] + np.array([border, 0])
                start_pos = np.array([start_pos + [i, 0] for i in range(beam_size)])
                target_pos = box2_bbox[2] + np.array([border, 1])
                target_pos = np.array([target_pos + [i, 0] for i in range(beam_size)])

            step = start_pos.copy()
            while not np.array_equal(step, target_pos):
                step += direction_to_unit_vector(direction)
                self.output_grid[step[:, 0], step[:, 1]] = self.boxes[box1][1]

                yield self.output_grid.copy()
