from collections import Counter
from itertools import cycle
from typing import cast

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction, clockwise_direction_90, \
    counterclockwise_direction_90, orthogonal_direction
from arc_puzzle_generator.geometry import unmask, PointSet, Point
from arc_puzzle_generator.model import Model
from arc_puzzle_generator.physics import direction_to_unit_vector
from arc_puzzle_generator.rule import RuleNode, identity_rule, DirectionRule, CollisionDirectionRule, \
    StayInGridRule, Rule, TrappedCollisionRule, LeftBottomRule
from arc_puzzle_generator.utils.entities import colour_count, find_connected_objects


def puzzle_eight(input_grid: np.ndarray) -> Model:
    """
    Generates the model for the eighth puzzle, which is puzzle where agents need to follow their border.
    :param input_grid: The input grid for the puzzle.
    :return: A Model object containing the simulation setup for the eighth puzzle.
    """

    sorted_colors = colour_count(input_grid)
    background_color = sorted_colors[0][0]
    line_color = sorted_colors[1][0]
    sequence_colors = [color for color, frequency in sorted_colors[2:]]

    # we initialise a border agent which other agents can follow
    agents = [Agent(
        position=unmask(input_grid == line_color),
        direction="none",
        label="border",
        node=RuleNode(cast(Rule, identity_rule)),
        colors=iter([line_color]),
        charge=0
    )]

    # puzzle eight follows the following sets of rules:
    # there are N arbitrary polygons and each polygon may have at most one sequence of colors associated with it
    # first we find all points that are associated with an instruction color
    points: dict[Point, int] = {}  # {point: color}
    for target_color in sequence_colors:
        target_mask = input_grid == target_color
        labeled_grid, bounding_box, num_objects = find_connected_objects(target_mask)
        for i in range(1, num_objects + 1):
            point = (
                bounding_box[(i - 1), 0][0].item(),
                bounding_box[(i - 1), 0][1].item(),
            )

            points[point] = target_color

    # next we find all polygons and their associated sequence
    point_set = PointSet(points.keys())
    color_pos = np.array([point for point in points])
    grid = input_grid.copy()
    grid[color_pos[:, 0], color_pos[:, 1]] = background_color
    lbl_grid, bboxes, num_polys = find_connected_objects(grid == background_color)

    for i in range(1, num_polys + 1):
        box_mask = lbl_grid == i
        box_points = unmask(box_mask)
        sequence_points = point_set & box_points

        if len(sequence_points) > 0:
            # once we found the starting colors, we want to find the corresponding corner of the polygon,
            direction_count = Counter()

            random_point = next(iter(sequence_points))
            for direction in ["top_left", "top_right", "bottom_left", "bottom_right"]:
                direction_unit = direction_to_unit_vector(direction)
                next_point = (random_point[0] + direction_unit[0], random_point[1] + direction_unit[1])

                while 0 < next_point[0] < grid.shape[0] and \
                        0 < next_point[1] < grid.shape[1] and box_mask[next_point]:
                    direction_count[direction] += 1
                    direction_unit = direction_to_unit_vector(direction)
                    next_point = (next_point[0] + direction_unit[0], next_point[1] + direction_unit[1])

            sequence_direction = orthogonal_direction(direction_count.most_common()[0][0], axis="diagonal")
            sorted_sequence_points = sorted(
                sequence_points,
                reverse=sequence_direction in ["bottom_left", "bottom_right"],
            )

            # we found the starting point, and we want to find the largest rectangle from this point
            start_point = sorted_sequence_points[0]

            if sequence_direction == "top_left":
                min_x = start_point[0]
                max_x = grid.shape[0]
                step_x = 1
                min_y = start_point[1]
                max_y = grid.shape[1]
                step_y = 1
            elif sequence_direction == "top_right":
                min_x = start_point[0]
                max_x = grid.shape[0]
                step_x = 1
                min_y = start_point[1]
                max_y = -1
                step_y = -1
            elif sequence_direction == "bottom_left":
                min_x = start_point[0]
                max_x = -1
                step_x = -1
                min_y = start_point[1]
                max_y = grid.shape[1]
                step_y = -1
            else:
                min_x = start_point[0]
                max_x = -1
                step_x = -1
                min_y = start_point[1]
                max_y = -1
                step_y = -1

            rectangle_size = (0, 0)

            for x in range(min_x, max_x, step_x):
                for y in range(min_y, max_y, step_y):
                    if box_mask[x, y]:
                        rectangle_size = (x - min_x, y - min_y)
                    else:
                        break

            # we use the shorted side of the rectangle to determine the number of agents,
            # as agents need to repeat on either side of the rectangle,
            # e.g. if a rectangle is 10x6, and we have 3 agents, we will have 3 agents on the left side and 3 agents on the right side, and one agent in the middle,
            # from this we can determine the number of agents we need to create, their positions, and we also let them run in both directions around the polygon,
            shortest_side = abs(min(rectangle_size))
            num_agents = (shortest_side // len(sorted_sequence_points)) + (shortest_side % len(sorted_sequence_points))
            print("HI")

    agents += [Agent(
        position=PointSet({(point[0], point[1])}),
        direction="right",
        label="point",
        node=RuleNode(
            TrappedCollisionRule(direction_rule=clockwise_direction_90, select_direction=True),
            alternative_node=RuleNode(
                CollisionDirectionRule(direction_rule=clockwise_direction_90, select_direction=True),
                alternative_node=RuleNode(
                    StayInGridRule(direction_rule=clockwise_direction_90, grid_size=input_grid.shape),
                    alternative_node=RuleNode(
                        LeftBottomRule(direction_rule=counterclockwise_direction_90),
                        alternative_node=RuleNode(
                            DirectionRule(direction_rule=identity_direction, select_direction=True)
                        )
                    )
                )
            )
        ),
        colors=cycle([target_color]),
        charge=-1
    ) for (point, target_color) in sorted_points]

    return Model(
        output_grid=input_grid,
        agents=agents,
        execution_mode="sequential",
        collision_mode="history",
    )
