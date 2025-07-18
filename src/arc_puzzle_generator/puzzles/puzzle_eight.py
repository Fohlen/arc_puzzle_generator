from itertools import cycle
from typing import cast

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction, clockwise_direction_90, \
    counterclockwise_direction_90
from arc_puzzle_generator.geometry import unmask, PointSet, Point
from arc_puzzle_generator.model import Model
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
            # once we found the starting colors, we want to try and find the largest rectangle that contains all of them,
            point_rectangle: dict[Point, Point] = {}

            for point in sequence_points:
                for x in range(point[0], grid.shape[0]):
                    for y in range(point[1], grid.shape[1]):
                        if box_mask[x, y]:
                            point_rectangle[point] = (x - point[0], y - point[1])

            sorted_sequence_points = sorted(
                sequence_points,
                key=lambda sort_point: point_rectangle[sort_point][0] * point_rectangle[sort_point][1],
                reverse=True
            )

            # we use the shorted side of the rectangle to determine the number of agents,
            # as agents need to repeat on either side of the rectangle,
            # e.g. if a rectangle is 10x6, and we have 3 agents, we will have 3 agents on the left side and 3 agents on the right side, and one agent in the middle,
            # from this we can determine the number of agents we need to create, their positions, and we also let them run in both directions around the polygon,
            shortest_side = min(point_rectangle[sorted_sequence_points[0]])
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
