from itertools import cycle

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.neighbourhood import moore_neighbours, resolve_point_set_neighbourhood
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import OutOfGridRule, RuleNode, CollisionConditionRule
from arc_puzzle_generator.utils.entities import find_connected_objects, relative_box_direction, get_bounding_box, \
    direction_to_numpy_unit_vector
from arc_puzzle_generator.utils.grid import unmask


def puzzle_thirtynine(
        input_grid: np.ndarray,
) -> Playground:
    """
    Generates a playground for puzzle thirty-nine based on the provided input grid.
    :param input_grid: The input grid representing the initial state of the puzzle.
    :return: A Playground instance configured for puzzle thirty-nine.
    """

    background_color = 8
    entity_color = 1
    agents: list[Agent] = []

    labels, bboxes, num_objects = find_connected_objects(input_grid == entity_color)
    for i in range(1, num_objects + 1):
        neighbours = resolve_point_set_neighbourhood(
            unmask(labels == i),
            moore_neighbours
        )

        neighbours_np = np.array(list(neighbours))
        neighbours_idx = np.where(
            input_grid[neighbours_np[:, 0], neighbours_np[:, 1]] != background_color
        )

        if len(neighbours_idx):
            neighbour_bbox = get_bounding_box(neighbours_np[neighbours_idx])

            direction = relative_box_direction(
                bboxes[i - 1],
                neighbour_bbox
            )

            points = [
                neighbours_np[idx]
                for idx in neighbours_idx[0]
            ]

            colors: list[list[int]] = [[] for _ in points]

            while not any(
                    point[0] < 0 or point[0] >= input_grid.shape[0] or
                    point[1] < 0 or point[1] >= input_grid.shape[1] or
                    input_grid[point[0], point[1]] == background_color
                    for point in points
            ):
                for index, point in enumerate(points):
                    points[index] = point + direction_to_numpy_unit_vector(direction)
                    colors[index].append(input_grid[point[0], point[1]].item())

            for point, color_sequence in zip(points, colors):
                if len(color_sequence) <= 2:  # Hard-coded condition for puzzle 39
                    agents.append(Agent(
                        position=PointSet([(point[0].item(), point[1].item())]),
                        direction=direction,
                        label=f"agent_{i}",
                        node=RuleNode(
                            OutOfGridRule(grid_size=input_grid.shape),
                            alternative_node=RuleNode(
                                CollisionConditionRule(
                                    direction_rule=identity_direction,
                                    conditions=[(False, "none")]
                                ),
                            ),
                        ),
                        colors=cycle(color_sequence),
                        charge=-1,
                    ))

    return Playground(
        output_grid=input_grid,
        agents=agents,
    )
