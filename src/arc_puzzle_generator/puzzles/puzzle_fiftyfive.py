from itertools import cycle

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.geometry import PointSet, in_grid
from arc_puzzle_generator.neighbourhood import moore_neighbours
from arc_puzzle_generator.physics import direction_to_unit_vector, shift
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, CollisionConditionRule
from arc_puzzle_generator.utils.entities import colour_count, find_connected_objects


def puzzle_fiftyfive(
        input_grid: np.ndarray,
) -> Playground:
    """
    Implementation of puzzle fiftyfive, a grid filling agent-based puzzle.
    :param input_grid: The input grid for the puzzle.
    :return: A Playground instance representing the puzzle.
    """

    sorted_colors = colour_count(input_grid)
    foreground_color = 2
    border_color = sorted_colors[1][0]

    points = PointSet()
    lines: list[PointSet] = []

    labels, bboxes, num_objects = find_connected_objects(input_grid == border_color, neighbourhood=moore_neighbours)
    for i in range(1, num_objects + 1):
        line = PointSet.from_numpy(np.argwhere(labels == i))
        lines.append(line)
        points.update(line)

    # one can brute-force puzzle 55 by running a pre-agent agent simulation
    agents: list[Agent] = []
    for line in lines:
        # sort the line by y-coordinate, then by x-coordinate
        for point_idx, point in enumerate(sorted(line, key=lambda p: (p[1], p[0]))):
            next_point = shift(point, direction_to_unit_vector("bottom_right"))
            charge = 1
            diagonal = False
            hit = False

            while not hit and in_grid(next_point, input_grid.shape):
                right_point = shift(next_point, direction_to_unit_vector("right"))
                bottom_point = shift(next_point, direction_to_unit_vector("down"))

                if input_grid[next_point[0], next_point[1]] == border_color:
                    hit = True
                elif in_grid(bottom_point, input_grid.shape) and input_grid[
                    bottom_point[0], bottom_point[1]] == border_color and \
                        in_grid(right_point, input_grid.shape) and input_grid[
                    right_point[0], right_point[1]] == border_color:
                    hit = True
                    diagonal = True
                else:
                    next_point = shift(next_point, direction_to_unit_vector("bottom_right"))
                    charge += 1

            if hit:
                agents.append(
                    Agent(
                        position=PointSet([shift(point, direction_to_unit_vector("bottom_right"))]),
                        direction="bottom_right",
                        label="agent",
                        colors=cycle([foreground_color]),
                        node=RuleNode(
                            CollisionConditionRule(
                                direction_rule=identity_direction,
                                conditions=[(False, "none")]
                            ),
                        ),
                        charge=charge if diagonal else charge - 1,
                    )
                )

                # all but the last point in the line should have an agent to the right
                if point_idx < len(line) - 1:
                    agents.append(
                        Agent(
                            position=PointSet([shift(point, direction_to_unit_vector("right"))]),
                            direction="bottom_right",
                            label="agent",
                            colors=cycle([foreground_color]),
                            node=RuleNode(
                                CollisionConditionRule(
                                    direction_rule=identity_direction,
                                    conditions=[(False, "none")]
                                ),
                            ),
                            charge=charge,
                        )
                    )

    return Playground(
        output_grid=input_grid.copy(),
        agents=agents,
    )
