from itertools import cycle

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.neighbourhood import moore_neighbours
from arc_puzzle_generator.physics import direction_to_unit_vector, shift
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, DirectionRule
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
    background_color = sorted_colors[0][0]
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
        for point in line:
            next_point = shift(point, direction_to_unit_vector("bottom_right"))
            charge = 1


            while 0 < next_point[0] < input_grid.shape[0] and 0 < next_point[1] < input_grid.shape[1]:
                if input_grid[next_point[0], next_point[1]] == background_color:
                    next_point = shift(next_point, direction_to_unit_vector("bottom_right"))
                    charge += 1
                else:
                    agents.append(Agent(
                        position=PointSet([
                            shift(point, direction_to_unit_vector("bottom_right")),
                            shift(point, direction_to_unit_vector("right"))
                        ]),
                        direction="bottom_right",
                        label="agent",
                        colors=cycle([foreground_color]),
                        node=RuleNode(
                            DirectionRule(direction_rule=identity_direction)
                        ),
                        charge=charge,
                    ))
                    break

    return Playground(
        output_grid=input_grid.copy(),
        agents=agents,
    )
