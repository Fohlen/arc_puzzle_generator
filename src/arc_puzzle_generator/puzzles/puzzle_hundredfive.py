from itertools import combinations, cycle

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.geometry import Point, PointSet, Direction
from arc_puzzle_generator.physics import shift, direction_to_unit_vector
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, OutOfGridRule, CollisionConditionRule
from arc_puzzle_generator.utils.grid import unmask


def same_diagonal(point1: Point, point2: Point) -> bool:
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) == abs(y1 - y2)


def puzzle_hundredfive(input_grid: np.ndarray) -> Playground:
    points_blue = unmask(input_grid == 1)
    points_purple = unmask(input_grid == 6)

    agents: list[Agent] = []

    for point1, point2 in combinations(sorted(points_blue), 2):
        if same_diagonal(point1, point2):
            direction: Direction
            beam1: Direction
            beam2: Direction
            if point1[1] < point2[1]:
                direction = "bottom_right"
                beam1 = "top_right"
                beam2 = "bottom_left"
            else:
                direction = "bottom_left"
                beam1 = "top_left"
                beam2 = "bottom_right"

            direction_vector = direction_to_unit_vector(direction)
            next_position = point1

            line_points: list[Point] = [
                point for point in points_purple if
                same_diagonal(point, point1) and
                same_diagonal(point, point2)
            ]
            colors = [1]
            while next_position != point2:
                next_position = shift(next_position, direction_vector)
                if next_position in line_points:
                    colors.append(6)
                else:
                    colors.append(1)

            colors.append(1) # to include the last point

            agents.append(Agent(
                position=PointSet([point1]),
                charge=len(colors) - 1,
                direction=direction,
                label="blue",
                colors=iter(colors),
                node=RuleNode(
                    CollisionConditionRule(
                        direction_rule=identity_direction,
                        conditions=[(False, "none")]
                    ),
                )
            ))

            agents.extend([
                Agent(
                    position=PointSet([point]),
                    charge=-1,
                    direction=beam1,
                    label="purple",
                    colors=cycle([6]),
                    node=RuleNode(
                        OutOfGridRule(grid_size=input_grid.shape),
                        alternative_node=RuleNode(
                            CollisionConditionRule(
                                direction_rule=identity_direction,
                                conditions=[(False, "none")]
                            ),
                        )
                    )
                ) for point in line_points
            ])
            agents.extend([
                Agent(
                    position=PointSet([point]),
                    charge=-1,
                    direction=beam2,
                    label="purple",
                    colors=cycle([6]),
                    node=RuleNode(
                        OutOfGridRule(grid_size=input_grid.shape),
                        alternative_node=RuleNode(
                            CollisionConditionRule(
                                direction_rule=identity_direction,
                                conditions=[(False, "none")]
                            ),
                        )
                    )
                ) for point in line_points
            ])

    return Playground(
        output_grid=input_grid,
        agents=agents,
    )
