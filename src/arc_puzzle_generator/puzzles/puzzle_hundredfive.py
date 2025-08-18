import math
from itertools import combinations, cycle

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.geometry import Point, PointSet
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, DirectionRule
from arc_puzzle_generator.utils.grid import unmask

def same_diagonal(point1: Point, point2: Point) -> bool:
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) == abs(y1 - y2)


def puzzle_hundredfive(input_grid: np.ndarray) -> Playground:
    points_blue = unmask(input_grid == 1)

    agents: list[Agent] = []

    for point1, point2 in combinations(points_blue, 2):
        if same_diagonal(point1, point2):
            agents.append(Agent(
                position=PointSet([point1]),
                charge=int(math.dist(point1, point2)),
                direction="bottom_right",
                label="blue",
                colors=cycle([1]),
                node=RuleNode(
                    DirectionRule(direction_rule=identity_direction)
                )
            ))

    return Playground(
        output_grid=input_grid,
        agents=agents,
    )
