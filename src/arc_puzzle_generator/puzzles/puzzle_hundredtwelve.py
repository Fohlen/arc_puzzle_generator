from itertools import cycle

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.geometry import Direction, PointSet
from arc_puzzle_generator.neighbourhood import MooreNeighbourhood
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, DirectionRule
from arc_puzzle_generator.utils.entities import find_connected_objects


def puzzle_hundredtwelve(
        input_grid: np.ndarray,
        orientation: Direction = "top_right",
) -> Playground:
    """
    Puzzle 112 is a puzzle in which the middle strand determines the length of all other strands.
    :param input_grid: The input grid.
    :param orientation: The orientation of the puzzle.
    :return: A Playground instance.
    """

    output_grid = input_grid.copy()
    background_color = 7

    agents: list[Agent] = []
    labels, bbox, num_objects = find_connected_objects(
        input_grid != background_color,
        neighbourhood=MooreNeighbourhood()
    )



    middle = (1 + num_objects + 1) // 2
    charge = np.sum(labels == middle)

    for i in range(1, num_objects + 1):
        color = input_grid[labels == i][0].item()
        start: list[int]
        if orientation in ["top_right", "top_left", "up"]:
            start = np.argwhere(labels == i)[-1].tolist()
        elif orientation in ["bottom_right", "bottom_left", "down"]:
            start = np.argwhere(labels == i)[0].tolist()
        elif orientation == "right":
            start = np.argwhere(labels == i)[0].tolist()
        else:
            start = np.argwhere(labels == i)[-1].tolist()

        agents.append(Agent(
            position=PointSet([(start[0], start[1])]),
            direction=orientation,
            label=f"agent_{i}",
            charge=charge,
            colors=cycle([color]),
            node=RuleNode(
                DirectionRule(direction_rule=identity_direction)
            )
        ))
        output_grid[labels == i] = background_color

    return Playground(
        output_grid=output_grid,
        agents=agents,
    )
