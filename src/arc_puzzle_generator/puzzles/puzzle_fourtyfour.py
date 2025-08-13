from itertools import cycle

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, OutOfGridRule, DirectionRule
from arc_puzzle_generator.utils.entities import find_connected_objects
from arc_puzzle_generator.utils.grid import unmask


def puzzle_fourtyfour(input_grid: np.ndarray) -> Playground:
    """
    Implements the 44th puzzle in the ARC dataset.
    In this puzzle a shooter follows a path of balloons.
    :param input_grid: The input grid for the puzzle.
    :return: A Playground object representing the puzzle.
    """

    output_grid = input_grid.copy()
    agent_tip_color = 4
    agent_color = 2
    background_color = 8

    agents: list[Agent] = []
    labels, bbox, num_blocks = find_connected_objects((input_grid == 3) | (input_grid == 1))

    for i in range(1, num_blocks + 1):
        agents.append(Agent(
            position=unmask(labels == i),
            direction="none",
            colors=iter([agent_color]),
            charge=0,
            label=f"ball_{i}",
        ))
        output_grid[labels == i] = background_color

    agents.append(Agent(
        position=unmask(input_grid == agent_tip_color),
        direction="up",
        label="shooter",
        colors=cycle([agent_color]),
        node=RuleNode(
            OutOfGridRule(grid_size=input_grid.shape),
            alternative_node=RuleNode(
                DirectionRule(direction_rule=identity_direction)
            )
        ),
        charge=-1
    ))

    return Playground(
        output_grid=output_grid,
        agents=agents
    )
