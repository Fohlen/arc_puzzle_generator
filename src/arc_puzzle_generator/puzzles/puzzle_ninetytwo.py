import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.geometry import Direction, PointSet
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, DirectionRule
from arc_puzzle_generator.utils.color_sequence_iterator import ColorSequenceIterator


def puzzle_ninetytwo(input_grid: np.ndarray, orientation: Direction = "right") -> Playground:
    """
    In puzzle 92 one needs to figure out a repeat pattern problem by row / column.
    :param input_grid: The input grid.
    :param orientation: The direction that the input grid is facing.
    :return: A Playground instance.
    """

    background_color = 0
    agents: list[Agent] = []

    for row_idx, row in enumerate(input_grid):
        if np.any(row != background_color):
            header = row[:4]
            pattern = [p for p in np.argwhere(header != background_color).squeeze().tolist()]
            color = row[pattern[0]]
            start = 0
            end = len(row)
            irregular_patterns = []

            for i in range(4, end):
                for idx in pattern[1:]:
                    if row[i] != background_color:
                        if i % (idx + 1) == 0:
                            color = row[i]
                            end = i
                            break
                        else:
                            irregular_patterns.append((i, row[i]))

            agents.append(Agent(
                position=PointSet([(row_idx, 0)]),
                direction=orientation,
                charge=end - start,
                label=f"agent_{row_idx}",
                colors=ColorSequenceIterator(
                    [
                        (color, p) for p in pattern[1:]
                    ] + irregular_patterns,
                    background_color=background_color,
                ),
                node=RuleNode(
                    DirectionRule(direction_rule=identity_direction)
                )
            ))

    return Playground(
        output_grid=input_grid,
        agents=agents
    )
