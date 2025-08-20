from itertools import cycle

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
            color: int
            end: int
            start = 0
            patterns: list[tuple[int, int]] = []

            # special cases
            if (header[0] != background_color and np.all(header[1:] == background_color)) or (
                    header[0] != background_color and header[1] != background_color and np.all(
                header[2:] == background_color)):
                end_idx = np.argwhere(row[4:] != background_color)
                # terminate at agent
                if len(end_idx) > 0:
                    color = row[4 + end_idx].item()
                    end = 4 + end_idx[-1].item() + 1
                else:
                    color = header[0]
                    end = len(row)

                patterns.append((color, 1))
            else:
                pattern = [p for p in np.argwhere(header != background_color).squeeze().tolist()]
                color = row[pattern[0]]
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

                patterns.extend([(color, p) for p in pattern[1:]])
                patterns.extend(irregular_patterns)

            agents.append(Agent(
                position=PointSet([(row_idx, 0)]),
                direction=orientation,
                charge=end - start,
                label=f"agent_{row_idx}",
                colors=ColorSequenceIterator(patterns),
                node=RuleNode(
                    DirectionRule(direction_rule=identity_direction)
                )
            ))

    return Playground(
        output_grid=input_grid,
        agents=agents
    )
