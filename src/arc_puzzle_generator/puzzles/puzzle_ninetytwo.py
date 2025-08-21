from itertools import cycle

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.geometry import Direction, PointSet
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, DirectionRule
from arc_puzzle_generator.utils.color_sequence_iterator import ColorSequenceIterator


def puzzle_ninetytwo(input_grid: np.ndarray, orientation: Direction = "right", cutoff: int = 5) -> Playground:
    """
    In puzzle 92 one needs to figure out a repeat pattern problem by row / column.
    :param input_grid: The input grid.
    :param orientation: The direction that the input grid is facing.
    :param cutoff: The cutoff for the repeat pattern.
    :return: A Playground instance.
    """

    background_color = 0
    agents: list[Agent] = []

    if orientation == "right":
        pattern_grid = input_grid.copy()
    elif orientation == "up":
        pattern_grid = np.rot90(input_grid, k=3)
    elif orientation == "left":
        pattern_grid = np.fliplr(input_grid)
    else:
        pattern_grid = np.rot90(input_grid, k=1)

    for row_idx, row in enumerate(pattern_grid):
        if np.any(row != background_color):
            header = row[:cutoff]
            color: int
            end: int
            start = 0
            patterns: list[tuple[int, int]] = []

            # special cases
            if (header[0] != background_color and np.all(header[1:] == background_color)) or (
                    header[0] != background_color and header[1] != background_color and np.all(
                header[2:] == background_color)):
                end_idx = np.argwhere(row[cutoff:] != background_color)
                # terminate at agent
                if len(end_idx) > 0:
                    color = row[cutoff + end_idx.min()].item()
                    end = cutoff + end_idx.min() + 1
                else:
                    color = header[0]
                    end = len(row)

                patterns.append((color, 1))
            else:
                pattern = [p for p in np.argwhere(header != background_color).squeeze().tolist()]
                color = row[pattern[0]]
                end = len(row)
                irregular_patterns = []

                for i in range(cutoff, end):
                    for idx in pattern[1:]:
                        if row[i] != background_color:
                            if i % idx == 0:
                                color = row[i]
                                end = i
                                break
                            else:
                                irregular_patterns.append((i, row[i]))

                patterns.extend([(color, p) for p in pattern[1:]])
                patterns.extend(irregular_patterns)

            if orientation == "right":
                position = PointSet([(row_idx, 0)])
            elif orientation == "up":
                position = PointSet([(input_grid.shape[0] - 1, row_idx)])
            elif orientation == "left":
                position = PointSet([(row_idx, input_grid.shape[1] - 1)])
            else:
                position = PointSet([(0, row_idx)])

            agents.append(Agent(
                position=position,
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
