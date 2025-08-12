from itertools import cycle

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import clockwise_direction_90, identity_direction
from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.neighbourhood import MooreNeighbourhood
from arc_puzzle_generator.physics import direction_to_unit_vector
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, OutOfGridRule, DirectionRule
from arc_puzzle_generator.utils.entities import colour_count, find_connected_objects
from arc_puzzle_generator.utils.grid import unmask


def puzzle_sixtyfour(input_grid: np.ndarray) -> Playground:
    """
    Generates a playground for the 64th puzzle in the ARC dataset.
    This puzzle involves rectangle spawners that spawn agents in a specific direction.
    :param input_grid: The input grid representing the puzzle state.
    :return: A Playground instance that simulates the puzzle.
    """

    background_colors = [0, 1]
    sorted_colors = colour_count(input_grid)

    boxes: list[tuple[np.ndarray]] = []  # [(mask), ...]
    beams: list[tuple[np.ndarray, int]] = []  # [(mask, color), ...]

    for color, count in sorted_colors:
        if color not in background_colors:
            labels, bbox, num_objects = find_connected_objects(input_grid == color, neighbourhood=MooreNeighbourhood())
            for i in range(1, num_objects + 1):
                if np.sum(labels == i) == 8:
                    boxes.append((labels == i))
                else:
                    beams.append((labels == i, color))

    # we walk through each box and beam and shift each beam in all four directions
    # if all shifted points of a beam are within the box, we can spawn the beam in the orthogonal direction
    agents: list[Agent] = []

    for box_labels in boxes:
        for beam_labels, beam_color in beams:
            beam_points = unmask(beam_labels)

            for direction in ["top_left", "top_right", "bottom_right", "bottom_left"]:
                shifted_points = beam_points.shift(direction_to_unit_vector(direction))
                if all(box_labels[point[0], point[1]] for point in shifted_points):
                    agent_direction = clockwise_direction_90(clockwise_direction_90(direction))
                    agents.extend([
                        Agent(
                            position=PointSet([min(beam_points)]),
                            direction=agent_direction,
                            label=f"{beam_color}_{agent_direction}_start",
                            colors=cycle([beam_color]),
                            node=RuleNode(
                                OutOfGridRule(grid_size=input_grid.shape),
                                alternative_node=RuleNode(
                                    DirectionRule(direction_rule=identity_direction)
                                )
                            ),
                            charge=-1,
                        ),
                        Agent(
                            position=PointSet([max(beam_points)]),
                            direction=agent_direction,
                            label=f"{beam_color}_{agent_direction}_end",
                            colors=cycle([beam_color]),
                            node=RuleNode(
                                OutOfGridRule(grid_size=input_grid.shape),
                                alternative_node=RuleNode(
                                    DirectionRule(direction_rule=identity_direction)
                                )
                            ),
                            charge=-1,
                        ),
                    ])
                    break

    return Playground(
        output_grid=input_grid.copy(),
        agents=agents,
    )
