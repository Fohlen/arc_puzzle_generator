from itertools import cycle

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import clockwise_direction_90, identity_direction, counterclockwise_direction_90
from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.neighbourhood import MooreNeighbourhood
from arc_puzzle_generator.physics import direction_to_unit_vector, shift
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, TerminateAtPoint, CollisionConditionDirectionRule, DirectionRule
from arc_puzzle_generator.topology import all_topology
from arc_puzzle_generator.utils.entities import find_connected_objects, extreme_point, find_holes
from arc_puzzle_generator.utils.grid import unmask


def puzzle_eightysix(input_grid: np.ndarray) -> Playground:
    """
    Puzzle 86 is a puzzle that requires agents to frame polygon shapes.
    :param input_grid: The input grid for the puzzle, represented as a 2D numpy array.
    :return: A Playground instance configured for puzzle 86.
    """

    output_grid = input_grid.copy()
    background = input_grid == 4

    agents: list[Agent] = []
    labels, bbox, num_objects = find_connected_objects(input_grid == 1)
    for i in range(1, num_objects + 1):
        mask = (labels == i)
        holes = find_holes(mask)
        unfilled_area = ((labels == i) & background) | holes

        # this is just the logic of the filling algorithm
        if np.any(unfilled_area):
            output_grid[unfilled_area] = 6
            fill_area = (labels == i) & ~background & ~holes
            output_grid[fill_area] = 8

        agents.append(Agent(
            position=unmask(mask),
            direction="none",
            label=f"block_{i}",
            colors=iter([1]),
            charge=0,
        ))

        # this one is a bit odd, one wants to shift the start point one to the right (or left), otherwise you will not get a full frame
        start_point = shift(
            extreme_point(labels == i, "up"),
            direction_to_unit_vector("top_right")
        )

        agents.append(Agent(
            position=PointSet([start_point]),
            direction="right",
            label=f"frame_{i}",
            colors=cycle([2]),
            charge=-1,
            node=RuleNode(
                TerminateAtPoint(PointSet([start_point]), direction_rule=identity_direction),
                alternative_node=RuleNode(
                    CollisionConditionDirectionRule(
                        direction_rule=counterclockwise_direction_90,
                        conditions=[
                            (True, "up"),
                            (True, "bottom_right"),
                            (True, "right"),
                        ]
                    ),
                    alternative_node=RuleNode(
                        CollisionConditionDirectionRule(
                            direction_rule=clockwise_direction_90,
                            conditions=[
                                (False, "right"),
                                (True, "bottom_right"),
                                (True, "down")
                            ]
                        ),
                        alternative_node=RuleNode(
                            DirectionRule(direction_rule=identity_direction, select_direction=True)
                        )
                    )

                )
            )
        ))

    return Playground(
        output_grid=output_grid,
        agents=agents,
        neighbourhood=MooreNeighbourhood(),
        topology=all_topology,
        collision_mode="history",
    )
