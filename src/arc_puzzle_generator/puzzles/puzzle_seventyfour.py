from itertools import cycle

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction, counterclockwise_direction_90, clockwise_direction_90
from arc_puzzle_generator.neighbourhood import MooreNeighbourhood
from arc_puzzle_generator.physics import direction_to_unit_vector
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, DirectionRule, StayInGridRule, TerminateAtPoint, \
    CollisionConditionDirectionRule
from arc_puzzle_generator.topology import all_topology
from arc_puzzle_generator.utils.grid import unmask


def puzzle_seventyfour(input_grid: np.ndarray) -> Playground:
    """
    Implements puzzle 74 in which the snake borders the grid and the snake is a single line that does not intersect itself.
    :param input_grid: The input grid as a 2D NumPy array.
    :return: The Playground object representing the puzzle.
    """

    start_point = unmask(input_grid == 6)

    agents: list[Agent] = [Agent(
        position=unmask(input_grid == 1),
        direction="none",
        label="obstacle",
        colors=iter([1]),
        charge=0,
    ), Agent(
        position=start_point.shift(direction_to_unit_vector("right")),
        direction="right",
        label="snake",
        node=RuleNode(
            TerminateAtPoint(target=start_point, direction_rule=identity_direction),
            alternative_node=RuleNode(
                StayInGridRule(grid_size=input_grid.shape, direction_rule=counterclockwise_direction_90),
                alternative_node=RuleNode(
                    CollisionConditionDirectionRule(
                        direction_rule=counterclockwise_direction_90,
                        conditions=[
                            (True, "up"),
                            (False, "left"),
                        ]
                    ),
                    alternative_node=RuleNode(
                        CollisionConditionDirectionRule(
                            direction_rule=clockwise_direction_90,
                            conditions=[
                                (True, "down"),
                                (True, "bottom_right"),
                                (False, "right"),
                            ]
                        ),
                        alternative_node=RuleNode(
                            DirectionRule(direction_rule=identity_direction, select_direction=True),
                        )
                    ),
                )
            )
        ),
        colors=cycle([7]),
        charge=-1,
    )]

    return Playground(
        output_grid=input_grid,
        agents=agents,
        neighbourhood=MooreNeighbourhood(),
        topology=all_topology,
        collision_mode="history",
    )
