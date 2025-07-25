from itertools import cycle

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import orthogonal_direction, snake_direction
from arc_puzzle_generator.geometry import unmask
from arc_puzzle_generator.neighbourhood import moore_neighbours
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, identity_rule, TrappedCollisionRule, ProximityRule, \
    CollisionDirectionRule
from arc_puzzle_generator.topology import all_topology


def puzzle_twentyfive(input_grid: np.ndarray) -> Playground:
    """
    Generates a puzzle for the 25th puzzle, which is a classic reward learning task.
    :param input_grid: A 2D numpy array representing the input grid for the puzzle.
    :return: A Playground object representing the generated puzzle.
    """

    foreground_color = 1
    background_color = 3
    goal_color = 2
    agent_color = 0

    foreground_points = unmask(input_grid == foreground_color)
    goal_points = unmask(input_grid == goal_color)

    background_points = unmask(input_grid == background_color)
    agent_points = unmask(input_grid == agent_color)

    agents = [
        Agent(
            background_points,
            direction="none",
            label="background",
            node=RuleNode(identity_rule),
            colors=iter([background_color]),
            charge=0,
        ),
        Agent(
            agent_points,
            direction="up",
            label="agent",
            node=RuleNode(
                TrappedCollisionRule(
                    direction_rule=snake_direction,
                    select_direction=True,
                ),
                next_node=RuleNode(
                    CollisionDirectionRule(
                        direction_rule=snake_direction,
                        select_direction=True,
                    )
                ),
                alternative_node=RuleNode(
                    ProximityRule(
                        target=goal_points,
                        points=foreground_points,
                    )
                )
            ),
            colors=cycle([agent_color]),
            charge=10,
        )
    ]

    return Playground(
        output_grid=input_grid.copy(),
        agents=agents,
        neighbourhood=moore_neighbours,
        topology=all_topology,
        backfill_color=foreground_color,
    )
