import random
from itertools import cycle
from typing import cast

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.geometry import unmask, PointSet
from arc_puzzle_generator.neighbourhood import moore_neighbours
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, identity_rule, OutOfGridRule, uncharge_rule, Rule, \
    GravityRule
from arc_puzzle_generator.topology import all_topology


def puzzle_eighteen(input_grid: np.ndarray) -> Playground:
    """
    Generate the output grid for puzzle eighteen simulating water flow.

    :return: Model instance with the output grid.
    """

    # set up random seed for reproducibility
    random.seed(42)

    colors = np.unique(input_grid).tolist()

    agents = [Agent(
        position=unmask(input_grid == target_color),
        direction="none",
        label="border",
        node=RuleNode(cast(Rule, identity_rule)),
        colors=cycle([target_color]),
        charge=0
    ) for target_color in colors if target_color not in [0, 1]]

    water_mask = input_grid == 1
    agents += [Agent(
        position=PointSet([point]),
        direction="none",
        label="water",
        node=RuleNode(
            OutOfGridRule(grid_size=input_grid.shape),
            alternative_node=RuleNode(
                GravityRule(grid_size=input_grid.shape),
                alternative_node=RuleNode(
                    cast(Rule, uncharge_rule)
                )
            ),
        ),
        colors=cycle([1]),
        charge=40 if point[0] < input_grid.shape[0] - 1 else 0
    ) for point in sorted(unmask(water_mask), reverse=True)]

    return Playground(
        input_grid.copy(),
        agents,
        neighbourhood=moore_neighbours,
        topology=all_topology,
        execution_mode="parallel",
        backfill_color=0
    )
