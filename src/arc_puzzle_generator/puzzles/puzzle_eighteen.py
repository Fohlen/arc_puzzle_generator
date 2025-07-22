import logging
from itertools import cycle
from typing import cast

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.geometry import unmask, PointSet
from arc_puzzle_generator.model import Model
from arc_puzzle_generator.neighbourhood import moore_neighbours
from arc_puzzle_generator.rule import RuleNode, identity_rule, OutOfGridRule, uncharge_rule, Rule, \
    BackFillColorRule, backtrack_rule, GravityRule
from arc_puzzle_generator.topology import all_topology

logging.basicConfig(level=logging.DEBUG)


def puzzle_eighteen(input_grid: np.ndarray) -> Model:
    """
    Generate the output grid for puzzle eighteen simulating water flow.

    :return: Model instance with the output grid.
    """

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
                next_node=RuleNode(
                    BackFillColorRule(fill_color=0),
                    next_node=RuleNode(
                        cast(Rule, backtrack_rule)
                    )
                ),
                alternative_node=RuleNode(
                    cast(Rule, uncharge_rule)
                )
            ),
        ),
        colors=cycle([1]),
        charge=20
    ) for point in sorted(unmask(water_mask))]

    return Model(
        input_grid.copy(),
        agents,
        neighbourhood=moore_neighbours,
        topology=all_topology,
        execution_mode="parallel"
    )
