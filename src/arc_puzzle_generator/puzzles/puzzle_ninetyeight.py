from itertools import cycle

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.geometry import Direction, PointSet
from arc_puzzle_generator.neighbourhood import resolve_point_set_neighbourhood, MooreNeighbourhood, \
    von_neumann_neighbours
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, OutOfGridRule, ProximityRule
from arc_puzzle_generator.topology import all_topology
from arc_puzzle_generator.utils.entities import find_connected_objects
from arc_puzzle_generator.utils.grid import unmask


def puzzle_ninetyeight(input_grid: np.ndarray) -> Playground:
    """
    In puzzle 98 a snake has to avoid obstacles and reach the other side of the grid.
    :param input_grid: The input grid for the puzzle, represented as a 2D numpy array.
    :return: A playground instance that simulates the puzzle environment.
    """

    output_grid = input_grid.copy()
    agents: list[Agent] = []
    labels_obstacles, bbox_obstacles, num_obstacles = find_connected_objects(
        (input_grid == 1) | (input_grid == 2)
    )
    for i in range(1, num_obstacles + 1):
        obstacle_size = np.sum((labels_obstacles == i) & (input_grid == 1))
        obstacle = unmask(labels_obstacles == i)
        neighbourhood = MooreNeighbourhood(size=obstacle_size)

        agents.append(Agent(
            position=resolve_point_set_neighbourhood(obstacle, neighbourhood),
            direction="none",
            label=f"obstacle_{i}",
            colors=iter([1]),
            node=None,
            charge=0,
        ))

        output_grid[labels_obstacles == i] = 2

    agent_color = 3
    agent_pos = unmask(input_grid == agent_color)
    start_pos = next(iter(agent_pos))
    target: PointSet

    direction: Direction
    if start_pos[0] == 0:
        direction = "down"
        target = PointSet([(input_grid.shape[0] - 1, i) for i in range(0, input_grid.shape[1])])
    elif start_pos[0] == input_grid.shape[0] - 1:
        direction = "up"
        target = PointSet([(0, i) for i in range(0, input_grid.shape[1])])
    elif start_pos[1] == 0:
        direction = "right"
        target = PointSet([(i, input_grid.shape[1] - 1) for i in range(0, input_grid.shape[1])])
    else:
        direction = "left"
        target = PointSet([(i, 0) for i in range(0, input_grid.shape[1])])

    grid = PointSet([
        (x, y)
        for x in range(0, input_grid.shape[0])
        for y in range(0, input_grid.shape[1])
        if input_grid[x, y] == 8
    ])
    proximity_rule = ProximityRule(
        target=target,
        points=grid
    )

    agents.append(Agent(
        position=agent_pos,
        direction=direction,
        label="snake",
        colors=cycle([agent_color]),
        node=RuleNode(
            OutOfGridRule(grid_size=input_grid.shape),
            alternative_node=RuleNode(
                proximity_rule,
            )
        ),
        charge=-1,
    ))

    return Playground(
        output_grid=output_grid,
        agents=agents,
        neighbourhood=von_neumann_neighbours,
        topology=all_topology,
        collision_mode="history",
    )
