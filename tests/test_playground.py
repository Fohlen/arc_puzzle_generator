from itertools import cycle
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.direction import identity_direction
from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.playground import Playground
from arc_puzzle_generator.rule import RuleNode, identity_rule, DirectionRule


class ModelTest(TestCase):
    def test_only_one_agent_steps_called_per_step(self):
        # Create two agents with mocked processing functions
        agent1 = Agent(
            position=PointSet([(0, 0)]),
            direction="right",
            label='A',
            node=RuleNode(DirectionRule(direction_rule=identity_direction)),
            colors=iter([1]),
            charge=1
        )
        agent2 = Agent(
            position=PointSet([(1, 1)]),
            direction="right",
            label='B',
            node=RuleNode(DirectionRule(direction_rule=identity_direction)),
            colors=iter([2]),
            charge=1
        )

        # Create model with sequential execution
        grid = np.zeros((2, 2), dtype=int)
        model1 = Playground(grid, [agent1, agent2])

        # Mock the _process_agent method to avoid actual processing
        model1._process_agent = MagicMock(return_value=None)

        # Step forward
        model1.step()

        # Assert that _process_agent was called for each agent
        self.assertEqual(agent2, model1._process_agent.call_args_list[-1].args[0])
        self.assertEqual(agent1, model1._process_agent.call_args_list[-2].args[0])

        # Step forward again
        model1.step()

        self.assertEqual(agent2, model1._process_agent.call_args_list[-1].args[0])
        self.assertEqual(agent1, model1._process_agent.call_args_list[-2].args[0])

    def test_model_sequential_processing(self):
        agent1 = Agent(
            position=PointSet([(0, 0)]),
            direction="right",
            label='A',
            node=RuleNode(DirectionRule(direction_rule=identity_direction)),
            colors=cycle([1]),
            charge=1
        )
        agent2 = Agent(
            position=PointSet([(1, 1)]),
            direction="right",
            label='B',
            node=RuleNode(DirectionRule(direction_rule=identity_direction)),
            colors=cycle([2]),
            charge=1
        )

        # Create model with sequential execution
        grid = np.zeros((2, 2), dtype=int)
        model1 = Playground(grid, [agent1, agent2], execution_mode='sequential')

        agent1.steps = MagicMock(side_effect=agent1.steps)
        agent2.steps = MagicMock(side_effect=agent2.steps)

        # Step forward
        model1.step()

        self.assertEqual(1, agent1.steps.call_count)
        self.assertEqual(0, agent2.steps.call_count)

        # Step forward again
        model1.step()
        self.assertEqual(1, agent1.steps.call_count)
        self.assertEqual(1, agent2.steps.call_count)
