from unittest import TestCase

from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.rule import identity_rule
from arc_puzzle_generator.state import AgentState


class RuleTest(TestCase):
    def test_identity_rule(self):
        states = [AgentState(
            PointSet([(0, 0)]),
            "none",
            0,
            0
        )]

        original_colors = [1, 2, 3]
        colors = iter(original_colors)

        output_state, output_colors = identity_rule(states, colors)

        self.assertEqual(output_state, states[0])
        self.assertEqual(original_colors, list(output_colors))
