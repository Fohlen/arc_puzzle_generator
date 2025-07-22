from unittest import TestCase

from arc_puzzle_generator.direction import orthogonal_direction
from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.physics import Direction
from arc_puzzle_generator.rule import identity_rule, DirectionRule, CollisionDirectionRule
from arc_puzzle_generator.state import AgentState


def dummy_direction_rule(direction: Direction) -> Direction:
    #  Returns a fixed direction for testing
    return "up"


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

    def test_direction_rule_no_collision(self):
        states = [AgentState(PointSet([(0, 0)]), "none", 1, 1)]
        colors = iter([1, 2])
        collision = PointSet()  # No collision

        rule = DirectionRule(dummy_direction_rule)
        result = rule(states, colors, collision)
        self.assertIsNotNone(result)
        new_state, new_colors = result
        self.assertIn((-1, 0), new_state.position)
        self.assertEqual("up", new_state.direction, )
        self.assertEqual(0, new_state.charge)
        self.assertEqual(1, new_state.color)

    def test_direction_rule_with_collision(self):
        states = [AgentState(PointSet([(0, 0)]), "none", 1, 1)]
        colors = iter([1, 2])
        collision = PointSet([(1, 0)])  # Collision present

        rule = DirectionRule(dummy_direction_rule)
        result = rule(states, colors, collision)
        self.assertIsNone(result)

    def test_collision_direction_rule_no_collision(self):
        states = [AgentState(PointSet([(0, 0)]), "left", 1, 1)]
        colors = iter([1, 2])
        collision = PointSet()  # No collision

        rule = CollisionDirectionRule(direction_rule=orthogonal_direction)
        result = rule(states, colors, collision, {})
        self.assertIsNone(result)

    def test_collision_direction_rule_with_collision(self):
        states = [AgentState(PointSet([(0, 0)]), "bottom_right", 1, 1)]
        colors = iter([1, 2])
        collision = PointSet([(1, 1)])  # Collision present

        rule = CollisionDirectionRule(orthogonal_direction)
        result = rule(states, colors, collision, {
            (1, 1): AgentState(PointSet([(1, 1)]), "none", 2, 0)
        })

        self.assertIsNotNone(result)
        new_state, new_colors = result
        self.assertEqual("top_right", new_state.direction)
        self.assertIn((-1, 1), new_state.position)
        self.assertEqual(0, new_state.charge)
        self.assertEqual(1, new_state.color)
