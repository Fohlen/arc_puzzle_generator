from unittest import TestCase

from arc_puzzle_generator.direction import orthogonal_direction, snake_direction
from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.physics import Direction
from arc_puzzle_generator.rule import identity_rule, DirectionRule, CollisionDirectionRule, OutOfGridRule, \
    collision_color_mapping_rule, TrappedCollisionRule, CollisionBorderRule, CollisionFillRule, backtrack_rule
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

    def test_out_of_grid_rule_inside(self):
        states = [AgentState(PointSet([(1, 1)]), "up", 1, 1)]
        colors = iter([1, 2])
        collision = PointSet()
        grid_size = (3, 3)
        rule = OutOfGridRule(grid_size)
        result = rule(states, colors, collision)
        self.assertIsNone(result)

    def test_out_of_grid_rule_outside(self):
        states = [AgentState(PointSet([(0, 0)]), "up", 1, 1)]
        colors = iter([1, 2])
        collision = PointSet()
        grid_size = (2, 2)
        rule = OutOfGridRule(grid_size)
        result = rule(states, colors, collision)
        self.assertIsNotNone(result)
        new_state, new_colors = result
        self.assertEqual(0, new_state.charge)
        self.assertEqual(PointSet([(0, 0)]), new_state.position)
        self.assertEqual("up", new_state.direction)
        self.assertEqual(1, new_state.color)

    def test_collision_color_mapping_rule_with_collision(self):
        states = [AgentState(PointSet([(0, 0)]), "down", 1, 1)]
        colors = iter([1, 2])
        collision = PointSet([(1, 0)])
        collision_mapping = {
            (1, 0): AgentState(PointSet([(1, 0)]), "down", 2, 1)
        }

        result = collision_color_mapping_rule(states, colors, collision, collision_mapping)

        self.assertIsNotNone(result)
        new_state, new_colors = result

        self.assertEqual(2, new_state.color)
        self.assertEqual(PointSet([(0, 0)]), new_state.position)
        self.assertEqual("down", new_state.direction)
        self.assertEqual(1, new_state.charge)

    def test_trapped_collision_rule_with_directionality(self):
        states = [AgentState(PointSet([(1, 1)]), "right", 1, 1)]
        colors = iter([1, 2])
        collision = PointSet([(0, 1), (1, 2)])
        collision_mapping = {
            (0, 1): AgentState(PointSet([(0, 1)]), "none", 1, 1),
            (1, 2): AgentState(PointSet([(1, 2)]), "none", 2, 1)
        }
        rule = TrappedCollisionRule(snake_direction, select_direction=True)
        result = rule(states, colors, collision, collision_mapping)
        self.assertIsNotNone(result)
        new_state, new_colors = result
        self.assertEqual(0, new_state.charge)
        self.assertEqual(PointSet([(1, 1)]), new_state.position)
        self.assertEqual("right", new_state.direction)

    def test_collision_border_rule_with_collision(self):
        states = [AgentState(PointSet([(0, 0)]), "right", 1, 1)]
        colors = iter([1, 2])
        collision = PointSet([(0, 1)])
        border_color = 9
        collision_mapping = {
            (0, 1): AgentState(PointSet([(0, 1)]), "none", 2, 1)  # Not border color
        }

        rule = CollisionBorderRule(border_color)
        result = rule(states, colors, collision, collision_mapping)
        self.assertIsNotNone(result)
        new_state, new_colors = result

        self.assertEqual(1, new_state.charge)
        self.assertEqual(PointSet([(0, 1)]), new_state.position)
        self.assertEqual("right", new_state.direction)
        self.assertEqual(9, new_state.color)

    def test_collision_file_rule_with_collision(self):
        states = [AgentState(PointSet([(0, 0)]), "right", 9, 1)]
        colors = iter([1, 2])
        collision = PointSet([(0, 1), (0, 2)])
        collision_mapping = {
            (0, 1): AgentState(PointSet([(0, 1)]), "none", 2, 1),
            (0, 2): AgentState(PointSet([(0, 2)]), "none", 3, 1)
        }

        rule = CollisionFillRule(fill_color=9)
        result = rule(states, colors, collision, collision_mapping)
        self.assertIsNotNone(result)
        new_state, new_colors = result

        self.assertEqual(9, new_state.color)
        self.assertEqual(PointSet([(0, 0), (0, 1), (0, 2)]), new_state.position)
        self.assertEqual(states[0].direction, new_state.direction)
        self.assertEqual(states[0].charge, new_state.charge)

    def test_backtrack_rule(self):
        states = [
            AgentState(PointSet([(0, 0)]), "down", 1, 1),
            AgentState(PointSet([(1, 0)]), "right", 2, 1)
        ]

        colors = iter([1, 2])
        collision = PointSet([(1, 0)])

        result = backtrack_rule(states, colors, collision, {})
        self.assertIsNotNone(result)
        new_state, new_colors = result

        self.assertEqual(states[0], new_state)

