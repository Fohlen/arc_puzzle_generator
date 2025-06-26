from unittest import TestCase

import numpy as np

from arc_puzzle_generator.collisions import directional_neighbourhood, orthogonal_direction, moore_neighbourhood, \
    identity_direction, axis_neighbourhood


class CollisionTest(TestCase):
    def test_moore_neighbourhood(self):
        point = np.array([5, 3])
        neighbours = np.array([
            [4, 2], [4, 3], [4, 4],
            [5, 2], [5, 4],
            [6, 2], [6, 3], [6, 4]
        ])
        self.assertTrue(np.array_equal(moore_neighbourhood(point), neighbours))

    def test_collision_direction_right(self):
        step = np.array([[5, 4]])
        self.assertTrue(np.array_equal(
            np.array([[5, 5]]),
            directional_neighbourhood(step, "right")
        ))

        step_two = np.array([
            [5, 3], [5, 4]
        ])
        self.assertTrue(np.array_equal(
            np.array([[5, 5]]),
            directional_neighbourhood(step_two, "right")
        ))

        step_three = np.array([
            [4, 3],
            [5, 3]
        ])
        self.assertTrue(np.array_equal(
            np.array([
                [4, 4],
                [5, 4]
            ]),
            directional_neighbourhood(step_three, "right")
        ))

    def test_collision_direction_left(self):
        step = np.array([[5, 3]])
        self.assertTrue(np.array_equal(
            [[5, 2]],
            directional_neighbourhood(step, "left")
        ))

        step_two = np.array([
            [5, 3],
            [5, 4]
        ])
        self.assertTrue(np.array_equal(
            [[5, 2]],
            directional_neighbourhood(step_two, "left")
        ))

        step_three = np.array([
            [4, 3],
            [5, 3]
        ])
        self.assertTrue(np.array_equal(
            [[4, 2], [5, 2]],
            directional_neighbourhood(step_three, "left")
        ))

    def test_collision_direction_up(self):
        step = np.array([[5, 3]])
        self.assertTrue(np.array_equal(
            [[4, 3]],
            directional_neighbourhood(step, "up")
        ))

        step_two = np.array([
            [4, 3],
            [5, 3],
        ])
        self.assertTrue(np.array_equal(
            [[3, 3]],
            directional_neighbourhood(step_two, "up")
        ))

        step_three = np.array([
            [4, 3], [4, 4]
        ])
        self.assertTrue(np.array_equal(
            [[3, 3], [3, 4]],
            directional_neighbourhood(step_three, "up")
        ))

    def test_collision_direction_down(self):
        step = np.array([[5, 3]])
        self.assertTrue(np.array_equal(
            [[6, 3]],
            directional_neighbourhood(step, "down")
        ))

        step_two = np.array([
            [4, 3],
            [5, 3],
        ])
        self.assertTrue(np.array_equal(
            [[6, 3]],
            directional_neighbourhood(step_two, "down")
        ))

        step_three = np.array([
            [5, 3], [5, 4]
        ])
        self.assertTrue(np.array_equal(
            [[6, 3], [6, 4]],
            directional_neighbourhood(step_three, "down")
        ))

    def test_collision_direction_top_left(self):
        step = np.array([[5, 3]])

        self.assertTrue(np.array_equal(
            [[4, 2], [4, 3], [5, 2]],
            directional_neighbourhood(step, "top_left")
        ))

        step_two = np.array([
            [5, 3],
            [6, 4]
        ])

        self.assertTrue(np.array_equal(
            [[4, 2], [4, 3], [5, 2]],
            directional_neighbourhood(step_two, "top_left")
        ))

    def test_collision_direction_top_right(self):
        step = np.array([[5, 3]])

        self.assertTrue(np.array_equal(
            [[4, 3], [4, 4], [5, 4]],
            directional_neighbourhood(step, "top_right")
        ))

        step_two = np.array([
            [5, 3],
            [6, 2]
        ])
        self.assertTrue(np.array_equal(
            [[4, 3], [4, 4], [5, 4]],
            directional_neighbourhood(step_two, "top_right")
        ))

    def test_collision_direction_bottom_left(self):
        step = np.array([[5, 3]])
        self.assertTrue(np.array_equal(
            [[5, 2], [6, 2], [6, 3]],
            directional_neighbourhood(step, "bottom_left")
        ))

    def test_collision_direction_bottom_right(self):
        step = np.array([[5, 3]])
        self.assertTrue(np.array_equal(
            [[5, 4], [6, 3], [6, 4]],
            directional_neighbourhood(step, "bottom_right")
        ))

    def test_identity_direction(self):
        self.assertEqual(identity_direction("right"), "right")

    def test_axis_neighbours(self):
        step = np.array([[5, 3]])
        grid_size = [10, 10]
        neighbours = axis_neighbourhood(step, "up", grid_size)
        self.assertEqual(neighbours.shape, (9, 2))
