from unittest import TestCase

import numpy as np

from arc_puzzle_generator.collisions import collision_neighbourhood


class CollisionTest(TestCase):
    def test_collision_direction_right(self):
        step = np.array([[5, 4]])
        self.assertTrue(np.array_equal(
            np.array([[5, 5]]),
            collision_neighbourhood(step, "right")
        ))

        step_two = np.array([
            [5, 3], [5, 4]
        ])
        self.assertTrue(np.array_equal(
            np.array([[5, 5]]),
            collision_neighbourhood(step_two, "right")
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
            collision_neighbourhood(step_three, "right")
        ))

    def test_collision_direction_left(self):
        step = np.array([[5, 3]])
        self.assertTrue(np.array_equal(
            [[5, 2]],
            collision_neighbourhood(step, "left")
        ))

        step_two = np.array([
            [5, 3],
            [5, 4]
        ])
        self.assertTrue(np.array_equal(
            [[5, 2]],
            collision_neighbourhood(step_two, "left")
        ))

        step_three = np.array([
            [4, 3],
            [5, 3]
        ])
        self.assertTrue(np.array_equal(
            [[4, 2], [5, 2]],
            collision_neighbourhood(step_three, "left")
        ))

    def test_collision_direction_up(self):
        step = np.array([[5, 3]])
        self.assertTrue(np.array_equal(
            [[4, 3]],
            collision_neighbourhood(step, "up")
        ))

        step_two = np.array([
            [4, 3],
            [5, 3],
        ])
        self.assertTrue(np.array_equal(
            [[3, 3]],
            collision_neighbourhood(step_two, "up")
        ))

        step_three = np.array([
            [4, 3], [4, 4]
        ])
        self.assertTrue(np.array_equal(
            [[3, 3], [3, 4]],
            collision_neighbourhood(step_three, "up")
        ))

    def test_collision_direction_down(self):
        step = np.array([[5, 3]])
        self.assertTrue(np.array_equal(
            [[6, 3]],
            collision_neighbourhood(step, "down")
        ))

        step_two = np.array([
            [4, 3],
            [5, 3],
        ])
        self.assertTrue(np.array_equal(
            [[6, 3]],
            collision_neighbourhood(step_two, "down")
        ))

        step_three = np.array([
            [5, 3], [5, 4]
        ])
        self.assertTrue(np.array_equal(
            [[6, 3], [6, 4]],
            collision_neighbourhood(step_three, "down")
        ))

    def test_collision_direction_top_left(self):
        step = np.array([[5, 3]])

        self.assertTrue(np.array_equal(
            [[4, 2], [4, 3], [5, 2]],
            collision_neighbourhood(step, "top_left")
        ))

        step_two = np.array([
            [5, 3],
            [6, 4]
        ])

        self.assertTrue(np.array_equal(
            [[4, 2], [4, 3], [5, 2]],
            collision_neighbourhood(step_two, "top_left")
        ))

    def test_collision_direction_top_right(self):
        step = np.array([[5, 3]])

        self.assertTrue(np.array_equal(
            [[4, 3], [4, 4], [5, 4]],
            collision_neighbourhood(step, "top_right")
        ))

        step_two = np.array([
            [5, 3],
            [6, 2]
        ])
        self.assertTrue(np.array_equal(
            [[4, 3], [4, 4], [5, 4]],
            collision_neighbourhood(step_two, "top_right")
        ))

        # step_three = np.array([
        #    [5, 3], [6, 4]
        # ])
        # step_three_neighbourhood = np.array([
        #    [4, 3], [4, 4], [5, 4],
        #    [5, 5], [6, 5]
        # ])

        # self.assertTrue(np.array_equal(
        #    step_three_neighbourhood,
        #    collision_neighbourhood(step_three, "top_right")
        # ))

    def test_collision_direction_bottom_left(self):
        step = np.array([[5, 3]])
        self.assertTrue(np.array_equal(
            [[5, 2], [6, 2], [6, 3]],
            collision_neighbourhood(step, "bottom_left")
        ))

    def test_collision_direction_bottom_right(self):
        step = np.array([[5, 3]])
        self.assertTrue(np.array_equal(
            [[5, 4], [6, 3], [6, 4]],
            collision_neighbourhood(step, "bottom_right")
        ))
