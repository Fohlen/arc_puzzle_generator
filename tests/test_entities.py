import unittest

import numpy as np

from arc_puzzle_generator.utils.data_loader import load_puzzle
from arc_puzzle_generator.utils.entities import find_connected_objects, is_l_shape, find_colors, extreme_point, \
    box_contained, starting_point, relative_box_direction, box_distance, get_bounding_box, find_5x5_grids_with_border
from tests.utils import test_dir


class EntityTestCase(unittest.TestCase):
    def setUp(self):
        file_path = test_dir / "data" / "142ca369.json"
        self.puzzle = load_puzzle(file_path)

    def test_colors(self):
        colors = find_colors(self.puzzle.train[0].input, background=0)
        self.assertEqual(len(colors), 4)

    def test_find_connected_objects(self):
        target_mask = self.puzzle.train[0].input == 2
        label_mask, bounding_boxes, object_count = find_connected_objects(target_mask)
        self.assertEqual(object_count, 2)

        # Check that we have the correct number of bounding boxes
        self.assertEqual(bounding_boxes.shape, (2, 4, 2))

        # Check that each bounding box has 4 points (bottom-left, top-left, top-right, bottom-right)
        for i in range(object_count):
            # Verify bottom-left and top-left have same column
            self.assertEqual(bounding_boxes[i, 0, 1], bounding_boxes[i, 1, 1])

            # Verify top-left and top-right have same row
            self.assertEqual(bounding_boxes[i, 1, 0], bounding_boxes[i, 2, 0])

            # Verify top-right and bottom-right have same column
            self.assertEqual(bounding_boxes[i, 2, 1], bounding_boxes[i, 3, 1])

            # Verify bottom-right and bottom-left have same row
            self.assertEqual(bounding_boxes[i, 3, 0], bounding_boxes[i, 0, 0])

        # Manually verify red L-shape in example puzzle
        bbox = np.array([[4, 4], [3, 4], [3, 5], [4, 5]])
        self.assertTrue(np.array_equal(bounding_boxes[0], bbox))

    def test_is_l_shape(self):
        # Example usage:
        array1 = np.array([[0, 1], [1, 1]])
        array2 = np.array([[1, 0], [1, 1]])
        array3 = np.array([[1, 1], [0, 1]])
        array4 = np.array([[1, 1], [1, 0]])
        array5 = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0]])
        array6 = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 1]])
        array7 = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 1]])
        array8 = np.array([[1, 1, 0], [1, 0, 0], [1, 0, 0]])
        array9 = np.array([[1, 1], [1, 1]])  # Not an L-shape
        array10 = np.array([[1, 0, 1], [1, 1, 1]])  # Not an L-shape
        array11 = np.array([[1, 1, 1]])  # Not an L-shape
        array12 = np.array([[1]])  # Not an L-shape
        array13 = np.array([[4, 4, 4], [0, 0, 0], [0, 0, 0]])  # Not an L-shape

        self.assertEqual(is_l_shape(array1), "bottom_right")
        self.assertEqual(is_l_shape(array2), "bottom_left")
        self.assertEqual(is_l_shape(array3), "top_right")
        self.assertEqual(is_l_shape(array4), "top_left")
        self.assertEqual(is_l_shape(array5), "bottom_left")
        self.assertEqual(is_l_shape(array6), "bottom_right")
        self.assertEqual(is_l_shape(array7), "top_right")
        self.assertEqual(is_l_shape(array8), "top_left")
        self.assertEqual(is_l_shape(array9), None)
        self.assertEqual(is_l_shape(array10), None)
        self.assertEqual(is_l_shape(array11), None)
        self.assertEqual(is_l_shape(array12), None)
        self.assertEqual(is_l_shape(array13), None)

    def test_box_distance(self):
        point_a = np.array([[5, 3], [5, 3], [5, 3], [5, 3]])
        point_b = np.array([[5, 2], [5, 2], [5, 2], [5, 2]])
        point_c = np.array([[5, 4], [5, 4], [5, 4], [5, 4]])
        point_d = np.array([[4, 4], [4, 4], [4, 4], [4, 4]])
        point_e = np.array([[4, 2], [4, 2], [4, 2], [4, 2]])

        self.assertEqual(box_distance(point_a, point_b, "left"), 1)
        self.assertEqual(box_distance(point_b, point_c, "right"), 2)
        self.assertEqual(box_distance(point_c, point_d, "down"), 1)
        self.assertEqual(box_distance(point_d, point_c, "up"), 1)
        self.assertEqual(box_distance(point_a, point_d, "top_right"), 1)
        self.assertEqual(box_distance(point_d, point_a, "bottom_left"), 1)
        self.assertEqual(box_distance(point_a, point_e, "top_left"), 1)
        self.assertEqual(box_distance(point_e, point_a, "bottom_right"), 1)

    def test_relative_box_direction(self):
        point_a = np.array([[5, 3], [5, 3], [5, 3], [5, 3]])
        point_b = np.array([[5, 2], [5, 2], [5, 2], [5, 2]])
        point_c = np.array([[4, 3], [4, 3], [4, 3], [4, 3]])
        point_d = np.array([[4, 2], [4, 2], [4, 2], [4, 2]])

        self.assertEqual(relative_box_direction(point_a, point_b), "left")
        self.assertEqual(relative_box_direction(point_b, point_a), "right")
        self.assertEqual(relative_box_direction(point_a, point_c), "up")
        self.assertEqual(relative_box_direction(point_c, point_a), "down")
        self.assertEqual(relative_box_direction(point_c, point_b), "bottom_left")
        self.assertEqual(relative_box_direction(point_b, point_c), "top_right")
        self.assertEqual(relative_box_direction(point_a, point_d), "top_left")
        self.assertEqual(relative_box_direction(point_d, point_a), "bottom_right")

    def test_starting_point(self):
        point_a = np.array([[5, 3], [5, 3], [5, 3], [5, 3]])

        step_a = np.array([[5, 3], [6, 3]])
        self.assertTrue(np.array_equal(
            starting_point(point_a, "left", 2),
            step_a
        ))

        self.assertTrue(np.array_equal(
            starting_point(point_a, "right", 2),
            step_a
        ))

        step_b = np.array([[5, 3], [5, 4]])
        self.assertTrue(np.array_equal(
            starting_point(point_a, "up", 2),
            step_b
        ))

        self.assertTrue(np.array_equal(
            starting_point(point_a, "down", 2),
            step_b
        ))

    def test_starting_point_diagonal(self):
        point_a = np.array([[5, 3], [5, 3], [5, 3], [5, 3]])

        step_c = np.array([[5, 3], [6, 4]])
        self.assertTrue(np.array_equal(
            starting_point(point_a, "top_right", 2),
            step_c
        ))

        step_d = np.array([[5, 3], [4, 4]])
        self.assertTrue(np.array_equal(
            starting_point(point_a, "bottom_right", 2),
            step_d
        ))

        step_e = np.array([[5, 3], [6, 4]])
        self.assertTrue(np.array_equal(
            starting_point(point_a, "bottom_left", 2),
            step_e
        ))

        step_f = np.array([[5, 3], [4, 4]])
        self.assertTrue(np.array_equal(
            starting_point(point_a, "top_left", 2),
            step_f
        ))

    def test_box_contained(self):
        box_a = np.array([
            [4, 0], [0, 0], [0, 4], [4, 4]
        ])

        box_b = np.array([
            [2, 2], [1, 2], [1, 3], [2, 3]
        ])

        box_c = np.array([
            [5, 5], [4, 5], [4, 6], [5, 6]
        ])

        self.assertTrue(box_contained(box_b, box_a))
        self.assertFalse(box_contained(box_a, box_b))
        self.assertFalse(box_contained(box_c, box_a))

    def test_extreme_point(self):
        mask = np.array([
            [False, False, True, False],
            [False, True, True, True],
            [False, False, True, False],
            [True, True, True, False],
        ])

        self.assertEqual((3, 0), extreme_point(mask, "left"))
        self.assertEqual((1, 3), extreme_point(mask, "right"))
        self.assertEqual((0, 2), extreme_point(mask, "up"))
        self.assertEqual((3, 0), extreme_point(mask, "down"))

    def test_bounding_box(self):
        # Test case 1: Simple square
        points = np.array([[1, 1], [1, 3], [3, 3], [3, 1]])
        expected_bbox = np.array([[1, 1], [1, 3], [3, 3], [3, 1]])
        result = get_bounding_box(points)
        self.assertTrue(np.array_equal(result, expected_bbox))

        # Test case 2: Random points
        points = np.array([[2, 5], [1, 3], [4, 6], [3, 2]])
        expected_bbox = np.array([[1, 2], [1, 6], [4, 6], [4, 2]])
        result = get_bounding_box(points)
        self.assertTrue(np.array_equal(result, expected_bbox))

        # Test case 3: Single point
        points = np.array([[2, 2]])
        expected_bbox = np.array([[2, 2], [2, 2], [2, 2], [2, 2]])
        result = get_bounding_box(points)
        self.assertTrue(np.array_equal(result, expected_bbox))

        # Test case 4: Vertical line
        points = np.array([[2, 1], [2, 3], [2, 2]])
        expected_bbox = np.array([[2, 1], [2, 3], [2, 3], [2, 1]])
        result = get_bounding_box(points)
        self.assertTrue(np.array_equal(result, expected_bbox))

        # Test case 5: Horizontal line
        points = np.array([[1, 2], [3, 2], [2, 2]])
        expected_bbox = np.array([[1, 2], [1, 2], [3, 2], [3, 2]])
        result = get_bounding_box(points)
        self.assertTrue(np.array_equal(result, expected_bbox))

    def test_find_5x5_grids_with_border(self):
        # Create a 10x10 grid with a 5x5 grid having a border of color 1
        grid = np.zeros((10, 10), dtype=int)
        grid[2:7, 2:7] = 1  # Fill a 5x5 area with border color
        grid[3:6, 3:6] = 0  # Inner area with a different color

        # Call the function
        result = find_5x5_grids_with_border(grid, border_color=1)

        # Expected bounding box for the 5x5 grid
        expected_bbox = np.array([
            [6, 2],  # bottom-left
            [2, 2],  # top-left
            [2, 6],  # top-right
            [6, 6]   # bottom-right
        ])

        # Assert the result contains the expected bounding box
        self.assertEqual(len(result), 1)
        self.assertTrue(np.array_equal(result[0], expected_bbox))

    def test_no_5x5_grids_with_border(self):
        # Create a grid with no valid 5x5 grids
        grid = np.zeros((10, 10), dtype=int)

        # Call the function
        result = find_5x5_grids_with_border(grid, border_color=1)

        # Assert no grids are found
        self.assertEqual(len(result), 0)
