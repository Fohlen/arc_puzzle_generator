import numpy as np

from src.arc_puzzle_generator.entities import find_colors, find_connected_objects, is_l_shape, is_point_adjacent
from src.arc_puzzle_generator.grid_utils import make_smallest_square_from_mask
from src.arc_puzzle_generator.physics import orientation_to_unit_vector, orthogonal_orientation, starting_point


def generate_48d8fb45(input_grid: np.ndarray) -> np.ndarray:
    output_grid = input_grid.copy()
    colors = find_colors(input_grid)

    l_shapes = []
    blocks = []

    for target_color in colors:
        target_mask = input_grid == target_color
        labeled_grid, bounding_box, num_objects = find_connected_objects(target_mask)

        for label in range(1, num_objects + 1):
            box = make_smallest_square_from_mask(output_grid, labeled_grid == label)

            orientation = is_l_shape(box)
            if orientation is not None:
                l_shapes.append((target_color, bounding_box[(label - 1), :], orientation))
            else:
                blocks.append((target_color, bounding_box[(label - 1), :]))

    bboxes = np.array([bbox for _, bbox in blocks])

    for color, bbox, orientation in l_shapes:
        current_color = color
        step = orientation_to_unit_vector(orientation) + starting_point(bbox, orientation)

        while input_grid.shape[0] > step[0] > -1 < step[1] < input_grid.shape[1]:
            colliding_blocks = is_point_adjacent(step, bboxes)

            if colliding_blocks is not None:
                # Determine if collision is horizontal by checking if the beam's x-coordinate
                # is adjacent to the block's vertical sides
                block_bbox = bboxes[colliding_blocks[0]]
                is_horizontal = (np.min(block_bbox[:, 0]) <= step[0] <= np.max(block_bbox[:, 0]))

                current_color = blocks[colliding_blocks[0]][0]
                orientation = orthogonal_orientation(orientation, is_horizontal)

            output_grid[step[0], step[1]] = current_color
            step += orientation_to_unit_vector(orientation)

    return output_grid
