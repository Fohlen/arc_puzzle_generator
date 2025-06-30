from itertools import cycle
from typing import Iterable

import numpy as np

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.collision_rules.FillColorRule import FillColorRule
from arc_puzzle_generator.collisions import identity_direction, axis_neighbourhood
from arc_puzzle_generator.entities import find_connected_objects
from arc_puzzle_generator.physics import Direction
from arc_puzzle_generator.puzzle_generator import PuzzleGenerator


class PuzzleFourteenPuzzleGenerator(PuzzleGenerator):
    def setup(self) -> Iterable[Agent]:
        background_color = self.input_grid[0, 0]
        start_color = 4
        fill_color = 3

        target_mask = self.input_grid == start_color
        labeled_grid, bounding_box, num_objects = find_connected_objects(target_mask)

        direction: Direction

        if bounding_box[0, 0, 0] == (self.output_grid.shape[0] - 1):
            direction = "up"
        elif bounding_box[0, 0, 0] == 0:
            direction = "down"
        elif bounding_box[0, 0, 1] == 0:
            direction = "right"
        elif bounding_box[0, 0, 1] == (self.output_grid.shape[1] - 1):
            direction = "left"
        else:
            raise ValueError("Puzzle scenario is not supported")

        beam_width = bounding_box[0, 3, 1] - bounding_box[0, 0, 1] + 1
        colors = cycle([
            start_color, background_color, start_color, background_color, fill_color, background_color
        ])

        return [Agent(
            output_grid=self.output_grid,
            bounding_box=bounding_box[0] - np.array([0, 1]),
            charge=-1,
            direction=direction,
            colors=colors,
            step_size=1,
            beam_width=beam_width,
            collision_rule=FillColorRule(
                background_color=background_color,
                fill_color=fill_color,
                direction_rule=identity_direction,
            ),
            neighbourhood_rule=axis_neighbourhood
        )]
