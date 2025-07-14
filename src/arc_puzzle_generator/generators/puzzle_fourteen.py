from itertools import cycle
from typing import Iterable

from abm.direction import identity_direction_rule
from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.collision_rules.FillColorRule import FillColorRule
from arc_puzzle_generator.collisions import AxisNeighbourHood
from abm.utils.entities import find_connected_objects
from arc_puzzle_generator.physics import Direction, bounding_box_to_points
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

        colors = cycle([
            start_color, background_color, start_color, background_color, fill_color, background_color
        ])

        return [Agent(
            output_grid=self.output_grid,
            step=bounding_box_to_points(bounding_box[0]),
            charge=-1,
            direction=direction,
            colors=colors,
            step_size=1,
            collision_rule=FillColorRule(
                background_color=background_color,
                fill_color=fill_color,
                direction_rule=identity_direction_rule,
            ),
            neighbourhood_rule=AxisNeighbourHood(self.input_grid.shape),
        )]
