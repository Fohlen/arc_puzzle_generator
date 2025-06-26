from itertools import cycle
from typing import Iterable, cast

from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.collision_rules.FillColorRule import FillColorRule
from arc_puzzle_generator.collisions import identity_direction, axis_neighbourhood
from arc_puzzle_generator.entities import colour_count, find_connected_objects
from arc_puzzle_generator.physics import Direction
from arc_puzzle_generator.puzzle_generator import PuzzleGenerator


class PuzzleFourteenPuzzleGenerator(PuzzleGenerator):
    def setup(self) -> Iterable[Agent]:
        sorted_colors = colour_count(self.input_grid)
        background_color = sorted_colors[0][0]

        target_mask = self.input_grid == 4
        labeled_grid, bounding_box, num_objects = find_connected_objects(target_mask)

        direction = cast(Direction, "down" if bounding_box[0, 0, 0] == 0 else "up")
        beam_width = bounding_box[0, 3, 1] - bounding_box[0, 0, 1] + 1

        return [Agent(
            output_grid=self.output_grid,
            bounding_box=bounding_box[0],
            charge=-1,
            direction=direction,
            colors=cycle([4, 4, 3]),
            step_size=2,
            beam_width=beam_width,
            collision_rule=FillColorRule(
                background_color=background_color,
                direction_rule=identity_direction,
            ),
            neighbourhood_rule=axis_neighbourhood
        )]
