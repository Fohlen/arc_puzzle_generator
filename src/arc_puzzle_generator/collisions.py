import numpy as np

from arc_puzzle_generator.physics import Direction


def collision_neighbourhood(point: np.ndarray, direction: Direction) -> np.ndarray:
    match direction:
        case "right":
            y_max = point[:, 1].max()
            x_min = point[:, 0].min()
            x_max = point[:, 0].max()

            return np.array([
                [x, y_max + 1]
                for x in range(x_min, x_max + 1)
            ])
        case "left":
            y_min = point[:, 1].min()
            x_min = point[:, 0].min()
            x_max = point[:, 0].max()

            return np.array([
                [x, y_min - 1]
                for x in range(x_min, x_max + 1)
            ])
        case "up":
            x_min = point[:, 0].min()
            y_min = point[:, 1].min()
            y_max = point[:, 1].max()

            return np.array([
                [x_min - 1, y]
                for y in range(y_min, y_max + 1)
            ])
        case "down":
            x_max = point[:, 0].max()
            y_min = point[:, 1].min()
            y_max = point[:, 1].max()

            return np.array([
                [x_max + 1, y]
                for y in range(y_min, y_max + 1)
            ])

    raise ValueError(f"Invalid direction: {direction}")
