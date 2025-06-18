import numpy as np

from arc_puzzle_generator.physics import Direction


def collision_neighbourhood(point: np.ndarray, direction: Direction) -> np.ndarray:
    """
    Determines the neighbourhood of a point based on a direction.
    :param point: The point to determine the neighbourhood for.
    :param direction: The direction to determine the neighbourhood into.
    :return: A 2D array of neighbourhood coordinates.
    """

    # TODO: Implement multiple step size diagonal collisions
    y_min = point[:, 1].min()
    y_max = point[:, 1].max()
    x_min = point[:, 0].min()
    x_max = point[:, 0].max()

    match direction:
        case "right":
            return np.array([
                [x, y_max + 1]
                for x in range(x_min, x_max + 1)
            ])
        case "left":
            return np.array([
                [x, y_min - 1]
                for x in range(x_min, x_max + 1)
            ])
        case "up":
            return np.array([
                [x_min - 1, y]
                for y in range(y_min, y_max + 1)
            ])
        case "down":
            return np.array([
                [x_max + 1, y]
                for y in range(y_min, y_max + 1)
            ])
        case "top_left":
            return np.array([
                (x_min - 1, y_min - 1), (x_min - 1, y_min), (x_min, y_min - 1)
            ])
        case "top_right":
            return np.array([
                (x_min - 1, y_max), (x_min - 1, y_max + 1), (x_min, y_max + 1)
            ])
        case "bottom_left":
            return np.array([
                (x_min, y_min - 1), (x_min + 1, y_min - 1), (x_min + 1, y_min)
            ])
        case "bottom_right":
            return np.array([
                (x_max, y_max + 1), (x_max + 1, y_max), (x_max + 1, y_max + 1)
            ])

    raise ValueError(f"Invalid direction: {direction}")
