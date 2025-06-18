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


def orthogonal_direction(
        point: np.ndarray,
        collision_block: np.ndarray,
        direction: Direction,
):
    """
    Calculates the orthogonal direction of a collision based on a point and the collision blocks.
    :param point: The point to determine the orthogonal direction for.
    :param collision_block: The block to collide with.
    :param direction: The direction to determine the orthogonal direction into.
    :return: A orthogonal direction.
    """

    x_min = point[:, 0].min()
    x_max = point[:, 0].max()

    if direction == "right":
        return "left"
    elif direction == "left":
        return "right"
    elif direction == "up":
        return "down"
    elif direction == "down":
        return "up"
    elif direction == "top_left":
        if collision_block[0].max() < x_min:
            return "bottom_left"
        else:
            return "top_right"
    elif direction == "top_right":
        if collision_block[0].max() < x_min:
            return "bottom_right"
        else:
            return "top_left"
    elif direction == "bottom_left":
        if x_max < collision_block[0].min():
            return "top_left"
        else:
            return "bottom_right"
    elif direction == "bottom_right":
        if x_max < collision_block[0].min():
            return "top_right"
        else:
            return "bottom_left"

    raise ValueError(f"Invalid direction: {direction}")
