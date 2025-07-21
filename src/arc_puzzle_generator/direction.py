from typing import Protocol

from arc_puzzle_generator.physics import Direction, Axis


class DirectionTransformer(Protocol):
    """
    A direction transformer determines the future direction of an agent based on the current direction and additional parameters.
    """

    def __call__(self, direction: Direction, *args, **kwargs) -> Direction:
        pass


def identity_direction(direction: Direction, *args, **kwargs) -> Direction:
    """
    A direction rule that returns the same direction.
    :param direction: The direction to follow.
    :return: A direction rule that returns the same direction.
    """

    return direction


def snake_direction(direction: Direction, *args, **kwargs) -> Direction:
    """
    Returns the opposite direction of the given direction, moving in a snake pattern.
    :param direction: The input direction
    :return: the opposite direction
    """
    if direction == "right":
        return "up"

    return "right"


def orthogonal_direction(direction: Direction, axis: Axis = "horizontal") -> Direction:
    """
    Returns the orthogonal direction of the given direction based on a collision axis.
    :param direction: The direction to convert.
    :param axis: The collision axis.
    :return: The orthogonal direction of the given direction.
    """

    match axis:
        case "vertical":
            # For vertical collisions (hitting vertical walls)
            match direction:
                case "bottom_left":
                    return "bottom_right"
                case "bottom_right":
                    return "bottom_left"
                case "top_left":
                    return "top_right"
                case "top_right":
                    return "top_left"
        case "horizontal":
            # For horizontal collisions (hitting horizontal walls)
            match direction:
                case "bottom_left":
                    return "top_left"
                case "bottom_right":
                    return "top_right"
                case "top_left":
                    return "bottom_left"
                case "top_right":
                    return "bottom_right"
        case "diagonal":
            match direction:
                case "bottom_left":
                    return "top_right"
                case "bottom_right":
                    return "top_left"
                case "top_left":
                    return "bottom_right"
                case "top_right":
                    return "bottom_left"

    raise ValueError("Unknown axis {}".format(axis))


def clockwise_direction_45(direction: Direction, *args, **kwargs) -> Direction:
    """
    Returns the clockwise direction of the given direction rotated by 45 degrees.
    :param direction: The direction to rotate.
    :return: The clockwise direction of the given direction.
    """

    match direction:
        case "left":
            return "top_left"
        case "top_left":
            return "up"
        case "up":
            return "top_right"
        case "top_right":
            return "right"
        case "right":
            return "bottom_right"
        case "bottom_right":
            return "down"
        case "down":
            return "bottom_left"
        case "bottom_left":
            return "left"

    raise ValueError("Unknown direction {}".format(direction))


def clockwise_direction_90(direction: Direction, *args, **kwargs) -> Direction:
    """
    Returns the clockwise direction of the given direction rotated by 90 degrees.
    :param direction: The input direction.
    :return: The rotated direction of the given direction.
    """
    return clockwise_direction_45(clockwise_direction_45(direction))


def counterclockwise_direction_45(direction: Direction, *args, **kwargs) -> Direction:
    """
    Returns the counterclockwise direction of the given direction rotated by 45 degrees.
    :param direction: The input direction.
    :return: The counterclockwise rotated direction of the given direction.
    """

    match direction:
        case "left":
            return "bottom_left"
        case "bottom_left":
            return "down"
        case "down":
            return "bottom_right"
        case "bottom_right":
            return "right"
        case "right":
            return "top_right"
        case "top_right":
            return "up"
        case "up":
            return "top_left"
        case "top_left":
            return "left"

    raise ValueError("Unknown direction {}".format(direction))


def counterclockwise_direction_90(direction: Direction, *args, **kwargs) -> Direction:
    """
    Returns the counterclockwise direction of the given direction rotated by 90 degrees.
    :param direction: The input direction.
    :return: The counterclockwise rotated direction of the given direction.
    """
    return counterclockwise_direction_45(counterclockwise_direction_45(direction))
