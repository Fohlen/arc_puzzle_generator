from typing import Callable


def get_callable_name(callable_obj: Callable | object) -> str:
    """
    Utility method to get the name of a callable object.
    :param callable_obj: The callable object (function, method, etc.) to get the name of.
    :return: The name of the callable object as a string.
    """
    if hasattr(callable_obj, '__name__'):
        return callable_obj.__name__
    elif hasattr(callable_obj, '__class__'):
        return callable_obj.__class__.__name__
    else:
        return str(callable_obj)
