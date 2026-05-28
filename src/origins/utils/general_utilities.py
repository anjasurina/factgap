from typing import Optional, Protocol, Any
import enum
import logging
import random
from datetime import datetime as dt
import numpy as np
import dataclasses

class ColorType(enum.Enum):
    DEFAULT = ""         # No ANSI code, uses terminal default
    BLACK = "\033[30m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    WHITE = "\033[97m"
    RESET = "\033[0m"

    @staticmethod
    def get_color(index: int) -> 'ColorType':
        """
        Get a color based on the index, cycling through available colors.

        Args:
            index (int): The index to determine the color.

        Returns:
            ColorType: The corresponding color.
        """
        colors = list(ColorType)[2:]  # Exclude DEFAULT and RESET
        return colors[index % len(colors)]


def print_c(message: str, color: ColorType = ColorType.DEFAULT, v=0, vmin=0) -> None:
    """
    Print a message with optional color formatting.

    Args:
        message (str): The message to print.
        color (ColorType): The color to format the message with.
                           Defaults to ColorType.DEFAULT (no color).
        v (int): Verbosity level of the message. Defaults to 0.
        vmin (int): Minimum verbosity level required to print the message. Defaults to 0.
    """
    if v < vmin:
        return

    if color == ColorType.DEFAULT:
        print(message)
    else:
        print(f"{color.value}{message}{ColorType.RESET.value}")


def log_c(
        message: str,
    logger_instance: "LoggerLike",
        color: ColorType = ColorType.DEFAULT,
        level: int = logging.INFO  # Changed log_type to level, type hint to int
) -> None:
    """
    Log a message with optional color formatting and specified logging level.

    Args:
        message (str): The message to log.
        logger_instance (logging.Logger): The logger instance to use.
        color (ColorType): The color to format the message with.
        level (int): The logging level to use (e.g., logging.INFO,
                     logging.WARNING, logging.ERROR). Defaults to logging.INFO.
    """
    if color != ColorType.DEFAULT:
        formatted_message = f"{color.value}{message}{ColorType.RESET.value}"
    else:
        # If no color is specified, use the message as is
        formatted_message = message

    # Use a dictionary to map levels to logger methods for cleaner code
    # or directly use logger_instance.log(level, formatted_message)
    logger_instance.log(level, formatted_message)


def get_timestamp() -> str:
    """
    Get the current timestamp in the format YYYYMMDD_HHMMSS.

    Returns:
        str: The current timestamp.
    """
    return dt.now().strftime("%Y%m%d_%H%M%S")


def json_default_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for objects not serializable by default json code.
    Handles dataclasses by converting them to dicts.

    Args:
        obj (Any): The object to serialize
    Returns:
        Any: The serialized object
    Raises:
        TypeError: If the object is not serializable
    """
    if dataclasses.is_dataclass(obj) and not isinstance(
        obj, type
    ):  # Check if it's an instance, not the class itself
        return dataclasses.asdict(obj)
    if hasattr(obj, "__iter__") and not isinstance(obj, list):
        return list(obj)

    if isinstance(obj, enum.Enum):
        return obj.value

    # If you had other custom types that are not dataclasses, you'd handle them here.
    # For example, if you were directly serializing a set:
    # if isinstance(obj, set):
    #     return list(obj)

    raise TypeError(
        f"Object of type {obj.__class__.__name__} is not JSON serializable")