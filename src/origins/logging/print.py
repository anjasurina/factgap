import os
from accelerate import Accelerator
import logging
from typing import Protocol, Any
from accelerate.logging import get_logger
logger = get_logger(__name__)

# ANSI escape codes for colors
class AnsiColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKMAGENTA = '\033[38;5;201m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'  # Resets the color
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DEBUG = '\033[38;5;240m'


def log_with_color(
        message: str,
        logger_instance: "LoggerLike",
        color_code: AnsiColors | str = "",
        level: int = logging.INFO,
        *args,
        **kwargs):
    """
    Log a message with a specified color and level.
    """
    if level == logging.INFO:
        if color_code:
            logger_instance.info(
                f"{color_code}{message}{AnsiColors.ENDC}", *args, **kwargs)
        else:
            logger_instance.info(message, *args, **kwargs)
    elif level == logging.WARNING:
        logger_instance.warning(
            f"{color_code}{message}{AnsiColors.WARNING}", *args, **kwargs)
    elif level == logging.ERROR:
        logger_instance.error(
            f"{color_code}{message}{AnsiColors.FAIL}", *args, **kwargs)
    elif level == logging.DEBUG:
        logger_instance.debug(
            f"{color_code}{message}{AnsiColors.DEBUG}", *args, **kwargs)
    else:
        raise ValueError(f"Invalid level: {level}")


class LoggerLike(Protocol):
    def log(self, level: int, msg: Any, *args: Any, **kwargs: Any) -> None: ...
    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None: ...
    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None: ...
    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None: ...


def print_with_color(message: str, color_code: AnsiColors | str = ""):
    """
    Print a message with a specified color.

    Args:
        message (str): The message to print.
        color_code (AnsiColors | str): The color code to use for printing.
    """
    if color_code:
        print(f"{color_code}{message}{AnsiColors.ENDC}")
    else:
        print(message)


def print_hello(accelerator: Accelerator):
    if accelerator.is_main_process:
        logger.info(f"Hello from main process")
    else:
        logger.info(
            f"Hello from non-main process with LOCAL_RANK: {os.getenv('LOCAL_RANK')}")
