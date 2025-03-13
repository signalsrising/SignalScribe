"""Color definitions for SignalScribe."""

import enum


class ConsoleColors(enum.Enum):
    BLACK = "black"
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"
    MAGENTA = "magenta"
    CYAN = "cyan"
    WHITE = "white"
    BRIGHT_BLACK = "bright_black"
    BRIGHT_RED = "bright_red"
    BRIGHT_GREEN = "bright_green"
    BRIGHT_YELLOW = "bright_yellow"
    BRIGHT_BLUE = "bright_blue"
    BRIGHT_MAGENTA = "bright_magenta"
    BRIGHT_CYAN = "bright_cyan"
    BRIGHT_WHITE = "bright_white"


class AppColors(enum.Enum):
    # File paths
    FILE_PATH = ConsoleColors.BRIGHT_CYAN.value
    FILE_NAME = ConsoleColors.YELLOW.value
    FILE_SIZE = ConsoleColors.CYAN.value
