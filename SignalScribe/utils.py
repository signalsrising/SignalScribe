"""Utility functions and constants for SignalScribe."""

import os
import argparse
from SignalScribe import __version__
from .defaults import (
    LOG_DIR_PATH,
    DEFAULT_MODEL,
    FILETYPES,
    DEFAULT_NUM_THREADS,
    MODEL_DIR_PATH,
)


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=f"SignalScribe {__version__} - Automatic Audio Transcription"
    )

    parser.add_argument(
        "folder",
        type=str,
        nargs="?",  # Make it optional
        default=None,
        help="Folder to watch for audio files",
    )

    parser.add_argument(
        "-M",
        "--list-models",
        action="store_true",
        default=False,
        help="List all available models",
    )

    parser.add_argument(
        "-c",
        "--csv-path",
        type=str,
        help="Path to CSV file for storing transcriptions.\nBy default this will be the name of the folder that the audio files are in, with a .csv extension.\nIf no csv filepath is provided, the default name will be used and stored in the same directory as the audio files.",
    )

    parser.add_argument(
        "-l",
        "--log-dir",
        type=str,
        default=str(LOG_DIR_PATH),
        help=f"Path to directory to store logging output.\nIf no log directory is provided, logs will be stored in {LOG_DIR_PATH}",
    )

    parser.add_argument(
        "-n",
        "--no-logging",
        "--no-logs",
        action="store_true",
        default=False,
        help="Disable logging to file completely",
    )

    parser.add_argument(
        "-f",
        "--formats",
        type=str,
        nargs="+",
        default=FILETYPES,
        help=f"File formats to watch for as comma-separated values with no spaces. Can be any ffmpeg-supported audio format. Default: {FILETYPES}",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Whisper model to use. Default: {DEFAULT_MODEL}. Use --list-models to see all available models.",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(MODEL_DIR_PATH),
        help="Directory containing model files. If not specified, uses ~/.signalscribe/models",
    )

    parser.add_argument(
        "-R",
        "--recursive",
        action="store_true",
        default=False,
        help="Watch subdirectories recursively",
    )

    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=False,
        help="Disable GPU acceleration",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads to use",
    )

    parser.add_argument(
        "--max-threads",
        type=int,
        default=DEFAULT_NUM_THREADS,  # Maximum number of logical CPU cores
        help="Maximum number of threads to use",
    )

    parser.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        default=False,
        help="Output logging to console",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode. Outputs all possible logging information to the console and log file. Overrides --verbose and --silent.",
    )

    parser.add_argument(
        "-S",
        "--silent",
        action="store_true",
        default=False,
        help="Disable console output",
    )

    # parser.add_argument(
    #     "--log-level",
    #     type=str,
    #     default=DEFAULT_LOG_LEVEL,
    #     choices=log_levels.keys(),
    #     help=f"Log level to output to log file and/or console (if --log-path or --verbose are specified) Default: {DEFAULT_LOG_LEVEL}",
    # )

    # parser.add_argument(
    #     "--realtime", action="store_true", help="Enable realtime transcription"
    # )

    return parser.parse_args(args)


def nested_dict_to_string(
    nested_dict: dict, indent: int = 2, include_none: bool = False
) -> str:
    """Recursively convert a nested dictionary to a string with increasing indent."""
    result = ""
    for key, value in nested_dict.items():
        if value is not None or include_none:
            if isinstance(value, dict):
                result += f"{' ' * indent}{key}:\n{nested_dict_to_string(value, indent + 2, include_none)}"
            else:
                result += f"{' ' * indent}{key}: {value}\n"
    return result


def insert_string(string: str, insert: str, position: int) -> str:
    """Insert a string into another string at a specific position."""
    return string[:position] + insert + string[position:]


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable string."""
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TiB"
