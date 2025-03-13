"""Utility functions and constants for SignalScribe."""

import argparse
from argparse import Namespace
import platform
import psutil
import traceback
import os

from .logging import console

from .defaults import (
    LOG_DIR_PATH,
    FILETYPES,
    MODEL_DIR_PATH,
    DEFAULT_MODEL,
)

from SignalScribe.version import __version__


class UserException(Exception):
    """Exception caused by user action that should gracefully exit the application."""

    pass


def check_ffmpeg():
    try:
        import subprocess

        # Run ffmpeg -version to check if it's available
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            console.print(
                "[bold red]Error: ffmpeg is not installed or not in the PATH.[/bold red]"
            )
            console.print(
                "[yellow]Please install ffmpeg before running SignalScribe.[/yellow]"
            )
            console.print(
                "[dim]Visit https://ffmpeg.org/download.html for installation instructions.[/dim]"
            )
            sys.exit(1)

    except FileNotFoundError:
        console.print(
            "[bold red]Error: ffmpeg is not installed or not in the PATH.[/bold red]"
        )
        console.print(
            "[yellow]Please install ffmpeg before running SignalScribe.[/yellow]"
        )
        console.print(
            "[dim]Visit https://ffmpeg.org/download.html for installation instructions.[/dim]"
        )
        sys.exit(1)


def compact_traceback(exc_type, exc_value, exc_traceback):
    """Format traceback in a compact way, showing only filename, line number and code."""
    tb_lines = []
    for frame in traceback.extract_tb(exc_traceback, limit=-3):
        filename = os.path.basename(frame.filename)
        tb_lines.append(f"[{filename}:{frame.lineno}] {frame.line}")

    error_msg = f"[{exc_type.__name__}] {exc_value}"
    return "\n".join(tb_lines + [error_msg])


def parse_args(args=None) -> Namespace:
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
        "-r",
        "--reload-models",
        action="store_true",
        default=False,
        help="Reload list of available models from the internet",
    )

    parser.add_argument(
        "-c",
        "--csv-path",
        type=str,
        help="""Path to place for storing transcriptions in CSV format.\n
                Can be a complete path including filename, or directory in which a name will be generated.
                If no path is provided, the default name will be used and stored in the same directory as the audio files.""",
    )

    parser.add_argument(
        "-l",
        "--log-path",
        type=str,
        default=str(LOG_DIR_PATH),
        help=f"""Path to place to store logging output.\n
                 Can be a complete path including filename, or directory in which a name will be generated.
                 If no log path is provided, logs will be stored in {LOG_DIR_PATH}""",
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
        default=None,
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
    """Insert a string into another string at a specific position. (used for colour highlighting)"""
    return string[:position] + insert + string[position:]


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable string."""
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TiB"


def get_system_info() -> list[str]:
    os_string = platform.version()
    cpu_string = (
        f"{psutil.cpu_count()} cores, {platform.processor()} ({platform.machine()})"
    )
    ram_total = format_size(round(psutil.virtual_memory().total))
    ram_available = format_size(round(psutil.virtual_memory().available))

    os_line = f"OS:  {os_string}"
    cpu_line = f"CPU: {cpu_string}"
    ram_line = f"RAM: {ram_available} available of {ram_total} total"

    return [os_line, cpu_line, ram_line]
