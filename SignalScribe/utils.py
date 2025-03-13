"""Utility functions and constants for SignalScribe."""

import os
import argparse
from rich.console import Console
from rich.logging import RichHandler
import logging
from pathlib import Path
from datetime import datetime
from SignalScribe import __version__

APP_NAME = "SignalScribe"

COMPUTE_TYPE = "float32"
DEFAULT_MODEL = "large-v3-turbo"

MODEL_CHOICES = ["large-v3-turbo"]
FILETYPES_SUPPORTED = ["mp3", "m4a", "wav"]
COLORS_SETTINGS_NAME = "colors.yaml"

# Set up logging
logger = logging.getLogger("rich")
console = Console(highlight=False)

log_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

DEFAULT_LOG_LEVEL = "DEBUG"


def setup_logging(
    verbose: bool = False, log_filepath: str = None, log_level: str = DEFAULT_LOG_LEVEL
) -> None:
    """Configure logging with rich handler and optional file output.

    Args:
        verbose: Whether to add a console handler for logging output
        log_filepath: Path to log file (if None, only console logging is used)
        log_level: Logging level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Set base log level
    base_level = log_levels[log_level]
    logger.setLevel(base_level)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Configure file handler first (this is always active if filepath provided)
    if log_filepath:
        try:
            file_handler = logging.FileHandler(log_filepath)
            file_handler.setLevel(base_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            console.print(f"[red]Warning: Failed to set up log file: {e}")

    # Add console handler if verbose is True
    if verbose:
        rich_handler = RichHandler(
            rich_tracebacks=True, markup=True, show_time=False, console=console
        )
        rich_handler.setLevel(base_level)
        logger.addHandler(rich_handler)


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


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} v{__version__} - Audio transcription tool"
    )

    parser.add_argument(
        "folder",
        type=str,
        nargs="?",  # Make it optional
        help="Folder to watch for audio files",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models",
    )

    parser.add_argument(
        "--csv-filepath",
        type=str,
        help="Path to CSV file for storing transcriptions.\nBy default this will be the name of the folder that the audio files are in, with a .csv extension.\nIf no csv filepath is provided, the default name will be used and stored in the same directory as the audio files.",
    )

    parser.add_argument(
        "--log-filepath",
        type=str,
        help="Path to file to store logging output.\nSet level with --log-level. If only a directory is provided, the log file will be named signscribe.log.\nIf no log filepath is provided at all, logs will be stored in ~/.signalscribe/logs",
    )

    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=FILETYPES_SUPPORTED,
        help=f"File formats to watch for. Default: {FILETYPES_SUPPORTED}",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=MODEL_CHOICES,
        help=f"Whisper model to use. Default: {DEFAULT_MODEL}",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory containing model files. If not specified, uses ~/.signalscribe/models",
    )

    parser.add_argument(
        "-R",
        "--recursive",
        action="store_true",
        help="Watch subdirectories recursively",
    )

    parser.add_argument(
        "--cpu-only", action="store_true", help="Disable GPU acceleration"
    )

    parser.add_argument(
        "--threads", type=int, default=4, help="Number of CPU threads to use"
    )

    parser.add_argument(
        "--max-threads",
        type=int,
        default=os.cpu_count(),  # Maximum number of logical CPU cores
        help="Maximum number of threads to use",
    )

    parser.add_argument(
        "-V", "--verbose", action="store_true", help="Output logging to console"
    )

    parser.add_argument(
        "-S", "--silent", action="store_true", help="Disable console output"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=log_levels.keys(),
        help=f"Log level to output to log file and/or console (if --log-path or --verbose are specified) Default: {DEFAULT_LOG_LEVEL}",
    )

    parser.add_argument(
        "--realtime", action="store_true", help="Enable realtime transcription"
    )

    return parser.parse_args(args)


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


def resolve_filepath(
    provided_path: str | None,
    default_dir: Path,
    default_name: str,
    default_extension: str = "",
    timestamp: bool = False,
    keep_last_n: int | None = None,
) -> Path:
    """Resolve a file path based on provided path or defaults.

    Args:
        provided_path: User-provided path (can be None)
        default_dir: Default directory to use if no path provided
        default_name: Default filename (without extension) to use in default dir
        default_extension: Extension to add to default name (include the dot)
        timestamp: Whether to add timestamp to default filename
        keep_last_n: If set, keeps only the last N files matching default_name* in default_dir

    Returns:
        Path: Resolved file path

    The resolution follows these rules:
    1. If no path provided: use default_dir/default_name[_timestamp]default_extension
    2. If absolute/relative path to existing file: use that file
    3. If absolute/relative path to existing directory: use directory/default_name[_timestamp]default_extension
    4. If path to non-existent location: fall back to default_dir/default_name[_timestamp]default_extension
    """
    try:
        # Handle cleanup of old files if requested
        if keep_last_n is not None and keep_last_n > 0:
            pattern = f"{default_name}*{default_extension}"
            old_files = sorted(
                default_dir.glob(pattern),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            for old_file in old_files[keep_last_n:]:
                try:
                    old_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove old file {old_file}: {e}")

        # Case 1: No path provided - use default directory
        if not provided_path:
            # Ensure default directory exists
            default_dir.mkdir(parents=True, exist_ok=True)

            # Build default filename
            filename = default_name
            if timestamp:
                filename = f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            filename = f"{filename}{default_extension}"

            return default_dir / filename

        # Convert to absolute path
        path = Path(provided_path).expanduser().resolve()

        if path.exists():
            if path.is_file():
                # Case 2: Existing file
                return path
            elif path.is_dir():
                # Case 3: Existing directory - use default name
                filename = f"{default_name}{default_extension}"
                return path / filename
        elif path.parent.exists():
            # Parent exists but file/dir doesn't
            if path.suffix:
                # Looks like a file path
                return path
            else:
                # Looks like a directory path
                logger.warning(
                    f"Directory {path} doesn't exist, falling back to default location"
                )
                return resolve_filepath(
                    None,
                    default_dir,
                    default_name,
                    default_extension,
                    timestamp,
                    keep_last_n,
                )
        else:
            # Parent directory doesn't exist
            logger.warning(
                f"Parent directory {path.parent} doesn't exist, falling back to default location"
            )
            return resolve_filepath(
                None,
                default_dir,
                default_name,
                default_extension,
                timestamp,
                keep_last_n,
            )

    except Exception as e:
        logger.error(f"Error resolving path {provided_path}: {e}")
        return resolve_filepath(
            None, default_dir, default_name, default_extension, timestamp, keep_last_n
        )
