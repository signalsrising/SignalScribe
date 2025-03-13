from pathlib import Path
from rich.logging import RichHandler
import logging
import logging.handlers

from .loggingconsole import LoggingConsole

# Set up logging
logger = logging.getLogger("rich")
from .defaults import (
    LOG_NAME,
    CONSOLE_OUTPUT_LOG_LEVEL,
    NUM_LOG_FILES_TO_KEEP,
)
from datetime import datetime

# Create console with the custom class
console = LoggingConsole(logger=logger, highlight=False)


def log_name() -> str:
    """Get the name of the log file."""

    # Logs are in the format: signalscribe-<date>-<time>.log
    # We use ISO style of YYYYMMDD as a compromise between human-readable and American formats
    return f"{LOG_NAME}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def setup_logging(
    log_file_path: str,
    verbose: bool = False,
    silent: bool = False,  # Not implemented yet
) -> str:
    """Configure logging with rich handler and optional file output.

    Args:
        param verbose: Whether to add a console handler for logging output
              i.e. logging also gets sent to console. Set with --verbose
        param log_filepath: Path to log file (if None, only console logging is used)
              Set with --no-logging.
    """
    # Convert log_file_path to Path object before anything else
    log_file_path = Path(log_file_path)

    # Ensure log_file_path has been given:
    if not log_file_path:
        raise ValueError("log_file_path must be provided")

    # If user provides a directory, put the log in there using
    # the standard incrementing filename
    if log_file_path.is_dir():
        log_file_path = log_file_path / log_name()

    # Check that the log file's parent dir exists and if not, make it
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Set base log level
    # (Actually not sure what effect this has since we set the handler levels below)
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Configure file handler first (this is always active if filepath provided)
    # Exceptions will be handled by the caller so we dont catch them here
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=1024 * 1024 * 10,  # 10MiB
        backupCount=10,
    )
    # Log everything, including console prints, to the file
    file_handler.setLevel(LoggingConsole.CONSOLE)

    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    )
    logger.addHandler(file_handler)

    # Add console handler if verbose is True
    if verbose:
        rich_handler = RichHandler(
            rich_tracebacks=True, markup=True, show_time=False, console=console
        )
        rich_handler.setLevel(CONSOLE_OUTPUT_LOG_LEVEL)
        logger.addHandler(rich_handler)

    # Cleanup old logs
    cleanup_old_logs(log_file_path.parent, keep_last_n=NUM_LOG_FILES_TO_KEEP)

    logger.debug(f"Logging setup complete")

    return log_file_path


def cleanup_old_logs(
    log_dir: Path,
    keep_last_n: int = 10,
) -> None:
    """Keep only the N most recent log files in the directory."""

    # Ensure log_dir has been given and is a directory
    if not log_dir:
        raise ValueError("log_dir must be provided")
    if not log_dir.is_dir():
        raise ValueError("log_dir must be a directory")

    # Get all log files in the directory
    log_files = sorted(
        log_dir.glob(f"{LOG_NAME}*.log"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    # Remove old files
    for old_log in log_files[keep_last_n:]:
        try:
            old_log.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove old log file {old_log}: {e}")
