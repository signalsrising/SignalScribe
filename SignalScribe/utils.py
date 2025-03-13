"""Utility functions and constants for SignalScribe."""

import os
import argparse
import pathlib
import rich.color
from rich.console import Console
from rich.logging import RichHandler
import logging

APP_NAME = "SignalScribe"
VERSION = "0.7.0"

COMPUTE_TYPE = "float32"
DEFAULT_MODEL = "large-v3-turbo"

MODEL_CHOICES = ["large-v3-turbo"]
FILETYPES_SUPPORTED = ["mp3", "m4a", "wav"]
COLORS_SETTINGS_NAME = "colors.yaml"

# Set up logging
logger = logging.getLogger("rich")
console = Console(highlight=False)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure rich handler
    rich_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=False,
        console=console
    )
    rich_handler.setLevel(log_level)
    
    # Configure logger
    logger.setLevel(log_level)
    logger.addHandler(rich_handler)
    
    # Remove any existing handlers
    for handler in logger.handlers[:-1]:
        logger.removeHandler(handler)


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} v{VERSION} - Audio transcription tool"
    )
    
    parser.add_argument(
        "--folder",
        type=str,
        default=".",
        help="Folder to watch for audio files"
    )
    
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=FILETYPES_SUPPORTED,
        help=f"File formats to watch for. Default: {FILETYPES_SUPPORTED}"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=MODEL_CHOICES,
        help=f"Whisper model to use. Default: {DEFAULT_MODEL}"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory containing model files. If not specified, uses ~/.signalscribe/models"
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Watch subdirectories recursively"
    )
    
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Disable GPU acceleration"
    )
    
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads to use"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--csv-filepath",
        type=str,
        default="transcriptions.csv",
        help="Path to CSV file for storing transcriptions"
    )
    
    args = parser.parse_args(args)
    
    # Verify model exists
    model_path = f"./models/ggml-{args.model}.bin"
    if not os.path.exists(model_path):
        available_models = [
            f.stem.split('-')[-1] 
            for f in pathlib.Path("./models").glob("ggml-*.bin")
        ]
        if available_models:
            console.print(
                f"[red]Model {args.model} not found! Available models: " +
                ", ".join(available_models)
            )
        else:
            console.print("[red]No models found in ./models directory!")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return args


def insert_string(string: str, insert: str, position: int) -> str:
    """Insert a string into another string at a specific position."""
    return string[:position] + insert + string[position:] 