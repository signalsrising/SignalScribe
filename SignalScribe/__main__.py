"""Main entry point for SignalScribe."""

# Todo list:
# Multiprocess logging
# Fix install on AMD64 + nVidia GPUs

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from .app import SignalScribeApp
from .logging import logger, setup_logging, log_name
from .utils import parse_args


def main():
    """Main entry point for the application."""
    args = parse_args()

    # Create and initialise application
    try:
        app = SignalScribeApp(args)

        if args.verbose:
            print(f"Parsed arguments: {args}")

    except Exception as e:
        print(f"Error initialising SignalScribe: {str(e)}")
        return 1

    # With the app initialised, set up logging and run the application:
    try:
        # Set up logging before setting up the app and running
        app.log_file_path = Path(args.log_dir) / log_name()

        setup_logging(
            log_file_path=app.log_file_path, verbose=args.verbose, silent=args.silent
        )

        logger.debug(f"Logging setup complete")

    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        return 1

    # Set up the application (initialise threads, processes, etc)
    try:
        app.setup()
        logger.debug(f"Application setup complete")
    except Exception as e:
        print(f"Error setting up application: {str(e)}")
        return 1

    # Run the application
    # No need to catch exceptions here, as the app handles them
    # in order to deal with graceful shutdowns
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
