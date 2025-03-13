"""Main entry point for SignalScribe."""

# Todo list:
# Multiprocess logging
# Fix install on AMD64 + nVidia GPUs
# Output thread

import sys
import os

# Add the parent directory to sys.path to allow relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SignalScribe.utils import setup_logging, parse_args
from SignalScribe.app import SignalScribeApp
from SignalScribe.utils import console, logger


def main():
    """Main entry point for the application."""
    args = parse_args()

    # Create and initialize application
    app = SignalScribeApp(args)

    try:
        # Set up logging before any other initialization
        log_filepath = (
            str(app._resolve_log_filepath())
            if hasattr(app, "_resolve_log_filepath")
            else None
        )

        setup_logging(
            verbose=args.verbose, log_filepath=log_filepath, log_level=args.log_level
        )

        logger.debug(f"Parsed arguments: {args}")

        if not app.start():
            return 1

        # Run the application
        return app.run()

    # Handle Ctrl-C to cleanly exit the application:
    except KeyboardInterrupt:
        if not args.silent:
            console.print("[red]SignalScribe interrupted by user. Exiting...")
        return 0
    # Handle all other exceptions:
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        if not args.silent:
            console.print(f"[red]An error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
