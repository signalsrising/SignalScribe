"""Main entry point for SignalScribe."""

# Todo list:
# Multiprocess logging
# Fix install on AMD64 + nVidia GPUs

import os
import sys
from pathlib import Path
import signal

# Add the parent directory to sys.path to allow relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from .app import SignalScribeApp
from .logging import console, logger, setup_logging, log_name
from .utils import UserException, parse_args, compact_traceback, check_ffmpeg


def main():
    """Main entry point for the application."""

    # Check if ffmpeg is installed and available in the PATH
    check_ffmpeg()

    # Custom exception formatting for uncaught exceptions
    # Should never be called but is safety net
    sys.excepthook = lambda exc_type, exc_value, exc_tb: console.print(
        compact_traceback(exc_type, exc_value, exc_tb), markup=False
    )

    # Parse command line arguments into Namespace object
    args = parse_args()

    # Set up logging before anything else
    try:
        log_file_path = setup_logging(
            log_file_path=args.log_path, verbose=args.verbose, silent=args.silent
        )
    except Exception as e:
        console.print(compact_traceback(type(e), e, sys.exc_info()[2]), markup=False)
        console.print(
            f"[bold red]Setting up logging for SignalScribe, aborting[/bold red]"
        )
        sys.exit(1)

    # Convenience function for brevity (requires logging and app to be setup first)
    def abort(message: str, exception: Exception = None):
        if exception:
            logger.debug(compact_traceback(type(e), e, sys.exc_info()[2]))
        console.print(f"[bold red]{message}[/bold red]")
        if exception:
            console.print(f"[dim]Log stored in {app.log_file_path}[/dim]")
        shutdown()

    # Create and initialise application
    try:
        app = SignalScribeApp(args)
        app.log_file_path = log_file_path

        # Function for graceful shutdown
        def shutdown(exitcode: int = 1):
            app.stop()
            sys.exit(exitcode)

        # Capture SIGINT
        def sigint_shutdown(sig, frame):
            shutdown(0)

        signal.signal(signal.SIGINT, sigint_shutdown)

        # Print all the arguments on a single line if --verbose
        args_str = ", ".join([f"{key}={value}" for key, value in vars(args).items()])
        logger.info(f"Starting SignalScribe with args: {args_str}")

    except UserException as e:
        abort(f"Quitting, reason: {e}")

    except Exception as e:
        abort("Error initialising SignalScribe, aborting", e)

    # Set up the application:
    # Initialise threads and processes
    # Resolve filepaths, user options etc
    try:
        app.setup()
    except UserException as e:
        abort(f"Quitting, reason: {e}")
    except Exception as e:
        abort("Error setting up SignalScribe, aborting", e)

    # Finally, run the application
    try:
        return app.run()
    except UserException as e:
        abort(f"Quitting, reason: {e}", False)
    except Exception as e:
        abort("Error running SignalScribe, aborting", e)


if __name__ == "__main__":
    sys.exit(main())
