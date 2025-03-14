"""Main entry point for SignalScribe."""

# TODO:
# Test non-Apple hardware acceleration
# Read wav file header to know true length of file
# Filter out nonverbal transcriptions (with user setting)
# Filter short transcriptions below threshold length (with user setting)
# Add use_scm_version=True to setup.py for versioning
# Add ffmpeg test to install and also catch in decoder subprocess
# Better show that hardware acceleration is working
# --show-queues: all queues always visible for debug
# Check that --threads works
# [NameError] cannot access free variable 'app' where it is not associated with a value in enclosing scope
# Add --no-csv option
# Change model, config and log storage to be platform-specific
# Improve responsiveness of ctrl-c
# Implement --version, --silent, --debug
# Have warnings and errors on by default
# -y option

import os
import sys
import signal

# Add the parent directory to sys.path to allow relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from .app import SignalScribeApp
from .logging import console, logger, setup_logging
from .utils import UserException, parse_args, compact_traceback, get_ffmpeg_version


def main():
    """Main entry point for the application."""

    # Custom exception formatting for uncaught exceptions
    # Should never be called but is safety net
    sys.excepthook = lambda exc_type, exc_value, exc_tb: console.print(
        compact_traceback(exc_type, exc_value, exc_tb), markup=False
    )

    # Parse command line arguments into Namespace object
    args = parse_args()

    # Set up logging before anything else
    try:
        log_file_path = setup_logging(log_file_path=args.log_path, verbose=args.verbose, silent=args.silent)
    except UserException as e:
        console.print(f"Error setting up logging for SignalScribe, aborting: {e}", style="bold red")
        sys.exit(1)
    except Exception as e:
        console.print(compact_traceback(type(e), e, sys.exc_info()[2]), markup=False)
        console.print("Error setting up logging for SignalScribe, aborting", style="bold red")
        sys.exit(1)

    # Check if ffmpeg is installed and available in the PATH
    ffmpeg_version = get_ffmpeg_version()
    if ffmpeg_version:
        logger.debug(f"Using ffmpeg version: {ffmpeg_version}")
    else:
        logger.fatal("ffmpeg is not installed or not in the PATH")
        sys.exit(1)

    # Convenience function for brevity (requires logging and app to be setup first)
    def abort(message: str, e: Exception = None):
        if e:
            logger.debug(compact_traceback(type(e), e, sys.exc_info()[2]))
        console.print(f"[bold red]{message}[/bold red]")
        if e:
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
        import traceback

        traceback.print_exc()
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
