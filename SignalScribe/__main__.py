"""Main entry point for SignalScribe."""

import sys
import os

# Add the parent directory to sys.path to allow relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SignalScribe.utils import setup_logging, parse_args
from SignalScribe.app import SignalScribeApp


def main():
    """Main entry point for the application."""
    args = parse_args()
    setup_logging(args.verbose)
    
    # Create and initialize application
    app = SignalScribeApp(args)
    if not app.initialize():
        return 1
        
    # Run the application
    return app.run()


if __name__ == '__main__':
    sys.exit(main()) 