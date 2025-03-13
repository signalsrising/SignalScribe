"""Main application class for SignalScribe."""

import sys
from pathlib import Path
from typing import Optional

from rich.panel import Panel

from .utils import logger, console, VERSION
from .model import WhisperModel
from .transcriber import Transcriber
from .watcher import FolderWatcher, FolderWatcherHandler


class SignalScribeApp:
    """Main application class that manages all components."""
    
    def __init__(self, args):
        """Initialize the application with command line arguments."""
        self.args = args
        self.model = None
        self.transcriber = None
        self.watcher = None
        self.handler = None
        
    def initialize(self) -> bool:
        """Initialize all components."""
        console.print(Panel(f"[bold][bright_cyan]SignalScribe\n[cyan]by Signals Rising[/bold]\n[bright_black]Version {VERSION}", expand=False))
        console.print("")
    
        try:
            # Initialize model management
            model_dir = self.args.model_dir if hasattr(self.args, 'model_dir') else None
            self.model = WhisperModel(self.args.model, model_dir)
            
            # Ensure model files exist
            if not self.model.ensure_models_exist():
                logger.error("Required model files are missing. Please download them and try again.")
                return False
                
            # Initialize transcriber
            self.transcriber = Transcriber(
                model_name=self.args.model,
                models_dir=str(self.model.model_dir),
                csv_filepath=self.args.csv_filepath,
                threads=self.args.threads,
                cpu_only=self.args.cpu_only
            )
            
            # Initialize watcher and handler
            self.watcher = FolderWatcher(
                folder=self.args.folder,
                formats=self.args.formats,
                recursive=self.args.recursive
            )
            
            self.handler = FolderWatcherHandler(
                folder=self.args.folder,
                formats=self.args.formats,
                transcriber=self.transcriber
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            return False
            
    def run(self) -> int:
        """Run the application."""
        try:
            # Start the watcher
            self.watcher.run(self.handler)
            return 0
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.shutdown()
            return 0
            
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            self.shutdown()
            return 1
            
    def shutdown(self) -> None:
        """Shutdown all components gracefully."""
        if self.transcriber:
            self.transcriber.shutdown() 