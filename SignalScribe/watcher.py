"""File system watching functionality for SignalScribe."""

import os
import time
import yaml
from yaml import YAMLError, SafeLoader
import rich.color
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
from .utils import logger, console, COLORS_SETTINGS_NAME
from .transcriber import Transcriber


class FolderWatcher:
    """Watches a folder for new audio files."""
    
    def __init__(self, folder: str, formats: list, recursive: bool = False):
        """Initialize the folder watcher."""
        self.folder = folder
        self.formats = formats
        self.recursive = recursive
        self.observer = PollingObserver(timeout=10)
        
    def run(self, handler):
        """Start watching the folder with the specified handler."""
        self.observer.schedule(
            event_handler=handler,
            path=self.folder,
            recursive=self.recursive
        )
        self.observer.start()
        logger.info(f"Started watching folder: {self.folder}")
        
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Stopping folder watcher...")
            self.observer.stop()
            
        self.observer.join()


class FolderWatcherHandler(PatternMatchingEventHandler):
    """Handles file system events for the folder watcher, acting as a producer."""
    
    def __init__(self, folder: str, formats: list, transcriber: Transcriber):
        """Initialize the event handler."""
        self.transcriber = transcriber
        self.folder = folder
        
        # Create patterns for watchdog
        patterns = ["*." + fmt for fmt in formats]
        patterns.append(COLORS_SETTINGS_NAME)
        super().__init__(patterns=patterns)
        
        # Set up color highlighting
        self.colors = dict()
        self._update_colors(os.path.join(folder, COLORS_SETTINGS_NAME))
        
        # Warn if watching current directory
        if folder == os.getcwd():
            logger.warning("You are watching the current working directory. This is not recommended.")
            
        # Print watch status
        formats_text = ", ".join(formats)
        logger.info(f"Watching for formats: {formats_text}")
        
    def _update_colors(self, colors_file_path: str) -> None:
        """Update color highlighting settings from YAML file."""
        if not os.path.exists(colors_file_path):
            return
            
        try:
            with open(colors_file_path, 'r') as colors_file:
                colors_dict = yaml.load(colors_file, Loader=SafeLoader)
        except YAMLError as e:
            logger.error(f"Invalid YAML syntax in colors file: {e}")
            return
            
        valid_colors = {}
        
        # Validate colors
        for color in colors_dict.keys():
            if color in rich.color.ANSI_COLOR_NAMES:
                phrases = colors_dict[color]
                if isinstance(phrases, list):
                    valid_colors[color] = [str(phrase) for phrase in phrases]
                    
        if valid_colors and valid_colors != self.colors:
            self.colors = valid_colors
            logger.info(f"Updated highlight settings from: {colors_file_path}")
            
    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events by producing transcription tasks."""
        file_path = event.src_path
        file_name = os.path.basename(file_path)
        
        if file_name == COLORS_SETTINGS_NAME:
            self._update_colors(file_path)
            return
            
        # Produce a new transcription task
        logger.info(f"New file detected: {file_path}")
        self.transcriber.add_task(file_path)
        
    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if os.path.basename(event.src_path) == COLORS_SETTINGS_NAME:
            self._update_colors(event.src_path)
            
    def on_closed(self, event: FileSystemEvent) -> None:
        """Handle file close events."""
        if os.path.basename(event.src_path) == COLORS_SETTINGS_NAME:
            self._update_colors(event.src_path) 