from queue import Queue
from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from yaml import YAMLError, SafeLoader
import os
import rich.color
import time
import yaml

from .transcription import Transcription
from .utils import logger, console, COLORS_SETTINGS_NAME

POLLING_INTERVAL = 10


class FolderWatcher:
    """Watches a folder for new audio files."""

    def __init__(
        self,
        queue: Queue,
        folder: str,
        formats: list,
        recursive: bool = False,
        polling: bool = True,
        polling_interval: int = POLLING_INTERVAL,
    ):
        """Initialize the folder watcher

        :param queue: Queue between watcher and decoder
        :param folder: Folder to watch (full or relative path)
        :param formats: List of formats to watch for
        :param recursive: Whether to watch the folder recursively (i.e. also watch all its subfolders)
        :param polling: Whether to use polling observer - user can force this if inotify observer is not working (e.g. on network drives)
        :param polling_interval: Polling interval in seconds when using polling observer
        """
        self.queue = queue
        self.folder = folder
        self.formats = formats
        self.recursive = recursive
        self.polling = polling
        self.polling_interval = polling_interval

        try:
            if not self.polling:
                self.observer = Observer()
                return
        except Exception as e:
            logger.warning(f"Error initializing observer: {e}")
            self.polling = True

        self.observer = PollingObserver(timeout=self.polling_interval)

    def run(self):
        """Start watching the folder with the specified handler."""

        handler = FolderWatcherHandler(
            queue=self.queue,
            folder=self.folder,
            formats=self.formats,
        )

        self.observer.schedule(
            event_handler=handler, path=self.folder, recursive=self.recursive
        )
        self.observer.start()
        logger.info(f"Started watching folder: {self.folder}")

        try:
            while True:
                time.sleep(self.polling_interval)
        except KeyboardInterrupt:
            logger.info("Stopping folder watcher...")
            self.observer.stop()

        self.observer.join()


class FolderWatcherHandler(PatternMatchingEventHandler):
    """Handles file system events for the folder watcher, acting as a producer."""

    def __init__(self, queue: Queue, folder: str, formats: list):
        """Initialize the event handler."""
        self.queue = queue
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
            logger.warning(
                "You are watching the current working directory. This is not recommended."
            )

        # Print watch status
        formats_text = ", ".join(formats)
        logger.info(f"Watching for formats: {formats_text}")

    def _update_colors(self, colors_file_path: str) -> None:
        """Update color highlighting settings from YAML file."""
        if not os.path.exists(colors_file_path):
            return

        logger.debug(f"Updating colors from: {colors_file_path}")

        try:
            with open(colors_file_path, "r") as colors_file:
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
        """Handle file creation events by producing decoding tasks."""
        file_path = event.src_path  # Full path to the new file
        file_name = os.path.basename(file_path)  # Name of the new file

        if file_name == COLORS_SETTINGS_NAME:
            self._update_colors(file_path)
            return

        # Produce a new decoding task
        logger.info(f"New audio file detected: {file_path}")
        self.queue.put(Transcription(file_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if os.path.basename(event.src_path) == COLORS_SETTINGS_NAME:
            self._update_colors(event.src_path)

    def on_closed(self, event: FileSystemEvent) -> None:
        """Handle file close events."""
        if os.path.basename(event.src_path) == COLORS_SETTINGS_NAME:
            self._update_colors(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events."""
        logger.info(f"File deleted: {event.src_path}")
