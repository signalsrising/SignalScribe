from multiprocessing import Queue
from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from yaml import YAMLError, SafeLoader
import os
import rich.color
import threading
import time
import yaml
import stat
import platform

from .transcription import Transcription
from .defaults import COLORS_FILE_NAME
from .logging import logger


POLLING_INTERVAL = 1


class FolderWatcher:
    """Watches a folder for new audio files."""

    def __init__(
        self,
        queue: Queue,
        folder: str,
        formats: list,
        shared_colors: dict = None,  # Shared colors dictionary
        shared_colors_lock=None,  # Lock for the shared colors dictionary
        recursive: bool = False,
        polling: bool = True,
        polling_interval: int = POLLING_INTERVAL,
    ):
        """Initialize the folder watcher

        :param queue: Queue between watcher and decoder
        :param folder: Folder to watch (full or relative path)
        :param formats: List of formats to watch for
        :param shared_colors: Dictionary to share colors between components
        :param shared_colors_lock: Lock to protect access to shared_colors
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
        self.shared_colors = shared_colors if shared_colors is not None else {}
        self.shared_colors_lock = shared_colors_lock
        self.running = False
        self.thread = None

        try:
            if not self.polling:
                self.observer = Observer()
                return
        except Exception as e:
            logger.warning(f"Error initialising observer: {e}")
            self.polling = True

        self.observer = PollingObserver(timeout=self.polling_interval)

    def run(self):
        """Watch the folder in a loop. This method runs in a separate thread."""
        handler = FolderWatcherHandler(
            queue=self.queue,
            folder=self.folder,
            formats=self.formats,
            shared_colors=self.shared_colors,
            shared_colors_lock=self.shared_colors_lock,
        )

        self.observer.schedule(
            event_handler=handler, path=self.folder, recursive=self.recursive
        )
        self.observer.start()
        logger.info(f"Started watching folder: {self.folder}")

        while self.running:
            time.sleep(self.polling_interval)

    def start(self):
        """Start the folder watcher in a separate thread."""
        if self.thread and self.thread.is_alive():
            logger.warning("Folder watcher is already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        logger.info("Folder watcher thread started")

    def stop(self):
        """Stop the folder watcher thread."""
        if not self.running:
            logger.warning("Folder watcher is not running")
            return

        logger.info("Stopping folder watcher...")
        self.running = False

        if self.observer:
            self.observer.stop()
            self.observer.join()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                logger.warning("Folder watcher thread did not terminate gracefully")

        logger.info("Folder watcher stopped")


class FolderWatcherHandler(PatternMatchingEventHandler):
    """Handles file system events for the folder watcher, acting as a producer."""

    def __init__(
        self,
        queue: Queue,
        folder: str,
        formats: list,
        shared_colors: dict = None,
        shared_colors_lock=None,
    ):
        self.queue = queue
        self.folder = folder
        self.shared_colors = shared_colors if shared_colors is not None else {}
        self.shared_colors_lock = shared_colors_lock

        # Create patterns for watchdog
        patterns = ["*." + fmt for fmt in formats]
        patterns.append(COLORS_FILE_NAME)
        super().__init__(patterns=patterns)

        # Set up color highlighting
        self.colors = dict()
        self._update_colors(os.path.join(folder, COLORS_FILE_NAME))

        # Warn if watching current directory
        if folder == os.getcwd():
            logger.warning(
                "You are watching the current working directory. This is not recommended."
            )

        # Print watch status
        formats_text = ", ".join(formats)
        logger.info(f"Watching for formats: {formats_text}")

    def _is_hidden(self, path: str) -> bool:
        """
        Check if a file or directory is hidden in a cross-platform way.
        
        :param path: Path to the file or directory
        :return: True if the file or directory is hidden, False otherwise
        """
        # Get the file/directory name
        name = os.path.basename(path)
        
        # On Unix-like systems (Linux, macOS), hidden files start with a dot
        if name.startswith('.'):
            return True
            
        # On Windows, check the hidden attribute
        if platform.system() == 'Windows':
            try:
                attrs = os.stat(path).st_file_attributes
                return bool(attrs & stat.FILE_ATTRIBUTE_HIDDEN)
            except (AttributeError, OSError):
                # If we can't get the attributes, fall back to just checking the name
                pass
                
        return False

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

            # Update the shared colors dictionary with lock protection
            if self.shared_colors is not None:
                if self.shared_colors_lock:
                    with self.shared_colors_lock:
                        self.shared_colors.clear()
                        self.shared_colors.update(valid_colors)
                else:
                    self.shared_colors.clear()
                    self.shared_colors.update(valid_colors)

            logger.info(f"Updated highlight settings from: {colors_file_path}")

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events by producing decoding tasks."""
        filepath = event.src_path  # Full path to the new file
        filename = os.path.basename(filepath)  # Name of the new file
        
        # Ignore hidden files and directories
        if self._is_hidden(filepath):
            return

        if filename == COLORS_FILE_NAME:
            self._update_colors(filepath)
            return

        # Produce a new decoding task
        logger.info(f"New audio file detected: {filepath}")
        self.queue.put(Transcription(filepath))

    # def on_any_event(self, event: FileSystemEvent) -> None:
    #     # Ignore hidden files and directories
    #     logger.info(f"File event ({event.event_type}): {event.src_path}")
    #     # if self._is_hidden(event.src_path):
    #         # return
            

    def on_moved(self, event: FileSystemEvent) -> None:
        # Ignore hidden files and directories

        if self._is_hidden(event.dest_path):
            return
        

        src_path = event.src_path
        dest_path = event.dest_path
        
        # If both are visible AND:
        # Source is ultimate parent of destination, ignore
        # OR
        # Destination is ultimate parent of source, ignore
        # This prevents decoding a file again when it is moved to a subfolder from a subfolder to parent

        
        if os.path.commonpath([src_path]) == os.path.commonpath([src_path, dest_path]) or \
           os.path.commonpath([dest_path]) == os.path.commonpath([src_path, dest_path]) and \
           not self._is_hidden(src_path) and \
           not self._is_hidden(dest_path):
            logger.debug(f"Ignoring move between parent-child directories: {src_path} -> {dest_path}")
            return
                    
        # Produce a new decoding task
        logger.info(f"New audio file detected [moved]: {event.dest_path}")
        self.queue.put(Transcription(event.dest_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        # Ignore hidden files and directories
        if self._is_hidden(event.src_path):
            return
            
        if os.path.basename(event.src_path) == COLORS_FILE_NAME:
            self._update_colors(event.src_path)

    def on_closed(self, event: FileSystemEvent) -> None:
        # Ignore hidden files and directories
        if self._is_hidden(event.src_path):
            return
            
        if os.path.basename(event.src_path) == COLORS_FILE_NAME:
            self._update_colors(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        # Ignore hidden files and directories
        if self._is_hidden(event.src_path):
            return
            
        logger.debug(f"File deleted: {event.src_path}")
