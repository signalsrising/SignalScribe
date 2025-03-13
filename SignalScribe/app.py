"""Main application class for SignalScribe."""

from pathlib import Path
from pywhispercpp.model import Model
from multiprocessing import Queue, Value, Lock
import threading
import time
from queue import Empty

from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn
from .utils import logger, console, nested_dict_to_string, resolve_filepath
from .model import ModelManager
from .transcriber import Transcriber
from .decoder import Decoder
from .watcher import FolderWatcher
from .version import __version__

# from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,


class TrackedQueue:
    """A queue wrapper that tracks its own size using shared memory."""
    
    def __init__(self, name="Queue", maxsize=0):
        self.name = name
        self.queue = Queue(maxsize=maxsize)  # Use composition instead of inheritance
        # Using multiprocessing.Value to share counter across processes
        self._size = Value('i', 0)
        self._mp_lock = Lock()
        
    def put(self, item, block=True, timeout=None):
        """Add an item to the queue and increment the size counter."""
        try:
            # First put the item in the queue
            self.queue.put(item, block=block, timeout=timeout)
            
            # Then increment the counter safely
            with self._mp_lock:
                self._size.value += 1
                
        except Exception as e:
            # If put fails, don't increment the counter
            logger.error(f"Error in TrackedQueue.put: {e}")
            raise
    
    def get(self, block=True, timeout=None):
        """Get an item from the queue and decrement the size counter."""
        try:
            # First get the item
            item = self.queue.get(block=block, timeout=timeout)
            
            # Then decrement the counter safely
            with self._mp_lock:
                if self._size.value > 0:
                    self._size.value -= 1
                    
            return item
        except Empty:
            # If the queue is empty, don't decrement, re-raise the exception
            raise
        except Exception as e:
            # If get fails for any other reason, log and re-raise
            logger.error(f"Error in TrackedQueue.get: {e}")
            raise
    
    def size(self):
        """Get the current size of the queue."""
        with self._mp_lock:
            return self._size.value


def _cleanup_old_logs(log_dir: Path, keep_last: int = 10) -> None:
    """Keep only the N most recent log files in the directory."""
    log_files = sorted(
        log_dir.glob("signalscribe_*.log"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    # Remove old files
    for old_log in log_files[keep_last:]:
        try:
            old_log.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove old log file {old_log}: {e}")


class SignalScribeApp:
    """Main application class that manages all components."""

    def __init__(self, args):
        """Initialize the application with command line arguments."""
        self.args = args
        self.model_manager = None
        self.transcriber = None
        self.decoder = None
        self.watcher = None
        self.handler = None
        
        # Status tracking - use multiprocessing Values for cross-process counters
        self.files_processed = 0  # This is updated in main process only
        self.files_transcribed_counter = Value('i', 0)  # Shared counter
        self.files_transcribed_lock = Lock()  # Lock for the counter
        
        self.status_lock = threading.Lock()  # For main process counters
        self.live_display = None
        self.stop_status_thread = threading.Event()
        self.status_thread = None

    def start(self) -> bool:
        """Initialize all components."""
        if not self.args.silent:
            self.print_banner()

        try:
            # Check that the folder exists and is accessible
            folder_path = Path(self.args.folder).expanduser().resolve()
            if not folder_path.exists():
                logger.error(f"Folder does not exist: {folder_path}")
                if not self.args.silent:
                    console.print(
                        f"[red]Error:[/red] The folder to observe does not exist: {folder_path}"
                    )
                return False

            if not folder_path.is_dir():
                logger.error(f"Path is not a directory: {folder_path}")
                if not self.args.silent:
                    console.print(
                        f"[red]Error:[/red] The specified path is not a directory: {folder_path}"
                    )
                return False

            # Initialize model management
            model_dir = self.args.model_dir if hasattr(self.args, "model_dir") else None
            self.model_manager = ModelManager(self.args.model, model_dir)

            # Ensure model files exist
            if not self.model_manager.ensure_models_exist():
                logger.error(
                    "Required model files are missing. Please download them and try again."
                )
                return False

            # Resolve file paths
            self.csv_filepath = self._resolve_csv_filepath()
            self.log_filepath = self._resolve_log_filepath()

            # Log the resolved paths
            logger.info(f"Using CSV file: {self.csv_filepath}")
            logger.info(f"Using log file: {self.log_filepath}")

            # Initialize queues for passing data between stages of the transcription pipeline
            # Use our tracked multiprocessing Queue
            self.decoding_queue = TrackedQueue(name="Decoding")
            self.transcribing_queue = TrackedQueue(name="Transcribing")

            # Initialize watcher - watches for new files in the folder and adds them to the decoding queue
            self.watcher = FolderWatcher(
                queue=self.decoding_queue,
                folder=self.args.folder,
                formats=self.args.formats,
                recursive=self.args.recursive,
                # polling=self.args.polling,
                # polling_interval=self.args.polling_interval,
            )

            # Initialize decoder - decodes audio files and adds them to the transcribing queue
            self.decoder = Decoder(
                decoding_queue=self.decoding_queue,
                transcribing_queue=self.transcribing_queue,
                silent=self.args.silent,
                file_processed_callback=self._on_file_processed
            )

            # Initialize transcriber - the transcriber manages its own worker process internally
            # Pass our shared counter tuple (counter, lock) to track completed transcriptions
            self.transcriber = Transcriber(
                task_queue=self.transcribing_queue,
                model_name=self.args.model,
                model_dir=self.model_manager.model_dir,
                n_threads=self.args.threads,
                csv_filepath=str(self.csv_filepath),
                silent=self.args.silent,
                print_progress=False,
                completed_counter=(self.files_transcribed_counter, self.files_transcribed_lock)
            )

            # Start status display if not in silent mode
            if not self.args.silent:
                self._start_status_display()

            self.print_parameters()
            return True

        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            # raise e
            return False
    
    def _on_file_processed(self):
        """Callback for when a file has been processed by the decoder."""
        with self.status_lock:
            self.files_processed += 1
    
    def _build_status_display(self):
        """Create a rich progress display for queue status."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            TextColumn("[cyan]{task.fields[status]}"),
            TimeElapsedColumn(),
            console=console,
            expand=False,
        )
        
        # Create all possible tasks but hide them initially
        self.listening_task_id = progress.add_task(
            "[blue]Listening for files", 
            total=None,  # Indeterminate total
            status="Waiting for new audio files...",
            visible=False
        )
        
        self.decoding_task_id = progress.add_task(
            "[yellow]Decoding Queue", 
            total=None,  # Indeterminate total
            status="",
            visible=False
        )
        
        self.transcribing_task_id = progress.add_task(
            "[green]Transcribing Queue", 
            total=None,  # Indeterminate total
            status="",
            visible=False
        )
        
        return progress
    
    def _update_status_display(self):
        """Thread function to update the status display."""
        try:
            # Create the progress display once
            progress = self._build_status_display()
            
            with Live(progress, refresh_per_second=4) as live:
                self.live_display = live
                
                while not self.stop_status_thread.is_set():
                    # Get current queue sizes
                    decoding_size = self.decoding_queue.size()
                    transcribing_size = self.transcribing_queue.size()
                    
                    # Get the transcribed count from the shared counter
                    with self.files_transcribed_lock:
                        files_transcribed = self.files_transcribed_counter.value
                    
                    # Update task visibility based on queue status
                    if decoding_size == 0 and transcribing_size == 0:
                        # Show only the listening task
                        progress.update(self.listening_task_id, visible=True)
                        progress.update(self.decoding_task_id, visible=False)
                        progress.update(self.transcribing_task_id, visible=False)
                    else:
                        # Hide the listening task
                        progress.update(self.listening_task_id, visible=False)
                        
                        # Show/hide and update the decoding task
                        if decoding_size > 0:
                            progress.update(
                                self.decoding_task_id, 
                                visible=True,
                                status=f"{decoding_size} pending, {self.files_processed} processed"
                            )
                        else:
                            progress.update(self.decoding_task_id, visible=False)
                        
                        # Show/hide and update the transcribing task
                        if transcribing_size > 0:
                            progress.update(
                                self.transcribing_task_id, 
                                visible=True,
                                status=f"{transcribing_size} pending, {files_transcribed} transcribed"
                            )
                        else:
                            progress.update(self.transcribing_task_id, visible=False)
                    
                    time.sleep(0.25)  # Update 4 times per second
        except Exception as e:
            logger.error(f"Error in status display: {e}")
    
    def _start_status_display(self):
        """Start the status display thread."""
        self.status_thread = threading.Thread(
            target=self._update_status_display,
            daemon=True
        )
        self.status_thread.start()

    def print_banner(self) -> None:
        """Print the intro message."""
        console.print(
            Panel(
                f"[bold][bright_cyan]SignalScribe\n[cyan]by Signals Rising[/bold]\n[bright_black]Version {__version__}",
                expand=False,
            )
        )

    def print_parameters(self) -> None:
        """Print runtime parameters for the application."""
        system_info_string = "Running in multiprocessing mode"

        grid = Table.grid(padding=(0, 2))

        grid.add_column(justify="right", style="yellow", no_wrap=True)
        grid.add_column()

        grid.add_row("Model", self.args.model)
        grid.add_row("Compute Info:", system_info_string)
        grid.add_row("CPU Threads", f"{self.args.threads}")
        grid.add_row("CSV File", str(self.csv_filepath))
        grid.add_row("Log File", str(self.log_filepath))

        console.print(grid)
        console.print("")

    def run(self) -> int:
        """Run the application."""
        try:
            # Start the watcher
            self.watcher.run()
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
        """Gracefully shutdown all components."""
        logger.info("Shutting down application...")
        
        # Stop the status display
        if self.status_thread:
            self.stop_status_thread.set()
            self.status_thread.join(timeout=1.0)
        
        # Shutdown transcriber first (it has a process)
        if self.transcriber:
            self.transcriber.shutdown()
        
        # Then shutdown decoder
        if self.decoder:
            self.decoder.shutdown()
            
        logger.info("Application shutdown complete")

    def _resolve_log_filepath(self) -> Path:
        """Resolve the log filepath based on user input or defaults."""
        return resolve_filepath(
            provided_path=self.args.log_filepath,
            default_dir=self._get_default_log_dir(),
            default_name="signalscribe",
            default_extension=".log",
            timestamp=True,
            keep_last_n=10,
        )

    def _get_default_log_dir(self) -> Path:
        """Get the default log directory."""
        return Path.home() / ".signalscribe" / "logs"

    def _resolve_csv_filepath(self) -> Path:
        """Resolve the CSV filepath based on user input or defaults."""
        # Get the monitored folder name for default CSV naming
        monitored_folder = Path(self.args.folder).expanduser().resolve()
        return resolve_filepath(
            provided_path=self.args.csv_filepath,
            default_dir=monitored_folder,
            default_name=monitored_folder.name,
            default_extension=".csv",
            timestamp=False,
            keep_last_n=None,
        )
