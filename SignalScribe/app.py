"""Main application class for SignalScribe."""

from pathlib import Path
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm
from rich.table import Table
from time import sleep
import threading

from .decoder import Decoder
from .logging import logger, console
from .model import ModelManager
from .output import Output
from .sdrtrunk import SDRTrunkDetector
from .trackedqueue import TrackedQueue
from .transcriber import Transcriber
from .version import __version__
from .utils import get_system_info, nested_dict_to_string
from .watcher import FolderWatcher


class SignalScribeApp:
    """Main application class that manages all pretty much everything apart from setting up logging."""

    @property
    def csv_file_path(self) -> Path:
        return self._csv_file_path

    @property
    def log_file_path(self) -> Path:
        return self._log_file_path

    @log_file_path.setter
    def log_file_path(self, file_path: Path):
        self._log_file_path = file_path
        logger.debug(f"Set log file path to {self._log_file_path}")

    def __init__(self, args):
        """Initialize the application with command line arguments."""
        self.args = args  # Command line arguments set by user
        self.monitoring_sdrtrunk = False  # Is app automatically monitoring SDRTrunk?

        # Shared state between components
        self.shared_colors = {}  # Dictionary to share colors between watcher and output
        self.shared_colors_lock = (
            threading.Lock()
        )  # Lock to protect access to shared_colors

        # Threads:
        self.model_manager = None  # Checks for models and downloads them if needed
        self.watcher = None  # Watches for new files
        self.decoder = None  # Reads audio files and adds to the transcribing queue
        self.transcriber = None  # Transcriber, transcribes audio files and adds them to the completed queue
        self.output = None  # Output, saves transcriptions to CSV and outputs to console

        # Rich live display
        self.live_display = None  # Displays length of the queues

    def setup(self) -> None:
        """Sets up the app to run, inlcuding:
        - Printing the banner
        - Setting up the recording folder
        - Initializing the model manager
        - Resolving the CSV and log file paths
        - Choosing which model to use
        """
        self._print_banner()

        self._set_up_recording_folder()
        self._set_up_csv_file()

        # For troubleshooting from 3rd party log files:
        logger.debug(nested_dict_to_string(get_system_info()))

        # Initialize model management - checks for required models and downloads them if needed
        model_dir = self.args.model_dir if hasattr(self.args, "model_dir") else None
        self.model_manager = ModelManager(model_dir, self.args.list_models)

        # Log the resolved paths
        logger.info(f"Using CSV file: {self._csv_file_path}")
        logger.info(f"Using log file: {self._log_file_path}")

        # Pipeline is:
        # FolderWatcher: Monitors for new files and if they are audio files adds
        #                them to decoding queue
        # Decoder:       Monitors decoding queue and any files in it are converted
        #                to raw numpy array then placed in transcribing queue.
        # Transcriber:   Monitors transcribing queue and transcribes any audio data
        #                into text which is then placed on output queue.
        # Output:        Prints transcriptions in the output queue to screen and
        #                saves them to disk. In future can do more with the texts.

        # Initialize queues for passing data between stages of the transcription pipeline
        self.decoding_queue = TrackedQueue(name="Decoding")
        self.transcribing_queue = TrackedQueue(name="Transcribing")
        self.output_queue = TrackedQueue(name="Output")

        # Initialize watcher - watches for new files in the folder and adds them to the decoding queue
        self.watcher = FolderWatcher(
            queue=self.decoding_queue,
            folder=self.args.folder,
            formats=self.args.formats,
            shared_colors=self.shared_colors,
            shared_colors_lock=self.shared_colors_lock,
            recursive=self.args.recursive,
            # polling=self.args.polling,
            # polling_interval=self.args.polling_interval,
        )

        # Initialize decoder - decodes audio files and adds them to the transcribing queue
        self.decoder = Decoder(
            decoding_queue=self.decoding_queue,
            transcribing_queue=self.transcribing_queue,
        )

        # Initialize transcriber - the transcriber manages its own worker process internally
        # Pass our shared counter tuple (counter, lock) to track completed transcriptions
        self.transcriber = Transcriber(
            transcribing_queue=self.transcribing_queue,
            output_queue=self.output_queue,
            model_name=self.args.model,
            model_dir=self.model_manager._model_dir,
            n_threads=self.args.threads,
        )

        # Initialize output - saves transcriptions to CSV and outputs to console
        self.output = Output(
            output_queue=self.output_queue,
            csv_file_path=self._csv_file_path,
            shared_colors=self.shared_colors,
            shared_colors_lock=self.shared_colors_lock,
        )

        # Print table of runtime parameters, like model stats, threads etc
        self.print_parameters()

    def _set_up_recording_folder(self):
        """
        Set up the recording folder:
        1. If folder is not specified, try to get SDRTrunk recording directory.
        2. If folder is specified, check that it exists and is a directory.
        3. If it doesn't exist, prompt the user to create it.
        """

        # If folder is not specified, try to get SDRTrunk recording directory
        if not self.args.folder:
            logger.info(
                "No folder specified, attempting to use SDRTrunk recording directory"
            )
            detector = SDRTrunkDetector()
            recording_dir = detector.get_recording_directory()

            if recording_dir:
                self.args.folder = str(recording_dir)
                logger.info(
                    f"Monitoring SDRTrunk recording directory: {self.args.folder}"
                )
                self.monitoring_sdrtrunk = True
                return
            else:
                console.print(
                    "[red]Error:[/red] No folder specified and could not find SDRTrunk recording directory."
                )
                console.print("Please either:")
                console.print("1. Specify a folder with --folder")
                console.print("2. Ensure SDRTrunk is installed and configured")
                console.print("   (SDRTrunk does not need to be running)")
                
                # This is fatal error by now so let's raise an exception and let the parent caller handle die gracefully...
                raise FileNotFoundError(
                    "No folder specified and could not find SDRTrunk recording directory."
                )

        else:
            # Check that the folder exists, and if not, prompt the user to create it
            self.folder_path = Path(self.args.folder).expanduser().resolve()

            if not self.folder_path.exists():
                if Confirm.ask(
                    f"[red]The folder you requested to observe does not exist[/red]\n Would you like to create it?",
                    default=True,
                ):
                    logger.debug(f"Creating folder: {self.folder_path}")

                    # Don't catch any exceptions here, let the caller deal with it
                    self.folder_path.mkdir(parents=True, exist_ok=True)
                    return

                else:
                    raise FileNotFoundError(
                        f"The specified folder does not exist: {self.folder_path}"
                    )

            # Check that the folder is a directory, and if not (i.e. a file), prompt the user to use its parent directory
            if not self.folder_path.is_dir():
                parent_dir = self.folder_path.parent
                if Confirm.ask(
                    f"[red]The given path is not a directory[/red]\n Would you like to use its parent directory instead? {parent_dir}",
                    default=False,
                ):
                    self.folder_path = parent_dir
                else:
                    raise FileExistsError(
                        f"The specified path is is a file, not a directory: {self.folder_path}"
                    )
                
    def _set_up_csv_file(self):
        """
        Set up the CSV file:
        1. If CSV file is not specified, use self.args.folder/signalscribe.csv.
        2. If CSV file is specified, check its parent directory exists. If not, prompt the user to create it.
        """
        logger.debug("Setting up CSV file")
        
        if not hasattr(self.args, "csv_file") or not self.args.csv_file:
            # Use default CSV file in the recording folder
            self._csv_file_path = self.folder_path / "signalscribe.csv"
            logger.debug(f"Using default CSV file path: {self._csv_file_path}")
        else:
            # Use the specified CSV file
            self._csv_file_path = Path(self.args.csv_file).expanduser().resolve()
            logger.debug(f"Using specified CSV file path: {self._csv_file_path}")
            
            # Check if the parent directory exists
            parent_dir = self._csv_file_path.parent
            if not parent_dir.exists():
                if Confirm.ask(
                    f"[red]The directory for the CSV file does not exist[/red]\n Would you like to create it?",
                    default=True,
                ):
                    logger.debug(f"Creating directory for CSV file: {parent_dir}")
                    
                    # Don't catch any exceptions here, let the caller deal with it
                    parent_dir.mkdir(parents=True, exist_ok=True)
                else:
                    raise FileNotFoundError(
                        f"The directory for the requested CSV file location does not exist: {parent_dir}"
                    )

    def _build_status_display(self):
        """Create a rich progress display for queue status."""

        logger.debug("Building status display")

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
            f"[blue]Monitoring {self.folder_path} for audio files...[/blue][dim](press CTRL+C to exit)[/dim]",
            total=None,  # Indeterminate total
            status="",
            visible=False,
        )

        self.decoding_task_id = progress.add_task(
            "[yellow]Decoding Queue",
            total=None,  # Indeterminate total
            status="",
            visible=False,
        )

        self.transcribing_task_id = progress.add_task(
            "[green]Transcribing Queue",
            total=None,  # Indeterminate total
            status="",
            visible=False,
        )

        return progress

    def _status_loop(self):
        # Create the progress display once

        progress = self._build_status_display()

        logger.debug("Entering status loop")

        with Live(progress, console=console, refresh_per_second=10) as live:
            logger.debug("Live display created")

            while not progress.finished:
                # Get current queue sizes
                decoding_queue_size = self.decoding_queue.size()
                transcribing_queue_size = self.transcribing_queue.size()

                # Update task visibility based on queue status
                if decoding_queue_size == 0 and transcribing_queue_size == 0:
                    # Show only the listening task
                    progress.update(self.listening_task_id, visible=True)
                    progress.update(self.decoding_task_id, visible=False)
                    progress.update(self.transcribing_task_id, visible=False)
                else:
                    # Hide the listening task
                    progress.update(self.listening_task_id, visible=False)

                    # Show/hide and update the decoding task
                    if decoding_queue_size > 0:
                        progress.update(
                            self.decoding_task_id,
                            visible=True,
                            status=f"{decoding_queue_size} pending",
                        )
                    else:
                        progress.update(self.decoding_task_id, visible=False)

                    # Show/hide and update the transcribing task
                    if transcribing_queue_size > 0:
                        progress.update(
                            self.transcribing_task_id,
                            visible=True,
                            status=f"{transcribing_queue_size} pending",
                        )
                    else:
                        progress.update(self.transcribing_task_id, visible=False)

                logger.debug("Updating status display")

                sleep(0.1)

    def _print_banner(self) -> None:
        """Print the intro message."""
        console.print(
            Panel(
                f"[bold][bright_cyan]SignalScribe[/bright_cyan][/bold]\n[cyan3]by Signals Rising[/cyan3]\n[dim]Version {__version__}",
                expand=False,
            )
        )

    def print_parameters(self) -> None:
        """Print runtime parameters for the application."""
        system_info_string = "Unknown"

        # Try to get system info from transcriber's shared dictionary
        # Wait a short time for the transcriber process to initialise and populate the shared dict
        max_wait = 20.0  # Maximum seconds to wait
        wait_interval = 0.1
        waited = 0

        system_info_string = None

        with console.status("Loading transcriber model..."):
            while waited < max_wait:
                # Check if transcriber exists and has populated the shared dict
                if (
                    hasattr(self, "transcriber")
                    and self.transcriber
                    and "system_info" in self.transcriber.shared_dict
                ):
                    system_info_string = self.transcriber.shared_dict["system_info"]
                    break

            # Wait a bit and try again
            sleep(wait_interval)
            waited += wait_interval

        if system_info_string is None:
            raise Exception("Transcriber process failed to load")

        grid = Table.grid(padding=(0, 2))

        grid.add_column(justify="right", style="yellow", no_wrap=True)
        grid.add_column()
        grid.add_column(style="dim", no_wrap=True)

        grid.add_row("Model", self.args.model, "Set with --model or -m")
        grid.add_row("Compute", system_info_string)
        grid.add_row("CPU Threads", f"{self.args.threads}", "Set with --threads or -t")
        grid.add_row(
            "CSV File",
            str(self._csv_file_path),
            "Set with --csv-path, remove with --no-csv",
        )
        grid.add_row(
            "Log File",
            str(self._log_file_path),
            "Set with --log-dir-path, remove with --no-logs",
        )

        if self.monitoring_sdrtrunk:
            grid.add_row(
                "Monitoring", f"{str(self.args.folder)} [red](from SDRTrunk)[/red]"
            )
        else:
            grid.add_row("Monitoring", str(self.args.folder))

        console.print(grid)
        console.print("")

    def run(self) -> int:
        """Run the application."""
        try:
            # Start the watcher
            self._build_status_display()

            logger.debug("Starting watcher")
            self.watcher.run()

            logger.debug("Starting transcriber")
            self.transcriber.start()

            logger.debug("Starting status loop")    
            self._status_loop()
            return 0

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self._shutdown()
            return 0

        except Exception as e:
            logger.fatal(f"An error occurred: {e}")
            self._shutdown()
            return 1

    def _shutdown(self) -> None:
        """Gracefully shutdown all components."""
        logger.info("Shutting down application...")

        # Shutdown transcriber first (it has a process)
        if self.transcriber:
            self.transcriber.shutdown()

        # Then shutdown threads:
        if self.decoder:
            self.decoder.shutdown()
        if self.output:
            self.output.shutdown()

        logger.info("Application shutdown complete")
