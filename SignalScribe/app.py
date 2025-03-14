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
from .transcriber import Transcriber, TranscriberStatus
from .version import __version__
from .utils import UserException, get_system_info, has_permission
from .watcher import FolderWatcher
from .defaults import DEFAULT_MODEL


class SignalScribeApp:
    """Main application class that manages all pretty much everything apart from setting up logging."""

    @property
    def csv_file_path(self) -> Path:
        return self._csv_file_path

    @csv_file_path.setter
    def csv_file_path(self, file_path: Path):
        self._csv_file_path = file_path
        logger.debug(f"Set CSV file path to {self._csv_file_path}")

    @property
    def log_file_path(self) -> Path:
        return self._log_file_path

    @log_file_path.setter
    def log_file_path(self, file_path: Path):
        self._log_file_path = file_path
        logger.debug(f"Set log file path to {self._log_file_path}")

    @property
    def recording_dir(self) -> Path:
        return self._recording_dir

    @recording_dir.setter
    def recording_dir(self, folder: Path):
        self._recording_dir = folder
        logger.debug(f"Set recording folder to {self._recording_dir}")

    def __init__(self, args):
        """Initialize the application with command line arguments."""
        self.args = args  # Command line arguments set by user
        self._monitoring_sdrtrunk = False  # Is app automatically monitoring SDRTrunk?

        self.running = False

        # Properties:
        self._csv_file_path = None
        self._log_file_path = None
        self._recording_dir = None

        # Shared state between components
        self.shared_colors = {}  # Dictionary to share colors between watcher and output
        self.shared_colors_lock = threading.Lock()  # Lock to protect access to shared_colors

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
        for line in get_system_info():
            logger.debug(line)

        # Initialize model management
        model_dir = self.args.model_dir if hasattr(self.args, "model_dir") else None
        self.model_manager = ModelManager(model_dir, self.args.list_models)

        # If a model was specified on the command line, use it
        if hasattr(self.args, "model") and self.args.model:
            logger.debug(f"Attempting to use model specified on command line: {self.args.model}")
            try:
                self.model_manager.selected_model = self.args.model
            except ValueError as e:
                raise UserException(
                    f"Given model '{self.args.model}' not found, valid choices are: {', '.join(self.model_manager.model_list())}"
                )

        else:
            # Otherwise use the default model
            logger.debug("Using default model")
            self.model_manager.selected_model = DEFAULT_MODEL

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
            folder=self._recording_dir,
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
            model_name=self.model_manager.selected_model,
            model_dir=self.model_manager.model_dir,
            n_threads=self.args.threads,
            show_whispercpp_logs=self.args.whisper_logs,
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
        logger.debug("Setup complete")

    def _set_up_recording_folder(self):
        """
        Set up the recording folder:
        1. If folder is not specified, try to get SDRTrunk recording directory.
        2. If folder is specified, check that it exists and is a directory.
        3. If it doesn't exist, prompt the user to create it.
        """
        logger.debug("Setting up recording folder")

        # If folder is not specified, try to get SDRTrunk recording directory
        if not self.args.folder:
            logger.info("No folder specified, attempting to use SDRTrunk recording directory")
            detector = SDRTrunkDetector()

            recording_dir = detector.get_recording_directory()

            if recording_dir:
                # Check if we have write permissions to the recording directory
                if not has_permission(recording_dir):
                    raise UserException(
                        f"You don't have write permission for SDRTrunk recording directory: {recording_dir}"
                    )

                self._recording_dir = recording_dir
                self._monitoring_sdrtrunk = True

                logger.info(f"Monitoring SDRTrunk recording directory: {self._recording_dir}")

                # Warn if SDRTrunk isn't running:
                if not detector.get_process():
                    logger.warning("SDRTrunk isn't running, monitoring its directory anyway")

                return
            else:
                console.print("[red]Error:[/red] No folder specified and could not find SDRTrunk recording directory.")
                console.print("Please either:")
                console.print("1. Specify a folder with --folder")
                console.print("2. Ensure SDRTrunk is installed and configured")
                console.print("   (SDRTrunk does not need to be running)")

                # This is fatal error by now so let's raise an exception and let the parent caller handle die gracefully...
                raise FileNotFoundError("No folder specified and could not find SDRTrunk recording directory.")

        else:
            # Check that the folder exists, and if not, prompt the user to create it
            self._recording_dir = Path(self.args.folder).expanduser().resolve()

            if not self._recording_dir.exists():
                if Confirm.ask(
                    "[red]The folder you requested to observe does not exist[/red]\n Would you like to create it?",
                    default=True,
                ):
                    logger.debug(f"Creating folder: {self._recording_dir}")

                    self._recording_dir.mkdir(parents=True, exist_ok=True)

            # Check that the folder is a directory, and if not (i.e. a file), prompt the user to use its parent directory
            if not self._recording_dir.is_dir():
                parent_dir = self._recording_dir.parent
                if Confirm.ask(
                    f"[red]The given path is not a directory[/red]\n Would you like to use its parent directory instead? {parent_dir}",
                    default=True,
                ):
                    self._recording_dir = parent_dir
                else:
                    raise FileExistsError(f"The specified path is is a file, not a directory: {self._recording_dir}")

        # Check if we have write permissions to the recording directory
        if not has_permission(self._recording_dir):
            raise UserException(f"You don't have write permission for directory: {self._recording_dir}")

    def _set_up_csv_file(self):
        """
        Set up the CSV file:
        1. If CSV file is not specified, use self.args.folder/signalscribe.csv.
        2. If CSV file is specified, check its parent directory exists. If not, prompt the user to create it.
        3. If a directory is specified, we put the CSV file there and name it after the recording directory.
        """
        logger.debug("Setting up CSV file")

        if not hasattr(self.args, "csv_path") or not self.args.csv_path:
            # Use default CSV file in the recording folder
            logger.debug("Using default CSV file path")
            self.csv_file_path = self._recording_dir / f"{self._recording_dir.name}.csv"
        else:
            # Use the specified CSV file
            logger.debug("Using CSV file path specified by user")
            csv_path = Path(self.args.csv_path).expanduser().resolve()

            # Check if the given path is a directory, if so then we put the CSV
            # file there and name it after the recording directory
            if csv_path.is_dir() or not csv_path.suffix:
                logger.debug("Given CSV path is a directory, using it with default CSV file name")
                self.csv_file_path = csv_path / f"{self._recording_dir.name}.csv"
                parent_dir = csv_path
            else:
                self.csv_file_path = csv_path
                parent_dir = csv_path.parent

            # Check if we have write permissions to the CSV file or its parent directory
            if not has_permission(self.csv_file_path):
                raise UserException(f"You don't have permission to write a CSV file to: {self.csv_file_path}")

            # Check if the parent directory exists
            if not parent_dir.exists():
                if Confirm.ask(
                    f"[red]The directory for the CSV {parent_dir} file does not exist[/red]\n Would you like to create it?",
                    default=True,
                ):
                    logger.debug(f"Creating directory for CSV file: {parent_dir}")
                    # Try to create the directory
                    parent_dir.mkdir(parents=True, exist_ok=True)

                else:
                    raise FileNotFoundError(
                        f"The directory for the requested CSV file location does not exist: {parent_dir}"
                    )

    def _build_status_display(self) -> Progress:
        """Create a rich progress display for queue status."""

        logger.debug("Building status display")

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            TextColumn("[cyan]{task.fields[status]}"),
            TimeElapsedColumn(),
            console=console,
            expand=False,
            transient=True,
        )

        # Create all possible tasks but hide them initially
        self.listening_task_id = progress.add_task(
            f"[cyan3]Monitoring {self._recording_dir} for audio files...[/cyan3] [dim]press CTRL+C to exit[/dim]",
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

        with Live(
            progress,
            console=console,
            refresh_per_second=10,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        ):
            logger.debug("Live display created")

            while self.running:
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

                sleep(0.1)

            logger.debug("Exiting status loop")
            progress.disable()

    def _print_banner(self) -> None:
        """Print the intro message."""
        logger.warning("Printing banner")
        console.print(
            Panel(
                f"[bold][bright_cyan]SignalScribe[/bright_cyan][/bold]\n[cyan3]by Signals Rising[/cyan3]\n[dim]Version {__version__}",
                expand=False,
            )
        )

    def print_parameters(self) -> None:
        """Print runtime parameters for the application."""

        # Try to get system info from transcriber's shared dictionary
        # Wait a short time for the transcriber process to initialise and populate the shared dict
        max_wait = 20.0  # Maximum seconds to wait
        wait_interval = 0.1
        waited = 0

        with console.status("Loading transcriber model..."):
            while waited < max_wait:
                status = self.transcriber.shared_dict["status"]
                # Check if transcriber exists and has populated the shared dict
                if status == TranscriberStatus.RUNNING:
                    break
                elif status == TranscriberStatus.ERROR or status == TranscriberStatus.SHUTDOWN:
                    raise Exception("Transcriber process failed to load")

            # Wait a bit and try again
            sleep(wait_interval)
            waited += wait_interval

        grid = Table.grid(padding=(0, 2))

        grid.add_column(justify="right", style="yellow", no_wrap=True)
        grid.add_column()
        grid.add_column(style="dim", no_wrap=True)

        grid.add_row("Model", self.model_manager.selected_model, "Set with --model or -m")
        grid.add_row("Compute", self.transcriber.shared_dict["system_info"])
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

        if self._monitoring_sdrtrunk:
            grid.add_row("Monitoring", f"{str(self._recording_dir)} [red](from SDRTrunk)[/red]")
        else:
            grid.add_row("Monitoring", str(self._recording_dir))

        console.print(grid)
        console.print("")

    def run(self) -> None:
        """Run the application."""
        # Start the watcher
        self.running = True

        self._build_status_display()

        logger.debug("Starting watcher")
        self.watcher.run()

        logger.debug("Starting status loop")
        self._status_loop()

    def stop(self) -> None:
        logger.info("Shutting down application...")

        self.running = False

        try:
            self._cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise e

        logger.info("Application shutdown complete")

    def _cleanup(self) -> None:
        # Shutdown transcriber first (it has a process)
        if self.transcriber:
            self.transcriber.stop()

        # Then shutdown threads:
        if self.watcher:
            self.watcher.stop()
        if self.decoder:
            self.decoder.stop()
        if self.output:
            self.output.stop()
