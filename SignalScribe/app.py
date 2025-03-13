"""Main application class for SignalScribe."""

from pathlib import Path
from pywhispercpp.model import Model
from multiprocessing import Queue, Value, Lock
from queue import Empty
from time import sleep
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, IntPrompt
from .utils import logger, console, nested_dict_to_string, resolve_filepath
from .model import ModelManager
from .transcriber import Transcriber
from .decoder import Decoder
from .watcher import FolderWatcher
from .version import __version__
from .trackedqueue import TrackedQueue
from .output import Output
import threading
import sys
from .sdrtrunk import SDRTrunkDetector


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
        self.args = args  # Command line arguments set by user
        self.monitoring_sdrtrunk = False # Is app automatically monitoring SDRTrunk?

        # Shared state between components
        self.shared_colors = {}  # Dictionary to share colors between watcher and output
        self.shared_colors_lock = threading.Lock()  # Lock to protect access to shared_colors

        # Threads:
        self.model_manager = None  # Checks for models and downloads them if needed
        self.watcher = None  # FWatches for new files
        self.decoder = None  # Reads audio files and adds to the transcribing queue
        self.transcriber = None  # Transcriber, transcribes audio files and adds them to the completed queue
        self.output = None  # Output, saves transcriptions to CSV and outputs to console

        # Rich live display
        self.live_display = None  # Displays length of the queues

    def start(self) -> bool:
        """Initialize all components."""
        self.print_banner()

        # If folder is not specified, try to get SDRTrunk recording directory
        if not self.args.folder:
            logger.info("No folder specified, attempting to use SDRTrunk recording directory")
            detector = SDRTrunkDetector()
            recording_dir = detector.get_recording_directory()
            
            if recording_dir:
                self.args.folder = str(recording_dir)
                logger.info(f"Monitoring SDRTrunk recording directory: {self.args.folder}")
                self.monitoring_sdrtrunk = True
            else:
                logger.error("Could not determine SDRTrunk recording directory")
                console.print("[red]Error:[/red] No folder specified and could not find SDRTrunk recording directory.")
                console.print("Please either:")
                console.print("  1. Specify a folder with --folder")
                console.print("  2. Ensure SDRTrunk is installed and properly configured")
                console.print("     (SDRTrunk does not need to be running)")
                return False
            
        else:
            # Check that the folder exists, and if not, prompt the user to create it
            self.folder_path = Path(self.args.folder).expanduser().resolve()

            if not self.folder_path.exists():
                console.print(
                    f"[red]Error:[/red] The folder to observe does not exist: {self.folder_path}"
                )
                if Confirm.ask(
                    f"Would you like to create it?",
                    default=False,
                ):
                    try:
                        self.folder_path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        console.print(
                            f"[red]Error:[/red] Failed to create folder: {self.folder_path}"
                        )
                        return False
                else:
                    return False

            # Check that the folder is a directory, and if not (i.e. a file), prompt the user to use its parent directory
            if not self.folder_path.is_dir():
                console.print(
                    f"[red]Error:[/red] The specified path is not a directory: {self.folder_path}"
                )
                parent_dir = self.folder_path.parent
                if Confirm.ask(
                    f"Would you like to use its parent directory instead? {parent_dir}",
                    default=False,
                ):
                    self.folder_path = parent_dir
                else:
                    return False

        # Initialize model management - checks for required models and downloads them if needed
        model_dir = self.args.model_dir if hasattr(self.args, "model_dir") else None
        self.model_manager = ModelManager(self.args.model_dir, self.args.list_models)

        if self.args.list_models:
            available_models = fetch_available_models()
            console.print(f"Available models:")
            if available_models:
                grid = Table.grid(padding=(0, 2))
                grid.add_column("Index", justify="left", style="blue", no_wrap=True)
                grid.add_column("Model", justify="left", style="bold", no_wrap=True)
                grid.add_column("Size", justify="left", style="dim", no_wrap=True)

                choices = []

                max_index = 0

                for i, (base_name, model_info) in enumerate(available_models.items()):
                    index = i+1
                    max_index = index
                    choices.append(str(index))
                    if sys.platform == "darwin":
                        grid.add_row(f"{index}", f"{base_name}", f"{model_info['bin_size']} (+ {model_info['coreml_size']} for CoreML)")
                    else:
                        grid.add_row(f"{index}", f"{base_name}", f"{model_info['bin_size']}")

                console.print(grid)
                
                selected_index = None

                while True:
                    selected_index = IntPrompt.ask(
                        f"\n[bold]Select which model to use[/bold] (Press CTRL+C to exit)"
                    )
                    if selected_index >= 1 and selected_index <= max_index:
                        break
                    console.print(f"[prompt.invalid]Number must be between 1 and {max_index}")
                        
                if selected_index:
                    self.args.model = list(available_models.keys())[selected_index-1]
                    console.print(f"Selected model: [bold][blue]{self.args.model}[/blue][/bold]")
                else:
                    console.print("[red]No model selected, quitting[/red]") # Should never happen
                    return False


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
            silent=self.args.silent,
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
            csv_filepath=self.csv_filepath,
            shared_colors=self.shared_colors,
            shared_colors_lock=self.shared_colors_lock,
        )

        # Print table of runtime parameters, like model stats, threads etc
        self.print_parameters()

        return True

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

        with Live(progress, console=console, refresh_per_second=10) as live:
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
                
                sleep(0.1)
                        



    def print_banner(self) -> None:
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
        # Wait a short time for the transcriber process to initialize and populate the shared dict
        max_wait = 3.0  # Maximum seconds to wait
        wait_interval = 0.1
        waited = 0
        
        while waited < max_wait:
            # Check if transcriber exists and has populated the shared dict
            if hasattr(self, 'transcriber') and self.transcriber and 'system_info' in self.transcriber.shared_dict:
                system_info_string = self.transcriber.shared_dict['system_info']
                break
            
            # Wait a bit and try again
            sleep(wait_interval)
            waited += wait_interval
        
        grid = Table.grid(padding=(0, 2))

        grid.add_column(justify="right", style="yellow", no_wrap=True)
        grid.add_column()
        grid.add_column(style="dim", no_wrap=True)

        grid.add_row("Model", self.args.model, "Set with --model")
        grid.add_row("Compute", system_info_string)
        grid.add_row("CPU Threads", f"{self.args.threads}", "Set with --threads")
        grid.add_row("CSV File", str(self.csv_filepath), "Set with --csv-filepath, remove with --no-csv")
        grid.add_row("Log File", str(self.log_filepath), "Set with --log-filepath, remove with --no-logs")

        if self.monitoring_sdrtrunk:
            grid.add_row("Monitoring", f"{str(self.args.folder)} [red]Autodetected from SDRTrunk[/red]")
        else:
            grid.add_row("Monitoring", str(self.args.folder))

        console.print(grid)
        console.print("")

    def run(self) -> int:
        """Run the application."""
        try:
            # Start the watcher
            self._build_status_display()

            self.watcher.run()
            
            self._status_loop()
            return 0

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.shutdown()
            return 0

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e
            self.shutdown()
            return 1

    def shutdown(self) -> None:
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
            keep_last_n=None
        )