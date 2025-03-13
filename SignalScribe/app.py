"""Main application class for SignalScribe."""

from pathlib import Path
from pywhispercpp.model import Model

from rich.panel import Panel
from rich.table import Table
from .utils import logger, console, nested_dict_to_string, resolve_filepath
from .model import ModelManager
from .transcriber import Transcriber
from .decoder import Decoder
from .watcher import FolderWatcher
from .version import __version__
from multiprocessing import Queue, Process

# from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,


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
        self.model = None
        self.transcriber = None
        self.decoder = None
        self.watcher = None
        self.handler = None

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

            self.model = self._load_model(
                model_name=self.args.model,
                models_dir=self.model_manager.model_dir,
                n_threads=self.args.threads,
                print_progress=False,
                print_timestamps=False,
                redirect_whispercpp_logs_to=None,
            )

            # Ensure parent directories exist
            # self.csv_filepath.parent.mkdir(parents=True, exist_ok=True)
            # self.log_filepath.parent.mkdir(parents=True, exist_ok=True)

            # Log the resolved paths
            logger.info(f"Using CSV file: {self.csv_filepath}")
            logger.info(f"Using log file: {self.log_filepath}")

            # Initialize queues for passing data between stages of the transcription pipeline
            # Use multiprocessing.Queue instead of queue.Queue for inter-process communication
            self.decoding_queue = Queue()  # From watcher to decoder
            self.transcribing_queue = Queue()  # From decoder to transcriber
            self.result_queue = Queue()  # From transcriber to result writer

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
            )

            # Create transcriber process
            self.transcriber_process = Process(
                target=self._start_transcriber,
                args=(
                    self.args.model,
                    self.model_manager.model_dir,
                    self.args.threads,
                    self.transcribing_queue,
                    self.result_queue,
                    self.csv_filepath,
                    self.args.silent,
                ),
                daemon=True,
            )
            self.transcriber_process.start()
            logger.info("Transcriber process started")

            self.print_parameters()
            return True

        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            # raise e
            return False

    def print_banner(self) -> None:
        """Print the intro message."""
        console.print(
            Panel(
                f"[bold][bright_cyan]SignalScribe\n[cyan]by Signals Rising[/bold]\n[bright_black]Version {__version__}",
                expand=False,
            )
        )

    def print_parameters(self) -> None:
        system_info_string = "Unknown"

        try:
            system_info = self.model.system_info()
            # Parse enabled features
            enabled_features = {
                part.split("=")[0].strip()
                for part in system_info.split("|")
                if part.split("=")[1].strip() == "1"
            }
            if enabled_features:
                system_info_string = ", ".join(enabled_features)

        except Exception as e:
            logger.error(f"Error getting system info: {e}")

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
            with console.status("Watching folder...", spinner="dots") as status:
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

    def _start_transcriber(
        self, 
        model_name, 
        models_dir, 
        n_threads,
        transcribing_queue, 
        result_queue, 
        csv_filepath, 
        silent
    ):
        """Initialize and run the transcriber in a separate process."""
        try:
            # Load model in this process
            model = Model(
                model=model_name,
                models_dir=models_dir,
                n_threads=n_threads,
                print_progress=False,
                print_timestamps=False,
            )
            logger.info(f"Loaded {model_name} model in transcriber process")
            
            # Initialize transcriber
            transcriber = Transcriber(
                transcribing_queue=transcribing_queue,
                result_queue=result_queue,
                model=model,
                csv_filepath=csv_filepath,
                silent=silent,
                print_progress=False,
            )
            
            # This process will stay alive as long as the transcriber thread is running
            # We just need to wait until it's done
            transcriber.transcriber_thread.join()
            
        except Exception as e:
            logger.error(f"Error in transcriber process: {e}")

    def shutdown(self) -> None:
        """Gracefully shutdown all components."""
        logger.info("Shutting down application...")
        
        # Terminate transcriber process
        if hasattr(self, 'transcriber_process') and self.transcriber_process.is_alive():
            self.transcriber_process.terminate()
            logger.info("Terminated transcriber process")
        
        if self.decoder:
            self.decoder.shutdown()
            
        logger.info("Application shutdown complete")

    def _load_model(
        self,
        model_name: str,
        models_dir: str,
        **params,
    ) -> Model:
        status_msg = f"Loading Whisper model {model_name}"
        logger.info(status_msg)

        def init_model() -> Model:
            logger.debug(
                f"Loading model with parameters:\n{nested_dict_to_string(params)}"
            )
            return Model(
                model=model_name,
                models_dir=models_dir,
                **params,
            )

        try:
            if not self.args.silent:
                with console.status(status_msg):
                    model = init_model()
            else:
                model = init_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        else:
            logger.info(f"Loaded {model_name} model successfully")
            logger.debug(
                f"Model loaded with parameters:\n{nested_dict_to_string(model.get_params())}"
            )

            return model

    def _get_default_log_dir(self) -> Path:
        """Get the default log directory."""
        return Path.home() / ".signalscribe" / "logs"

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
