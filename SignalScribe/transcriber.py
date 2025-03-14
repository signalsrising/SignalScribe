from multiprocessing import Process, Queue, Manager, Event
from pywhispercpp.model import Model
from queue import Empty
import logging
import time
from enum import Enum
from pathlib import Path

from .logging import logger, console
from .transcription import Transcription
from .trackedqueue import TrackedQueue
from logging.handlers import QueueHandler, QueueListener

"""
Whisper seems to lock the GIL while transcribing so even if its
not in the main python thread it blocks the main thread and prevents
things like decoding and status updates from happening.
Only way round it for now is to run it in a separate process,
which is implemented here.

Logging is handled by QueueHandler in the child process which serialises
the log messages and sends them to the main process over a multiprocessing
Queue. These are received by QueueListener in the main process and output
to the console.
"""

mp_logger = None


class TranscriberStatus(Enum):
    """Status of the transcriber process."""

    INITIALISED = "Initialised"
    LOADING = "Loading"
    RUNNING = "Running"
    SHUTDOWN = "Shutdown"
    ERROR = "Error"


# Am defining the worker functions outside of any class to avoid a pickling issues
# (serialisation/deserialisation) maybe can be inside the class but idk.
# works for now
def transcriber_entry(
    model_name: str,
    model_dir: str,
    n_threads: int,
    transcribing_queue: Queue,
    output_queue: Queue,
    shared_dict: dict,
    log_queue: Queue,
    show_whispercpp_logs: bool,
) -> None:
    """Entry point to transcriber worker process, wraps all work in try/except so that we can flag to the main process that an error has occurred"""
    try:
        shared_dict["status"] = TranscriberStatus.LOADING

        # Set up logging queue first:
        # Configure root logger in worker process
        mp_logger = logging.getLogger("transcriber_process")
        mp_logger.setLevel(logging.DEBUG)  # Set appropriate level

        for handler in mp_logger.handlers[:]:
            mp_logger.removeHandler(handler)

        # Add queue handler for our code
        mp_logger.addHandler(QueueHandler(log_queue))

        # Redirect pywhispercpp logs to our log queue
        # so that they can printed in main process and
        # written to its log file
        pwc_logger = logging.getLogger("pywhispercpp")
        pwc_logger.setLevel(logging.DEBUG)  # Set appropriate level

        # Remove any existing handlers to avoid duplication
        for handler in pwc_logger.handlers[:]:
            pwc_logger.removeHandler(handler)

        # Add queue handler for pywhispercpp
        pwc_logger.addHandler(QueueHandler(log_queue))

        # Run the transcriber worker loop:
        transcriber_main(
            model_name,
            model_dir,
            n_threads,
            transcribing_queue,
            output_queue,
            shared_dict,
            mp_logger,
            show_whispercpp_logs,   
        )

    except Exception as e:
        mp_logger.fatal(f"Error in transcriber process: {e}")
        shared_dict["status"] = TranscriberStatus.ERROR
        pass

    shared_dict["status"] = TranscriberStatus.SHUTDOWN


def transcriber_main(
    model_name: str,
    model_dir: str,
    n_threads: int,
    transcribing_queue: Queue,
    output_queue: Queue,
    shared_dict: dict,
    mp_logger: logging.Logger,
    show_whispercpp_logs: bool,
) -> None:
    """Worker process function that runs transcription tasks."""
    mp_logger.info(f"Loading {model_name} model in worker process")

    # Last ditch check to make sure the model exists otherwise pywhispercpp will try to download it
    # which we really dont want to happen (we handle this ourselves in ModelManager)
    bin_file_name = f"ggml-{model_name}.bin"
    bin_file_path = Path(model_dir) / bin_file_name
    if not bin_file_path.exists():
        mp_logger.error(f"Model file {bin_file_path} does not exist")
        shared_dict["status"] = TranscriberStatus.ERROR
        return

    # Load the model actually in this process
    model = Model(
        model=model_name,
        models_dir=model_dir,
        n_threads=n_threads,
        print_progress=False,
        print_timestamps=False,
        redirect_whispercpp_logs_to=False if show_whispercpp_logs else None,
    )

    mp_logger.info(f"Loaded {model_name} model in worker process")

    # Get system info and store it in the shared dictionary
    system_info = model.system_info()
    logger.debug("Got system info")

    # Parse enabled features
    enabled_features = []

    # System info is a string of the form:
    # FMA = 0 | NEON = 1 | ARM_FMA = 1 ... etc

    # We want to extract the feature names where the value is 1
    # I've noticed that this format changes subtly between versions of pywhispercpp
    # so need to make sure not to let it bring down the process with uncaught exceptions
    try:
        for feature in system_info.split("|"):
            if feature:
                feature_parts = feature.split("=")
                if len(feature_parts) == 2:
                    feature_name = feature_parts[0].strip()
                    feature_value = feature_parts[1].strip()
                    if feature_value == "1":
                        enabled_features.append(feature_name)
    except Exception as e:
        mp_logger.error(f"Error parsing system info, continuing anyway: {e}")
        shared_dict["system_info"] = "<unknown>"
    else:
        shared_dict["system_info"] = ", ".join(enabled_features)

    # Process tasks until None sentinel is received
    mp_logger.info("Transcriber process started")

    shared_dict["status"] = TranscriberStatus.RUNNING
    while True:
        try:
            # Get the next task with a timeout to keep the process responsive
            transcription = transcribing_queue.get(timeout=0.5)

            # Check for None sentinel (shutdown signal)
            if transcription is None:
                mp_logger.debug("Transcriber process received shutdown sentinel")
                break

            mp_logger.debug(f"Transcribing: {transcription.filepath}")

            # Process the audio
            transcription = transcribe_audio(transcription, model)

            # Add the result to the result queue
            output_queue.put(transcription)

            mp_logger.debug(f"Transcribed:  {transcription.filepath}")

        except Empty:
            continue  # No tasks available, just keep checking
        except Exception as e:
            shared_dict["error_count"] += 1
            mp_logger.error(f"Error transcribing audio data: {e}")


def transcribe_audio(
    transcription: Transcription,
    model: Model,
) -> Transcription:
    """Process a single transcription task."""
    # Clock to time how long each transcription takes
    start_time = time.monotonic()

    # Use the decoded audio data for transcription
    segments = model.transcribe(transcription.audio, print_progress=False)

    transcription.text = " ".join(segment.text for segment in segments).strip()
    transcription.duration = time.monotonic() - start_time

    return transcription


class Transcriber:
    """Handles audio transcription tasks as a consumer."""

    def __init__(
        self,
        transcribing_queue: TrackedQueue,
        output_queue: TrackedQueue,
        model_name: str,
        model_dir: str,
        n_threads: int,
        silent: bool = False,
        show_whispercpp_logs: bool = False,
    ):
        """Initialize the transcriber with the specified model and settings."""
        self.transcribing_queue = transcribing_queue
        self.output_queue = output_queue

        self.stop_event = Event()  # This is only used in the main process
        self.silent = silent

        # Save these for the process
        self.model_name = model_name
        self.model_dir = model_dir
        self.n_threads = n_threads
        self.show_whispercpp_logs = show_whispercpp_logs
        # Create a manager for shared objects
        # n.b. unlike the watcher and output threads, this is a separate process with its own
        #      memory space so we need to use a manager to share objects between the processes.
        #      specifically we're using it to share model/whisper engine info back to main process
        #      so it can be displayed in the UI
        self.manager = Manager()
        self.shared_dict = self.manager.dict()
        self.shared_dict["status"] = TranscriberStatus.INITIALISED
        self.shared_dict["error_count"] = 0
        self.shared_dict["system_info"] = "Unknown"

        # Receive log messages from the worker process and log them in main process
        self.log_queue = Queue()
        self.logging_handler = QueueListener(self.log_queue, *logger.handlers, respect_handler_level=True)
        self.logging_handler.start()

        # Start worker process
        self.worker_process = Process(
            name="transcriber_process",
            target=transcriber_entry,
            args=(
                model_name,
                model_dir,
                n_threads,
                transcribing_queue,
                self.output_queue,
                self.shared_dict,
                self.log_queue,
                self.show_whispercpp_logs,
            ),
        )
        self.start()

    @property
    def is_alive(self) -> bool:
        return self.worker_process.is_alive()

    def start(self) -> None:
        self.worker_process.start()
        logger.info("Transcriber worker process started")

    def stop(self) -> None:
        logger.info("Shutting down transcriber...")
        self.stop_event.set()

        # Send None sentinel to signal the worker process to stop
        self.transcribing_queue.put(None)

        max_wait_time = 10.0  # seconds

        # Terminate the worker process if it's still running
        logger.debug("Waiting for transcriber worker process to terminate")
        while self.worker_process.is_alive():
            # Give it a moment to shut down gracefully
            time.sleep(0.1)
            max_wait_time -= 0.1

            if max_wait_time <= 0:
                logger.warning("Worker process did not terminate in time")
                self.worker_process.terminate()
                logger.info("Terminated transcriber worker process")
                break

        self.logging_handler.stop()

        logger.info("Transcriber shutdown complete")
