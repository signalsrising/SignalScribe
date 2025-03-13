from multiprocessing import Process, Queue, Manager, Event
from pywhispercpp.model import Model
from queue import Empty
import logging
import time

from .transcription import Transcription
from .utils import logger, console, APP_NAME, MODEL_CHOICES, DEFAULT_MODEL
from logging import Logger

"""
Whisper seems to lock the GIL while transcribing so even if its
not in the main python thread it blocks the main thread and prevents
things like decoding and status updates from happening.
Only way round it for now is to run it in a separate process,
which is implemented here.
"""

# Am defining the worker function outside of any class to avoid a pickling issue
# maybe can be inside the class but idk. works for now
def transcriber_worker(
    model_name: str,
    model_dir: str,
    n_threads: int,
    task_queue: Queue,
    output_queue: Queue,
    shared_dict,
    logger: logging.Logger = None,
):
    """Worker process function that runs transcription tasks."""

    # if not logger:
    logger = logging.getLogger("transcriber_process")

    logger.info(f"Loading {model_name} model in worker process")
    # Load the model actually in this process
    model = Model(
        model=model_name,
        models_dir=model_dir,
        n_threads=n_threads,
        print_progress=False,
        print_timestamps=False,
        redirect_whispercpp_logs_to=None,
    )
    logger.info(f"Loaded {model_name} model in worker process")

    # Get system info and store it in the shared dictionary
    system_info = model.system_info()

    # Parse enabled features
    enabled_features = {
        part.split("=")[0].strip()
        for part in system_info.split("|")
        if part.split("=")[1].strip() == "1"
    }
    if enabled_features:
        system_info_string = ", ".join(enabled_features)
        shared_dict["system_info"] = system_info_string

    # Process tasks until None sentinel is received
    logger.info("Transcriber process started")
    while True:
        try:
            # Get the next task with a timeout to keep the process responsive
            transcription = task_queue.get(timeout=0.5)

            # Check for None sentinel (shutdown signal)
            if transcription is None:
                logger.info("Transcriber process received shutdown sentinel")
                break

            logger.debug(
                f"Transcriber process got audio data for {transcription.filepath}"
            )

            # Process the audio
            transcription = transcribe_audio(transcription, model)

            # Add the result to the result queue
            output_queue.put(transcription)

            logger.debug(
                f"Transcriber process completed transcription for {transcription.filepath}"
            )

        except Empty:
            continue  # No tasks available, just keep checking
        except Exception as e:
            logger.error(f"Error transcribing audio data: {e}")


def transcribe_audio(
    transcription: Transcription,
    model: Model,
) -> Transcription:
    """Process a single transcription task."""
    # Clock to time how long each transcription takes
    start_time = time.monotonic()

    # Use the decoded audio data for transcription
    segments = model.transcribe(transcription.audio, print_progress=False)

    transcription.text = "".join(segment.text for segment in segments).strip()
    transcription.duration = time.monotonic() - start_time

    return transcription


class Transcriber:
    """Handles audio transcription tasks as a consumer."""

    def __init__(
        self,
        transcribing_queue,  # multiprocessing.Queue
        output_queue,  # multiprocessing.Queue
        model_name: str,
        model_dir: str,
        n_threads: int,
        silent: bool = False,
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

        # Create a manager for shared objects
        self.manager = Manager()
        self.shared_dict = self.manager.dict()

        # Start worker process
        self.worker_process = Process(
            target=transcriber_worker,
            args=(
                model_name,
                model_dir,
                n_threads,
                transcribing_queue,
                self.output_queue,
                self.shared_dict,  # Pass the shared dictionary
            ),
            daemon=True,
        )
        self.worker_process.start()
        logger.info("Transcriber worker process started")


    def shutdown(self) -> None:
        logger.info("Shutting down transcriber...")
        self.stop_event.set()

        # Send None sentinel to signal the worker process to stop
        self.transcribing_queue.put(None)
        logger.debug("Sent shutdown sentinel to transcriber process")

        max_wait_time = 10.0

        # Terminate the worker process if it's still running
        while self.worker_process.is_alive():
            # Give it a moment to shut down gracefully
            time.sleep(0.1)
            max_wait_time -= 0.1

            if max_wait_time <= 0:
                logger.warning("Worker process did not terminate in time")
                self.worker_process.terminate()
                logger.info("Terminated transcriber worker process")
                break

        logger.info("Transcriber shutdown complete")
