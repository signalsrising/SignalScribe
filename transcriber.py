import logging
from multiprocessing import Queue, Empty
from pathlib import Path
from whispercpp import Model
from transcriber_utils import transcribe_audio

def transcriber_worker(
    model_name: str,
    bin_file_path: Path,
    n_threads: int,
    task_queue: Queue,
    output_queue: Queue,
    shared_dict,
    log_queue,
):
    """Worker process function that runs transcription tasks."""

    # Set up logging queue first:
    # Configure root logger in worker process
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # Set appropriate level
    
    # Remove any existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Add queue handler
    queue_handler = QueueHandler(log_queue)
    root_logger.addHandler(queue_handler)

    # Now we can log
    logging.info(f"Loading {model_name} model in worker process")

    # Last ditch check to make sure the model exists otherwise pywhispercpp will try to download it
    # which we really dont want to happen (we should handle this ourselves)
    if not bin_file_path.exists():
        logging.error(f"Model file not found at {bin_file_path}")
        return

    # Load the model actually in this process
    model = Model(
        model=model_name,
        models_dir=None,
        n_threads=n_threads,
        print_progress=False,
        print_timestamps=False,
        redirect_whispercpp_logs_to=None,
        suppress_non_speech_tokens=True,
    )
    logging.info(f"Loaded {model_name} model in worker process")

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
    logging.info("Transcriber process started")
    while True:
        try:
            # Get the next task with a timeout to keep the process responsive
            transcription = task_queue.get(timeout=0.5)

            # Check for None sentinel (shutdown signal)
            if transcription is None:
                logging.info("Transcriber process received shutdown sentinel")
                break

            logging.debug(
                f"Transcriber process got audio data for {transcription.filepath}"
            )

            # Process the audio
            transcription = transcribe_audio(transcription, model)

            # Add the result to the result queue
            output_queue.put(transcription)

            logging.debug(
                f"Transcriber process completed transcription for {transcription.filepath}"
            )

        except Empty:
            continue  # No tasks available, just keep checking
        except Exception as e:
            logging.error(f"Error transcribing audio data: {e}")