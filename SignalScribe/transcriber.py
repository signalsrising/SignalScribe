"""Transcription handling for SignalScribe."""

import os
import sys
from datetime import datetime
import time
from queue import Empty  # Keep using this for exception handling
from threading import Thread, Event
from multiprocessing import Process, Queue
import csv
import numpy as np
from pywhispercpp.model import Model, utils
from .utils import logger, console, APP_NAME, MODEL_CHOICES, DEFAULT_MODEL
from .transcription import Transcription


# Define the worker function outside of any class to avoid pickling issues
def transcriber_worker(model_name, model_dir, n_threads, task_queue, result_queue, 
                      control_queue, csv_filepath, silent, print_progress):
    """Worker process function that runs transcription tasks."""
    try:
        # Load the model in this process
        model = Model(
            model=model_name,
            models_dir=model_dir,
            n_threads=n_threads,
            print_progress=print_progress,
            print_timestamps=False,
        )
        logger.info(f"Loaded {model_name} model in worker process")
        
        # Ensure CSV file exists
        csv_filepath = os.path.abspath(csv_filepath)
        if not os.path.exists(csv_filepath):
            with open(csv_filepath, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Timestamp", "File Path", "Duration", "Transcription"])
        
        # Process tasks until told to stop
        running = True
        while running:
            # First check if we should stop
            try:
                if not control_queue.empty():
                    signal = control_queue.get_nowait()
                    if signal == "STOP":
                        logger.info("Transcriber process received stop signal")
                        running = False
                        break
            except:
                pass  # Continue if there's an error checking the control queue
                
            # Process tasks if we're still running
            if running:
                try:
                    # Short timeout to check control queue frequently
                    # print("Transcriber process waiting for audio data")
                    transcription = task_queue.get(timeout=0.5)  # 0.5 second timeout
                    print(f"\nGot {len(transcription.audio)} Bytes")
                    # print(f"Transcriber process got audio data for {transcription.filepath}")
                    
                    # Process the audio
                    process_task(transcription, model, result_queue, csv_filepath, silent)
                    
                    logger.debug(f"Transcriber process completed transcription for {transcription.filepath}")
                    
                except Empty:
                    continue  # No tasks available, check control queue again
                except Exception as e:
                    logger.error(f"Error transcribing audio data: {e}")
    
    except Exception as e:
        logger.error(f"Error in transcriber worker process: {e}")
        sys.exit(1)


def process_task(transcription, model, result_queue, csv_filepath, silent):
    """Process a single transcription task."""
    start_time = time.monotonic()
    logger.info(f"Processing transcription for {transcription.filepath}")

    try:
        # Use the decoded audio data for transcription
        segments = model.transcribe(transcription.audio, print_progress=False)

        text = "".join(segment.text for segment in segments).strip()
        duration = time.monotonic() - start_time

        # Save to CSV
        with open(csv_filepath, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    transcription.filepath,
                    f"{duration:.2f}",
                    text,
                ]
            )

        # Add the result to the result queue
        result_queue.put((transcription.filepath, text))
        
        logger.info(f"Completed transcription of {transcription.filepath} in {duration:.2f}s")
        return text

    except Exception as e:
        error_msg = f"Failed to transcribe {transcription.filepath}: {e}"
        logger.error(error_msg)
        return None


class Transcriber:
    """Handles audio transcription tasks as a consumer."""

    def __init__(
        self,
        task_queue,  # multiprocessing.Queue
        model_name: str,
        model_dir: str,
        n_threads: int,
        csv_filepath: str,
        silent: bool = False,
        print_progress: bool = False,
    ):
        """Initialize the transcriber with the specified model and settings."""
        self.task_queue = task_queue
        self.result_queue = Queue()  # multiprocessing queue for results
        self.control_queue = Queue()  # Control queue for signaling process shutdown
        
        self.stop_event = Event()  # This is only used in the main process
        self.silent = silent
        
        # Save these for the process
        self.model_name = model_name
        self.model_dir = model_dir
        self.n_threads = n_threads
        self.csv_filepath = csv_filepath
        self.print_progress = print_progress
        
        # Start worker process
        self.worker_process = Process(
            target=transcriber_worker,
            args=(
                model_name,
                model_dir,
                n_threads,
                task_queue,
                self.result_queue,
                self.control_queue,
                csv_filepath,
                silent,
                print_progress,
            ),
            daemon=True,
        )
        self.worker_process.start()
        logger.info("Transcriber worker process started")
        
        # Start a thread in the main process to handle results from the worker
        self.result_thread = Thread(
            target=self._handle_results,
            daemon=True,
        )
        self.result_thread.start()
        logger.info("Result handling thread started")

    def add_task(self, task: Transcription) -> None:
        """Add a transcription task to the queue."""
        if self.worker_process.is_alive():
            logger.debug(f"Adding transcription task for {task.filepath} to queue")
            try:
                # Add the task to the multiprocessing queue
                self.task_queue.put(task)
                logger.debug(f"Added transcription task for {task.filepath}")
            except Exception as e:
                logger.error(f"Error adding task to transcriber queue: {e}")
                if not self.silent:
                    console.print(f"[red]Error adding task to transcriber queue: {e}")
        else:
            logger.warning(f"Cannot add task: worker process is not running")

    def _handle_results(self) -> None:
        """Handle results coming back from the worker process."""
        while not self.stop_event.is_set():
            try:
                # Get result from queue with a short timeout
                result = self.result_queue.get(timeout=0.5)
                if result:
                    filepath, text = result
                    completion_msg = f"Completed transcription of {filepath}"
                    logger.info(completion_msg)
                    if not self.silent:
                        console.print(f"[green]{completion_msg}\n[blue]{text}[/blue]")
                
            except Empty:
                continue  # No results available, check stop_event again
            except Exception as e:
                logger.error(f"Error handling transcription result: {e}")

    def shutdown(self) -> None:
        """Gracefully shutdown the transcriber."""
        logger.info("Shutting down transcriber...")
        self.stop_event.set()
        
        # Send stop signal to worker process
        try:
            self.control_queue.put("STOP")
            logger.info("Sent stop signal to transcriber process")
        except:
            logger.warning("Failed to send stop signal to transcriber process")

        # Wait for result thread to finish
        self.result_thread.join(timeout=5.0)

        # Terminate the worker process if it's still running
        if self.worker_process.is_alive():
            # Give it a moment to shut down gracefully
            import time
            time.sleep(1.0)
            
            if self.worker_process.is_alive():
                self.worker_process.terminate()
                logger.info("Terminated transcriber worker process")

        logger.info("Transcriber shutdown complete")
