"""Transcription handling for SignalScribe."""

import os
from datetime import datetime
import time
from queue import Empty  # Keep using this for exception handling
from threading import Thread, Event
from dataclasses import dataclass
import csv
from typing import Optional
import numpy as np
from pywhispercpp.model import Model, utils
from .utils import logger, console, APP_NAME, MODEL_CHOICES, DEFAULT_MODEL
from .transcription import Transcription


class Transcriber:
    """Handles audio transcription tasks as a consumer."""

    def __init__(
        self,
        transcribing_queue,  # multiprocessing.Queue
        result_queue,  # multiprocessing.Queue
        model: Model,
        csv_filepath: str,
        silent: bool = False,
        print_progress: bool = False,
    ):
        """Initialize the transcriber with the specified model and settings."""
        self.csv_filepath = csv_filepath
        self.transcribing_queue = transcribing_queue
        self.result_queue = result_queue

        self.stop_event = Event()
        self.silent = silent
        self.model = model
        self.print_progress = print_progress

        # Initialize CSV file if it doesn't exist
        if not os.path.exists(self.csv_filepath):
            with open(self.csv_filepath, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Timestamp", "File Path", "Duration", "Transcription"])

        # Start consumer thread
        self.transcriber_thread = Thread(
            target=self._transcription_loop,
            args=(
                self.model,
                self.transcribing_queue,
                self.result_queue,
                self.stop_event,
            ),
            daemon=True,
        )
        self.transcriber_thread.start()
        logger.info("Transcriber thread started")

    # def add_task(self, task: TranscriptionTask) -> None:
    #     """Add a transcription task to the queue."""
    #     logger.debug(
    #         f"Adding transcription task for {task.filepath} to queue (queue size: {self.task_queue.qsize()})"
    #     )
    #     try:
    #         # We use put (blocking) to ensure we don't drop any transcription tasks
    #         # This is important because by this point we've already done the work to decode the audio
    #         self.task_queue.put(task, block=False, timeout=10)  # Wait up to 10 seconds
    #         logger.debug(
    #             f"Added transcription task for {task.filepath} (new queue size: {self.task_queue.qsize()})"
    #         )
    #     except Full:
    #         logger.error(
    #             f"Transcriber queue is full even after waiting, dropping task for {task.filepath}"
    #         )
    #         if not self.silent:
    #             console.print(
    #                 f"[red]Error: Transcriber queue is full, dropping task for {task.filepath}"
    #             )
    #     except Exception as e:
    #         logger.error(f"Error adding task to transcriber queue: {e}")
    #         if not self.silent:
    #             console.print(f"[red]Error adding task to transcriber queue: {e}")

    def _transcription_loop(
        self,
        model: Model,
        transcribing_queue,  # multiprocessing.Queue
        result_queue,  # multiprocessing.Queue
        stop_event: Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                # Use a shorter timeout to check stop_event more frequently
                logger.debug(
                    f"Transcriber thread waiting for audio data")
                transcription = transcribing_queue.get(timeout=0.5)  # 0.5 second timeout
                logger.debug(f"Transcriber thread got audio data for {transcription.filepath}")
                
                # Process the audio in this thread
                result_text = self._process_task(transcription, model)
                
                # Mark the task as done (multiprocessing queues don't have task_done())
                # transcribing_queue.task_done()  # This doesn't exist in multiprocessing.Queue
                
                logger.debug(f"Transcriber thread completed transcription for {transcription.filepath}")
                
                # Add result to result queue if needed
                if result_text:
                    result_queue.put((transcription, result_text))
                
            except Empty:
                continue  # No tasks available, check stop_event again
            except Exception as e:
                logger.error(f"Error transcribing audio data: {e}")

    @staticmethod
    def _process_task(transcription: Transcription, model: Model) -> Optional[str]:
        """Process a single transcription task."""
        start_time = time.monotonic()
        logger.info(f"Processing transcription for {transcription.filepath}")

        try:
            # Use the decoded audio data for transcription
            segments = model.transcribe(transcription.audio, print_progress=False)

            text = "".join(segment.text for segment in segments).strip()
            duration = time.monotonic() - start_time

            # Save to CSV (use the absolute path to avoid issues with different working directories)
            csv_filepath = os.path.abspath(getattr(model, 'csv_filepath', 'transcriptions.csv'))
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

            # Log completion and optionally print to console
            completion_msg = f"Completed transcription of {transcription.filepath} in {duration:.2f}s"
            logger.info(completion_msg)
            console.print(f"[green]{completion_msg}\n[blue]{text}[/blue]")
            return text

        except Exception as e:
            error_msg = f"Failed to transcribe {transcription.filepath}: {e}"
            logger.error(error_msg)
            console.print(f"[red]{error_msg}")
            return None

    def shutdown(self) -> None:
        """Gracefully shutdown the transcriber."""
        logger.info("Shutting down transcriber...")
        self.stop_event.set()

        # Wait for consumer thread to finish
        self.transcriber_thread.join(timeout=5.0)

        logger.info("Transcriber shutdown complete")
