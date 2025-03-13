"""Transcription handling for SignalScribe."""

import os
from datetime import datetime
import time
from queue import Queue
from threading import Thread, Event
from dataclasses import dataclass
import csv
from typing import Optional
from pywhispercpp.model import Model
from .utils import logger, console, APP_NAME, VERSION, MODEL_CHOICES, DEFAULT_MODEL


@dataclass
class TranscriptionTask:
    """Represents a single transcription task."""
    filepath: str
    timestamp: datetime = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


class Transcriber:
    """Handles audio transcription tasks as a consumer."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL, 
                 models_dir: str = "./models/",
                 csv_filepath: str = "transcriptions.csv", 
                 threads: int = 4,
                 cpu_only: bool = False):
        """Initialize the transcriber with the specified model and settings."""
        self.csv_filepath = csv_filepath
        self.task_queue = Queue()
        self.stop_event = Event()
        
        # Load model
        try:
            with console.status(f"Loading Whisper model {model_name}...") as status:
                self.model = Model(
                    model=model_name,
                    models_dir=models_dir,
                    print_progress=False,
                    redirect_whispercpp_logs_to="whisper.log"
                )
                logger.info(f"Loaded {model_name} model successfully")
        except Exception as e:
            logger.error(f"Fatal error loading Whisper: {e}")    
            raise
            
        # Initialize CSV file if it doesn't exist
        if not os.path.exists(self.csv_filepath):
            with open(self.csv_filepath, 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Timestamp", "File Path", "Duration", "Transcription"])
                
        # Start consumer thread
        self.consumer_thread = Thread(target=self._consume_tasks, daemon=True)
        self.consumer_thread.start()
        logger.info("Transcriber consumer thread started")
    
    def add_task(self, filepath: str) -> None:
        """Add a new transcription task to the queue."""
        task = TranscriptionTask(filepath)
        self.task_queue.put(task)
        logger.debug(f"Added transcription task for {filepath}")
    
    def _consume_tasks(self) -> None:
        """Consume tasks from the queue until stopped."""
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=1.0)  # 1 second timeout
                self._process_task(task)
                self.task_queue.task_done()
            except Queue.Empty:
                continue  # No tasks available
            except Exception as e:
                logger.error(f"Error consuming task: {e}")
                
    def _process_task(self, task: TranscriptionTask) -> Optional[str]:
        """Process a single transcription task."""
        start_time = time.monotonic()
        logger.info(f"Processing transcription for {task.filepath}")
        
        try:
            segments = self.model.transcribe(task.filepath, print_progress=False)
            text = "".join(segment.text for segment in segments).strip()
            duration = time.monotonic() - start_time
            
            # Save to CSV
            with open(self.csv_filepath, 'a', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    task.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    task.filepath,
                    f"{duration:.2f}",
                    text
                ])
                
            logger.info(f"Completed transcription of {task.filepath} in {duration:.2f}s")
            return text
            
        except Exception as e:
            logger.error(f"Failed to transcribe {task.filepath}: {e}")
            return None
            
    def shutdown(self) -> None:
        """Gracefully shutdown the transcriber."""
        logger.info("Shutting down transcriber...")
        self.stop_event.set()
        
        # Wait for consumer thread to finish
        self.consumer_thread.join(timeout=5.0)
        
        # Process any remaining items in the queue
        remaining_tasks = []
        while not self.task_queue.empty():
            remaining_tasks.append(self.task_queue.get())
            
        if remaining_tasks:
            logger.info(f"Processing {len(remaining_tasks)} remaining tasks...")
            for task in remaining_tasks:
                self._process_task(task)
                
        logger.info("Transcriber shutdown complete") 