"""Audio decoding functionality for SignalScribe."""

import os
import time
import shutil
import tempfile
import subprocess
import numpy as np
from queue import Empty  # Keep this for exception handling
from threading import Thread, Event
from typing import Optional
from datetime import datetime

from .utils import logger, console
from .transcription import Transcription


class Decoder:
    """Decodes audio files into numpy arrays, acting as a middle component between watcher and transcriber."""

    def __init__(
        self, decoding_queue, transcribing_queue, silent: bool = False, 
        file_processed_callback=None
    ):
        """Initialize the decoder with the specified settings."""
        self.stop_event = Event()
        self.silent = silent
        self.file_processed_callback = file_processed_callback

        # Start consumer thread
        self.decoder_thread = Thread(
            target=self._decode_loop,
            daemon=True,
            args=(decoding_queue, transcribing_queue, self.stop_event),
        )
        self.decoder_thread.start()
        logger.info("Decoder thread started")

    def _decode_loop(self, decoding_queue, transcribing_queue, stop_event: Event) -> None:
        """Consume tasks from the queue until stopped."""
        while not stop_event.is_set():
            try:
                logger.debug(f"Decoder thread waiting for task")
                # Use shorter timeout to check stop_event more frequently
                transcription = decoding_queue.get(timeout=0.5)
                
                # Process the file if we're not stopping
                if not stop_event.is_set():
                    logger.debug(f"Decoder thread got task for {transcription.filepath}")
                    start_time = time.monotonic()
                    logger.info(f"Decoding audio for {transcription.filepath}")
                    
                    try:
                        # Decode the audio file
                        audio_data = self._load_audio(transcription.filepath)
                        
                        # Update the transcription object with the audio data
                        transcription.audio = audio_data
                        
                        # Pass to transcriber queue
                        if not stop_event.is_set():
                            transcribing_queue.put(transcription)
                            
                            # Log completion
                            duration = time.monotonic() - start_time
                            logger.info(f"Completed decoding of {transcription.filepath} in {duration:.2f}s")
                            
                            # Call callback if provided
                            if self.file_processed_callback:
                                self.file_processed_callback()
                                
                    except Exception as e:
                        error_msg = f"Failed to decode {transcription.filepath}: {e}"
                        logger.error(error_msg)
                        if not self.silent:
                            console.print(f"[red]{error_msg}")
            
            except Empty:
                continue  # No files available to decode
            except Exception as e:
                logger.warning(f"Error in decoder thread: {e}")

    @staticmethod
    def _load_audio(media_file_path: str) -> np.ndarray:
        """
        Helper method to return a `np.array` object from a media file
        If the media file is not a WAV file, it will try to convert it using ffmpeg

        :param media_file_path: Path of the media file
        :return: Numpy array
        """

        def wav_to_np(file_path):
            with open(file_path, "rb") as f:
                f.read(44)  # Skip WAV header
                raw_data = f.read()
                samples = np.frombuffer(raw_data, dtype=np.int16)
            audio_array = samples.astype(np.float32) / np.iinfo(np.int16).max
            return audio_array

        if media_file_path.endswith(".wav"):
            return wav_to_np(media_file_path)
        else:
            if shutil.which("ffmpeg") is None:
                raise Exception(
                    "FFMPEG is not installed or not in PATH. Please install it, or provide a WAV file instead!"
                )

            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file_path = temp_file.name
            temp_file.close()
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        media_file_path,
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        temp_file_path,
                        "-y",
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return wav_to_np(temp_file_path)
            finally:
                os.remove(temp_file_path)

    def shutdown(self) -> None:
        """Gracefully shutdown the decoder."""
        logger.info("Shutting down decoder...")
        self.stop_event.set()

        # Wait for consumer thread to finish
        self.decoder_thread.join(timeout=5.0)

        logger.info("Decoder shutdown complete")
