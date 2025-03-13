import os
import time
import shutil
import tempfile
import subprocess
import numpy as np
from queue import Empty  # Keep this for exception handling
from threading import Thread, Event

from .logging import logger, console


class Decoder:
    """Decodes audio files into numpy arrays, acting as a middle component between watcher and transcriber."""

    def __init__(
        self,
        decoding_queue,
        transcribing_queue,
    ):
        """Initialize the decoder with the specified settings."""
        self.stop_event = Event()

        # Start consumer thread
        self.decoder_thread = Thread(
            target=self._decode_loop,
            daemon=True,
            args=(decoding_queue, transcribing_queue, self.stop_event),
        )
        self.decoder_thread.start()
        logger.info("Decoder thread started")

    def _decode_loop(
        self, decoding_queue, transcribing_queue, stop_event: Event
    ) -> None:
        """Decode audio files from the queue until stopped."""
        while not stop_event.is_set():
            try:
                # For debug only (should be removed really):
                # logger.debug(f"Decoder thread waiting for task")

                # Use shorter timeout to check stop_event more frequently
                transcription = decoding_queue.get(timeout=0.5)

                # Process the file if we're not stopping
                if not stop_event.is_set():
                    logger.debug(
                        f"Decoder thread got task for {transcription.filepath}"
                    )
                    start_time = time.monotonic()

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
                            logger.info(
                                f"Decoded {transcription.filepath} in {duration:.2f}s"
                            )

                    except Exception as e:
                        logger.error(f"Failed to decode {transcription.filepath}: {e}")

            except Empty:
                continue  # No files available to decode
            except Exception as e:
                logger.warning(f"Error in decoder thread: {e}")

    @staticmethod
    def _load_audio(media_file_path: str) -> np.ndarray:
        """
        Helper method to return a `np.array` object from a media file
        If the media file is not a WAV file, it will try to convert it using ffmpeg
        via a temporary file.

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

    def stop(self) -> None:
        logger.info("Shutting down decoder...")
        self.stop_event.set()

        # Wait for consumer thread to finish
        self.decoder_thread.join(timeout=5.0)

        logger.info("Decoder shutdown complete")
