from queue import Empty 
from rich.padding import Padding
from threading import Thread, Event
import csv
import os
import re

from .trackedqueue import TrackedQueue
from .utils import logger, console, insert_string


class Output:
    """
    Reads completed transcriptions from the queue, saves to CSV and outputs to console until stopped.
    Any additional outputs (JSON, websocket, whatever) should go here.
    """

    def __init__(
        self,
        output_queue: TrackedQueue,
        csv_filepath: str,
        shared_colors: dict = None,
        shared_colors_lock = None,
    ):
        self.stop_event = Event()
        self.shared_colors = shared_colors if shared_colors is not None else {}
        self.shared_colors_lock = shared_colors_lock

        # Start consumer thread
        self.output_thread = Thread(
            target=self._output_loop,
            daemon=True,
            args=(output_queue, csv_filepath, self.stop_event, self.shared_colors, self.shared_colors_lock),
        )
        self.output_thread.start()
        logger.info("Output thread started")

    def _output_loop(
        self,
        output_queue: TrackedQueue,
        csv_filepath: str,
        stop_event: Event,
        shared_colors: dict,
        shared_colors_lock,
    ) -> None:
        """Read transcriptions from the queue, save to CSV and output to console until stopped."""

        while not stop_event.is_set():
            try:
                logger.debug(f"Output thread waiting for transcription")
                transcription = output_queue.get(timeout=1)

                text = transcription.text
                filename = transcription.filename
                filepath = transcription.filepath
                duration = transcription.duration
                added_timestamp = transcription.added_timestamp.strftime("%Y-%m-%d %H:%M:%S")

                # Ensure CSV file exists
                csv_filepath = os.path.abspath(csv_filepath)
                if not os.path.exists(csv_filepath):
                    with open(
                        csv_filepath, "w", newline="", encoding="utf-8"
                    ) as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(
                            ["Timestamp", "File Path", "Duration", "Transcription"]
                        )

                # Save to CSV
                with open(csv_filepath, "a", newline="", encoding="utf-8") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(
                        [
                            added_timestamp,
                            filepath,
                            f"{duration:.2f}",
                            text,
                        ]
                    )

                # Print the results:
                console.print(f"{added_timestamp}", end=" | ")
                console.print(
                    f"[blue]{filename}", style=f"link {filepath}"
                )

                if not text:
                    console.print(
                        Padding(
                            "<no transcription>",
                            (0, len(added_timestamp) + 3),
                        )
                    )
                    continue
                
                # Apply color highlighting to the text
                highlighted_text = self._highlight_text(text, shared_colors, shared_colors_lock)
                
                # Print the highlighted text with padding
                console.print(
                    Padding(
                        highlighted_text,
                        (0, len(added_timestamp) + 3),
                    )
                )

            except Empty:
                continue  # No transcriptions available
            except Exception as e:
                logger.error(f"Error in output thread: {e}")

    def shutdown(self) -> None:
        logger.info("Shutting down output thread...")
        self.stop_event.set()

        # Wait for consumer thread to finish
        self.output_thread.join(timeout=5.0)

        logger.info("Output thread shutdown complete")

    def _highlight_text(self, text: str, shared_colors: dict, shared_colors_lock) -> str:
        """Apply color highlighting to text based on the shared colors dictionary."""
        if not text:
            return text
            
        # Make a copy of the colors dictionary with lock protection
        colors_copy = {}
        if shared_colors_lock:
            with shared_colors_lock:
                if shared_colors:
                    colors_copy = shared_colors.copy()
        else:
            if shared_colors:
                colors_copy = shared_colors.copy()
                
        if not colors_copy:
            return text
            
        # Make a copy of the text for highlighting
        highlighted_text = text
        
        # Lower-case copy for case-insensitive searching
        search_text = text.lower()
        
        for color, phrases in colors_copy.items():
            opening_tag = f"[{color}]"
            closing_tag = f"[/{color}]"
            
            for phrase in phrases:
                # Find all positions of the phrase
                positions = [
                    (pos.start(), pos.end())
                    for pos in re.finditer(re.escape(phrase.lower()), search_text, re.IGNORECASE)
                ]
                
                # Highlight from end to start to preserve positions
                for start, end in reversed(positions):
                    highlighted_text = insert_string(highlighted_text, closing_tag, end)
                    highlighted_text = insert_string(highlighted_text, opening_tag, start)
                    
                    # Replace the matched phrase in search_text with spaces to avoid double-highlighting
                    search_text = search_text[:start] + ' ' * (end - start) + search_text[end:]
                    
        return highlighted_text
