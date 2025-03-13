from datetime import datetime
import numpy as np
import os

class Transcription:
    """Represents a single transcription task that can be passed between processes."""

    def __init__(self, filepath: str):
        self.added_timestamp = datetime.now()  # When the task was added to the queue
        self.audio = None  # Audio data as a numpy array
        self.duration = None  # Duration of the actual audio
        self.filepath = filepath  # Absolute filepath of the audio file on the system
        self.filename = os.path.basename(filepath)  # Name of the audio file
        self.realtime = None  # Transcription speed as multiple of realtime
        self.text = None  # Transcription text
        self.transcription_time = None  # How long it took to transcribe

    def __getstate__(self):
        """Define what gets pickled to ensure compatibility with multiprocessing."""
        # Create a copy of the object's state
        state = self.__dict__.copy()
        # Make sure the audio data is a numpy array which is picklable
        if state["audio"] is not None and not isinstance(state["audio"], np.ndarray):
            state["audio"] = np.array(state["audio"])
        return state

    def __setstate__(self, state):
        """Define how to unpickle the object."""
        self.__dict__.update(state)
