from datetime import datetime


class Transcription:
    """Represents a single transcription task."""

    def __init__(self, filepath: str):
        self.added = datetime.now()  # When the task was added to the queue
        self.decoded = None  # When the task was decoded (audio file -> numpy array)
        self.transcribed = None  # When the task was transcribed (numpy array -> text via Whisper)
        self.filepath = filepath  # Absolute filepath of the audio file on the system
        self.audio = None  # Audio data as a numpy array
        self.transcription = None  # Transcription text
