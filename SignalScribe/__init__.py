"""
SignalScribe - A tool for transcribing and processing audio signals
"""

import os
from pathlib import Path
from .version import __version__

# Load environment variables for hardware acceleration if they exist
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

# # Import the SDRTrunkDetector class for external use
# from .sdrtrunk import SDRTrunkDetector
