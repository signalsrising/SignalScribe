from pathlib import Path
import os
import logging

CONFIG_DIR_PATH = Path.home() / ".signalscribe"

LOG_DIR_PATH = CONFIG_DIR_PATH / "logs"
LOG_NAME = "signalscribe"

MODEL_DIR_PATH = CONFIG_DIR_PATH / "models"
MODEL_LIST_FILEPATH = CONFIG_DIR_PATH / "models.json"

DEFAULT_MODEL = "large-v3-turbo"

FILETYPES = ["mp3", "m4a", "wav"]

COLORS_FILE_NAME = "colors.yaml"

DEFAULT_NUM_THREADS = int(os.cpu_count() / 2)

CONSOLE_OUTPUT_LOG_LEVEL = logging.DEBUG

NUM_LOG_FILES_TO_KEEP = 10
