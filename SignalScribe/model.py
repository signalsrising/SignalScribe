import hashlib
import platform
import requests
import zipfile
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from rich.prompt import Confirm
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)
from .utils import logger, console, format_size
from .colors import AppColors, ConsoleColors

# Model information
MODEL_INFO = {
    "large-v3-turbo": {
        "base_url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main",
        "files": {
            "ggml": {
                "filename": "ggml-large-v3-turbo.bin",
                "sha256": "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
            },
            "coreml": {
                "filename": "ggml-large-v3-turbo-encoder.mlmodelc.zip",
                "sha256": "84bedfe895bd7b5de6e8e89a0803dfc5addf8c0c5bc4c937451716bf7cf7988a",
            },
        },
    }
}


class ModelManager:
    """Manages Whisper model files."""

    def __init__(self, model_name: str, model_dir: Optional[str] = None):
        """Initialize the model manager."""
        self.model_name = model_name
        self.model_info = MODEL_INFO[model_name]

        # Set up model directory
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = Path.home() / ".signalscribe" / "models"

        self.model_dir = self.model_dir.absolute()
        logger.info(f"Using model directory: {self.model_dir}")

        # Create model directory if it doesn't exist
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create model directory: {e}")
            return False

        # Determine required files based on platform
        self.required_files = ["ggml"]
        if platform.system() == "Darwin":  # macOS
            self.required_files.append("coreml")

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate the SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def _get_file_size(self, url: str) -> Optional[int]:
        """Get file size in bytes using a HEAD request.
        For letting user know how big the download will be."""
        try:
            response = requests.head(url, allow_redirects=True)
            response.raise_for_status()
            return int(response.headers.get("content-length", 0))
        except Exception as e:
            logger.error(f"Failed to get file size for {url}: {e}")
            return None

    def _download_file(self, url: str, target_path: Path) -> bool:
        """Download a file with progress indication."""
        try:
            # Get file size first
            file_size = self._get_file_size(url)
            if file_size is None:
                return False

            # Start download
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:

                # Create the progress bar
                task = progress.add_task(
                    f"Downloading {target_path.name}", total=file_size
                )

                # Download with progress updates
                with open(target_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))

            return True

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False

    def _extract_coreml_model(self, zip_path: Path) -> bool:
        """Extract CoreML model from zip file."""
        try:
            with console.status(f"Extracting {zip_path.name}..."):
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(self.model_dir)
            return True
        except Exception as e:
            logger.error(f"Failed to extract CoreML model: {e}")
            return False

    def get_model_files(self) -> List[Path]:
        """Get paths to all required model files."""
        files = []
        for file_type in self.required_files:
            file_info = self.model_info["files"][file_type]
            filename = file_info["filename"]

            # For CoreML, we need the directory
            if file_type == "coreml":
                filename = filename.replace(".zip", "")

            files.append(self.model_dir / filename)

        return files

    def _get_missing_files(self) -> List[Tuple[str, Dict]]:
        """Get list of missing or invalid files."""
        missing = []

        for file_type in self.required_files:
            file_info = self.model_info["files"][file_type]
            file_path = self.model_dir / file_info["filename"]

            # For CoreML, check the directory exists
            if file_type == "coreml":
                model_dir = file_path.with_suffix("")  # Remove .zip
                if not model_dir.exists():
                    missing.append((file_type, file_info))
                continue

            # For other files, check they exist and have correct hash
            if (
                file_path.exists()
                and self._calculate_hash(file_path) == file_info["sha256"]
            ):
                continue

            # Either file doesn't exist or hash is incorrect
            missing.append((file_type, file_info))

        return missing

    def ensure_models_exist(self) -> bool:
        """Ensure all required model files exist and are valid."""
        missing_files = self._get_missing_files()

        if not missing_files:
            return True

        # Get file sizes and calculate total
        file_sizes = {}
        total_size = 0

        for file_type, file_info in missing_files:
            url = f"{self.model_info['base_url']}/{file_info['filename']}"
            size = self._get_file_size(url)
            if size is None:
                return False
            file_sizes[file_type] = size
            total_size += size

        console.print(
            f"Couldn't find {len(missing_files)} required model file(s) in [green]{self.model_dir}[/green]:"
        )

        for file_type, info in missing_files:
            console.print(
                f"  • {info['filename']}: {format_size(file_sizes[file_type])}"
            )
        console.print(f"Total download size: {format_size(total_size)}")

        if not Confirm.ask(
            f"Would you like to download the missing model files to {self.model_dir}?\n(This can be changed with the --model-dir flag)",
            default=True,
        ):
            return False

        # Download missing files
        for file_type, file_info in missing_files:
            file_path = self.model_dir / file_info["filename"]

            # Download the file
            url = f"{self.model_info['base_url']}/{file_info['filename']}"
            if not self._download_file(url, file_path):
                return False

            # Verify the download
            calculated_hash = self._calculate_hash(file_path)

            if not calculated_hash == file_info["sha256"]:
                logger.error(f"Downloaded file {file_path} has incorrect hash!")
                logger.error(f"Expected hash:   {file_info['sha256']}")
                logger.error(f"Calculated hash: {calculated_hash}")
                return False

            # Extract CoreML model if needed
            if file_type == "coreml":
                if not self._extract_coreml_model(file_path):
                    return False

        return True
