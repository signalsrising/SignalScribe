from typing import Optional, Dict, Tuple
from pathlib import Path
import requests
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
import json
from bs4 import BeautifulSoup
import zipfile
import hashlib
from .utils import logger, console
import platform


"""
Contains all static functions for ModelManager to download and read model info files.
"""

def get_file_details(filename: str) -> Optional[Tuple[str, str]]:
    """
    Fetches the SHA256 hash for a file directly from Hugging Face,
    without downloading the file itself.

    Args:
        filename: Name of the file to get hash for

    Returns:
        Tuple of (size, hash), or None if not found
    """
    logger.debug(f"Getting file details for {filename}")

    try:
        # Construct the blob URL for the file
        blob_url = (
            f"https://huggingface.co/ggerganov/whisper.cpp/blob/main/{filename}"
        )

        download_url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{filename}"

        size = None
        hash = None

        response = requests.get(blob_url)
        response.raise_for_status()

        # Parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Look for the Git LFS details section
        sha_text = None
        for strong_tag in soup.find_all("strong"):
            if "SHA256:" in strong_tag.text:
                # The hash is in the next sibling text
                sha_text = strong_tag.parent.text.strip()
                hash = sha_text.split(":")[1].strip()
                continue
            # if "Size" in strong_tag.text:
            #     size = strong_tag.parent.text.split(":")[1].strip()
            #     continue

        size = get_download_size(download_url)

        if size and hash:
            return size, hash

        logger.warning(f"Could not find SHA256 hash for {filename} on Hugging Face")
        return None

    except Exception as e:
        logger.error(f"Failed to fetch hash for {filename}: {e}")
        return None


def fetch_available_models(model_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Fetches specifically the models that have CoreML versions available.
    Only includes models that have both:
    - A standard .bin file starting with 'ggml-'
    - A corresponding CoreML file ending with '-encoder.mlmodelc.zip'

    Returns:
        A dictionary of model information with both standard and CoreML variants:
        {
            "model_name": {
                "bin": "URL to the standard .bin model file",
                "bin_size": "Size of the bin file",
                "bin_sha256": "SHA256 hash of the bin file",
                "coreml": "URL to the CoreML encoder.mlmodelc.zip file",
                "coreml_size": "Size of the coreml file",
                "coreml_sha256": "SHA256 hash of the coreml file"
            },
            ...
        }
    """

    logger.debug("Fetching available models")

    # Base URL for the Hugging Face repository
    repo_url = "https://huggingface.co/ggerganov/whisper.cpp/tree/main"
    base_download_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

    try:
        # Fetch the repository page
        response = requests.get(repo_url)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # Initialize results dictionary
        coreml_models = {}

        # Extract all file links
        file_links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag.get("href", "")
            if "/blob/main/" in href and not href.endswith("/"):
                # Extract filename from URL
                filename = href.split("/")[-1]
                file_links.append(filename)

        # Filter for only CoreML (compatible) models
        # N.b.: Users could technically user any model they want, but
        #       we're artifically limiting all users to only CoreML compatible
        #       models for now to make cross platform development easier.
        coreml_compatible_models = [
            f
            for f in file_links
            if f.startswith("ggml-") and f.endswith("-encoder.mlmodelc.zip")
        ]

        if not coreml_compatible_models or len(coreml_compatible_models) == 0:
            logger.warning("No CoreML compatible models found")
            return {}

        with console.status(
            f"Found {len(coreml_compatible_models)} models, fetching details (1/{len(coreml_compatible_models)})"
        ) as status:

            # For each CoreML file, find the corresponding .bin file
            for i, coreml_model_file in enumerate(coreml_compatible_models):
                # TODO: REMOVE:
                if i > 2:
                    break
                # Extract base model name by removing the "-encoder.mlmodelc.zip" suffix
                model_name = coreml_model_file.replace("-encoder.mlmodelc.zip", "")
                display_name = model_name.replace("ggml-", "")
                bin_model_file = f"{model_name}.bin"

                status.update(
                    f"Found {len(coreml_compatible_models)} models, "
                    f"fetching details of {display_name} "
                    f"({i+1}/{len(coreml_compatible_models)})"
                )

                if bin_model_file in file_links:
                    bin_url = f"{base_download_url}/{bin_model_file}"
                    coreml_url = f"{base_download_url}/{coreml_model_file}"

                    # Check our models dir to see if the files are present
                    bin_filepath = model_dir / bin_model_file

                    bin_downloaded = bin_filepath.exists()
                    

                    coreml_models[display_name]["bin"] = {
                        "url": bin_url,
                        "size": None,
                        "sha256": None,
                        "downloaded": bin_downloaded,
                    }

                    # Get bin file info
                    bin_size, bin_hash = get_file_details(bin_model_file)
                    if bin_size and bin_hash:
                        coreml_models[display_name]["bin_size"] = bin_size
                        coreml_models[display_name]["bin_sha256"] = bin_hash

                    if platform.system() == "Darwin":
                        # Get CoreML file info
                        coreml_size, coreml_hash = get_file_details(
                            coreml_model_file
                        )
                        if coreml_size and coreml_hash:
                            coreml_models[display_name]["coreml_size"] = coreml_size
                            coreml_models[display_name][
                                "coreml_sha256"
                            ] = coreml_hash
                            
                        coreml_filepath = model_dir / coreml_model_file
                        coreml_downloaded = coreml_filepath.exists()

                        coreml_models[display_name]["coreml"] = {
                            "url": coreml_url,
                            "filepath": coreml_filepath,
                            "size": None,
                            "sha256": None,
                            "downloaded": coreml_downloaded,
                        }

        return coreml_models

    except Exception as e:
        logger.error(f"Failed to fetch CoreML models: {e}")
        return {}
    
def get_download_size(url: str) -> Optional[int]:
    """Get file size in bytes using a HEAD request.
    For letting user know how big the download will be."""

    logger.debug(f"Getting file size for {url} using HEAD HTTP request")

    try:
        response = requests.head(url, allow_redirects=True)
        response.raise_for_status()
        return int(response.headers.get("content-length", 0))
    except Exception as e:
        logger.error(f"Failed to get file size for {url}: {e}")
        return None

def download_file(url: str, target_path: Path) -> bool:
    """Download a file with progress indication."""

    logger.debug(f"Downloading file from {url} to {target_path}")

    try:
        # Get file size first
        file_size = get_download_size(url)
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
    
def read_model_info_file(file_path: Path) -> Dict:
    """
    Read a JSON file into a Python dictionary.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary containing the JSON data, or empty dict if file doesn't exist or is invalid
    """
    logger.debug(f"Reading model info file from {file_path}")

    try:
        if not file_path.exists():
            logger.debug(f"Model info file does not exist: {file_path}")
            return {}

        with open(file_path, "r") as f:
            model_info = json.loads(f.read())
            return model_info
    except Exception as e:
        logger.error(f"Failed to read model info file {file_path}: {e}")
        return {}

def write_model_info_file(file_path: Path, model_info: Dict) -> bool:
    """
    Write a downloaded model info out to a JSON file.

    Args:
        file_path: Path where the JSON file should be written
        model_info: Model info dictionary to write to the file

    Returns:
        True if successful, False otherwise
    """
    logger.debug(f"Writing model info JSON file to {file_path}")

    try:
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(model_info, f, indent=4)
        return True
    except Exception as e:
        raise Exception(f"Failed to write model info file {file_path}: {e}")


def calculate_hash(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    logger.debug(f"Calculating hash for {file_path}")

    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def extract_coreml_model(zip_path: Path) -> bool:
    """Extract CoreML model from zip file."""

    folder = zip_path.parent
    logger.debug(f"Extracting CoreML model: {zip_path} to {folder}")

    try:
        with console.status(f"Extracting {zip_path.name}..."):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(folder)
        return True
    except Exception as e:
        logger.error(f"Failed to extract CoreML model: {e}")
        return False
