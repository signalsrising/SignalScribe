import platform
from pathlib import Path
from typing import List, Optional, Dict
from rich.prompt import Confirm

from .utils import logger, console, format_size
from .modelutils import (
# Static functions to reduce clutter in this file
    get_file_details,
    fetch_available_models,
    get_download_size,
    download_file,
    read_model_info_file,
    write_model_info_file,
)

MODEL_INFO_FILENAME = "models.json"
DEFAULT_MODEL_DIR = Path.home() / ".signalscribe" / "models"

class ModelManager:
    """Manages Whisper model files."""

    def __init__(
        self,
        model_dir: Optional[str] = None,  #  (--model-dir)
        reload_model_list: bool = False,  #  (--list-models)
    ):
        """Initialize the model manager."""

        logger.debug(f"Initializing model manager at {model_dir}, reload_model_list: {reload_model_list}")

        # Steps:
        # 1. Check to see if model directory *and* model info file exist

        # Scenarios:
        # Model file doesn't exist (so we must load it from huggingface)
        # Model file exists, and user doesn't request to reload
        # Model file exists, user requests to reload and we are online
        # Model file exists, user requests to reload and we are offline

        self.model_info = {}
        self.load_model_list = False

        if model_dir is None:
            model_dir = DEFAULT_MODEL_DIR

        model_dir_path = Path(model_dir)

        if not model_dir_path.exists():
            logger.debug(f"Model directory does not exist")

            if not Confirm.ask(f"Model directory [bold]{model_dir_path}[/bold] does not exist. Create it?", default=True):
                raise Exception(f"Model directory does not exist and user chose not to create it.")
            model_dir_path.mkdir()
        else:
            model_info_file = model_dir_path / MODEL_INFO_FILENAME
            if not model_info_file.exists():
                logger.debug(f"Model info file does not exist")
                load_model_list = True
            else:
                logger.debug(f"Model info file exists")
        
        model_info_file_updated = False

        # 2. If user has requested to reload the model list,
        #    fetch the available models from huggingface:
        if reload_model_list:
            try:
                self.model_info = fetch_available_models()
                model_info_file_updated = True
            except Exception as e:
                # If we fail to fetch the available models...
                logger.warning(f"Failed to fetch available models: {e}")
                # ...try to use read local model info file:
                try:
                    self.model_info = read_model_info_file(model_info_file)
                    console.print(f"Failed to fetch available models, using local model info file: {model_info_file}")
                except Exception as e:
                    raise Exception(f"Failed to fetch available model from the internet: {e}")
        
        # 3. If user hasn't requested to reload the model list, but we can't find an existing model info file,
        #    attempt to fetch the available models from huggingface:
        elif load_model_list:
            try:
                self.model_info = fetch_available_models()
                model_info_file_updated = True
            except Exception as e:
                # If we fail to fetch the available models from the internet,
                # and no local model info file exists, quit:
                raise Exception(f"Failed to fetch available model from the internet: {e}")

        # If we don't need to load the model list, try to read the model info file:
        else:
            try:
                self.model_info = read_model_info_file(model_info_file)
            except Exception as e:
                console.print(f"Valid model info file not found")
                console.print(f"Attempting to fetch available models from the internet...")
                try:
                    self.model_info = fetch_available_models()
                    model_info_file_updated = True
                except Exception as e:
                    raise Exception(f"Failed to fetch available model from the internet: {e}")

        # 5. If we got this far it means we have a valid model info file...
        logger.info(f"Detected models: {str(self.model_info.keys())}")

        # If we've fetched a valid model info file, write it to disk for future use:
        if model_info_file_updated:
            write_model_info_file(model_info_file, self.model_info)
            # Raises an exception if it fails, will be caught by app and fail loudly

        downloaded_models = []

        # ... so for each model, set a downloaded key to True if the model files are present
        # (n.b. we don't check the sha256 hashes here as it could take a long time to read all
        # models. Instead we do it when attempting load a specific model)
        for model_name in self.model_info.keys():
            if self._model_is_downloaded(model_name):           
                downloaded_models.append(model_name)

        if len(downloaded_models) > 0:
            logger.info(f"Downloaded models: {', '.join(downloaded_models)}")
        else:
            logger.info(f"No models downloaded")

        # If we get this far then we have a valid model info file and are ready
        # to load a mode

    # Not a static method as we need to access the model_info dict
    def _model_is_downloaded(self, model_name: str) -> bool:
        bin_downloaded = Path(self.model_dir / self.model_info[model_name]["bin"]).exists()
        self.model_info[model_name]["bin_downloaded"] = bin_downloaded

        if platform.system() == "Darwin":
            coreml_downloaded = Path(self.model_dir / self.model_info[model_name]["coreml"]).exists()
            self.model_info[model_name]["coreml_downloaded"] = coreml_downloaded
            return bin_downloaded and coreml_downloaded
        
        return bin_downloaded

    @property
    def selected_model(self) -> str:
        return self._selected_model

    @selected_model.setter
    def set_selected_model(self, model_name: str):
        if model_name not in self.model_info.keys():
            logger.error(f"Model {model_name} not found")
            raise ValueError(f"Model {model_name} not found")

        self._selected_model = model_name

        self._ensure_model_exists(model_name)


    def _ensure_model_exists(self, model_name: str) -> bool:
        """Ensure all required model files exist and are valid."""

        logger.info(f"Ensuring model files exist for {model_name}")

        # 1. Check if necessary file(s) exist
        # 2. If they do, check their sha256 hashes
        # 3. If either they don't exist or the hashes don't match, download the missing files
        # 4. If the files are downloaded and the hashes match, return True



        if not self._model_info[model_name]["bin_downloaded"]:
            pass
        
        # self._model_info[model_name]["coreml_downloaded"]:


        # Get file sizes and calculate total
        file_sizes = {}
        total_size = 0

        for file_type, file_info in missing_files:
            url = f"{self._model_info['base_url']}/{file_info['filename']}"
            size = self.get_download_size(url)
            if size is None:
                return False
            file_sizes[file_type] = size
            total_size += size

        console.print(
            f"Couldn't find {len(missing_files)} required model file(s) in [green]{self._model_dir}[/green]:"
        )

        for file_type, info in missing_files:
            console.print(
                f"  • {info['filename']}: {format_size(file_sizes[file_type])}"
            )
        console.print(f"Total download size: {format_size(total_size)}")

        if not Confirm.ask(
            f"Would you like to download the missing model files to {self._model_dir}?\n(This can be changed with the --model-dir flag)",
            default=True,
        ):
            return False

        # Download missing files
        for file_type, file_info in missing_files:
            file_path = self._model_dir / file_info["filename"]

            # Download the file
            url = f"{self._model_info['base_url']}/{file_info['filename']}"
            if not self.download_file(url, file_path):
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


# # Model information
# MODEL_INFO = {
#     "large-v3-turbo": {
#         "base_url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main",
#         "files": {
#             "ggml": {
#                 "filename": "ggml-large-v3-turbo.bin",
#                 "sha256": "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
#             },
#             "coreml": {
#                 "filename": "ggml-large-v3-turbo-encoder.mlmodelc.zip",
#                 "sha256": "84bedfe895bd7b5de6e8e89a0803dfc5addf8c0c5bc4c937451716bf7cf7988a",
#             },
#         },
#     }
# }


    # def get_model_files(self) -> List[Path]:
    #     """Get paths to all required model files."""

    #     logger.debug("Getting model files")

    #     files = []
    #     for file_type in self.required_files:
    #         file_info = self._model_info["files"][file_type]
    #         filename = file_info["filename"]

    #         # For CoreML, we need the directory
    #         if file_type == "coreml":
    #             filename = filename.replace(".zip", "")

    #         files.append(self._model_dir / filename)

    #     return files


    # def _get_file_hash(self, url: str) -> Optional[str]:
    #     """
    #     Attempts to download and calculate the SHA256 hash for a file.
    #     Uses a streaming approach to avoid loading entire file into memory.

    #     Args:
    #         url: URL of the file to hash

    #     Returns:
    #         SHA256 hash as a string, or None if download failed
    #     """

    #     logger.debug(f"Getting file hash for {url}")

    #     try:
    #         with console.status(f"Calculating hash for {url.split('/')[-1]}..."):
    #             response = requests.get(url, stream=True)
    #             response.raise_for_status()

    #             # Calculate hash
    #             sha256_hash = hashlib.sha256()
    #             for chunk in response.iter_content(chunk_size=8192):
    #                 if chunk:
    #                     sha256_hash.update(chunk)

    #             return sha256_hash.hexdigest()
    #     except Exception as e:
    #         logger.error(f"Failed to calculate hash for {url}: {e}")
    #         return None