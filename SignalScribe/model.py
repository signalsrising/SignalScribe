import platform
import json
from pathlib import Path
from typing import List, Optional, Dict
from rich.prompt import Confirm
import os
from .logging import logger, console
from .utils import format_size, nested_dict_to_string
from .modelutils import (
    # Static functions to reduce clutter in this file
    fetch_available_models,
    read_model_info_file,
    write_model_info_file, 
    validate_model_info,
    validate_file_hash,
    download_file,
    extract_coreml_model,
)

MODEL_INFO_FILENAME = "models.json"
DEFAULT_MODEL_DIR = Path.home() / ".signalscribe" / "models"


class ModelManager:
    """Manages Whisper model files."""

    def __init__(
        self,
        model_dir: Optional[str] = None,  #  (--model-dir)
        user_requested_model_list: bool = False,  #  (--list-models)
    ):
        """Initialize the model manager."""

        logger.debug(
            f"Initializing model manager at {model_dir}, reload_model_list: {user_requested_model_list}"
        )

        # Steps:
        # 1. Check to see if model directory *and* model info file exist
        self._model_info = {}
        self._model_dir = model_dir

        if self._model_dir is None:
            self._model_dir = DEFAULT_MODEL_DIR

        self._model_dir = Path(self._model_dir)

        # 1. Check to see if model directory *and* model info file exist
        if not self._model_dir.exists():
            logger.info(f"Model directory does not exist, creating it")

            # if not Confirm.ask(f"Model directory [bold]{self._model_dir}[/bold] does not exist. Create it?", default=True):
            # raise Exception(f"Model directory does not exist and user chose not to create it.")
            self._model_dir.mkdir(exist_ok=True)
            logger.debug(f"Model directory created at {self._model_dir}")

        model_info_file = self._model_dir / MODEL_INFO_FILENAME

        # 2. Check to see if model info file exists
        try:
            self._model_info = read_model_info_file(model_info_file)
        except FileNotFoundError:
            logger.info(f"Model info file does not exist")
        except json.JSONDecodeError:
            logger.info(f"Model info file is not valid JSON")
            model_info_file.unlink() # delete the file as it's junk
        except KeyboardInterrupt:
            logger.info("User interrupted model info file read")
            raise KeyboardInterrupt
        except Exception as e:
            raise Exception(f"Failed to read model info file, "
                            f"this is indicative of the file being in use by another process. "
                            f"Please close any other instances of SignalScribe and try again.")

        model_info_file_updated = False
        
        # 3. Validate the model info file
        model_info_file_is_valid = validate_model_info(self._model_info)
        if not model_info_file_is_valid:
            logger.info(f"Model info file is invalid, will attempt to reload it")
            model_info_file.unlink(missing_ok=True) # delete the file as it's junk


        # 4. If user has requested to reload the model list from internet,
        #    fetch the available models from huggingface:
        if user_requested_model_list:
            try:
                self._model_info = fetch_available_models(self._model_dir)
                self._model_info_file_updated = True
            except KeyboardInterrupt:
                logger.info("User interrupted model list re-download")
                raise KeyboardInterrupt
            except Exception as e:
                import traceback
                traceback.print_exc()
                # If we fail to fetch the available models try to use read local model info file:
                
                if model_info_file_is_valid:
                    console.print(
                        f"Failed to fetch available models from the internet, using local model info file: {model_info_file}"
                    )
                else:
                    raise Exception(
                        f"Failed to fetch available model from the internet and no valid local model info file exists. Quitting: {e}"
                    )

        # 5. If user hasn't requested to reload the model list, but we can't find an existing model info file,
        #    attempt to fetch the available models from huggingface:

        elif not model_info_file_is_valid:
            try:
                self._model_info = fetch_available_models(self._model_dir)
                model_info_file_updated = True
            except KeyboardInterrupt:
                logger.info("User interrupted model list download")
                raise KeyboardInterrupt
            except Exception as e:
                # If we fail to fetch the available models from the internet,
                # and no local model info file exists, quit:
                raise Exception(
                    f"Failed to fetch available model from the internet: {e}"
                )

        # 5. If we got this far it means we have a valid model info file...
        logger.info(f"Detected models: {', '.join(list(self._model_info.keys()))}")

        # If we've fetched a valid model info file, write it to disk for future use:
        if model_info_file_updated:
            write_model_info_file(model_info_file, self._model_info)
            # Raises an exception if it fails, will be caught by app and fail loudly

        # downloaded_models = []

        # # ... so for each model, set a downloaded key to True if the model files are present
        # # (n.b. we don't check the sha256 hashes here as it could take a long time to read all
        # # models. Instead we do it when attempting load a specific model)
        # for model_name in self._model_info.keys():
        #     if self._model_is_downloaded(model_name):
        #         downloaded_models.append(model_name)

        # if len(downloaded_models) > 0:
        #     logger.info(f"Downloaded models: {', '.join(downloaded_models)}")
        # else:
        #     logger.info(f"No models downloaded")

        # logger.debug(f"Models: {nested_dict_to_string(self._model_info)}")

        # If we get this far then we have a valid model info file and are ready
        # to load a model

    # Not a static method as we need to access the model_info dict
    def _model_is_downloaded(self, model_name: str) -> bool:
        bin_file_path = self._model_dir / self._model_info[model_name]["bin"]["filename"]
        bin_downloaded = bin_file_path.exists()
        self._model_info[model_name]["bin"]["downloaded"] = bin_downloaded
        
        if bin_downloaded:
            size_on_disk = os.path.getsize(bin_file_path)
            console.print(f"Bin expected size: {self._model_info[model_name]['bin']['size']}")
            console.print(f"Bin size on disk: {size_on_disk}")

        if platform.system() == "Darwin":
            coreml_file_path = Path(
                self._model_dir / self._model_info[model_name]["coreml"]["filename"]
            )
            coreml_downloaded = coreml_file_path.exists()
            self._model_info[model_name]["coreml"]["downloaded"] = coreml_downloaded

            if coreml_downloaded:
                size_on_disk = os.path.getsize(coreml_file_path)
                console.print(f"CoreML expected size: {self._model_info[model_name]['coreml']['size']}")
                console.print(f"CoreML size on disk: {size_on_disk}")

            return bin_downloaded and coreml_downloaded

        return bin_downloaded
    
    def prompt_validate_file(self, model_name: str, expected_hash: str, file_path: Path) -> bool:
        console.status(f"Validating integrity")
        if not validate_file_hash(file_path, expected_hash):
            logger.warning(f"Hash validation failed for {model_name} model file.")

            # Ask user if they want to delete the corrupted file
            if Confirm.ask(
                f"[bold red]The downloaded model file for {model_name} appears to be corrupted.[/] "
                f"Would you like to delete it and download again?",
                default=True
            ):
                try:
                    file_path.unlink()
                    raise Exception(f"Model file for {model_name} has been deleted. Please restart SignalScribe to download the model again.")
                except Exception as e:
                    logger.fatal(f"Couldn't delete the model file for {model_name}. Please delete it manually.")
                    logger.fatal(f"Full file path to delete: {file_path}")
                    raise e
            else:
                logger.warning(f"Keeping the corrupted file. Model may not work correctly.")

    @property
    def selected_model(self) -> str:
        return self._selected_model

    @selected_model.setter
    def set_selected_model(self, model_name: str):
        if model_name not in self._model_info.keys():
            logger.error(f"Model {model_name} not found")
            raise ValueError(f"Model {model_name} not found")

        self._selected_model = model_name

        self._ensure_model_exists(model_name)

    @property
    def model_dir(self) -> str:
        return self._model_dir

    def _ensure_model_exists(self, model_name: str):
        """Ensure all required model files exist and are valid."""
        logger.info(f"Ensuring {model_name} model files exist")

        missing_files = []
        total_bytes_needed = 0

        if not self._model_info[model_name]["bin"]["downloaded"]:
            missing_files.append("bin")
            total_bytes_needed += self._model_info[model_name]["bin"]["size"]

        if platform.system() == "Darwin":
            if not self._model_info[model_name]["coreml"]["downloaded"]:
                missing_files.append("coreml")
                total_bytes_needed += self._model_info[model_name]["coreml"]["size"]

        if not missing_files:
            logger.info(f"{model_name} model files exist, don't need to download anything")
            return
        
        logger.info(f"Downloading missing model files for {model_name}")


        console.print(
            f"Couldn't find {len(missing_files)} required model file(s) in [green]{self._model_dir}[/green]:"
        )

        for file_type in missing_files:
            console.print(
                f"  • {self._model_info[model_name][file_type]['filename']}: {format_size(self._model_info[model_name][file_type]['size'])}"
            )
        console.print(f"Total download size: {format_size(total_bytes_needed)}")

        if not Confirm.ask(
            f"Would you like to download the missing model files to {self._model_dir}?\n(This can be changed with the --model-dir flag)",
            default=True,
        ):
            raise FileNotFoundError(f"Model files for {model_name} not downloaded")

        # Download missing files
        for file_type in missing_files: 
            file_info = self._model_info[model_name][file_type]

            file_path = self._model_dir / file_info["filename"]

            # Download the file (will throw an exception if it fails - let the caller handle it)
            download_file(file_info["url"], file_path)

            # Verify the download
            self.prompt_validate_file(model_name, file_info["sha256"], file_path)

        # Extract CoreML model if needed
        if "coreml" in missing_files:
            extract_coreml_model(file_path)


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


# from app.py:
# if self.args.list_models:
#     available_models = self.model_manager._fetch_available_models()
#     console.print(f"Available models:")
#     if available_models:
#         grid = Table.grid(padding=(0, 2))
#         grid.add_column("Index", justify="left", style="blue", no_wrap=True)
#         grid.add_column("Model", justify="left", style="bold", no_wrap=True)
#         grid.add_column("Size", justify="left", style="dim", no_wrap=True)

#         choices = []

#         max_index = 0

#         for i, (base_name, model_info) in enumerate(available_models.items()):
#             index = i+1
#             max_index = index
#             choices.append(str(index))
#             if sys.platform == "darwin":
#                 grid.add_row(f"{index}", f"{base_name}", f"{model_info['bin_size']} (+ {model_info['coreml_size']} for CoreML)")
#             else:
#                 grid.add_row(f"{index}", f"{base_name}", f"{model_info['bin_size']}")

#         console.print(grid)

#         selected_index = None

#         while True:
#             selected_index = IntPrompt.ask(
#                 f"\n[bold]Select which model to use[/bold] (Press CTRL+C to exit)"
#             )
#             if selected_index >= 1 and selected_index <= max_index:
#                 break
#             console.print(f"[prompt.invalid]Number must be between 1 and {max_index}")

#         if selected_index:
#             self.args.model = list(available_models.keys())[selected_index-1]
#             console.print(f"Selected model: [bold][blue]{self.args.model}[/blue][/bold]")
#         else:
#             console.print("[red]No model selected, quitting[/red]") # Should never happen
#             return False
