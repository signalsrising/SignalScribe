#!/bin/bash

# Exit on any error
set -e

# Default values
PY_VERSION=python3
VULKAN=0    # 0 means don't force vulkan, 1 means force vulkan (i.e. use Vulkan even if nvidia card is detected)
FAIL=0

ANE_MODEL=ggml-large-v3-turbo-encoder.mlmodelc
ANE_MODEL_ZIP_SHA_256=84bedfe895bd7b5de6e8e89a0803dfc5addf8c0c5bc4c937451716bf7cf7988a
GGML_MODEL=ggml-large-v3-turbo.bin
GGML_MODEL_SHA_256=1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69

MODEL_REPO_URL=https://huggingface.co/ggerganov/whisper.cpp/resolve/main
ANE_MODEL_URL=$MODEL_REPO_URL/$ANE_MODEL.zip
GGML_MODEL_URL=$MODEL_REPO_URL/$GGML_MODEL

MODELS_DIR=./models
ANE_MODEL_ZIP_PATH=$MODELS_DIR/$ANE_MODEL.zip
ANE_MODEL_PATH=$MODELS_DIR/$ANE_MODEL
GGML_MODEL_PATH=$MODELS_DIR/$GGML_MODEL

# check if user has specified a python version and whether to force vulkan
for i in "$@"; do
  case $i in
    --python-version=*)
      PY_VERSION="${i#*=}"
      shift 
      ;;
    --vulkan)
      VULKAN=1
      shift 
      ;;
    --cpu)
      CPU=1 
      shift 
      ;;
    *)
      echo "Unknown option: $i"
      exit 1
      ;;
  esac
done

if ! command -v $PY_VERSION --version &> /dev/null; then
    echo "Error: Python $PY_VERSION is not installed. Please install Python $PY_VERSION first."
    exit 1
else
    PY_VERSION_USING=$($PY_VERSION --version 2>&1)
    echo "Using Python version $PY_VERSION_USING"
fi

# check if OS is mac or linux
if ! [[ "$OSTYPE" == "darwin"* ]] && ! [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "SignalScribe is only intended to be run on GNU/Linux or Macintosh. Will attempt to install anyway."
fi


# First check if ffmpeg, curl and unzip are installed.
# On Macintosh we can try to install them using homebrew.
# On Linux we don't try to install them as we don't want to ask for sudo - user will have to install them manually.
echo "Checking for ffmpeg, curl and unzip..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    # check if homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Error: Homebrew is not installed. Please install Homebrew first."
        echo "Homebrew is required to install ffmpeg, curl and unzip."
        echo "Visit https://brew.sh for installation instructions."
        exit 1
    fi

    echo "Checking for ffmpeg, curl and unzip are installed..."
    if ! command -v ffmpeg &> /dev/null; then
        echo "Error: ffmpeg is not installed. Attempting to install ffmpeg..."

        if ! brew install ffmpeg; then
            echo "Error: Failed to install ffmpeg. Please install ffmpeg manually."
            exit 1
        fi
    fi

    # check if curl is installed
    if ! command -v curl &> /dev/null; then
        echo "Error: curl is not installed. Attempting to install curl..."

        if ! brew install curl; then
            echo "Error: Failed to install curl. Please install curl manually."
            exit 1
        fi
    fi

    # check if unzip is installed
    if ! command -v unzip &> /dev/null; then
        echo "Error: unzip is not installed. Attempting to install unzip..."

        if ! brew install unzip; then
            echo "Error: Failed to install unzip. Please install unzip manually."
            exit 1
        fi
    fi
else
    if ! command -v ffmpeg &> /dev/null; then
        echo "Error: ffmpeg is not installed."
        FAIL=1
    fi

    if ! command -v curl &> /dev/null; then
        echo "Error: curl is not installed."
        FAIL=1
    fi

    if ! command -v unzip &> /dev/null; then
        echo "Error: unzip is not installed."
        FAIL=1
    fi

    # Add check for Python development headers
    if ! (dpkg -l python3-dev &>/dev/null || rpm -q python3-devel &>/dev/null); then
        echo "Error: Python development headers are not installed."
        FAIL=1
    fi

    if [[ $FAIL -eq 1 ]]; then
        echo "Error: Failed to install required packages, please run this:"
        echo "sudo apt install ffmpeg curl unzip python3-dev   # on Debian/Ubuntu"
        echo "sudo dnf install ffmpeg curl unzip python3-devel # on Fedora"
        exit 1
    fi
fi

# Ok that's good, we have ffmpeg, curl and unzip installed. Now we can try to install the python requirements.

# Macintosh:
if [[ "$OSTYPE" == "darwin"* ]]; then
        # Install Python requirements
        echo "Installing Python requirements..."
        WHISPER_COREML=1 $PY_VERSION -m pip install -r requirements.txt || {
            echo "Error: Failed to install Python requirements"
            exit 1
        }
else    
    # Check if nvidia gpu is installed or --vulkan flag is not set
    if command -v nvidia-smi &> /dev/null && [[ $VULKAN -ne 1 ]]; then 
        # read output of nvidia-smi --version and check if it contains "CUDA"
        if ! nvcc --version | grep -i cuda; then
            echo "NVIDIA card detected, but CUDA is not installed. Please install CUDA first then run this script again."
            echo "To install CUDA, please visit https://developer.nvidia.com/cuda-downloads"
            echo "Ensure that nvcc is installed using 'sudo apt install nvidia-cuda-toolkit'"
            echo "---"
            echo "If you want to force use of Vulkan on NVIDIA cards, please run this script with the --vulkan flag"
            echo "If you want to force installation without GPU support, please run this script with the --cpu flag"
            exit 1
        else
            echo "NVIDIA card detected, CUDA is installed. Installing Whisper.cpp with CUDA support..."
            GGML_CUDA=1 $PY_VERSION -m pip install -r requirements.txt || {
                echo "Error: Failed to install Python requirements"
                exit 1
            }
        fi
    else # either nvidia-smi is not installed or --vulkan flag is set
        if [[ $CPU -eq 1 ]]; then
            echo "Attempting to install Whisper.cpp without GPU support..."
            $PY_VERSION -m pip install -r requirements.txt || {
                echo "Error: Failed to install Python requirements"
                exit 1
            }
        else
            echo "Attempting to install Whisper.cpp with Vulkan support..."
            GGML_VULKAN=1 $PY_VERSION -m pip install -r requirements.txt || {
                echo "Error: Failed to install Python requirements"
                exit 1
            }
        fi
    fi
fi

echo "Whisper installed successfully, attempting to download models to $MODELS_DIR directory"

# Create models directory if it doesn't exist
mkdir -p models

# For  debugging:
# echo MODEL_REPO_URL: $MODEL_REPO_URL
# echo ANE_MODEL_URL: $ANE_MODEL_URL
# echo GGML_MODEL_URL: $GGML_MODEL_URL

# echo MODELS_DIR: $MODELS_DIR
# echo ANE_MODEL_ZIP_PATH: $ANE_MODEL_ZIP_PATH
# echo ANE_MODEL_PATH: $ANE_MODEL_PATH
# echo GGML_MODEL_PATH: $GGML_MODEL_PATH

# Download and extract ANE model if it doesn't exist
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [ ! -d "$ANE_MODEL_PATH" ]; then
        echo "Downloading Apple Neural Engine (ANE) model from:"
        echo "$ANE_MODEL_URL"
        curl -L $ANE_MODEL_URL -o $ANE_MODEL_ZIP_PATH || {
            echo "Error: Failed to download encoder model"
            exit 1
        }
        
        ANE_MODEL_ZIP_SHA_256=$(shasum -a 256 $ANE_MODEL_ZIP_PATH | awk '{print $1}')

        if ! [[ "$ANE_MODEL_ZIP_SHA_256" == "$ANE_MODEL_ZIP_SHA_256" ]]; then
            echo "Error: SHA256 of ANE model zip file does not match, please delete $ANE_MODEL_ZIP_PATH and run this script again."
            exit 1
        fi

        echo "Extracting ANE model..."
        unzip $ANE_MODEL_ZIP_PATH -d $MODELS_DIR || {
            echo "Error: Failed to extract ANE model"
            rm $ANE_MODEL_ZIP_PATH
            exit 1
        }
        echo "ANE model extracted successfully"
        rm $ANE_MODEL_ZIP_PATH

    else
        echo "ANE model already exists, skipping download"
    fi
fi

# Download GGML model if it doesn't exist
if [ ! -f "$GGML_MODEL_PATH" ]; then
    echo "Downloading GGML Model model from:"
    echo "$GGML_MODEL_URL"
    curl -L $GGML_MODEL_URL -o $GGML_MODEL_PATH || {
        echo "Error: Failed to download GGML model"
        exit 1
    }
else
    echo "GGML model already exists, skipping download"
fi


GGML_MODEL_SHA_256=$(shasum -a 256 $GGML_MODEL_PATH | awk '{print $1}')

if ! [[ "$GGML_MODEL_SHA_256" == "$GGML_MODEL_SHA_256" ]]; then
    echo "Error: SHA256 of GGML model does not match, please delete $GGML_MODEL_PATH and run this script again."
    exit 1
fi

echo "Installation of SignalScribe completed successfully!"