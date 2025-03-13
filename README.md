# Signals Rising SignalScribe
signalscribe@signalsrising.org

Version 0.6
Please do not distribute this software. After a period of closed beta testing it will be released under the GPL Affero licence.

SignalScribe makes used of recordings output by SDRTrunk and similar SDR-based digital radios by monitoring for new recordings in a folder and transcribing them using OpenAI's Whisper model.

Transcriptions are logged to a CSV file named the same as the monitored folder.
The CSV is also stored in the same folder as the recordings.

## Installation

To install, run `./install.sh`.
This will install the Python dependencies and download the GGML model (and Apple Neural Engine model if required) to ./models/
These models are between 1 and 2GiB in size so may take a while to download.

You can provide the name of a particular verison of Python to use with the `--python-version` option, e.g. `./install.sh --python-version python3.11`.
By default it will just try to use 'python3'.

The script attempts to detect whether you have Apple Silicon or a CUDA-enabled GPU and will install the appropriate libraries and models.

For now only recent versions of MacOS and Ubuntu(-based) Linux are supported. Other operating systems may work but we can't make any guarantees.

## Running

To run, run `./python3 signalscribe.py <path_to_folder_with_recordings>`.
To stop, press Ctrl+C.
For help, run `./python3 signalscribe.py --help`.

(N.b. You must use the same version of Python that you used to install the dependencies.)

## Colors

If you place a file named `colors.yaml` in the same folder as the recordings, it will be used to highlight the transcriptions in the console.
An example `colors.yaml` file is provided with this file.

