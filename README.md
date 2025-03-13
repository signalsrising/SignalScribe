# SignalScribe by Signals Rising
signalscribe@signalsrising.org

**Version 0.8.0** 2025-03-13

SignalScribe makes use of recordings output by SDRTrunk and similar SDR-based digital radios by monitoring a folder for new recordings and transcribing them using OpenAI's Whisper model.

Transcriptions are logged to a CSV file named the same as the monitored folder.
The CSV is also stored in the same folder as the recordings.

## Installation using pip via pipy.org

Depending on your hardware, you have to manually prepend the following to the pip install command:

- NVIDIA GPU with CUDA:
  ```
  GGML_CUDA=1 pip install git+https://github.com/signalsrising/signalscribe.git
  ```

- AMD/Intel GPU with Vulkan:
  ```
  GGML_VULKAN=1 pip install git+https://github.com/signalsrising/signalscribe.git
  ```

- Apple Silicon with Neural Engine:
  ```
  GGML_COREML=1 pip install git+https://github.com/signalsrising/signalscribe.git
  ```

- CPU only (for compatibility):
  ```
  pip install git+https://github.com/signalsrising/signalscribe.git
  ```
  
For hardware acceleration to work, you need to have the necessary drivers and libraries installed.

- For nVidia GPUs, hardware acceleration is provided via CUDA. Visit [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

- For AMD or Intel based GPUs, hardware acceleration can be provided by Vulkan. Visit [https://vulkan.lunarg.com/sdk/home](https://vulkan.lunarg.com/sdk/home).

- For Apple Silicon, hardware acceleration is provided by CoreML automatically.


## Installation from source

```
git clone https://github.com/signalsrising/signalscribe.git
cd signalscribe
pip install -e .
```

## Running

To run:

`$ signalscribe <path_to_folder_with_recordings>`.

To stop, press Ctrl+C.

To automatically monitor the SDRTrunk recordings folder, just run `signalscribe` without any folder argument:

`$ signalscribe`

SignalScribe has a number of optional arguments, to list them, run:

`$ signalscribe --help`

## Colors

If you place a file named `colors.yaml` in the same folder as the recordings, it will be used to highlight the transcriptions in the console. Example color.yaml file:


```
green:
- one
- two
- three

red:
- four
- five six seven
- 8
- 9 10

yellow:
- ELEVEN

purple:
- 12
- ThirtEEN

blue:
- one
- two
- three
```

Highlights take precedence the higher up in a file they are.
For instance 'one' will be highlighted green, despite the fact that it's also included under 'blue:', because green comes higher up in the file than blue.

Highlighting is case insensitive, so Thirteen, THIRTEEN and thirteen will all be highlighted purple according to this example

Highlighting happens by phrase, rather than by word. So 'five six seven' will be highlighted only if all 3 words occur together

Supported color names can be found here: https://rich.readthedocs.io/en/stable/appendix/colors.html

This colors file can be updated at any time while SignalScribe is running and the changes will be reflected in future console output.

