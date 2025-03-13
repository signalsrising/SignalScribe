# SignalScribe by Signals Rising
signalscribe@signalsrising.org

**Version 0.7.1** 2025-02-25

SignalScribe makes use of recordings output by SDRTrunk and similar SDR-based digital radios by monitoring a folder for new recordings and transcribing them using OpenAI's Whisper model.

Transcriptions are logged to a CSV file named the same as the monitored folder.
The CSV is also stored in the same folder as the recordings.

## Installation using pip via pipy.org

Simply run:

```
pip install signalscribe
```

Setup will try to automatically detect your hardware and install the appropriate acceleration.

- For hardware acceleration to work, you need to have the necessary drivers and libraries installed. For nVidia GPUs, hardware acceleration is provided via CUDA. Visit [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

- For AMD or Intel based GPUs, hardware acceleration can be provided by Vulkan. Visit [https://vulkan.lunarg.com/sdk/home](https://vulkan.lunarg.com/sdk/home).

For Apple Silicon, hardware acceleration is provided by CoreML automatically.

### Manual hardware API selection

If you want to choose a specific acceleration method:

- NVIDIA GPU with CUDA:
  ```
  pip install "signalscribe[cuda]"
  ```

- AMD/Intel GPU with Vulkan:
  ```
  pip install "signalscribe[vulkan]"
  ```

- Apple Silicon with Neural Engine:
  ```
  pip install "signalscribe[coreml]"
  ```

- CPU only (for compatibility):
  ```
  pip install "signalscribe[cpu]"
  ```

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

For help:

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

# Java Preferences Access from Python

This repository contains three different approaches to access Java preferences from Python, specifically targeting the SDRTrunk application's recording directory preference.

## Background

Java applications store user preferences using the `java.util.prefs.Preferences` API. These preferences are stored in different locations depending on the operating system:

- **Windows**: In the Windows Registry
- **macOS**: In `~/Library/Preferences/com.apple.java.util.prefs.plist`
- **Linux**: In `~/.java/.userPrefs/` as XML files

## Approaches

### 1. Using JPype (check_java_prefs.py)

This approach uses the JPype library to directly interact with the Java Preferences API from Python. It's the most direct approach but requires installing the JPype library.

**Requirements:**
- JPype (`pip install JPype1`)
- Java JDK installed

**Usage:**
```bash
python check_java_prefs.py
```

### 2. Direct File Access (check_java_prefs_direct.py)

This approach directly accesses the preference storage files on the filesystem. It's more complex but doesn't require additional libraries beyond Python's standard library.

**Requirements:**
- Python 3.6+

**Usage:**
```bash
python check_java_prefs_direct.py
```

### 3. Java Helper (check_java_prefs_jna.py)

This approach creates a small Java program on-the-fly to access the preferences and returns the results to Python. It's the most reliable cross-platform approach but requires Java to be installed and accessible from the command line.

**Requirements:**
- Java JDK installed and accessible in PATH

**Usage:**
```bash
python check_java_prefs_jna.py
```

## Recommendation

For most users, the Java Helper approach (option 3) is recommended as it:
- Works across all platforms
- Doesn't require additional Python libraries
- Is the most reliable as it uses Java's native API directly

If you're already using JPype in your project, option 1 might be more integrated with your existing code.

## Example Output

All scripts will output something like:

```
All stored keys:
- directory.recording = /path/to/recordings
- other.key = other.value
Recording path: /path/to/recordings
```

