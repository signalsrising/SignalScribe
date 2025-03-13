# Signals Rising SignalScribe

SignalScribe makes used of recordings output by SDRTrunk and similar SDR-based digital radios by monitoring for new recordings in a folder and transcribing them using OpenAI's Whisper model.

Transcriptions are logged to a CSV file named the same as the monitored folder.

Clone repo, run `pip install -r requirements.txt` then run `python signalscribe.py --help` to show usage.
