import argparse
import rich.color
from rich.console import Console
from rich.logging import RichHandler
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
import pathlib
import re
import os
import yaml
import time
from datetime import datetime
import logging
from faster_whisper import WhisperModel
from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
import csv

APP_NAME = "SignalScribe"
VERSION = "0.3"

COMPUTE_TYPE = "float32"
DEFAULT_MODEL = "medium.en"

MODEL_CHOICES = ["large-v3", "large-v2", "medium", "medium.en"]
FILETYPES_SUPPORTED = ["mp3", "m4a", "wav"]
COLORS_SETTINGS_NAME = "colors.yaml"

logger = logging.getLogger("rich")
console = Console(highlight=False)

def insert_string(string, insert, position):
    return string[:position] + insert + string[position:]
        
class FolderWatcher:
    def __init__(self, options: list):

        self.options = options
        self.observer = Observer()

    def run(self):
        event_handler = FolderWatcherHandler(self.options)
        self.observer.schedule(event_handler, self.options.folder, self.options.recursive)
        self.observer.start()

        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()

        self.observer.join()


class FolderWatcherHandler(PatternMatchingEventHandler):

    def __init__(self, options: list) -> None:
        super().__init__()
        
        # Command line options:
        self.options = options
        if options.cpu_only:
            device = "cpu"
            device_string = "CPU Only"
        else:
            device = "auto" # Attempt to use CUDA but fall-back to CPU if it doesn't work out
            device_string = "CUDA (if available)"
    
        
        # Reformat the command-line inputs (mp3,mp4 etc) into wildcard+extention ("*.mp3", "*.mp4"):
        patterns = ["*." + format for format in options.formats]
        patterns.append(COLORS_SETTINGS_NAME)
        
        # Init superclass:
        PatternMatchingEventHandler.__init__(self, patterns=patterns)

        # Load model:
        try:
            with console.status("Loading Whisper") as status:
                self.model = WhisperModel(options.model, device=device, compute_type=COMPUTE_TYPE, cpu_threads=options.threads)
        except Exception as e:
            console.print(f"[red bold]Fatal error loading Whisper: {e}")    
            quit()
        
        console.print(f"Loaded Whisper with settings:")
        
        grid = Table.grid(padding=(0,2))

        grid.add_column(justify="right", style="yellow", no_wrap=True)
        grid.add_column()

        grid.add_row("Model", options.model)
        grid.add_row("Compute Device", device_string)
        grid.add_row("Compute Type", COMPUTE_TYPE)
        grid.add_row("CPU Threads", str(options.threads))
        # table.add_row("R", "")

        console.print(grid)
        console.print("")
        
        # Set up dict to store our highlighting colors
        self.colors = dict()
        self.update_colors(os.path.join(self.options.folder, COLORS_SETTINGS_NAME))
        console.print("")

        formats_text = "[blue]" + "[default], [blue]".join(options.formats)
        if options.recursive:
            console.print(f"Watching folder: [green]{options.folder}[default] and all subfolders for formats: {formats_text}.")
        else:
            console.print(f"Watching folder: [green]{options.folder}[default] for formats: {formats_text}.")

        logger.debug(f"Created new FolderWatcherHandler for formats: {options.formats}")
       
        
    def update_colors(self, colors_file_path: str) -> bool:     
        try:
            with open(colors_file_path, 'r') as colors_file:
                colors_dict = yaml.safe_load(colors_file)
        except OSError as e:
            console.print(f"No highlight settings file found at {colors_file_path}")
            return
        except Exception as e:
            console.print(e)
            return
        
        valid_colors = dict()
        
        # sanity checks:
        for color in colors_dict.keys():
            # if the color name is valid ANSI color..
            if color in rich.color.ANSI_COLOR_NAMES:
                phrases = colors_dict[color]
                # .. and the entry for this color contains a list..
                if isinstance(phrases, list):
                    # then ensure all the entries are strings and add to our new color dict
                    valid_colors[color] = [str(phrase) for phrase in phrases]
        
        if valid_colors and valid_colors != self.colors:        
            self.colors = valid_colors
            console.print(f"Updated highlight settings from: {colors_file_path}")
    
    def on_closed(self, event: FileSystemEvent) -> None:
        file_name_ext = os.path.basename(event.src_path) # file name with extention
        
        if file_name_ext == COLORS_SETTINGS_NAME:
            self.update_colors(event.src_path)
    
    
    
    def on_created(self, event: FileSystemEvent) -> None:
        # logger.info(f"Detected new file {event.src_path}")
        file_name_ext = os.path.basename(event.src_path) # file name with extention
        
        if file_name_ext == COLORS_SETTINGS_NAME:
            self.update_colors(event.src_path)
            return

        start_time = time.monotonic() # for debug/logging/benchmarking
        file_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_datetime_tag = f"[{file_datetime}] "

        file_name = os.path.splitext(file_name_ext)[0] # just the file name, no path, no extension
        file_uri = pathlib.Path(event.src_path).absolute().as_uri()
        

        try:
            console.log(f"Transcribing file: {event.src_path}")
            segments, info = self.model.transcribe(event.src_path, beam_size=5, vad_filter=True)
        except ValueError as e:
            console.print(f"Error transcribing potentially empty file: {file_name}", style="red")
            return

        text = ""
        try:
            for segment in segments:
                text += segment.text
        except Exception as e:
            console.print(f"Error transcribing file {file_name}: {str(e)}", style="red") 
            return

        text = text.strip()
        
        try:
            with open(self.options.csv_filepath, 'a', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file, dialect="excel")
                writer.writerow([file_datetime,event.src_path,info.duration,text])
        except Exception as e:
            console.print(f"[red]Error writing to CSV file: {e}")

        if not text:
            return

        # Lower-case copy of our transcription that's used for searching for phrases to highlight.
        # As we highlight phrases in the main text we replace them in the search-text with spaces.
        # This means that we don't highlight a word or phrase twice if it appears in the color YAML file more than once.
        search_text = text.lower()
        
        for color, phrases in self.colors.items():
        
            opening_tag = f"[{color}]"
            closing_tag = f"[/{color}]"
            
            for phrase in phrases:
                # find all positions that the phrase occurs         
                positions = [(pos.start(), pos.end()) for pos in re.finditer(phrase, search_text, re.IGNORECASE)]
                
                # If we find at least 1 instance
                if positions:
                    # From back to front (end to start) we highlight the found phrases
                    # (Because this way it doesn't change the indexes found by the regex search above each time we insert a [tag])
                    for (start, end) in reversed(positions):
                        text = insert_string(text, closing_tag, end)
                        text = insert_string(text, opening_tag, start)
                    
                    # then remove all instances of this phrase from the search text so we don't try and colorize it again:
                    tagged_phrase = opening_tag + phrase + closing_tag
                    search_text = re.compile(re.escape(tagged_phrase), re.IGNORECASE).sub(" " * len(tagged_phrase), text)
                    
        # Print our results:
        console.print(f"{file_datetime_tag}", end="")
        console.print(f"[blue]{file_name}", style=f"link {file_uri}")
        console.print(Padding(text,(0,len(file_datetime_tag)), style="bold"))

        elapsed = time.monotonic() - start_time
        speedup = info.duration / elapsed

        logging.debug(f"Transcribed file {event.src_path} (language: {info.language}) in {elapsed:.1f} seconds ({speedup:.1f}x realtime)")


def main(command_line=None):

    parser = argparse.ArgumentParser(prog="signalscribe",
                                     description="""Signals Rising SignalScribe:
                                                Monitors a folder for additions,
                                                automatically transcribes them using a given transformer model,
                                                then writes the text out to a file or database""")

    # parser.add_argument("-c",
    #                     "--csv",
    #                     action="store_true",
    #                     help="Write output to CSV file. File will be named after the current folder.")

    # parser.add_argument("-d",
    #                     "--database",
    #                     nargs="?", # not working
    #                     action="store",
    #                     help="Write output to sqlite database. If used alone will name the sqlite file after the current folder. A filename can be provided.")

    parser.add_argument("-l",
                        "--log-file",
                        # nargs="?", # not working
                        action="store_true",
                        help="Write program log to file. Log file will be named after the current folder.")

    parser.add_argument("--cpu-only",
                        action="store_true",
                        help="Forces CPU-only operation. Otherwise will attempt to use GPU and fall back on CPU if not available.")

    parser.add_argument("-r",
                        "--recursive",
                        action="store_true",
                        help="Also monitor subfolders for changes, as opposed to just this folder.")

    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="Verbose mode. Logs all debug info to terminal output.")

    parser.add_argument("-f",
                        "--formats",
                        choices=FILETYPES_SUPPORTED,
                        default=FILETYPES_SUPPORTED,
                        help="Audio formats to accept for processing. By default, all common audio types are enabled.")

    parser.add_argument("-t",
                        "--threads",
                        type=int,
                        default=int(os.cpu_count()/2), # improve this
                        help=f"Number of threads to use for processing. Default: {int(os.cpu_count()/2)}. Max: {os.cpu_count()}")

    parser.add_argument("-m",
                        "--model",
                        choices=MODEL_CHOICES,
                        default=DEFAULT_MODEL,
                        help=f"Transformer model to use for transcription. Default: {DEFAULT_MODEL}")

    parser.add_argument("folder", type=str, nargs='?',  default=".")

    options = parser.parse_args(command_line)

    if options.folder == ".":
        filename = pathlib.PurePath(os.getcwd()).name
    else:
        filename = pathlib.PurePath(options.folder).name

    if filename == "":
        filename = "signalscribe"

    if options.threads >= os.cpu_count():
        options.threads = os.cpu_count()
        
    console.print(Panel(f"[bold][bright_cyan]{APP_NAME}\n[cyan]by Signals Rising[/bold]\n[bright_black]Version {VERSION}", expand=False))
    console.print("")
    # console.print(f"Version {VERSION}")

    folder_path = pathlib.Path(options.folder)

    options.log_filename = filename + ".log"
    options.csv_filename = filename + ".csv"

    options.log_filepath = folder_path / options.log_filename
    options.csv_filepath = folder_path / options.csv_filename

    if options.verbose:
        console.log("Verbose mode")
        logging.basicConfig()
        logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

    # logger.debug(f"Logging:  {options.log_filepath}")
    logger.debug(f"CSV file: {options.csv_filepath}")

    watch = FolderWatcher(options)
    watch.run()



if __name__ == '__main__':
    main()
