import logging
from rich.progress import Progress
from rich.status import Status
from rich.live import Live
from rich.prompt import Prompt, Confirm
from rich.console import Console as RichConsole
from rich.text import Text
import inspect


class LoggingConsole(RichConsole):
    """Custom Console that logs all print calls to the logger,
    but ignores interactive elements like progress bars and status spinners."""

    # Classes that should not be logged when printed
    INTERACTIVE_CLASSES = (
        Progress,  # Progress bars
        Status,  # Status spinners
        Live,  # Live updating displays
        Prompt,  # Interactive prompts
        Confirm,  # Yes/no prompts
    )

    # New logging level for console prints
    CONSOLE = 5

    def __init__(
        self,
        logger: logging.Logger,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logger = logger

        # Define a new logging level below DEBUG so that we can
        # log console prints to the log file but avoid having them
        # on screen twice.
        logging.addLevelName(self.CONSOLE, "CONSOLE")

        # Add a console_log method to the logger class
        def console(self, message, *args, **kwargs):
            if self.isEnabledFor(self.CONSOLE):
                self._log(self.CONSOLE, message, args, **kwargs)

        # Add the console_log method to the Logger class
        logging.Logger.console_log = console

    def print(self, *args, **kwargs):
        # Skip logging for interactive elements
        should_log = True

        for arg in args:
            # Check if it's an instance of any interactive class
            if isinstance(arg, self.INTERACTIVE_CLASSES):
                should_log = False
                break

            # Check for _live attribute (used by many Rich live components)
            if hasattr(arg, "_live"):
                should_log = False
                break

            # Check if it's being used in an interactive context
            # (e.g., Table or Panel that will be used in a Live context)
            caller_frame = inspect.currentframe().f_back
            if caller_frame and "self" in caller_frame.f_locals:
                caller_self = caller_frame.f_locals["self"]
                if isinstance(caller_self, self.INTERACTIVE_CLASSES):
                    should_log = False
                    break

        if should_log:
            # Convert args to string and log it
            try:
                valid_args = []
                for arg in args:
                    if isinstance(arg, Text):
                        valid_args.append(arg.plain)
                    elif isinstance(arg, str):
                        valid_args.append(arg)

                if valid_args:
                    # Use the CONSOLE level instead of INFO
                    self.logger.console_log(" ".join(valid_args))

            except Exception:
                # If we can't convert to string, just skip logging this message
                pass

        # Call the original print method
        return super().print(*args, **kwargs)
