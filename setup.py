from setuptools import setup, find_packages
import os
import re
import subprocess
import warnings

"""
SignalScribe Installation Summary:

1. Check for ffmpeg (required dependency) and warn if not found (but doesn't abort/fail)
2. Install the base SignalScribe package
"""


# Dirty hack to import version from the package even when it's not installed
# lmk if you're reading this and there's a better way to do it...
def get_version():
    """Extract version from version.py"""
    version_file = os.path.join(os.path.dirname(__file__), "SignalScribe", "version.py")

    with open(version_file, "r") as f:
        version_line = [
            line for line in f.readlines() if line.startswith("__version__")
        ][0]

    # Version string like '0.1.0'
    return re.match(r"__version__ = ['\"]([^'\"]+)['\"]", version_line).group(1)


setup(
    name="signalscribe",
    version=get_version(),
    packages=find_packages(),
)
