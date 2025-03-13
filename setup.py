from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys
import os
import warnings
import platform
import re

"""
SignalScribe Installation Summary:

1. Check for ffmpeg (required dependency) and warn if not found (but doesn't abort/fail)
2. Install the base SignalScribe package
3. Attempt to detect available hardware and install appropriate acceleration
4. Allow manual selection of acceleration with:

- pip install "signalscribe[cuda]"    # For NVIDIA GPUs
- pip install "signalscribe[vulkan]"  # For AMD/Intel GPUs
- pip install "signalscribe[coreml]"  # For Apple Silicon
- pip install "signalscribe[cpu]"     # For CPU-only mode
- pip install "signalscribe[auto]"    # For automatic detection (default)
"""


def check_ffmpeg():
    """Check if ffmpeg is installed and available in PATH"""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        print("✅ ffmpeg found")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        warnings.warn(
            "\n⚠️  WARNING: ffmpeg not found in PATH. SignalScribe requires ffmpeg to function properly.\n"
            "Please install ffmpeg before using SignalScribe:\n"
            "- On macOS: brew install ffmpeg\n"
            "- On Ubuntu/Debian: apt install ffmpeg\n"
            "- On Windows: https://ffmpeg.org/download.html\n"
        )
        return False


def configure_acceleration(env_var=None, env_value=None):
    """Configure hardware acceleration by setting environment variables.

    Args:
        env_var: Optional environment variable name to set
        env_value: Value to set for the environment variable
    """
    if env_var and env_value:
        os.environ[env_var] = env_value
        print(f"Setting {env_var}={env_value}")

        # Create a .env file in the package directory to persist this setting
        env_file = os.path.join(os.path.dirname(__file__), "SignalScribe", ".env")
        with open(env_file, "w") as f:
            f.write(f"{env_var}={env_value}\n")
        print(f"Saved acceleration setting to {env_file}")


class CustomInstallCommand(install):
    """Custom installation for handling acceleration options."""

    def run(self):
        # Check for ffmpeg
        ffmpeg_available = check_ffmpeg()

        # Run the standard installation first
        install.run(self)

        # Check if we're installing with extras
        install_args = sys.argv[1:]

        # If 'auto' extra is selected or no accelerator is specified
        if any("auto" in arg for arg in install_args) or not any(
            acc in " ".join(install_args) for acc in ["cuda", "vulkan", "coreml", "cpu"]
        ):
            print("Auto-detecting hardware acceleration...")
            self.auto_detect_and_configure()
        elif any("cuda" in arg for arg in install_args):
            print("Configuring with CUDA acceleration...")
            configure_acceleration("GGML_CUDA", "1")
        elif any("vulkan" in arg for arg in install_args):
            print("Configuring with Vulkan acceleration...")
            configure_acceleration("GGML_VULKAN", "1")
        elif any("coreml" in arg for arg in install_args):
            print("Configuring with CoreML acceleration...")
            configure_acceleration("WHISPER_COREML", "1")
        elif any("cpu" in arg for arg in install_args):
            # CPU install doesn't need special environment variables
            print("Configuring CPU-only version...")
        else:
            # Fallback if none of the above match but somehow we got here
            print("No acceleration method specified, using CPU only...")

        # Final message based on dependency check
        if not ffmpeg_available:
            print("\n⚠️  IMPORTANT: ffmpeg was not found during installation.")
            print("   SignalScribe will not work properly until ffmpeg is installed.")

            # Print detailed ffmpeg installation instructions based on OS
            print("\n📋 ffmpeg Installation Instructions:")
            system = platform.system()

            if system == "Darwin":  # macOS
                print("  macOS:")
                print("    Using Homebrew: brew install ffmpeg")
                print("    Using MacPorts: sudo port install ffmpeg")
                print("    Manual download: https://ffmpeg.org/download.html#build-mac")

            elif system == "Linux":
                # Try to detect distribution
                try:
                    with open("/etc/os-release") as f:
                        os_info = dict(
                            line.strip().split("=", 1) for line in f if "=" in line
                        )
                    distro = os_info.get("ID", "").lower()

                    if distro in ["ubuntu", "debian", "linuxmint"]:
                        print("  Ubuntu/Debian/Mint:")
                        print("    sudo apt update && sudo apt install ffmpeg")
                    elif distro in ["fedora", "rhel", "centos"]:
                        print("  Fedora/RHEL/CentOS:")
                        print("    sudo dnf install ffmpeg")
                    elif distro == "arch":
                        print("  Arch Linux:")
                        print("    sudo pacman -S ffmpeg")
                    elif distro == "opensuse":
                        print("  openSUSE:")
                        print("    sudo zypper install ffmpeg")
                    else:
                        print("  Linux (generic):")
                        print(
                            "    Use your distribution's package manager to install ffmpeg"
                        )
                        print(
                            "    For example: sudo apt install ffmpeg, sudo dnf install ffmpeg, etc."
                        )
                except (FileNotFoundError, IOError):
                    print("  Linux (generic):")
                    print(
                        "    Use your distribution's package manager to install ffmpeg"
                    )

            elif system == "Windows":
                print("  Windows:")
                print("    1. Download from: https://ffmpeg.org/download.html")
                print("    2. Extract the ZIP file to a folder (e.g., C:\\ffmpeg)")
                print("    3. Add the bin folder to your PATH environment variable:")
                print("       - Right-click on 'This PC' or 'My Computer'")
                print("       - Select 'Properties' → 'Advanced system settings'")
                print("       - Click 'Environment Variables'")
                print(
                    "       - Edit the 'Path' variable and add the path to ffmpeg's bin folder"
                )
                print("    Alternative: Install using Chocolatey: choco install ffmpeg")

            else:
                print(f"  {system}:")
                print(
                    "    Please visit https://ffmpeg.org/download.html for installation instructions"
                )

            print(
                "\nFFmpeg is required for SignalScribe to process audio files properly.\n"
            )

    def auto_detect_and_configure(self):
        """Auto-detect platform and configure appropriate acceleration"""
        import platform

        # Check if on macOS with Apple Silicon
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            print("Detected Apple Silicon - enabling CoreML support")
            configure_acceleration("WHISPER_COREML", "1")
            return

        # Check for NVIDIA GPU with CUDA
        try:
            subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, check=True)
            print("Detected NVIDIA GPU - enabling CUDA support")
            configure_acceleration("GGML_CUDA", "1")
            return
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Check for Vulkan support
        try:
            # This is a simplified check - you'd need a more robust check in practice
            if platform.system() == "Linux":
                vulkan_check = subprocess.run(
                    ["vulkaninfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                if vulkan_check.returncode == 0:
                    print("Detected Vulkan support - enabling Vulkan acceleration")
                    configure_acceleration("GGML_VULKAN", "1")
                    return
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Fallback to CPU
        print("No GPU acceleration detected - using CPU only")


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
    cmdclass={
        "install": CustomInstallCommand,
    },
)
