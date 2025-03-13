import os
import platform
import xml.etree.ElementTree as ET
from pathlib import Path
import re
from typing import Optional
import psutil

# Platform-specific imports
system = platform.system()
if system == "Windows":
    import winreg
elif system == "Darwin":  # macOS
    import plistlib

from .utils import logger

class SDRTrunkDetector:
    """
    SDRTrunk integration for SignalScribe.

    This module provides functionality to detect if SDRTrunk is running and find its recording directory.
    """

    def get_process(self) -> Optional[psutil.Process]:
        """Get the process of SDRTrunk."""
        for proc in psutil.process_iter(['pid','name']):
            name = proc.info["name"]
            pid = proc.info["pid"]
            if "java" in name and "sdrtrunk" in str(proc.cmdline()).lower():
                return proc
        
        return None
    
    def get_recording_directory(self) -> Optional[Path]:
        """Get the recording directory for SDRTrunk if it's running.
        
        Returns:
            Optional[Path]: The path to the recording directory, or None if SDRTrunk is not running
                           or the directory cannot be determined.
        """
        
        # First try to get the recording directory from Java preferences
        recording_dir = self._get_sdrtrunk_recording_directory()
        
        if recording_dir:
            logger.info(f"Found SDRTrunk recording directory from preferences: {recording_dir}")
            return Path(recording_dir)
        
        # If preferences method failed, try to find it in logs
        logger.info("Could not find recording directory in preferences, trying logs...")
        log_recording_dir = self._find_recording_dir_in_logs()
        
        if log_recording_dir:
            logger.info(f"Found SDRTrunk recording directory from logs: {log_recording_dir}")
            return Path(log_recording_dir)
        
        logger.warning("Could not determine SDRTrunk recording directory")
        return None


    def _get_log_file(self) -> Optional[str]:
        """Get the log file of SDRTrunk."""
        proc = self.get_process()

        if proc:
            logger.info(f"Found SDRTrunk running with PID: {str(proc.pid)})")
            for file in proc.open_files():
                if "sdrtrunk_app.log" in file.path:
                    logger.info(f"Found SDRTrunk log file at: {file.path}")
                    return file.path
                
            logger.warning(f"SDRTrunk running but cannot find open log file.")
            logger.debug(f"Open files: {", ".join([openfile.path for openfile in proc.open_files()])}")
        return None
        
    def _find_recording_dir_in_logs(self) -> Optional[str]:
        """Find the recording directory by parsing log files.
        
        Returns:
            Optional[str]: The recording directory path, or None if it cannot be found.
        """
        log_file = self._get_log_file()

        if not log_file:
            return None
            
        try:
            recording_dir = None
            with open(log_file, 'r', errors='ignore') as f:
                for line in f:
                    if "Recordings:" in line:
                        # Extract the path after "Recordings:" but before any bracket or other stats
                        match = re.search(r'Recordings:\s+([^[\r\n]+)', line)
                        if match:
                            # Trim any trailing whitespace
                            recording_dir = match.group(1).strip()
            
            return recording_dir
        
        except Exception as e:
            logger.warning(f"Error reading SDRTrunk log file: {e}")
            return None

    def _get_sdrtrunk_recording_directory(self):
        """
        Get the recording directory configured in SDRTrunk across platforms
        
        Returns:
            str: The recording directory path or None if not found
        """
        system = platform.system()
        
        if system == "Windows":
            return self._get_windows_recording_dir()
        elif system == "Darwin":  # macOS
            return self._get_macos_recording_dir()
        elif system == "Linux":
            return self._get_linux_recording_dir()
        else:
            logger.warning(f"Unsupported platform: {system}")
            return None

    @staticmethod
    def _get_windows_recording_dir():
        """Get SDRTrunk recording directory from Windows registry"""
        try:
            reg_path = "Software\\JavaSoft\\Prefs\\io\\github\\dsheirer\\preference\\directory"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path) as registry_key:
                value, _ = winreg.QueryValueEx(registry_key, "directory.recording")
                return value
        except Exception as e:
            logger.debug(f"Error reading Windows registry: {e}")
            return None

    @staticmethod
    def _get_macos_recording_dir():
        """Get SDRTrunk recording directory from macOS plist file"""
        try:
            # Check the application-specific plist file
            sdrtrunk_plist = os.path.expanduser("~/Library/Preferences/io.github.dsheirer.plist")
            if os.path.exists(sdrtrunk_plist):
                with open(sdrtrunk_plist, 'rb') as fp:
                    prefs = plistlib.load(fp)
                
                # Navigate to the preference
                if "/io/github/dsheirer/" in prefs:
                    app_prefs = prefs["/io/github/dsheirer/"]
                    if "preference/" in app_prefs and "directory/" in app_prefs["preference/"]:
                        dir_prefs = app_prefs["preference/"]["directory/"]
                        if "directory.recording" in dir_prefs:
                            return dir_prefs["directory.recording"]
            
            return None
        except Exception as e:
            logger.debug(f"Error reading macOS preferences: {e}")
            return None

    @staticmethod
    def _get_linux_recording_dir():
        """Get SDRTrunk recording directory from Linux XML files"""
        try:
            # Java preferences on Linux are stored in XML files
            prefs_dir = os.path.expanduser("~/.java/.userPrefs/")
            xml_path = os.path.join(prefs_dir, "io", "github", "dsheirer", 
                                "preference", "directory", "prefs.xml")
            
            if not os.path.exists(xml_path):
                return None
                
            # Parse the XML file
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Find the entry with the matching key
            for entry in root.findall(".//entry"):
                if entry.get("key") == "directory.recording":
                    return entry.get("value")
                    
            return None
        except Exception as e:
            logger.debug(f"Error reading Linux preferences: {e}")
            return None 
        



if __name__ == "__main__":
    detector = SDRTrunkDetector()
    
    # Check if SDRTrunk is running
    if detector.get_process():
        print("SDRTrunk is running")
    else:
        print("SDRTrunk is not running")
        # Get the recording directory
    
    recording_dir = detector.get_recording_directory()
    if recording_dir:
        print(f"SDRTrunk recording directory: `{recording_dir}`")
    else:
        print("Could not determine SDRTrunk recording directory")
