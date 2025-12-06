"""
Platform detection utility for identifying target platforms.
"""

import platform
import sys
from typing import Tuple


def is_intel_mac() -> bool:
    """Check if running on Intel Mac (not Apple Silicon)."""
    if platform.system() != "Darwin":
        return False
    
    # Check processor architecture
    machine = platform.machine()
    processor = platform.processor()
    
    # Intel Macs typically report 'x86_64' or 'i386'
    # Apple Silicon reports 'arm64'
    if machine == 'x86_64' or machine == 'i386':
        # Additional check: processor name
        if processor and ('Intel' in processor or 'i386' in processor or 'x86_64' in processor):
            return True
    
    return False


def is_raspberry_pi() -> bool:
    """Check if running on Raspberry Pi."""
    if platform.system() != "Linux":
        return False
    
    # Check for Raspberry Pi by looking at /proc/cpuinfo
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Raspberry Pi' in cpuinfo or 'BCM' in cpuinfo:
                return True
    except (IOError, FileNotFoundError):
        pass
    
    # Alternative check: machine architecture
    machine = platform.machine()
    if machine.startswith('arm') or machine.startswith('aarch64'):
        # Additional check: check if it's likely a Pi
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().lower()
                if 'raspberry' in model:
                    return True
        except (IOError, FileNotFoundError):
            pass
    
    return False


def is_non_gpu_platform() -> bool:
    """Check if running on a non-GPU platform (Intel Mac or Raspberry Pi)."""
    return is_intel_mac() or is_raspberry_pi()


def get_platform_info() -> Tuple[str, bool]:
    """
    Get platform information.
    
    Returns:
        Tuple of (platform_name, is_non_gpu)
    """
    if is_raspberry_pi():
        return ("Raspberry Pi", True)
    elif is_intel_mac():
        return ("Intel Mac", True)
    elif platform.system() == "Darwin":
        return ("Apple Silicon Mac", False)
    elif platform.system() == "Linux":
        return ("Linux", False)
    elif platform.system() == "Windows":
        return ("Windows", False)
    else:
        return (platform.system(), False)


if __name__ == "__main__":
    # Test platform detection
    print(f"Platform: {platform.system()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Is Intel Mac: {is_intel_mac()}")
    print(f"Is Raspberry Pi: {is_raspberry_pi()}")
    print(f"Is Non-GPU Platform: {is_non_gpu_platform()}")
    platform_name, is_non_gpu = get_platform_info()
    print(f"Platform Info: {platform_name}, Non-GPU: {is_non_gpu}")

