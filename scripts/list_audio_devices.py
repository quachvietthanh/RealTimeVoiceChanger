"""Audio device discovery script.

Discovers input and output audio devices (such as physical microphones
and VB-Audio Virtual Cable interfaces). Uses available audio drivers
or system query tools.
"""

from __future__ import annotations

import sys


def list_devices() -> None:
    print("=" * 60)
    print("Audio Device Discovery")
    print("=" * 60)

    # Check if sounddevice or pyaudio is installed
    try:
        import sounddevice as sd  # type: ignore

        print("Querying via 'sounddevice' library:\n")
        devices = sd.query_devices()
        print(devices)
        return
    except ImportError:
        pass

    try:
        import pyaudio  # type: ignore

        print("Querying via 'pyaudio' library:\n")
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get("deviceCount", 0)
        for i in range(0, numdevices):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            max_in = device_info.get("maxInputChannels", 0)
            max_out = device_info.get("maxOutputChannels", 0)
            name = device_info.get("name")
            print(f"[{i:2d}] {name} (Inputs: {max_in}, Outputs: {max_out})")
        p.terminate()
        return
    except ImportError:
        pass

    print("Audio libraries ('sounddevice' or 'pyaudio') are not yet installed.")
    print("Audio backend selection and installation is scheduled for Milestone 1 & 3.")
    print("To enumerate Windows audio devices from PowerShell, run:")
    print("  Get-CimInstance Win32_SoundDevice | Select-Object Name, Status, DeviceID")


def main() -> None:
    list_devices()


if __name__ == "__main__":
    main()
