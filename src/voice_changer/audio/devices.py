"""Audio device management and enumeration.

Provides hardware audio device discovery and selection using sounddevice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import sounddevice as sd

# Name fragments of virtual audio cable endpoints (VB-Audio Virtual Cable / Hi-Fi Cable / VoiceMeeter).
# The playback side of VB-Cable is named "CABLE Input"; apps such as Discord record from "CABLE Output".
VIRTUAL_CABLE_MARKERS = ("vb-audio", "cable input", "cable output", "hi-fi cable", "voicemeeter")


@dataclass(frozen=True)
class AudioDevice:
    """Represents a physical or virtual audio endpoint."""

    device_id: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0

    @property
    def is_virtual_cable(self) -> bool:
        """Heuristic, name-based: True for VB-Audio / VoiceMeeter style virtual endpoints."""
        name = self.name.lower()
        return any(m in name for m in VIRTUAL_CABLE_MARKERS)


class AudioDeviceManager:
    """Manages audio endpoint discovery and selection."""

    def __init__(self) -> None:
        try:
            self._host_apis = {i: api["name"] for i, api in enumerate(sd.query_hostapis())}
        except Exception:
            self._host_apis = {}

    def _query_all_devices(self) -> List[AudioDevice]:
        devices = []
        raw_devices = sd.query_devices()
        for i, d in enumerate(raw_devices):
            host_api_name = self._host_apis.get(d.get("hostapi", 0), f"API {d.get('hostapi', 0)}")
            devices.append(
                AudioDevice(
                    device_id=i,
                    name=str(d.get("name", f"Device {i}")),
                    host_api=host_api_name,
                    max_input_channels=int(d.get("max_input_channels", 0)),
                    max_output_channels=int(d.get("max_output_channels", 0)),
                    default_sample_rate=float(d.get("default_samplerate", 44100.0)),
                )
            )
        return devices

    def list_input_devices(self) -> List[AudioDevice]:
        """Enumerate available microphone and input devices."""
        return [d for d in self._query_all_devices() if d.is_input]

    def list_output_devices(self) -> List[AudioDevice]:
        """Enumerate available speakers and virtual cable output devices."""
        return [d for d in self._query_all_devices() if d.is_output]

    def get_device(self, device_id: int) -> Optional[AudioDevice]:
        """Return the device with this PortAudio index, or None if it does not exist now.

        Indices are only valid for the current device enumeration; they can change when devices are
        plugged in or removed. Persist (name, host_api) instead, and resolve with find_device().
        """
        for d in self._query_all_devices():
            if d.device_id == device_id:
                return d
        return None

    def find_device(self, name: str, host_api: Optional[str], is_input: bool) -> Optional[AudioDevice]:
        """Find a device by exact name (and host API, if given): the stable way to re-select a device."""
        candidates = self.list_input_devices() if is_input else self.list_output_devices()
        for d in candidates:
            if d.name == name and (host_api is None or d.host_api == host_api):
                return d
        return None

    def check_device(self, device_id: Optional[int], is_input: bool, sample_rate: int, channels: int) -> Optional[str]:
        """Return why the device cannot stream with these settings, or None if PortAudio accepts them.

        `device_id=None` checks the system default device. Does not open a stream.
        """
        kind = "input" if is_input else "output"
        if device_id is not None:
            d = self.get_device(device_id)
            if d is None:
                return f"{kind} device {device_id} does not exist (devices may have been re-enumerated)"
            available = d.max_input_channels if is_input else d.max_output_channels
            if available < 1:
                return f"device {device_id} '{d.name}' has no {kind} channels"
        check = sd.check_input_settings if is_input else sd.check_output_settings
        try:
            check(device=device_id, channels=channels, samplerate=sample_rate, dtype="float32")
        except Exception as err:
            return f"{kind} device {device_id if device_id is not None else '(default)'} rejects {sample_rate} Hz, {channels} ch: {err}"
        return None

    def find_device_by_name(self, name_pattern: str, is_input: bool = True) -> Optional[AudioDevice]:
        """Find a device whose name matches the given pattern (case-insensitive)."""
        pattern_lower = name_pattern.lower()
        candidates = self.list_input_devices() if is_input else self.list_output_devices()
        for d in candidates:
            if pattern_lower in d.name.lower():
                return d
        return None

    def get_default_device(self, is_input: bool = True) -> Optional[AudioDevice]:
        """Retrieve system default audio input or output device."""
        try:
            kind = "input" if is_input else "output"
            d = sd.query_devices(kind=kind)
            device_id = int(d.get("index", 0))
            host_api_name = self._host_apis.get(d.get("hostapi", 0), "Default API")
            return AudioDevice(
                device_id=device_id,
                name=str(d.get("name", "Default Device")),
                host_api=host_api_name,
                max_input_channels=int(d.get("max_input_channels", 0)),
                max_output_channels=int(d.get("max_output_channels", 0)),
                default_sample_rate=float(d.get("default_samplerate", 44100.0)),
            )
        except Exception:
            return None
