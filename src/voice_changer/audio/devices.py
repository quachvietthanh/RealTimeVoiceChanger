"""Audio device management and enumeration.

FUTURE COMPONENT: To be implemented in Milestone 3 once an audio backend
(e.g., sounddevice or pyaudio) is selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


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


class AudioDeviceManager:
    """Manages audio endpoint discovery and selection."""

    def list_input_devices(self) -> List[AudioDevice]:
        """Enumerate available microphone and input devices."""
        raise NotImplementedError("Audio device discovery will be implemented in Milestone 3.")

    def list_output_devices(self) -> List[AudioDevice]:
        """Enumerate available speakers and virtual cable output devices."""
        raise NotImplementedError("Audio device discovery will be implemented in Milestone 3.")

    def find_device_by_name(self, name_pattern: str, is_input: bool = True) -> Optional[AudioDevice]:
        """Find a device whose name matches the given pattern (e.g. 'CABLE Input')."""
        raise NotImplementedError("Audio device filtering will be implemented in Milestone 3.")
