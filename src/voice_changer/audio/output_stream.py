"""Audio output stream playback.

FUTURE COMPONENT: Audio output routing (e.g. to VB-Audio Virtual Cable),
to be implemented in Milestone 3.
"""

from __future__ import annotations

from typing import Optional


class AudioOutputStream:
    """Manages audio playback to a selected output endpoint."""

    def __init__(
        self,
        device_id: Optional[int] = None,
        sample_rate: int = 40000,
        channels: int = 1,
    ) -> None:
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.channels = channels
        self._is_active = False

    @property
    def is_active(self) -> bool:
        return self._is_active

    def start(self) -> None:
        """Initialize and open the output audio stream."""
        raise NotImplementedError("Real-time output streaming will be implemented in Milestone 3.")

    def write(self, audio_data: bytes) -> None:
        """Write a block of audio data to the output device."""
        raise NotImplementedError("Real-time output streaming will be implemented in Milestone 3.")

    def stop(self) -> None:
        """Flush and close the output audio stream."""
        raise NotImplementedError("Real-time output streaming will be implemented in Milestone 3.")
