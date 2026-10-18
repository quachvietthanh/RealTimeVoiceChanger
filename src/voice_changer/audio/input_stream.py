"""Audio input stream capture.

FUTURE COMPONENT: Real-time microphone capture interface, to be implemented
in Milestone 3.
"""

from __future__ import annotations

from typing import Callable, Optional


class AudioInputStream:
    """Manages audio capture from a selected input device."""

    def __init__(
        self,
        device_id: Optional[int] = None,
        sample_rate: int = 40000,
        block_size: int = 2048,
        channels: int = 1,
    ) -> None:
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels
        self._is_active = False

    @property
    def is_active(self) -> bool:
        return self._is_active

    def start(self, callback: Callable[[bytes], None]) -> None:
        """Start capturing audio blocks and dispatching them to the callback."""
        raise NotImplementedError("Real-time input streaming will be implemented in Milestone 3.")

    def stop(self) -> None:
        """Stop capturing audio."""
        raise NotImplementedError("Real-time input streaming will be implemented in Milestone 3.")
