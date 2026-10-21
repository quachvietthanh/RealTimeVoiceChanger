"""Audio input stream capture using sounddevice.

Manages non-blocking real-time audio capture from microphone or input device.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


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
        self._stream: Optional[sd.InputStream] = None
        self._is_active = False
        self._overflow_count = 0
        self._status_flag_count = 0
        self._device_latency_s: Optional[float] = None

    @property
    def is_active(self) -> bool:
        return self._is_active and self._stream is not None and self._stream.active

    @property
    def overflow_count(self) -> int:
        return self._overflow_count

    @property
    def device_latency_s(self) -> Optional[float]:
        """Input latency PortAudio reported for the last opened stream (None if never opened)."""
        return self._device_latency_s

    @property
    def status_flag_count(self) -> int:
        """Callbacks that reported any PortAudio status flag (overflow included)."""
        return self._status_flag_count

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        """Start capturing audio blocks and dispatching them to the callback."""
        if self._is_active:
            return

        def _audio_callback(indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
            # Counters only: logging can block, and this runs on the PortAudio thread.
            if status:
                self._status_flag_count += 1
                if status.input_overflow:
                    self._overflow_count += 1
            callback(indata.copy())

        self._overflow_count = 0
        self._status_flag_count = 0
        stream = sd.InputStream(
            device=self.device_id,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
            callback=_audio_callback,
        )
        try:
            stream.start()
        except BaseException:
            stream.close()
            raise
        self._stream = stream
        self._device_latency_s = float(stream.latency)
        self._is_active = True

    def stop(self) -> None:
        """Stop capturing audio."""
        if not self._is_active:
            return
        self._is_active = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as err:
                logger.debug("Error stopping input stream: %s", err)
            finally:
                self._stream = None
