"""Audio output stream playback using sounddevice.

Provides non-blocking audio playback to physical speakers or virtual audio cables.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioOutputStream:
    """Manages audio playback to a selected output endpoint."""

    def __init__(
        self,
        device_id: Optional[int] = None,
        sample_rate: int = 40000,
        channels: int = 2,
        block_size: int = 2048,
        max_buffer_seconds: float = 4.0,
        prebuffer_seconds: float = 0.15,
    ) -> None:
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.max_buffer_samples = int(sample_rate * max_buffer_seconds)
        self.prebuffer_seconds = max(float(prebuffer_seconds), 0.0)

        self._stream: Optional[sd.OutputStream] = None
        self._is_active = False
        self._lock = threading.Lock()
        self._buffer: np.ndarray = np.zeros((0, self.channels), dtype=np.float32)

        self._underflow_count = 0
        self._startup_underflow_count = 0
        self._steady_underflow_count = 0
        self._shutdown_underflow_count = 0
        self._first_chunk_received = False
        self._first_chunk_time: Optional[float] = None
        self._playback_started = False
        self._is_stopping = False
        self._overflow_count = 0
        self._steady_underflow_timestamps: list[float] = []

    @property
    def is_active(self) -> bool:
        return self._is_active and self._stream is not None and self._stream.active

    @property
    def playback_started(self) -> bool:
        with self._lock:
            return self._playback_started

    @property
    def underflow_count(self) -> int:
        return self._underflow_count

    @property
    def startup_underflow_count(self) -> int:
        return self._startup_underflow_count

    @property
    def steady_underflow_count(self) -> int:
        return self._steady_underflow_count

    @property
    def shutdown_underflow_count(self) -> int:
        return self._shutdown_underflow_count

    @property
    def steady_underflow_timestamps(self) -> list[float]:
        with self._lock:
            return list(self._steady_underflow_timestamps)

    @property
    def overflow_count(self) -> int:
        return self._overflow_count

    @property
    def buffered_samples(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def buffered_seconds(self) -> float:
        with self._lock:
            return len(self._buffer) / self.sample_rate

    def _classify_underflow(self) -> None:
        """Classify underrun into startup silence, steady-state interruption, or shutdown drain."""
        self._underflow_count += 1
        if not self._playback_started:
            self._startup_underflow_count += 1
        elif self._is_stopping:
            self._shutdown_underflow_count += 1
        else:
            self._steady_underflow_count += 1
            if len(self._steady_underflow_timestamps) < 100:
                self._steady_underflow_timestamps.append(time.perf_counter())

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("Audio output status flag: %s", status)

        with self._lock:
            if not self._playback_started:
                if self._is_stopping:
                    self._playback_started = True
                elif not self._first_chunk_received:
                    outdata.fill(0.0)
                    self._startup_underflow_count += 1
                    self._underflow_count += 1
                    return
                else:
                    elapsed = (
                        time.perf_counter() - self._first_chunk_time
                        if self._first_chunk_time is not None
                        else 0.0
                    )
                    if elapsed < self.prebuffer_seconds:
                        outdata.fill(0.0)
                        self._startup_underflow_count += 1
                        self._underflow_count += 1
                        return
                    self._playback_started = True

            available = len(self._buffer)
            if available >= frames:
                outdata[:] = self._buffer[:frames]
                self._buffer = self._buffer[frames:]
            elif available > 0:
                outdata[:available] = self._buffer
                outdata[available:].fill(0.0)
                self._buffer = np.zeros((0, self.channels), dtype=np.float32)
                self._classify_underflow()
            else:
                outdata.fill(0.0)
                self._classify_underflow()

    def start(self) -> None:
        """Initialize and open the output audio stream."""
        if self._is_active:
            return

        with self._lock:
            self._buffer = np.zeros((0, self.channels), dtype=np.float32)
            self._first_chunk_received = False
            self._first_chunk_time = None
            self._playback_started = False
            self._is_stopping = False
            self._underflow_count = 0
            self._startup_underflow_count = 0
            self._steady_underflow_count = 0
            self._shutdown_underflow_count = 0
            self._overflow_count = 0
            self._steady_underflow_timestamps = []

        self._stream = sd.OutputStream(
            device=self.device_id,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        self._is_active = True

    def write_chunk(self, audio_data: np.ndarray) -> None:
        """Enqueue converted audio samples (float32, 1D mono or 2D stereo)."""
        if not self._is_active:
            return

        arr = np.asarray(audio_data, dtype=np.float32)
        if arr.ndim == 1:
            if self.channels == 2:
                arr = np.column_stack([arr, arr])
            else:
                arr = arr[:, np.newaxis]
        elif arr.ndim == 2:
            if arr.shape[1] == 1 and self.channels == 2:
                arr = np.column_stack([arr[:, 0], arr[:, 0]])
            elif arr.shape[1] > self.channels:
                arr = arr[:, :self.channels]

        with self._lock:
            if not self._first_chunk_received:
                self._first_chunk_received = True
                self._first_chunk_time = time.perf_counter()
                if self.prebuffer_seconds <= 0.0:
                    self._playback_started = True
            self._buffer = np.vstack([self._buffer, arr])
            if len(self._buffer) > self.max_buffer_samples:
                excess = len(self._buffer) - self.max_buffer_samples
                self._buffer = self._buffer[excess:]
                self._overflow_count += 1

    def write(self, audio_data: bytes) -> None:
        """Write raw audio bytes (PCM float32)."""
        arr = np.frombuffer(audio_data, dtype=np.float32)
        self.write_chunk(arr)

    def prepare_shutdown(self) -> None:
        """Mark output stream as entering shutdown phase so drain silence is classified properly."""
        with self._lock:
            self._is_stopping = True
            self._playback_started = True

    def stop(self) -> None:
        """Flush and close the output audio stream."""
        if not self._is_active:
            return
        self.prepare_shutdown()
        self._is_active = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as err:
                logger.debug("Error stopping output stream: %s", err)
            finally:
                self._stream = None
