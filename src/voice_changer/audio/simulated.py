"""Device-free audio streams for headless tests and simulated streaming runs.

SimulatedInputStream plays an array into the pipeline in real-time blocks, like a microphone.
SimulatedOutputStream is the real AudioOutputStream ring buffer, prebuffering and underflow
accounting, driven by a clock thread instead of a PortAudio device. Neither opens a sound device,
so results describe the software pipeline only, never physical audio latency.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np

from voice_changer.audio.output_stream import AudioOutputStream


class SimulatedInputStream:
    """Delivers `audio` to the callback in blocks of `block_size`, paced at `sample_rate` (real time)."""

    def __init__(self, audio: np.ndarray, sample_rate: int = 40000, block_size: int = 2048) -> None:
        arr = np.asarray(audio, dtype=np.float32)
        self._audio = arr.mean(axis=1) if arr.ndim > 1 else arr
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.overflow_count = 0
        self.status_flag_count = 0
        self.finished = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fail_after_blocks: Optional[int] = None

    @property
    def is_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def simulate_device_loss(self, after_blocks: int) -> None:
        """Make the stream stop by itself after `after_blocks` blocks, as an unplugged device does."""
        self._fail_after_blocks = after_blocks

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        if self.is_active:
            return
        self._stop.clear()
        self.finished.clear()

        def run() -> None:
            # After the audio runs out, keep delivering silence (like an idle microphone) until stop(),
            # so the stream stays active and the last partial chunk is completed. `finished` marks the end.
            t0 = time.perf_counter()
            n = -(-len(self._audio) // self.block_size)  # the last partial block is zero-padded, not dropped
            padded = np.zeros(n * self.block_size, dtype=np.float32)
            padded[: len(self._audio)] = self._audio
            silence = np.zeros((self.block_size, 1), dtype=np.float32)
            i = 0
            while not self._stop.is_set():
                if self._fail_after_blocks is not None and i >= self._fail_after_blocks:
                    break
                if i < n:
                    callback(padded[i * self.block_size : (i + 1) * self.block_size][:, np.newaxis].copy())
                else:
                    self.finished.set()
                    callback(silence.copy())
                i += 1
                delay = t0 + i * self.block_size / self.sample_rate - time.perf_counter()
                if delay > 0:
                    self._stop.wait(delay)
            self.finished.set()

        self._thread = threading.Thread(target=run, daemon=True, name="SimulatedInput")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class SimulatedOutputStream(AudioOutputStream):
    """AudioOutputStream whose callback is driven by a real-time clock thread instead of a device.

    If `capture` is True, every block handed to the "device" is kept for inspection (`captured()`).
    """

    def __init__(self, *args, capture: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._capture = capture
        self._blocks: list[np.ndarray] = []
        self._clock: Optional[threading.Thread] = None
        self._clock_stop = threading.Event()

    @property
    def is_active(self) -> bool:
        return self._is_active and self._clock is not None and self._clock.is_alive()

    def start(self) -> None:
        if self._is_active:
            return
        self._reset_state()
        self._blocks = []
        self._clock_stop.clear()

        def run() -> None:
            t0 = time.perf_counter()
            i = 0
            out = np.zeros((self.block_size, self.channels), dtype=np.float32)
            while not self._clock_stop.is_set():
                self._audio_callback(out, self.block_size, {}, None)  # type: ignore[arg-type]
                if self._capture:
                    self._blocks.append(out.copy())
                i += 1
                delay = t0 + i * self.block_size / self.sample_rate - time.perf_counter()
                if delay > 0:
                    self._clock_stop.wait(delay)

        self._is_active = True
        self._clock = threading.Thread(target=run, daemon=True, name="SimulatedOutputClock")
        self._clock.start()

    def stop(self) -> None:
        if not self._is_active:
            return
        self.prepare_shutdown()
        self._is_active = False
        self._clock_stop.set()
        if self._clock is not None:
            self._clock.join(timeout=2.0)
            self._clock = None

    def captured(self) -> np.ndarray:
        """All blocks played so far, concatenated (frames x channels)."""
        if not self._blocks:
            return np.zeros((0, self.channels), dtype=np.float32)
        return np.concatenate(self._blocks)
