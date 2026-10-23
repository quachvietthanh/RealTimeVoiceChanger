"""Device-free audio streams for headless tests and simulated streaming runs.

SimulatedInputStream plays an array into the pipeline in real-time blocks, like a microphone.
SimulatedOutputStream is the real AudioOutputStream ring buffer, prebuffering and underflow
accounting, driven by a clock thread instead of a PortAudio device. Neither opens a sound device,
so results describe the software pipeline only, never physical audio latency.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Optional

import numpy as np

from voice_changer.audio.output_stream import AudioOutputStream


def next_tick(t0: float, period_s: float, now: float) -> tuple[int, float]:
    """First tick of the absolute schedule t0 + k * period_s (k >= 1) that is strictly after `now`.

    For periodic sampling that does not drift: a late iteration skips the ticks it missed instead of
    shifting every later sample by its own lateness.
    """
    if period_s <= 0:
        raise ValueError(f"period_s must be positive, got {period_s}")
    k = max(int((now - t0) // period_s) + 1, 1)
    return k, t0 + k * period_s


class WriteTelemetry:
    """Bounded per-write record of a SimulatedOutputStream (the oldest rows are evicted).

    One row per write_chunk(): session write index, time since the stream started, buffer level just
    before and after the write, output callbacks that ran between those two reads (0: the pair is exact),
    cumulative underflow/overflow counters, plus whatever `probe` returns (e.g. queue depth, drops).
    `probe` runs on the writing thread, so it must be cheap and must not block.
    """

    def __init__(self, max_rows: int = 4096, probe: Optional[Callable[[], dict[str, Any]]] = None) -> None:
        if max_rows < 1:
            raise ValueError(f"max_rows must be >= 1, got {max_rows}")
        self.probe = probe
        self.rows: deque[dict[str, Any]] = deque(maxlen=max_rows)
        self.evicted = 0
        self._writes = 0

    def reset(self) -> None:
        self.rows.clear()
        self.evicted = 0
        self._writes = 0

    def record(self, row: dict[str, Any]) -> None:
        self._writes += 1
        if len(self.rows) == self.rows.maxlen:
            self.evicted += 1
        self.rows.append({"index": self._writes, **row})

    def summary(self) -> dict[str, Any]:
        """Buffer levels around writes and the latest counters; `before_trend_ms` is the mean level
        before a write in the last quarter of the rows minus the first quarter (drift, not oscillation)."""
        rows = list(self.rows)
        out: dict[str, Any] = {"rows": len(rows), "evicted": self.evicted,
                               "inexact_rows": sum(r["callbacks_during_write"] > 0 for r in rows)}
        if not rows:
            return out
        for key in ("buffer_before_ms", "buffer_after_ms"):
            v = [r[key] for r in rows]
            out[key] = {"min": round(min(v), 1), "mean": round(float(np.mean(v)), 1), "max": round(max(v), 1)}
        q = max(len(rows) // 4, 1)
        before = [r["buffer_before_ms"] for r in rows]
        out["before_trend_ms"] = round(float(np.mean(before[-q:]) - np.mean(before[:q])), 1)
        if "queue_depth" in rows[0]:
            out["max_queue_depth"] = max(r["queue_depth"] for r in rows)
        out["last"] = {k: v for k, v in rows[-1].items() if k.endswith(("_count", "_dropped"))}
        return out


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
    With `telemetry`, every write is recorded there (reset at start()); the callback is not instrumented.
    """

    def __init__(self, *args, capture: bool = False, telemetry: Optional[WriteTelemetry] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._capture = capture
        self.telemetry = telemetry
        self._blocks: list[np.ndarray] = []
        self._clock: Optional[threading.Thread] = None
        self._clock_stop = threading.Event()
        self._callbacks = 0  # device callbacks run this session (clock thread only)
        self._t_start = 0.0

    @property
    def is_active(self) -> bool:
        return self._is_active and self._clock is not None and self._clock.is_alive()

    def start(self) -> None:
        if self._is_active:
            return
        self._reset_state()
        self._blocks = []
        self._clock_stop.clear()
        self._callbacks = 0
        self._t_start = time.perf_counter()
        if self.telemetry is not None:
            self.telemetry.reset()

        def run() -> None:
            t0 = time.perf_counter()
            i = 0
            out = np.zeros((self.block_size, self.channels), dtype=np.float32)
            while not self._clock_stop.is_set():
                self._audio_callback(out, self.block_size, {}, None)  # type: ignore[arg-type]
                self._callbacks += 1
                if self._capture:
                    self._blocks.append(out.copy())
                i += 1
                delay = t0 + i * self.block_size / self.sample_rate - time.perf_counter()
                if delay > 0:
                    self._clock_stop.wait(delay)

        self._is_active = True
        self._clock = threading.Thread(target=run, daemon=True, name="SimulatedOutputClock")
        self._clock.start()

    def write_chunk(self, audio_data: np.ndarray) -> None:
        tel = self.telemetry
        if tel is None or not self._is_active:
            super().write_chunk(audio_data)
            return
        # Two separate lock acquisitions: a callback can run in between (counted, row marked inexact).
        callbacks = self._callbacks
        before = self.buffered_seconds
        super().write_chunk(audio_data)
        after = self.buffered_seconds
        row = {
            "t_s": round(time.perf_counter() - self._t_start, 4),
            "buffer_before_ms": round(before * 1000.0, 1), "buffer_after_ms": round(after * 1000.0, 1),
            "callbacks_during_write": self._callbacks - callbacks,
            "startup_underflow_count": self.startup_underflow_count,
            "steady_underflow_count": self.steady_underflow_count,
            "shutdown_underflow_count": self.shutdown_underflow_count,
            "overflow_count": self.overflow_count,
        }
        if tel.probe is not None:
            row.update(tel.probe())
        tel.record(row)

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
