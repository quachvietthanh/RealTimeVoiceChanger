"""Latency and performance measurement tools.

Tracks chunk ingestion, inference duration, crossfading, and output delays.
Used during Milestone 6 to profile and optimize real-time streaming constraints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class LatencyStats:
    """Summary of measured latencies over a sampling window."""

    sample_count: int
    mean_ms: float
    min_ms: float
    max_ms: float
    underrun_count: int


@dataclass
class LatencyMonitor:
    """Records timing intervals for audio chunks in the conversion pipeline."""

    window_size: int = 100
    _inference_times: List[float] = field(default_factory=list)
    _underruns: int = 0

    def record_inference(self, duration_seconds: float) -> None:
        """Record the elapsed time taken for a single conversion chunk."""
        self._inference_times.append(duration_seconds * 1000.0)
        if len(self._inference_times) > self.window_size:
            self._inference_times.pop(0)

    def record_underrun(self) -> None:
        """Record an audio buffer starvation/underrun event."""
        self._underruns += 1

    def get_stats(self) -> LatencyStats:
        """Calculate latency statistics over the current sampling window."""
        if not self._inference_times:
            return LatencyStats(
                sample_count=0,
                mean_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                underrun_count=self._underruns,
            )

        return LatencyStats(
            sample_count=len(self._inference_times),
            mean_ms=sum(self._inference_times) / len(self._inference_times),
            min_ms=min(self._inference_times),
            max_ms=max(self._inference_times),
            underrun_count=self._underruns,
        )

    def reset(self) -> None:
        """Clear recorded latency measurements."""
        self._inference_times.clear()
        self._underruns = 0
