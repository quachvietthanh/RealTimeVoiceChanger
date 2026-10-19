"""Real-time voice changer audio processing pipeline.

Orchestrates non-blocking microphone capture, bounded worker-thread inference,
and continuous output streaming without blocking real-time audio callbacks.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np

from voice_changer.audio.input_stream import AudioInputStream
from voice_changer.audio.output_stream import AudioOutputStream
from voice_changer.conversion.base import ConversionConfig, VoiceConverter

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Telemetry and performance metrics for the real-time pipeline."""

    total_chunks_captured: int = 0
    total_chunks_converted: int = 0
    total_chunks_dropped: int = 0
    input_overflow_count: int = 0
    output_underflow_count: int = 0
    startup_underflow_count: int = 0
    steady_underflow_count: int = 0
    shutdown_underflow_count: int = 0
    inference_errors: int = 0
    sample_count_mismatches: int = 0
    expected_chunk_samples: int = 0
    last_converted_chunk_samples: int = 0

    inference_times_ms: list[float] = field(default_factory=list)
    queue_wait_times_ms: list[float] = field(default_factory=list)
    turnaround_times_ms: list[float] = field(default_factory=list)
    software_e2e_times_ms: list[float] = field(default_factory=list)
    output_buffer_occupancy_ms: list[float] = field(default_factory=list)
    speech_onset_latency_ms: float = 0.0

    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def session_duration_s(self) -> float:
        if self.start_time == 0.0:
            return 0.0
        finish = self.end_time if self.end_time > 0.0 else time.perf_counter()
        return finish - self.start_time

    @property
    def avg_inference_ms(self) -> float:
        return float(np.mean(self.inference_times_ms)) if self.inference_times_ms else 0.0

    @property
    def median_inference_ms(self) -> float:
        return float(np.median(self.inference_times_ms)) if self.inference_times_ms else 0.0

    @property
    def min_inference_ms(self) -> float:
        return float(np.min(self.inference_times_ms)) if self.inference_times_ms else 0.0

    @property
    def max_inference_ms(self) -> float:
        return float(np.max(self.inference_times_ms)) if self.inference_times_ms else 0.0

    @property
    def p95_inference_ms(self) -> float:
        return float(np.percentile(self.inference_times_ms, 95)) if self.inference_times_ms else 0.0

    @property
    def avg_queue_wait_ms(self) -> float:
        return float(np.mean(self.queue_wait_times_ms)) if self.queue_wait_times_ms else 0.0

    @property
    def median_queue_wait_ms(self) -> float:
        return float(np.median(self.queue_wait_times_ms)) if self.queue_wait_times_ms else 0.0

    @property
    def p95_queue_wait_ms(self) -> float:
        return float(np.percentile(self.queue_wait_times_ms, 95)) if self.queue_wait_times_ms else 0.0

    @property
    def avg_software_e2e_ms(self) -> float:
        return float(np.mean(self.software_e2e_times_ms)) if self.software_e2e_times_ms else 0.0

    @property
    def median_software_e2e_ms(self) -> float:
        return float(np.median(self.software_e2e_times_ms)) if self.software_e2e_times_ms else 0.0

    @property
    def p95_software_e2e_ms(self) -> float:
        return float(np.percentile(self.software_e2e_times_ms, 95)) if self.software_e2e_times_ms else 0.0

    @property
    def avg_buffer_occupancy_ms(self) -> float:
        return float(np.mean(self.output_buffer_occupancy_ms)) if self.output_buffer_occupancy_ms else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "session_duration_s": round(self.session_duration_s, 2),
            "chunks_captured": self.total_chunks_captured,
            "chunks_converted": self.total_chunks_converted,
            "chunks_dropped": self.total_chunks_dropped,
            "input_overflows": self.input_overflow_count,
            "output_underflows": self.output_underflow_count,
            "startup_underflows": self.startup_underflow_count,
            "steady_underflows": self.steady_underflow_count,
            "shutdown_underflows": self.shutdown_underflow_count,
            "inference_errors": self.inference_errors,
            "expected_chunk_samples": self.expected_chunk_samples,
            "last_converted_chunk_samples": self.last_converted_chunk_samples,
            "sample_count_mismatches": self.sample_count_mismatches,
            "avg_inference_ms": round(self.avg_inference_ms, 1),
            "median_inference_ms": round(self.median_inference_ms, 1),
            "min_inference_ms": round(self.min_inference_ms, 1),
            "max_inference_ms": round(self.max_inference_ms, 1),
            "p95_inference_ms": round(self.p95_inference_ms, 1),
            "avg_queue_wait_ms": round(self.avg_queue_wait_ms, 2),
            "median_queue_wait_ms": round(self.median_queue_wait_ms, 2),
            "p95_queue_wait_ms": round(self.p95_queue_wait_ms, 2),
            "speech_onset_latency_ms": round(self.speech_onset_latency_ms, 1),
            "avg_software_e2e_ms": round(self.avg_software_e2e_ms, 1),
            "median_software_e2e_ms": round(self.median_software_e2e_ms, 1),
            "p95_software_e2e_ms": round(self.p95_software_e2e_ms, 1),
            "avg_buffer_occupancy_ms": round(self.avg_buffer_occupancy_ms, 1),
        }


class RealTimeVoiceChangerPipeline:
    """Coordinates real-time audio capture, asynchronous inference, and playback."""

    def __init__(
        self,
        converter: VoiceConverter,
        conversion_config: Optional[ConversionConfig] = None,
        input_device_id: Optional[int] = None,
        output_device_id: Optional[int] = None,
        sample_rate: int = 40000,
        chunk_duration_s: float = 0.5,
        max_queue_size: int = 3,
        prebuffer_duration_s: float = 0.15,
    ) -> None:
        self.converter = converter
        self.config = conversion_config or ConversionConfig()
        self.input_device_id = input_device_id
        self.output_device_id = output_device_id
        self.sample_rate = sample_rate
        self.chunk_duration_s = chunk_duration_s
        self.chunk_samples = int(sample_rate * chunk_duration_s)
        self.prebuffer_duration_s = prebuffer_duration_s

        self._input_queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
        self._accumulator: list[np.ndarray] = []
        self._accumulated_samples: int = 0
        self._accum_lock = threading.Lock()
        self._chunk_first_sample_t: Optional[float] = None

        self.input_stream = AudioInputStream(
            device_id=self.input_device_id,
            sample_rate=self.sample_rate,
            block_size=2048,
            channels=1,
        )
        self.output_stream = AudioOutputStream(
            device_id=self.output_device_id,
            sample_rate=self.sample_rate,
            channels=2,
            block_size=2048,
            prebuffer_seconds=self.prebuffer_duration_s,
        )

        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        self.metrics = PipelineMetrics()

    @property
    def is_running(self) -> bool:
        return self._running

    def _on_audio_input(self, block: np.ndarray) -> None:
        """Lightweight non-blocking callback invoked by the input audio thread."""
        if not self._running:
            return

        with self._accum_lock:
            samples_1d = block[:, 0] if block.ndim > 1 else block

            if self._accumulated_samples == 0:
                self._chunk_first_sample_t = time.perf_counter()

            self._accumulator.append(samples_1d)
            self._accumulated_samples += len(samples_1d)

            if self._accumulated_samples >= self.chunk_samples:
                full_chunk = np.concatenate(self._accumulator)
                ready_chunk = full_chunk[: self.chunk_samples]
                remainder = full_chunk[self.chunk_samples :]

                t_first = self._chunk_first_sample_t or time.perf_counter()
                t_ready = time.perf_counter()

                self._accumulator = [remainder] if len(remainder) > 0 else []
                self._accumulated_samples = len(remainder)
                self._chunk_first_sample_t = time.perf_counter() if len(remainder) > 0 else None

                self.metrics.total_chunks_captured += 1
                item = (ready_chunk, t_first, t_ready)

                try:
                    self._input_queue.put_nowait(item)
                except queue.Full:
                    try:
                        _ = self._input_queue.get_nowait()
                        self.metrics.total_chunks_dropped += 1
                        self._input_queue.put_nowait(item)
                        logger.warning("Input queue backlog: dropped oldest chunk to preserve real-time latency")
                    except Exception:
                        pass

    def _worker_loop(self) -> None:
        """Background thread executing neural model inference."""
        logger.info("Real-time inference worker thread started.")
        while self._running:
            try:
                item = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if isinstance(item, tuple) and len(item) == 3:
                    chunk, t_first, t_ready = item
                else:
                    chunk = item
                    t_first = t_ready = time.perf_counter()

                t_dequeue = time.perf_counter()
                queue_wait_ms = (t_dequeue - t_ready) * 1000.0
                self.metrics.queue_wait_times_ms.append(queue_wait_ms)

                chunk_arr = np.asarray(chunk, dtype=np.float32)

                t0 = time.perf_counter()
                if hasattr(self.converter, "convert_chunk"):
                    converted = self.converter.convert_chunk(
                        chunk_arr,
                        config=self.config,
                        input_sr=self.sample_rate,
                        target_sr=self.sample_rate,
                    )
                else:
                    raw_bytes = chunk_arr.tobytes()
                    out_bytes = self.converter.convert(raw_bytes, self.config)
                    converted = np.frombuffer(out_bytes, dtype=np.float32)

                infer_duration_ms = (time.perf_counter() - t0) * 1000.0
                self.metrics.inference_times_ms.append(infer_duration_ms)
                self.metrics.total_chunks_converted += 1

                if self.chunk_samples > 0:
                    self.metrics.expected_chunk_samples = self.chunk_samples
                    self.metrics.last_converted_chunk_samples = len(converted)
                    if len(converted) != self.chunk_samples:
                        self.metrics.sample_count_mismatches += 1

                self.output_stream.write_chunk(converted)
                t_written = time.perf_counter()

                turnaround_ms = (t_written - t_ready) * 1000.0
                e2e_ms = (t_written - t_first) * 1000.0
                self.metrics.turnaround_times_ms.append(turnaround_ms)
                self.metrics.software_e2e_times_ms.append(e2e_ms)
                if self.metrics.speech_onset_latency_ms == 0.0:
                    self.metrics.speech_onset_latency_ms = e2e_ms
                self.metrics.output_buffer_occupancy_ms.append(self.output_stream.buffered_seconds * 1000.0)

            except Exception as err:
                self.metrics.inference_errors += 1
                logger.error("Error during real-time chunk conversion: %s", err)
            finally:
                self._input_queue.task_done()

        logger.info("Real-time inference worker thread terminated.")

    def start(self) -> None:
        """Start the real-time audio pipeline."""
        if self._running:
            return

        logger.info("Starting real-time voice changer pipeline...")
        self.metrics = PipelineMetrics()
        self.metrics.start_time = time.perf_counter()

        with self._accum_lock:
            self._accumulator = []
            self._accumulated_samples = 0
            self._chunk_first_sample_t = None

        self.output_stream.start()

        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="RVCInferenceWorker")
        self._worker_thread.start()

        self.input_stream.start(self._on_audio_input)
        logger.info("Pipeline started successfully.")

    def stop(self) -> None:
        """Stop audio streaming and shutdown background worker."""
        if not self._running:
            return

        logger.info("Stopping real-time voice changer pipeline...")
        self.input_stream.stop()

        self.output_stream.prepare_shutdown()

        self._running = False
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

        self.output_stream.stop()

        self.metrics.end_time = time.perf_counter()
        self.metrics.input_overflow_count = self.input_stream.overflow_count
        self.metrics.output_underflow_count = self.output_stream.underflow_count
        self.metrics.startup_underflow_count = self.output_stream.startup_underflow_count
        self.metrics.steady_underflow_count = self.output_stream.steady_underflow_count
        self.metrics.shutdown_underflow_count = self.output_stream.shutdown_underflow_count
        logger.info("Pipeline stopped cleanly.")
