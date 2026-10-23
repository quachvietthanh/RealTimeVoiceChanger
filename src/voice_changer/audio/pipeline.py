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

from voice_changer.audio.crossfade import ChunkStitcher
from voice_changer.audio.input_stream import AudioInputStream
from voice_changer.audio.noise_suppression import NoiseSuppressionSettings, NoiseSuppressionStage
from voice_changer.audio.output_stream import AudioOutputStream
from voice_changer.conversion.base import ConversionConfig, VoiceConverter

logger = logging.getLogger(__name__)

# Per-chunk timings are kept (and logged once) for this many chunks at the start of a session.
STARTUP_DIAGNOSTIC_CHUNKS = 5
# First-chunk processing up to this much slower than expected keeps the full prebuffer (normal jitter;
# measured p95 - median inference is ~15-25 ms).
STARTUP_LATENESS_TOLERANCE_S = 0.05


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
    consecutive_inference_errors: int = 0
    last_inference_error: str = ""
    output_overflow_count: int = 0
    input_status_flags: int = 0
    output_status_flags: int = 0
    sample_count_mismatches: int = 0
    expected_chunk_samples: int = 0
    last_converted_chunk_samples: int = 0
    stitch_resets: int = 0
    algorithmic_delay_ms: float = 0.0
    noise_suppression_times_ms: list[float] = field(default_factory=list)
    noise_suppression_enabled: bool = False
    noise_suppression_mode: str = ""
    noise_suppression_active: bool = False
    noise_suppression_failed: bool = False
    noise_suppression_errors: int = 0
    noise_suppression_last_error: str = ""
    noise_suppression_latency_ms: float = 0.0

    inference_times_ms: list[float] = field(default_factory=list)
    queue_wait_times_ms: list[float] = field(default_factory=list)
    turnaround_times_ms: list[float] = field(default_factory=list)
    software_e2e_times_ms: list[float] = field(default_factory=list)
    output_buffer_occupancy_ms: list[float] = field(default_factory=list)
    speech_onset_latency_ms: float = 0.0
    # First STARTUP_DIAGNOSTIC_CHUNKS converted chunks: software timings and playback-buffer level.
    startup_chunks: list[dict[str, Any]] = field(default_factory=list)
    expected_inference_ms: float = 0.0
    prebuffer_credit_ms: float = 0.0

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

    @staticmethod
    def _pct(values: list[float], q: float) -> float:
        return float(np.percentile(values, q)) if values else 0.0

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
            "consecutive_inference_errors": self.consecutive_inference_errors,
            "last_inference_error": self.last_inference_error,
            "output_overflows": self.output_overflow_count,
            "input_status_flags": self.input_status_flags,
            "output_status_flags": self.output_status_flags,
            "expected_chunk_samples": self.expected_chunk_samples,
            "last_converted_chunk_samples": self.last_converted_chunk_samples,
            "sample_count_mismatches": self.sample_count_mismatches,
            "avg_inference_ms": round(self.avg_inference_ms, 1),
            "median_inference_ms": round(self.median_inference_ms, 1),
            "min_inference_ms": round(self.min_inference_ms, 1),
            "max_inference_ms": round(self.max_inference_ms, 1),
            "p95_inference_ms": round(self.p95_inference_ms, 1),
            "p99_inference_ms": round(self._pct(self.inference_times_ms, 99), 1),
            "avg_queue_wait_ms": round(self.avg_queue_wait_ms, 2),
            "median_queue_wait_ms": round(self.median_queue_wait_ms, 2),
            "p95_queue_wait_ms": round(self.p95_queue_wait_ms, 2),
            "p99_queue_wait_ms": round(self._pct(self.queue_wait_times_ms, 99), 2),
            "max_queue_wait_ms": round(max(self.queue_wait_times_ms, default=0.0), 2),
            "speech_onset_latency_ms": round(self.speech_onset_latency_ms, 1),
            "startup_chunks": [dict(c) for c in self.startup_chunks],
            "expected_inference_ms": round(self.expected_inference_ms, 1),
            "prebuffer_credit_ms": round(self.prebuffer_credit_ms, 1),
            "avg_software_e2e_ms": round(self.avg_software_e2e_ms, 1),
            "median_software_e2e_ms": round(self.median_software_e2e_ms, 1),
            "p95_software_e2e_ms": round(self.p95_software_e2e_ms, 1),
            "p99_software_e2e_ms": round(self._pct(self.software_e2e_times_ms, 99), 1),
            "max_software_e2e_ms": round(max(self.software_e2e_times_ms, default=0.0), 1),
            "avg_buffer_occupancy_ms": round(self.avg_buffer_occupancy_ms, 1),
            "p95_buffer_occupancy_ms": round(self._pct(self.output_buffer_occupancy_ms, 95), 1),
            "max_buffer_occupancy_ms": round(max(self.output_buffer_occupancy_ms, default=0.0), 1),
            "stitch_resets": self.stitch_resets,
            "algorithmic_delay_ms": round(self.algorithmic_delay_ms, 1),
            "noise_suppression_enabled": self.noise_suppression_enabled,
            "noise_suppression_mode": self.noise_suppression_mode,
            "noise_suppression_active": self.noise_suppression_active,
            "noise_suppression_failed": self.noise_suppression_failed,
            "noise_suppression_errors": self.noise_suppression_errors,
            "noise_suppression_last_error": self.noise_suppression_last_error,
            "noise_suppression_latency_ms": round(self.noise_suppression_latency_ms, 1),
            "avg_noise_suppression_ms": round(float(np.mean(self.noise_suppression_times_ms))
                                              if self.noise_suppression_times_ms else 0.0, 2),
            "p95_noise_suppression_ms": round(self._pct(self.noise_suppression_times_ms, 95), 2),
            "p99_noise_suppression_ms": round(self._pct(self.noise_suppression_times_ms, 99), 2),
            "max_noise_suppression_ms": round(max(self.noise_suppression_times_ms, default=0.0), 2),
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
        input_stream: Optional[AudioInputStream] = None,
        output_stream: Optional[AudioOutputStream] = None,
        context_duration_s: float = 0.2,
        crossfade_duration_s: float = 0.05,
        sola_search_s: float = 0.01,
        noise_suppression: Optional[NoiseSuppressionSettings] = None,
        expected_inference_s: Optional[float] = None,
    ) -> None:
        """Create a stopped pipeline.

        `input_stream` / `output_stream` default to sounddevice streams on the given devices; tests
        and simulations may pass objects with the same interface.

        Chunk seams: each model call also sees `context_duration_s` of the previous input, and
        consecutive outputs are joined with a `crossfade_duration_s` SOLA crossfade (see
        audio/crossfade.py). This adds about crossfade + search/2 of delay plus the inference cost of
        the context. All three 0 restores plain end-to-end joining, which leaves a seam at every chunk
        boundary.

        Noise suppression: with `noise_suppression` settings (the service always passes them) the worker
        runs NoiseSuppressionStage on each captured chunk before inference. The stage always delays the
        audio by one 20 ms frame, on or off, so it can be toggled live without shifting the timeline.
        None means no stage at all (the exact behaviour before noise suppression existed).

        Startup: with continuous playback every sample has the latency of the first chunk, so a slow
        first chunk (e.g. the GPU waking up after idling) would add its lateness to the whole session.
        With `expected_inference_s` (steady-state conversion time of one chunk), the first chunk's
        processing time beyond expected + STARTUP_LATENESS_TOLERANCE_S is taken off the output stream's
        prebuffer wait (at most the prebuffer minus one output block), so playback starts as if it had
        been on time. Only when the first chunk finished before the second could be ready; otherwise the
        worker is behind and the full prebuffer is kept. No audio is discarded; lateness beyond the credit
        remains. None keeps the full prebuffer.

        Raises:
            ValueError: If the sample rate, chunk duration, queue size, prebuffer, seam lengths, noise
                suppression settings or expected inference time are invalid.
        """
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if int(sample_rate * chunk_duration_s) <= 0:
            raise ValueError(f"chunk_duration_s must give at least one sample, got {chunk_duration_s}")
        if max_queue_size < 1:
            raise ValueError(f"max_queue_size must be >= 1, got {max_queue_size}")
        if prebuffer_duration_s < 0:
            raise ValueError(f"prebuffer_duration_s must be >= 0, got {prebuffer_duration_s}")
        if expected_inference_s is not None and not (np.isfinite(expected_inference_s) and expected_inference_s >= 0):
            raise ValueError(f"expected_inference_s must be a finite value >= 0, got {expected_inference_s}")
        self.converter = converter
        self.expected_inference_s = expected_inference_s
        self.config = conversion_config or ConversionConfig()
        self.input_device_id = input_device_id
        self.output_device_id = output_device_id
        self.sample_rate = sample_rate
        self.chunk_duration_s = chunk_duration_s
        self.chunk_samples = int(sample_rate * chunk_duration_s)
        self.prebuffer_duration_s = prebuffer_duration_s
        self._stitcher = ChunkStitcher(
            self.chunk_samples,
            context_samples=int(round(sample_rate * context_duration_s)),
            crossfade_samples=int(round(sample_rate * crossfade_duration_s)),
            search_samples=int(round(sample_rate * sola_search_s)),
        )
        self._noise_stage: Optional[NoiseSuppressionStage] = (
            NoiseSuppressionStage(sample_rate, noise_suppression) if noise_suppression is not None else None
        )

        self._input_queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
        self._accumulator: list[np.ndarray] = []
        self._accumulated_samples: int = 0
        self._accum_lock = threading.Lock()
        self._chunk_first_sample_t: Optional[float] = None
        self._chunk_seq = 0

        self.input_stream = input_stream if input_stream is not None else AudioInputStream(
            device_id=self.input_device_id,
            sample_rate=self.sample_rate,
            block_size=2048,
            channels=1,
        )
        self.output_stream = output_stream if output_stream is not None else AudioOutputStream(
            device_id=self.output_device_id,
            sample_rate=self.sample_rate,
            channels=2,
            block_size=2048,
            prebuffer_seconds=self.prebuffer_duration_s,
        )

        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stuck = False

        self.metrics = PipelineMetrics()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def worker_stuck(self) -> bool:
        """True if the last stop() returned while the worker was still inside an inference call."""
        return self._worker_stuck and self._worker_thread is not None and self._worker_thread.is_alive()

    def wait_for_worker(self, timeout: float) -> bool:
        """Wait for a worker left running by stop() (see worker_stuck). True when no worker is running."""
        thread = self._worker_thread
        if thread is not None and not self._running:
            thread.join(timeout=timeout)
            if thread.is_alive():
                return False
            self._worker_thread = None
        return self._worker_thread is None or self._running

    def update_config(self, config: ConversionConfig) -> None:
        """Replace the conversion settings; takes effect from the next chunk.

        The worker reads `self.config` once per chunk and the reference is replaced atomically, so a
        chunk never sees a half-updated config. The caller must not mutate `config` afterwards.
        """
        self.config = config

    def update_noise_suppression(self, settings: NoiseSuppressionSettings) -> None:
        """Change noise suppression; applied from the next chunk with a short ramp.

        Raises:
            RuntimeError: the pipeline was built without a noise suppression stage.
            ValueError: invalid settings.
        """
        if self._noise_stage is None:
            raise RuntimeError("pipeline was created without a noise suppression stage")
        self._noise_stage.update(settings)

    def health(self) -> list[str]:
        """Problems detected while running (empty list if healthy, or if stopped).

        Detects a stream that stopped on its own (device unplugged, driver error) and a worker
        thread that died. Cheap; safe to poll from another thread.
        """
        if not self._running:
            return []
        problems = []
        if not self.input_stream.is_active:
            problems.append("input audio stream is no longer active (device disconnected or driver error)")
        if not self.output_stream.is_active:
            problems.append("output audio stream is no longer active (device disconnected or driver error)")
        if self._worker_thread is None or not self._worker_thread.is_alive():
            problems.append("inference worker thread is not running")
        return problems

    def refresh_stream_counters(self) -> None:
        """Copy the audio streams' event counters into `metrics` (called on stop and on demand)."""
        self.metrics.input_overflow_count = self.input_stream.overflow_count
        self.metrics.input_status_flags = getattr(self.input_stream, "status_flag_count", 0)
        self.metrics.output_underflow_count = self.output_stream.underflow_count
        self.metrics.startup_underflow_count = self.output_stream.startup_underflow_count
        self.metrics.steady_underflow_count = self.output_stream.steady_underflow_count
        self.metrics.shutdown_underflow_count = self.output_stream.shutdown_underflow_count
        self.metrics.output_overflow_count = self.output_stream.overflow_count
        self.metrics.output_status_flags = getattr(self.output_stream, "status_flag_count", 0)
        stage = self._noise_stage
        if stage is not None:
            settings = stage.settings
            self.metrics.noise_suppression_enabled = settings.enabled
            self.metrics.noise_suppression_mode = settings.mode
            self.metrics.noise_suppression_active = stage.active
            self.metrics.noise_suppression_failed = stage.failed
            self.metrics.noise_suppression_errors = stage.errors
            self.metrics.noise_suppression_last_error = stage.last_error

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
                self._chunk_seq += 1
                item = (ready_chunk, t_first, t_ready, self._chunk_seq)

                # Overload policy: drop the OLDEST queued chunk so latency stays bounded. Every drop
                # is counted in metrics.total_chunks_dropped. No logging here: this is the audio thread.
                try:
                    self._input_queue.put_nowait(item)
                except queue.Full:
                    try:
                        _ = self._input_queue.get_nowait()
                        self._input_queue.task_done()
                        self.metrics.total_chunks_dropped += 1
                        self._input_queue.put_nowait(item)
                    except (queue.Empty, queue.Full):
                        self.metrics.total_chunks_dropped += 1

    def _worker_loop(self) -> None:
        """Background thread executing neural model inference."""
        logger.info("Real-time inference worker thread started.")
        last_seq: Optional[int] = None
        while self._running:
            try:
                item = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                seq: Optional[int] = None
                if isinstance(item, tuple) and len(item) == 4:
                    chunk, t_first, t_ready, seq = item
                elif isinstance(item, tuple) and len(item) == 3:
                    chunk, t_first, t_ready = item
                else:
                    chunk = item
                    t_first = t_ready = time.perf_counter()

                t_dequeue = time.perf_counter()
                queue_wait_ms = (t_dequeue - t_ready) * 1000.0
                self.metrics.queue_wait_times_ms.append(queue_wait_ms)

                chunk_arr = np.asarray(chunk, dtype=np.float32)
                # A dropped chunk breaks the timeline: context or a held tail from before the gap must
                # not be blended into the audio after it.
                if seq is not None and last_seq is not None and seq != last_seq + 1:
                    self._stitcher.reset()
                    self.metrics.stitch_resets += 1
                    if self._noise_stage is not None:
                        self._noise_stage.reset(keep_noise_estimate=True)
                last_seq = seq
                ns_ms: Optional[float] = None
                if self._noise_stage is not None:
                    t_ns = time.perf_counter()
                    chunk_arr = self._noise_stage.process(chunk_arr)
                    ns_ms = (time.perf_counter() - t_ns) * 1000.0
                    self.metrics.noise_suppression_times_ms.append(ns_ms)
                model_in = self._stitcher.model_input(chunk_arr)

                t0 = time.perf_counter()
                if hasattr(self.converter, "convert_chunk"):
                    converted = self.converter.convert_chunk(
                        model_in,
                        config=self.config,
                        input_sr=self.sample_rate,
                        target_sr=self.sample_rate,
                    )
                else:
                    raw_bytes = model_in.tobytes()
                    out_bytes = self.converter.convert(raw_bytes, self.config)
                    converted = np.frombuffer(out_bytes, dtype=np.float32)

                infer_duration_ms = (time.perf_counter() - t0) * 1000.0
                self.metrics.inference_times_ms.append(infer_duration_ms)
                self.metrics.total_chunks_converted += 1
                self.metrics.consecutive_inference_errors = 0

                # Length check is on the raw converter output; join() then emits exactly one chunk.
                self.metrics.expected_chunk_samples = len(model_in)
                self.metrics.last_converted_chunk_samples = len(converted)
                if len(converted) != len(model_in):
                    self.metrics.sample_count_mismatches += 1
                converted = self._stitcher.join(converted)

                startup = len(self.metrics.startup_chunks) < STARTUP_DIAGNOSTIC_CHUNKS
                buffer_before_ms = self.output_stream.buffered_seconds * 1000.0 if startup else 0.0
                if self.metrics.total_chunks_converted == 1:
                    self._apply_prebuffer_credit(time.perf_counter() - t_ready)
                self.output_stream.write_chunk(converted)
                t_written = time.perf_counter()

                turnaround_ms = (t_written - t_ready) * 1000.0
                e2e_ms = (t_written - t_first) * 1000.0
                self.metrics.turnaround_times_ms.append(turnaround_ms)
                self.metrics.software_e2e_times_ms.append(e2e_ms)
                if self.metrics.speech_onset_latency_ms == 0.0:
                    self.metrics.speech_onset_latency_ms = e2e_ms
                buffer_after_ms = self.output_stream.buffered_seconds * 1000.0
                self.metrics.output_buffer_occupancy_ms.append(buffer_after_ms)
                if startup:
                    self._record_startup_chunk(seq, queue_wait_ms, ns_ms, infer_duration_ms, turnaround_ms,
                                               e2e_ms, buffer_before_ms, buffer_after_ms)

            except Exception as err:
                # The chunk is lost; the output buffer covers the gap or underflows (counted).
                self._stitcher.reset()
                self.metrics.stitch_resets += 1
                self.metrics.inference_errors += 1
                self.metrics.consecutive_inference_errors += 1
                self.metrics.last_inference_error = f"{type(err).__name__}: {err}"
                logger.error("Error during real-time chunk conversion: %s", err, exc_info=True)
            finally:
                self._input_queue.task_done()

        logger.info("Real-time inference worker thread terminated.")

    def _apply_prebuffer_credit(self, processing_s: float) -> None:
        """Called before the session's first write: offset a late first chunk against the prebuffer.

        Withheld unless the first chunk is written before the second can be ready: input arrives in whole
        blocks, so that is at the earliest floor(chunk / input block) blocks after the first. A later first
        chunk means the worker is already behind, and that backlog is carried by every following chunk;
        only the full prebuffer absorbs it.
        """
        if self.expected_inference_s is None:
            return
        block = getattr(self.input_stream, "block_size", None)
        if not isinstance(block, int) or block <= 0:
            return  # unknown input timing: keep the full prebuffer
        next_ready_s = (self.chunk_samples // block) * block / self.sample_rate
        if not processing_s < next_ready_s:
            return
        lateness_s = processing_s - self.expected_inference_s - STARTUP_LATENESS_TOLERANCE_S
        # Optional part of the output-stream interface; streams without it keep the full prebuffer.
        apply = getattr(self.output_stream, "apply_prebuffer_credit", None)
        if lateness_s > 0.0 and apply is not None:
            self.metrics.prebuffer_credit_ms = float(apply(lateness_s)) * 1000.0

    def _record_startup_chunk(self, seq: Optional[int], queue_wait_ms: float, ns_ms: Optional[float],
                              inference_ms: float, turnaround_ms: float, e2e_ms: float,
                              buffer_before_ms: float, buffer_after_ms: float) -> None:
        """Keep one startup chunk's software timings; log all of them once the set is complete."""
        self.metrics.startup_chunks.append({
            "seq": seq, "queue_wait_ms": round(queue_wait_ms, 2),
            "noise_suppression_ms": None if ns_ms is None else round(ns_ms, 2),
            "inference_ms": round(inference_ms, 1), "turnaround_ms": round(turnaround_ms, 1),
            "software_e2e_ms": round(e2e_ms, 1), "buffer_before_write_ms": round(buffer_before_ms, 1),
            "buffer_after_write_ms": round(buffer_after_ms, 1),
        })
        if len(self.metrics.startup_chunks) == STARTUP_DIAGNOSTIC_CHUNKS:
            logger.info("startup chunks (software timings; expected inference %.1f ms, prebuffer credit %.1f ms): %s",
                        self.metrics.expected_inference_ms, self.metrics.prebuffer_credit_ms,
                        self.metrics.startup_chunks)

    def start(self) -> None:
        """Start the real-time audio pipeline."""
        if self._running:
            return

        if self._worker_thread is not None and self._worker_thread.is_alive():
            raise RuntimeError("previous inference worker is still finishing a chunk; retry start() shortly")

        logger.info("Starting real-time voice changer pipeline...")
        self.metrics = PipelineMetrics()
        self.metrics.start_time = time.perf_counter()
        self.metrics.expected_inference_ms = (self.expected_inference_s or 0.0) * 1000.0
        ns_latency = self._noise_stage.latency_samples if self._noise_stage is not None else 0
        self.metrics.noise_suppression_latency_ms = ns_latency / self.sample_rate * 1000.0
        self.metrics.algorithmic_delay_ms = (self._stitcher.nominal_delay_samples + ns_latency) / self.sample_rate * 1000.0
        self._stitcher.reset()
        if self._noise_stage is not None:
            self._noise_stage.reset()

        with self._accum_lock:
            self._accumulator = []
            self._accumulated_samples = 0
            self._chunk_first_sample_t = None
            self._chunk_seq = 0
        # Chunks left from a previous session must not be played in this one.
        while True:
            try:
                self._input_queue.get_nowait()
                self._input_queue.task_done()
            except queue.Empty:
                break
        self._worker_stuck = False

        # Roll back whatever already started if a later step fails (e.g. invalid input device).
        try:
            self.output_stream.start()
            self._running = True
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="RVCInferenceWorker")
            self._worker_thread.start()
            self.input_stream.start(self._on_audio_input)
        except BaseException:
            self._running = False
            self.input_stream.stop()
            self.output_stream.stop()
            if self._worker_thread is not None:
                self._worker_thread.join(timeout=2.0)
                self._worker_thread = None
            self.metrics.end_time = time.perf_counter()
            raise
        logger.info("Pipeline started successfully.")

    def stop(self) -> None:
        """Stop audio streaming and shutdown background worker."""
        if not self._running:
            return

        logger.info("Stopping real-time voice changer pipeline...")
        self.input_stream.stop()

        self.output_stream.prepare_shutdown()

        self._running = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            if self._worker_thread.is_alive():
                # A chunk is still in inference; the thread exits after it. start() refuses until then.
                self._worker_stuck = True
                logger.error("Inference worker did not finish within 2 s of stop().")
            else:
                self._worker_thread = None

        self.output_stream.stop()

        self.metrics.end_time = time.perf_counter()
        self.refresh_stream_counters()
        logger.info("Pipeline stopped%s.", " (worker still finishing a chunk)" if self._worker_stuck else "")
