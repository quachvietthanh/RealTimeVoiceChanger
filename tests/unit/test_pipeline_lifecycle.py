"""Lifecycle and failure-handling tests for RealTimeVoiceChangerPipeline with fake streams (no devices, no GPU)."""

import sys
import threading
import time
import unittest
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np

from voice_changer.audio.noise_suppression import NoiseSuppressionSettings
from voice_changer.audio.pipeline import RealTimeVoiceChangerPipeline
from voice_changer.conversion.base import ConversionConfig

SR = 40000
CHUNK = 20000


class FakeInput:
    def __init__(self, fail_on_start=False):
        self.fail_on_start = fail_on_start
        self.callback = None
        self.active = False
        self.overflow_count = 0
        self.status_flag_count = 0
        self.starts = 0

    @property
    def is_active(self):
        return self.active

    def start(self, callback):
        self.starts += 1
        if self.fail_on_start:
            raise RuntimeError("PortAudio: invalid device")
        self.callback = callback
        self.active = True

    def stop(self):
        self.active = False

    def push(self, samples):
        self.callback(np.asarray(samples, dtype=np.float32)[:, np.newaxis])

    def push_chunks(self, n):
        # The pipeline emits at most one chunk per callback, as with real 2048-frame blocks.
        for i in range(n):
            self.push(np.full(CHUNK, float(i), dtype=np.float32))


class FakeOutput:
    def __init__(self):
        self.active = False
        self.chunks = []
        self.underflow_count = self.startup_underflow_count = self.steady_underflow_count = 0
        self.shutdown_underflow_count = self.overflow_count = self.status_flag_count = 0
        self.buffered_seconds = 0.0

    @property
    def is_active(self):
        return self.active

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def prepare_shutdown(self):
        pass

    def write_chunk(self, arr):
        if self.active:
            self.chunks.append(np.array(arr))


class FakeConverter:
    def __init__(self, fail=False, delay=0.0):
        self.fail = fail
        self.delay = delay
        self.configs = []

    def convert_chunk(self, audio_data, config=None, input_sr=SR, target_sr=None):
        self.configs.append(config)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return np.asarray(audio_data) * 0.5


def wait_for(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


def make(converter=None, inp=None, out=None, **kw):
    inp = inp or FakeInput()
    out = out or FakeOutput()
    p = RealTimeVoiceChangerPipeline(converter or FakeConverter(), input_stream=inp, output_stream=out, **kw)
    return p, inp, out


class PipelineLifecycleTests(unittest.TestCase):
    def test_converts_whole_chunks_with_exact_sample_count(self):
        p, inp, out = make()
        p.start()
        inp.push(np.ones(CHUNK + 500))
        self.assertTrue(wait_for(lambda: len(out.chunks) == 1))
        p.stop()
        self.assertEqual(len(out.chunks[0]), CHUNK)
        self.assertEqual(p.metrics.total_chunks_converted, 1)
        self.assertEqual(p.metrics.sample_count_mismatches, 0)

    def test_repeated_start_stop(self):
        p, inp, out = make()
        for _ in range(5):
            p.start()
            self.assertTrue(p.is_running)
            self.assertEqual(p.health(), [])
            p.stop()
            self.assertFalse(p.is_running)
            self.assertFalse(inp.active or out.active)
            self.assertIsNone(p._worker_thread)

    def test_stale_chunks_are_not_played_in_next_session(self):
        conv = FakeConverter(delay=0.05)
        p, inp, out = make(conv)
        p.start()
        p._running = False  # freeze the worker so chunks stay queued, then stop normally
        p._worker_thread.join()
        p._running = True
        inp.push_chunks(2)
        self.assertEqual(p._input_queue.qsize(), 2)
        p.stop()
        p.start()
        self.assertEqual(p._input_queue.qsize(), 0)
        time.sleep(0.15)
        p.stop()
        self.assertEqual(conv.configs, [])

    def test_failed_input_start_rolls_back(self):
        p, inp, out = make(inp=FakeInput(fail_on_start=True))
        with self.assertRaises(RuntimeError):
            p.start()
        self.assertFalse(p.is_running)
        self.assertFalse(out.active)
        self.assertIsNone(p._worker_thread)
        inp.fail_on_start = False
        p.start()  # recoverable
        self.assertTrue(p.is_running)
        p.stop()

    def test_overload_drops_oldest_and_counts_it(self):
        conv = FakeConverter(delay=0.3)
        p, inp, out = make(conv, max_queue_size=2)
        p.start()
        inp.push_chunks(6)
        self.assertGreaterEqual(p.metrics.total_chunks_dropped, 3)
        self.assertEqual(p.metrics.total_chunks_captured, 6)
        p.stop()

    def test_inference_errors_are_visible(self):
        p, inp, out = make(FakeConverter(fail=True))
        p.start()
        inp.push_chunks(3)
        self.assertTrue(wait_for(lambda: p.metrics.inference_errors == 3))
        p.stop()
        self.assertEqual(p.metrics.consecutive_inference_errors, 3)
        self.assertIn("CUDA out of memory", p.metrics.last_inference_error)
        self.assertIn("last_inference_error", p.metrics.summary())

    def test_health_reports_lost_stream(self):
        p, inp, out = make()
        p.start()
        inp.active = False  # device unplugged
        problems = p.health()
        p.stop()
        self.assertTrue(any("input audio stream" in x for x in problems))
        self.assertEqual(p.health(), [])  # stopped pipelines report nothing

    def test_update_config_applies_to_next_chunk(self):
        conv = FakeConverter()
        p, inp, out = make(conv, conversion_config=ConversionConfig(pitch_shift=12))
        p.start()
        inp.push(np.ones(CHUNK))
        self.assertTrue(wait_for(lambda: len(conv.configs) == 1))
        p.update_config(ConversionConfig(pitch_shift=16))
        inp.push(np.ones(CHUNK))
        self.assertTrue(wait_for(lambda: len(conv.configs) == 2))
        p.stop()
        self.assertEqual([c.pitch_shift for c in conv.configs], [12, 16])

    def test_stop_while_inference_in_progress_blocks_restart_until_done(self):
        conv = FakeConverter(delay=2.6)
        p, inp, out = make(conv)
        p.start()
        inp.push(np.ones(CHUNK))
        self.assertTrue(wait_for(lambda: len(conv.configs) == 1))
        p.stop()  # join times out after 2 s
        self.assertTrue(p.worker_stuck)
        with self.assertRaises(RuntimeError):
            p.start()
        self.assertTrue(wait_for(lambda: not p.worker_stuck, timeout=2.0))
        p.start()
        p.stop()

    def test_converter_gets_left_context_and_output_stays_one_chunk(self):
        seen = []

        class Recording(FakeConverter):
            def convert_chunk(self, audio_data, config=None, input_sr=SR, target_sr=None):
                seen.append(len(audio_data))
                return super().convert_chunk(audio_data, config, input_sr, target_sr)

        p, inp, out = make(Recording())
        p.start()
        inp.push_chunks(3)
        self.assertTrue(wait_for(lambda: len(out.chunks) == 3))
        p.stop()
        self.assertEqual(seen, [CHUNK + int(0.2 * SR)] * 3)  # default 200 ms context
        self.assertEqual([len(c) for c in out.chunks], [CHUNK] * 3)
        self.assertEqual(p.metrics.sample_count_mismatches, 0)
        self.assertEqual(p.metrics.algorithmic_delay_ms, 55.0)  # 50 ms crossfade + half the 10 ms search

    def test_dropped_chunk_resets_seam_state(self):
        conv = FakeConverter(delay=0.3)
        p, inp, out = make(conv, max_queue_size=1)
        p.start()
        inp.push_chunks(1)
        self.assertTrue(wait_for(lambda: len(conv.configs) == 1))  # worker is inside chunk 1
        inp.push_chunks(3)  # 2 and 3 are dropped from the 1-slot queue, 4 stays
        self.assertTrue(wait_for(lambda: p.metrics.total_chunks_converted == 2, timeout=3.0))
        p.stop()
        self.assertEqual(p.metrics.total_chunks_dropped, 2)
        self.assertEqual(p.metrics.stitch_resets, 1)

    def test_inference_error_resets_seam_state(self):
        p, inp, out = make(FakeConverter(fail=True))
        p.start()
        inp.push_chunks(2)
        self.assertTrue(wait_for(lambda: p.metrics.inference_errors == 2))
        p.stop()
        self.assertEqual(p.metrics.stitch_resets, 2)

    def test_noise_suppression_runs_on_worker_thread_not_in_callback(self):
        p, inp, out = make(noise_suppression=NoiseSuppressionSettings(enabled=True))
        threads = []
        original = p._noise_stage.process

        def recording(chunk):
            threads.append(threading.current_thread().name)
            return original(chunk)

        p._noise_stage.process = recording
        p.start()
        inp.push_chunks(2)  # push() runs the input callback on this (test) thread
        self.assertTrue(wait_for(lambda: len(out.chunks) == 2))
        p.stop()
        self.assertEqual(threads, ["RVCInferenceWorker"] * 2)
        self.assertEqual([len(c) for c in out.chunks], [CHUNK] * 2)
        s = p.metrics.summary()
        self.assertTrue(s["noise_suppression_enabled"])
        self.assertEqual(s["noise_suppression_latency_ms"], 20.0)
        self.assertEqual(s["algorithmic_delay_ms"], 75.0)
        self.assertGreater(s["max_noise_suppression_ms"], 0.0)

    def test_without_noise_stage_converter_input_is_unchanged(self):
        seen = []

        class Recording(FakeConverter):
            def convert_chunk(self, audio_data, config=None, input_sr=SR, target_sr=None):
                seen.append(np.array(audio_data))
                return super().convert_chunk(audio_data, config, input_sr, target_sr)

        p, inp, out = make(Recording(), context_duration_s=0.0, crossfade_duration_s=0.0, sola_search_s=0.0)
        p.start()
        inp.push(np.arange(CHUNK, dtype=np.float32))
        self.assertTrue(wait_for(lambda: len(out.chunks) == 1))
        p.stop()
        np.testing.assert_array_equal(seen[0], np.arange(CHUNK, dtype=np.float32))
        self.assertIsNone(p._noise_stage)

    def test_disabled_noise_stage_only_delays_converter_input(self):
        seen = []

        class Recording(FakeConverter):
            def convert_chunk(self, audio_data, config=None, input_sr=SR, target_sr=None):
                seen.append(np.array(audio_data))
                return super().convert_chunk(audio_data, config, input_sr, target_sr)

        p, inp, out = make(Recording(), context_duration_s=0.0, crossfade_duration_s=0.0, sola_search_s=0.0,
                           noise_suppression=NoiseSuppressionSettings(enabled=False))
        p.start()
        x = np.arange(1, 2 * CHUNK + 1, dtype=np.float32)
        inp.push(x[:CHUNK])
        inp.push(x[CHUNK:])
        self.assertTrue(wait_for(lambda: len(out.chunks) == 2))
        p.stop()
        lat = p._noise_stage.latency_samples
        np.testing.assert_array_equal(np.concatenate(seen)[lat:], x[:-lat])

    def test_dropped_chunk_resets_noise_stage_audio_but_keeps_noise_estimate(self):
        conv = FakeConverter(delay=0.3)
        p, inp, out = make(conv, max_queue_size=1, noise_suppression=NoiseSuppressionSettings(enabled=True))
        resets = []
        original = p._noise_stage.reset
        p._noise_stage.reset = lambda keep_noise_estimate=False: (resets.append(keep_noise_estimate),
                                                                  original(keep_noise_estimate))
        p.start()
        self.assertEqual(resets, [False])  # session start: full reset
        inp.push_chunks(1)
        self.assertTrue(wait_for(lambda: len(conv.configs) == 1))
        inp.push_chunks(3)
        self.assertTrue(wait_for(lambda: p.metrics.total_chunks_converted == 2, timeout=3.0))
        p.stop()
        self.assertEqual(resets, [False, True])

    def test_update_noise_suppression_needs_a_stage(self):
        p, inp, out = make()
        with self.assertRaises(RuntimeError):
            p.update_noise_suppression(NoiseSuppressionSettings(enabled=True))
        p2, _, _ = make(noise_suppression=NoiseSuppressionSettings())
        p2.update_noise_suppression(NoiseSuppressionSettings(enabled=True, mode="light"))
        self.assertEqual(p2._noise_stage.settings.mode, "light")

    def test_constructor_validation(self):
        for kw in ({"sample_rate": 0}, {"chunk_duration_s": 0.0}, {"max_queue_size": 0}, {"prebuffer_duration_s": -1},
                   {"context_duration_s": 0.01}, {"crossfade_duration_s": -0.1},
                   {"noise_suppression": NoiseSuppressionSettings(mode="max")},
                   {"expected_inference_s": -0.1}, {"expected_inference_s": float("nan")}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                make(**kw)


class CreditOutput(FakeOutput):
    """FakeOutput with the optional prebuffer-credit hook (0.15 s prebuffer)."""

    def __init__(self):
        super().__init__()
        self.credits = []

    def apply_prebuffer_credit(self, seconds):
        self.credits.append((seconds, len(self.chunks)))  # and how many chunks were written before
        return min(seconds, 0.15)


def make_credit(converter, **kw):
    """Pipeline with CreditOutput and a FakeInput that reports 2048-frame blocks (next chunk >= 460.8 ms)."""
    p, inp, out = make(converter, out=CreditOutput(), **kw)
    inp.block_size = 2048
    return p, inp, out


class StartupLatencyTests(unittest.TestCase):
    def test_late_first_chunk_is_credited_once_before_its_write(self):
        conv = FakeConverter(delay=0.3)
        p, inp, out = make_credit(conv, expected_inference_s=0.1)
        p.start()
        inp.push_chunks(3)
        self.assertTrue(wait_for(lambda: len(out.chunks) == 3, timeout=3.0))
        p.stop()
        self.assertEqual(len(out.credits), 1)
        lateness, written_before = out.credits[0]
        self.assertEqual(written_before, 0)
        self.assertGreaterEqual(lateness, 0.3 - 0.1 - 0.05 - 0.005)
        self.assertLess(lateness, 0.3 - 0.1 - 0.05 + 0.05)
        self.assertAlmostEqual(p.metrics.prebuffer_credit_ms, 150.0)
        self.assertEqual(p.metrics.expected_inference_ms, 100.0)

    def test_first_chunk_within_tolerance_keeps_the_full_prebuffer(self):
        # Processing is a little above expected (sleep + overhead), but well inside the 50 ms tolerance.
        p, inp, out = make_credit(FakeConverter(delay=0.1), expected_inference_s=0.1)
        p.start()
        inp.push_chunks(2)
        self.assertTrue(wait_for(lambda: len(out.chunks) == 2))
        p.stop()
        self.assertEqual(out.credits, [])
        self.assertEqual(p.metrics.prebuffer_credit_ms, 0.0)

    def test_first_chunk_finished_after_the_next_could_be_ready_is_not_credited(self):
        # 0.5 s > 9 input blocks (460.8 ms): the worker is behind, so the full prebuffer is kept.
        p, inp, out = make_credit(FakeConverter(delay=0.5), expected_inference_s=0.1)
        p.start()
        inp.push_chunks(1)
        self.assertTrue(wait_for(lambda: len(out.chunks) == 1, timeout=3.0))
        p.stop()
        self.assertEqual((out.credits, p.metrics.prebuffer_credit_ms), ([], 0.0))

    def test_input_without_block_size_is_not_credited(self):
        p, inp, out = make(FakeConverter(delay=0.3), out=CreditOutput(), expected_inference_s=0.1)
        p.start()
        inp.push_chunks(1)
        self.assertTrue(wait_for(lambda: len(out.chunks) == 1, timeout=3.0))
        p.stop()
        self.assertEqual(out.credits, [])

    def test_without_expected_inference_nothing_is_credited(self):
        p, inp, out = make_credit(FakeConverter(delay=0.3))
        p.start()
        inp.push_chunks(1)
        self.assertTrue(wait_for(lambda: len(out.chunks) == 1))
        p.stop()
        self.assertEqual(out.credits, [])

    def test_streams_without_the_hook_still_work(self):
        p, inp, out = make(FakeConverter(delay=0.3), expected_inference_s=0.0)
        p.start()
        inp.push_chunks(2)
        self.assertTrue(wait_for(lambda: len(out.chunks) == 2))
        p.stop()
        self.assertEqual((p.metrics.inference_errors, p.metrics.prebuffer_credit_ms), (0, 0.0))

    def test_each_session_gets_its_own_credit(self):
        p, inp, out = make_credit(FakeConverter(delay=0.3), expected_inference_s=0.1)
        for session in (1, 2):
            p.start()
            inp.push_chunks(1)
            self.assertTrue(wait_for(lambda: len(out.chunks) == session, timeout=3.0))
            p.stop()
            self.assertAlmostEqual(p.metrics.prebuffer_credit_ms, 150.0)
        self.assertEqual(len(out.credits), 2)

    def test_first_five_chunks_are_recorded_and_logged_once(self):
        p, inp, out = make(FakeConverter(delay=0.01), noise_suppression=NoiseSuppressionSettings())
        p.start()
        with self.assertLogs("voice_changer.audio.pipeline", level="INFO") as logs:
            for n in range(1, 8):  # one at a time: a full queue would drop chunks
                inp.push(np.zeros(CHUNK, dtype=np.float32))
                self.assertTrue(wait_for(lambda: len(out.chunks) == n))
        p.stop()
        self.assertEqual(p.metrics.total_chunks_dropped, 0)
        rows = p.metrics.startup_chunks
        self.assertEqual([r["seq"] for r in rows], [1, 2, 3, 4, 5])
        for r in rows:
            self.assertGreaterEqual(r["inference_ms"], 10.0)
            self.assertIsNotNone(r["noise_suppression_ms"])
            self.assertGreaterEqual(r["software_e2e_ms"], r["turnaround_ms"])
            self.assertIn("buffer_before_write_ms", r)
            self.assertIn("buffer_after_write_ms", r)
        self.assertEqual(sum("startup chunks" in line for line in logs.output), 1)
        summary = p.metrics.summary()
        self.assertEqual(summary["startup_chunks"], rows)
        self.assertIsNot(summary["startup_chunks"][0], rows[0])  # a copy, safe to hand to other threads


if __name__ == "__main__":
    unittest.main()
