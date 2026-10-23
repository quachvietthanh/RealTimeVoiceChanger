"""Unit tests verifying AudioOutputStream ring-buffer and PipelineMetrics telemetry."""

import sys
import unittest
from pathlib import Path

# Ensure src/ is on sys.path
src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
from voice_changer.audio.output_stream import AudioOutputStream
from voice_changer.audio.pipeline import PipelineMetrics


class TestAudioOutputStream(unittest.TestCase):
    def test_output_stream_initialization(self):
        stream = AudioOutputStream(sample_rate=40000, channels=2, block_size=2048)
        self.assertEqual(stream.sample_rate, 40000)
        self.assertEqual(stream.channels, 2)
        self.assertEqual(stream.underflow_count, 0)
        self.assertEqual(stream.startup_underflow_count, 0)
        self.assertEqual(stream.steady_underflow_count, 0)
        self.assertEqual(stream.shutdown_underflow_count, 0)

    def test_underflow_classification(self):
        stream = AudioOutputStream(sample_rate=40000, channels=2, block_size=2048)

        # 1. Before first chunk is written: startup underflows
        stream._classify_underflow()
        stream._classify_underflow()
        self.assertEqual(stream.startup_underflow_count, 2)
        self.assertEqual(stream.steady_underflow_count, 0)
        self.assertEqual(stream.shutdown_underflow_count, 0)
        self.assertEqual(stream.underflow_count, 2)

        # 2. Write a chunk: marks stream as active streaming
        dummy = np.zeros(2048, dtype=np.float32)
        stream._first_chunk_received = True
        stream._playback_started = True

        # Steady-state underflow (e.g. buffer starved during active speech)
        stream._classify_underflow()
        self.assertEqual(stream.startup_underflow_count, 2)
        self.assertEqual(stream.steady_underflow_count, 1)
        self.assertEqual(stream.shutdown_underflow_count, 0)
        self.assertEqual(stream.underflow_count, 3)

        # 3. Enter shutdown: marks stream as stopping
        stream.prepare_shutdown()
        stream._classify_underflow()
        stream._classify_underflow()
        self.assertEqual(stream.startup_underflow_count, 2)
        self.assertEqual(stream.steady_underflow_count, 1)
        self.assertEqual(stream.shutdown_underflow_count, 2)
        self.assertEqual(stream.underflow_count, 5)

    def test_prebuffering_logic(self):
        # With prebuffer_seconds = 0.15s
        stream = AudioOutputStream(sample_rate=40000, channels=2, block_size=2048, prebuffer_seconds=0.15)
        self.assertFalse(stream.playback_started)

        # Write first chunk
        chunk = np.zeros(20000, dtype=np.float32)
        stream._is_active = True
        stream.write_chunk(chunk)
        self.assertTrue(stream._first_chunk_received)
        # Should not start playback immediately if prebuffer > 0
        self.assertFalse(stream.playback_started)

        # Prebuffer with 0.0s should start immediately
        stream_zero = AudioOutputStream(sample_rate=40000, channels=2, block_size=2048, prebuffer_seconds=0.0)
        stream_zero._is_active = True
        stream_zero.write_chunk(chunk)
        self.assertTrue(stream_zero.playback_started)

    @staticmethod
    def _pull(stream, frames=2048):
        out = np.full((frames, stream.channels), -1.0, dtype=np.float32)
        stream._audio_callback(out, frames, {}, None)
        return out

    def test_prebuffer_credit_shortens_only_the_startup_wait(self):
        stream = AudioOutputStream(sample_rate=40000, channels=2, block_size=2048, prebuffer_seconds=0.15)
        plain = AudioOutputStream(sample_rate=40000, channels=2, block_size=2048, prebuffer_seconds=0.15)
        stream._is_active = plain._is_active = True
        self.assertAlmostEqual(stream.apply_prebuffer_credit(0.09), 0.09)
        for s in (stream, plain):
            s.write_chunk(np.ones(20000, dtype=np.float32))
            self.assertFalse(s.playback_started)
            s._first_chunk_time -= 0.07  # 70 ms after the first write
        # Within the full 0.15 s prebuffer, past the shortened 0.06 s one.
        np.testing.assert_array_equal(self._pull(plain), 0.0)
        self.assertFalse(plain.playback_started)
        np.testing.assert_array_equal(self._pull(stream), 1.0)
        self.assertTrue(stream.playback_started)
        self.assertAlmostEqual(stream.prebuffer_credit_s, 0.09)

    def test_prebuffer_credit_is_clamped_and_only_accepted_before_the_first_chunk(self):
        stream = AudioOutputStream(sample_rate=40000, channels=2, block_size=2048, prebuffer_seconds=0.15)
        stream._is_active = True
        self.assertEqual(stream.apply_prebuffer_credit(-1.0), 0.0)
        self.assertEqual(stream.apply_prebuffer_credit(float("nan")), 0.0)
        # Never more than the prebuffer minus one 2048-frame callback block (51.2 ms reserve).
        self.assertAlmostEqual(stream.apply_prebuffer_credit(5.0), 0.15 - 2048 / 40000)
        stream.write_chunk(np.ones(2048, dtype=np.float32))
        self.assertFalse(stream.playback_started)  # the reserve still has to elapse
        stream._first_chunk_time += 1.0  # pin "just written" regardless of scheduling delays
        np.testing.assert_array_equal(self._pull(stream), 0.0)
        stream._first_chunk_time -= 1.0 + 2048 / 40000
        np.testing.assert_array_equal(self._pull(stream), 1.0)
        self.assertEqual(stream.apply_prebuffer_credit(0.05), 0.0)
        self.assertAlmostEqual(stream.prebuffer_credit_s, 0.15 - 2048 / 40000)
        stream._reset_state()  # new session
        self.assertEqual(stream.prebuffer_credit_s, 0.0)
        self.assertFalse(stream.playback_started)

    def test_prebuffer_credit_keeps_every_sample_in_order(self):
        stream = AudioOutputStream(sample_rate=40000, channels=1, block_size=2048, prebuffer_seconds=0.15)
        stream._is_active = True
        stream.apply_prebuffer_credit(0.15)
        ramp = np.arange(1, 50001, dtype=np.float32)
        for part in (ramp[:20000], ramp[20000:40000], ramp[40000:]):
            stream.write_chunk(part)
        stream._first_chunk_time -= 0.06  # past the 51.2 ms that remain of the prebuffer
        played = np.concatenate([self._pull(stream) for _ in range(25)])[:, 0]  # 51,200 frames
        np.testing.assert_array_equal(played[: len(ramp)], ramp)
        np.testing.assert_array_equal(played[len(ramp):], 0.0)
        self.assertEqual(stream.overflow_count, 0)
        self.assertEqual(stream.steady_underflow_count, 1)  # only the last, partial block


class TestPipelineMetrics(unittest.TestCase):
    def test_metrics_calculations(self):
        metrics = PipelineMetrics()
        metrics.inference_times_ms = [200.0, 220.0, 240.0, 260.0, 300.0]
        metrics.queue_wait_times_ms = [1.0, 2.0, 3.0]
        metrics.software_e2e_times_ms = [700.0, 720.0, 740.0]
        metrics.speech_onset_latency_ms = 700.0

        self.assertAlmostEqual(metrics.avg_inference_ms, 244.0, places=1)
        self.assertAlmostEqual(metrics.median_inference_ms, 240.0, places=1)
        self.assertAlmostEqual(metrics.min_inference_ms, 200.0, places=1)
        self.assertAlmostEqual(metrics.max_inference_ms, 300.0, places=1)
        self.assertEqual(metrics.speech_onset_latency_ms, 700.0)

        summary = metrics.summary()
        self.assertIn("startup_underflows", summary)
        self.assertIn("steady_underflows", summary)
        self.assertIn("shutdown_underflows", summary)
        self.assertIn("speech_onset_latency_ms", summary)
        self.assertIn("median_software_e2e_ms", summary)


if __name__ == "__main__":
    unittest.main()
