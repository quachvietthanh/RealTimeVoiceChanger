"""Streaming integration test: real pipeline + real output ring buffer, simulated devices, identity converter.

Checks sample continuity end to end (no lost, duplicated or reordered samples) and steady-state
underflows over a few seconds of real-time streaming. CPU only, ~4 s.
"""

import sys
import time
import unittest
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np

from voice_changer.audio.noise_suppression import NoiseSuppressionSettings
from voice_changer.audio.pipeline import RealTimeVoiceChangerPipeline
from voice_changer.audio.simulated import SimulatedInputStream, SimulatedOutputStream

SR = 40000


class IdentityConverter:
    def convert_chunk(self, audio_data, config=None, input_sr=SR, target_sr=None):
        time.sleep(0.05)  # stand-in for inference time
        return np.asarray(audio_data, dtype=np.float32)


class DelayedIdentityConverter:
    def __init__(self, delays, steady_s):
        self.delays, self.steady_s, self.calls = list(delays), steady_s, 0

    def convert_chunk(self, audio_data, config=None, input_sr=SR, target_sr=None):
        time.sleep(self.delays[self.calls] if self.calls < len(self.delays) else self.steady_s)
        self.calls += 1
        return np.asarray(audio_data, dtype=np.float32)


class SimulatedStreamingTests(unittest.TestCase):
    def _stream_ramp(self, **seam_kwargs):
        seconds = 3.0
        ramp = (np.arange(int(SR * seconds), dtype=np.float32) + 1.0) / 1e6  # unique, strictly increasing
        inp = SimulatedInputStream(ramp, sample_rate=SR)
        out = SimulatedOutputStream(sample_rate=SR, channels=2, block_size=2048, prebuffer_seconds=0.15, capture=True)
        p = RealTimeVoiceChangerPipeline(IdentityConverter(), input_stream=inp, output_stream=out, **seam_kwargs)
        p.start()
        self.assertTrue(inp.finished.wait(seconds + 2))
        time.sleep(1.3)  # let the last ramp chunk (and the seam tail held back from it) play out
        self.assertEqual(p.health(), [])
        p.stop()
        self.assertEqual(p.metrics.total_chunks_dropped, 0)
        self.assertEqual(p.metrics.inference_errors, 0)
        self.assertEqual(p.metrics.sample_count_mismatches, 0)
        self.assertEqual(p.metrics.steady_underflow_count, 0)
        self.assertEqual(p.metrics.output_overflow_count, 0)
        self.assertEqual(p.metrics.stitch_resets, 0)
        played = out.captured()[:, 0]
        return ramp, played[played > 0], p  # drop startup/padding silence

    def test_continuity_legacy_joining(self):
        ramp, signal, p = self._stream_ramp(context_duration_s=0.0, crossfade_duration_s=0.0, sola_search_s=0.0)
        expected_chunks = len(ramp) // p.chunk_samples
        self.assertEqual(len(signal), expected_chunks * p.chunk_samples)
        np.testing.assert_array_equal(signal, ramp[: len(signal)])

    def test_continuity_with_context_and_crossfade(self):
        # No lag search: the identity converter's two estimates of each seam span are identical, so the
        # crossfade must reproduce the input exactly (to float rounding), only later by the held-back span.
        ramp, signal, p = self._stream_ramp(context_duration_s=0.2, crossfade_duration_s=0.05, sola_search_s=0.0)
        self.assertEqual(p.metrics.algorithmic_delay_ms, 50.0)
        self.assertEqual(len(signal), len(ramp))
        np.testing.assert_allclose(signal, ramp, rtol=0, atol=1e-7)

    def test_noise_suppression_in_real_time_path(self):
        # Real suppressor, real ring buffer, identity "model": noise in pauses must drop, speech must stay,
        # output must stay finite, unclipped and gap-free, with the timeline unchanged.
        seconds = 6.0
        t = np.arange(int(SR * seconds)) / SR
        gate = (np.sin(2 * np.pi * 0.5 * t) > 0).astype(np.float32)
        clean = (0.2 * np.sin(2 * np.pi * 220 * t) * gate).astype(np.float32)
        noise = np.random.default_rng(0).normal(0, 0.02, len(t)).astype(np.float32)
        inp = SimulatedInputStream(clean + noise, sample_rate=SR)
        out = SimulatedOutputStream(sample_rate=SR, channels=2, block_size=2048, prebuffer_seconds=0.15, capture=True)
        p = RealTimeVoiceChangerPipeline(IdentityConverter(), input_stream=inp, output_stream=out,
                                         noise_suppression=NoiseSuppressionSettings(enabled=True, mode="medium"))
        p.start()
        self.assertTrue(inp.finished.wait(seconds + 2))
        time.sleep(1.3)
        self.assertEqual(p.health(), [])
        p.stop()
        m = p.metrics
        self.assertEqual((m.total_chunks_dropped, m.inference_errors, m.steady_underflow_count, m.stitch_resets),
                         (0, 0, 0, 0))
        self.assertFalse(m.noise_suppression_failed)
        played = out.captured()[:, 0]
        self.assertTrue(np.all(np.isfinite(played)))
        self.assertLess(np.abs(played).max(), 1.0)
        # Align the output to the input by the known delays (startup silence, 75 ms algorithmic delay).
        delay = int(np.argmax(np.abs(played) > 0.05)) - int(np.argmax(np.abs(clean) > 0.05))
        y = played[delay: delay + len(clean)]
        rms = lambda a: float(np.sqrt(np.mean(np.square(a, dtype=np.float64))))
        pause, speech = slice(int(SR * 5.2), int(SR * 5.8)), slice(int(SR * 4.2), int(SR * 4.8))
        self.assertLess(rms(y[pause]) / rms(noise[pause]), 10 ** (-9 / 20))  # medium floor is -12 dB
        self.assertAlmostEqual(rms(y[speech]) / rms(clean[speech] + noise[speech]), 1.0, delta=0.1)

    def _stream_with_delays(self, delays, expected_s):
        """3 s ramp through the real ring buffer; inference `delays` for the first chunks, then 0.27 s.

        Seams are on without lag search, so an identity model reproduces the input exactly (see above).
        Returns (ramp, played, pipeline); played includes the startup silence.
        """
        seconds = 3.0
        ramp = (np.arange(int(SR * seconds), dtype=np.float32) + 1.0) / 1e6
        inp = SimulatedInputStream(ramp, sample_rate=SR)
        out = SimulatedOutputStream(sample_rate=SR, channels=2, block_size=2048, prebuffer_seconds=0.15, capture=True)
        p = RealTimeVoiceChangerPipeline(DelayedIdentityConverter(delays, 0.27), input_stream=inp, output_stream=out,
                                         sola_search_s=0.0, expected_inference_s=expected_s)
        p.start()
        self.assertTrue(inp.finished.wait(seconds + 2))
        time.sleep(1.6)
        self.assertEqual(p.health(), [])
        p.stop()
        m = p.metrics
        self.assertEqual((m.total_chunks_dropped, m.inference_errors, m.steady_underflow_count, m.stitch_resets,
                          m.output_overflow_count), (0, 0, 0, 0, 0))
        self.assertEqual([r["seq"] for r in m.startup_chunks], [1, 2, 3, 4, 5])
        return ramp, out.captured()[:, 0], p

    def _check_no_sample_lost(self, ramp, played):
        start = int(np.argmax(played > 0))  # the ramp is > 0 from its first sample: onset is not cut
        np.testing.assert_array_equal(played[:start], 0.0)
        np.testing.assert_allclose(played[start: start + len(ramp)], ramp, rtol=0, atol=1e-7)
        return start / SR  # software startup delay: output clock start -> first input sample played

    def test_slow_first_chunk_latency_is_offset_against_the_prebuffer(self):
        # Without an expected time a 430 ms first chunk adds ~160 ms to the whole session; with it, the
        # lateness beyond the 50 ms tolerance (110 ms) is taken off the 150 ms prebuffer, capped at
        # 150 - 51.2 ms (one output block stays in reserve).
        ramp, normal, p_normal = self._stream_with_delays([0.27], expected_s=0.27)
        _, slow_old, p_old = self._stream_with_delays([0.43], expected_s=None)
        _, slow_new, p_new = self._stream_with_delays([0.43], expected_s=0.27)
        d_normal = self._check_no_sample_lost(ramp, normal)
        d_old = self._check_no_sample_lost(ramp, slow_old)
        d_new = self._check_no_sample_lost(ramp, slow_new)
        self.assertEqual(p_normal.metrics.prebuffer_credit_ms, 0.0)
        self.assertEqual(p_old.metrics.prebuffer_credit_ms, 0.0)
        self.assertAlmostEqual(p_new.metrics.prebuffer_credit_ms, 98.8, places=6)
        # Playback starts on a 51.2 ms output block, so each run's delay is off by up to one block, plus
        # sleep jitter.
        self.assertAlmostEqual(d_old - d_normal, 0.16, delta=0.08)
        self.assertAlmostEqual(d_old - d_new, 0.0988, delta=0.08)
        steady = lambda p: float(np.mean(p.metrics.output_buffer_occupancy_ms[2:]))  # noqa: E731
        self.assertAlmostEqual(steady(p_old) - steady(p_normal), 160.0, delta=80.0)
        self.assertAlmostEqual(steady(p_old) - steady(p_new), 98.8, delta=80.0)

    def test_consecutive_slow_chunks_keep_the_full_prebuffer_and_do_not_underflow(self):
        # 530 ms per chunk is slower than real time: the next chunk is already waiting when the first is
        # written, so nothing is credited and the session behaves as without an expected time.
        ramp, base, p_base = self._stream_with_delays([0.53, 0.53], expected_s=None)
        d_base = self._check_no_sample_lost(ramp, base)
        for delays in ([0.53, 0.53], [0.53, 0.53, 0.53]):
            with self.subTest(delays=delays):
                _, played, p = self._stream_with_delays(delays, expected_s=0.27)  # asserts 0 steady underflows
                self.assertEqual(p.metrics.prebuffer_credit_ms, 0.0)
                d = self._check_no_sample_lost(ramp, played)
                self.assertAlmostEqual(d, d_base, delta=0.08)

    def test_device_loss_is_detected(self):
        inp = SimulatedInputStream(np.zeros(SR * 5, dtype=np.float32), sample_rate=SR)
        inp.simulate_device_loss(after_blocks=5)
        out = SimulatedOutputStream(sample_rate=SR, channels=2, block_size=2048)
        p = RealTimeVoiceChangerPipeline(IdentityConverter(), input_stream=inp, output_stream=out)
        p.start()
        deadline = time.monotonic() + 2
        while not p.health() and time.monotonic() < deadline:
            time.sleep(0.02)
        problems = p.health()
        p.stop()
        self.assertTrue(any("input audio stream" in x for x in problems))


if __name__ == "__main__":
    unittest.main()
