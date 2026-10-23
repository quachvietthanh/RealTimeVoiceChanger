"""Startup prebuffer credit vs. consecutive slow chunks: deterministic, virtual clock, no threads or devices.

Drives the real AudioOutputStream callback and the pipeline's credit decision on a virtual clock. Input
chunks become ready on 2048-frame input blocks, one worker converts them in order (identity) with given
delays, and the output callback takes 2048-frame blocks every 51.2 ms at a given phase. Each scenario is
compared with the pre-credit baseline (expected_inference_s=None) over a sweep of output phases and input
block offsets.
"""

import math
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np

from voice_changer.audio import output_stream as output_stream_module
from voice_changer.audio.output_stream import AudioOutputStream
from voice_changer.audio.pipeline import RealTimeVoiceChangerPipeline

SR = 40000
CHUNK = 20000
BLOCK = 2048
BLOCK_S = BLOCK / SR  # 51.2 ms
PREBUFFER_S = 0.15
EXPECTED_S = 0.27
PHASES = [i * BLOCK_S / 16 for i in range(16)]
OFFSETS = (0, 1000)  # input samples already accumulated before the session's first block


class BlockInput:
    """Only what the credit decision reads from an input stream."""
    block_size = BLOCK


class IdentityConverter:
    def convert_chunk(self, audio_data, config=None, input_sr=SR, target_sr=None):
        return np.asarray(audio_data, dtype=np.float32)


def simulate(delays, expected_s, *, phase_s=0.0, offset=0, prebuffer_s=PREBUFFER_S, stream=None,
             force_credit_s=None, stop_after=None):
    """Play len(delays) ramp chunks; return timing, underflow and integrity results.

    force_credit_s sets the stream's credit directly (bypassing the decision and its cap) to measure
    other reserves. stop_after=n calls prepare_shutdown() right after the n-th write.
    """
    now = [0.0]
    with mock.patch.object(output_stream_module, "time", types.SimpleNamespace(perf_counter=lambda: now[0])):
        if stream is None:
            stream = AudioOutputStream(sample_rate=SR, channels=1, block_size=BLOCK, prebuffer_seconds=prebuffer_s)
        stream._reset_state()  # what start() does, without opening a device
        stream._is_active = True
        pipeline = RealTimeVoiceChangerPipeline(IdentityConverter(), input_stream=BlockInput(), output_stream=stream,
                                                prebuffer_duration_s=prebuffer_s, context_duration_s=0.0,
                                                crossfade_duration_s=0.0, sola_search_s=0.0,
                                                expected_inference_s=expected_s)
        ready = [math.ceil((offset + k * CHUNK) / BLOCK) * BLOCK_S for k in range(1, len(delays) + 1)]
        written, t = [], 0.0
        for r, d in zip(ready, delays):
            t = max(r, t) + d  # one worker, chunks in order
            written.append(t)
        ramp = np.arange(1, CHUNK * len(delays) + 1, dtype=np.float32)

        played, start_t, steady_underflows, k, n = [], None, 0, 0, 0
        while True:
            t_cb = phase_s + n * BLOCK_S
            if k < len(written) and written[k] < t_cb:  # a write at the same instant comes after the callback
                now[0] = written[k]
                if k == 0:
                    if force_credit_s is not None:
                        stream._prebuffer_credit_s = force_credit_s
                    else:
                        pipeline._apply_prebuffer_credit(written[0] - ready[0])
                stream.write_chunk(ramp[k * CHUNK:(k + 1) * CHUNK])
                k += 1
                if stop_after == k:
                    stream.prepare_shutdown()
                continue
            if k == len(written) and stream.buffered_samples == 0:
                break
            now[0] = t_cb
            before = stream.steady_underflow_count
            out = np.full((BLOCK, 1), -1.0, dtype=np.float32)
            stream._audio_callback(out, BLOCK, {}, None)
            if stream.playback_started:
                if start_t is None:
                    start_t = t_cb
                played.append(out[:, 0].copy())
                if stream.steady_underflow_count > before and k < len(written):  # not the final drain
                    steady_underflows += 1
            n += 1

    y = np.concatenate(played)
    body = y[: np.flatnonzero(y)[-1] + 1]
    return {
        "credit_ms": stream.prebuffer_credit_s * 1000.0,
        "metric_credit_ms": pipeline.metrics.prebuffer_credit_ms,
        "latency_ms": (start_t - ready[0]) * 1000.0,  # chunk 1 ready -> its first sample played
        "steady_underflows": steady_underflows,
        "silence_ms": float(np.sum(body == 0.0)) / SR * 1000.0,  # inserted inside the played audio
        "intact": bool(np.array_equal(body[body != 0.0], ramp)),  # every sample once, in order
        "stream": stream,
    }


def sweep(delays, expected_s, **kw):
    return [simulate(delays, expected_s, phase_s=ph, offset=off, **kw) for off in OFFSETS for ph in PHASES]


class StartupCreditScenarioTests(unittest.TestCase):
    def compare(self, delays):
        """Fixed vs. baseline at every phase/offset: never worse; identical when nothing is credited."""
        base, fixed = sweep(delays, None), sweep(delays, EXPECTED_S)
        for b, f in zip(base, fixed):
            self.assertTrue(b["intact"] and f["intact"])
            self.assertEqual(b["credit_ms"], 0.0)
            self.assertLessEqual(f["latency_ms"], b["latency_ms"] + 1e-9)
            self.assertLessEqual(f["steady_underflows"], b["steady_underflows"])
            self.assertLessEqual(f["silence_ms"], b["silence_ms"])
            if f["credit_ms"] == 0.0:
                for key in ("latency_ms", "steady_underflows", "silence_ms"):
                    self.assertEqual(f[key], b[key])
        return base, fixed

    def test_1_normal_first_chunk_is_not_credited(self):
        base, fixed = self.compare([0.27] * 6)
        self.assertTrue(all(f["credit_ms"] == 0.0 and f["steady_underflows"] == 0 for f in fixed))

    def test_2_slow_first_chunk_past_the_next_chunk_keeps_the_full_prebuffer(self):
        # 530 ms > 460.8 ms: chunk 2 is already waiting when chunk 1 is written (the worker is behind).
        base, fixed = self.compare([0.53] + [0.27] * 5)
        self.assertTrue(all(f["credit_ms"] == 0.0 and f["steady_underflows"] == 0 for f in fixed))

    def test_3_two_consecutive_slow_chunks_do_not_underflow(self):
        base, fixed = self.compare([0.53, 0.53] + [0.27] * 4)
        self.assertTrue(all(f["steady_underflows"] == 0 and f["silence_ms"] == 0.0 for f in fixed))

    def test_4_three_consecutive_slow_chunks_do_not_underflow(self):
        base, fixed = self.compare([0.53] * 3 + [0.27] * 3)
        self.assertTrue(all(f["steady_underflows"] == 0 and f["silence_ms"] == 0.0 for f in fixed))

    def test_5_mixed_delays_do_not_underflow(self):
        base, fixed = self.compare([0.53, 0.45, 0.53, 0.40] + [0.27] * 2)
        self.assertTrue(all(f["steady_underflows"] == 0 and f["silence_ms"] == 0.0 for f in fixed))

    def test_6_normal_first_then_slow_second_matches_the_baseline(self):
        # Not caused by the credit: a 530 ms chunk after a fast start underflows in the baseline too.
        base, fixed = self.compare([0.27, 0.53] + [0.27] * 4)
        self.assertTrue(all(b["steady_underflows"] > 0 for b in base))

    def test_late_first_chunk_before_the_next_one_is_credited_and_keeps_its_latency_gain(self):
        # 450 ms: lateness 450 - 270 - 50 = 130 ms, capped at 150 - 51.2 = 98.8 ms.
        base, fixed = self.compare([0.45] * 2 + [0.27] * 4)
        for b, f in zip(base, fixed):
            self.assertAlmostEqual(f["credit_ms"], 98.8, places=6)
            self.assertAlmostEqual(f["metric_credit_ms"], 98.8, places=6)
            self.assertEqual((f["steady_underflows"], f["silence_ms"]), (0, 0.0))
            # Playback starts on an output block, so the gain is the credit +- less than one block.
            self.assertGreater(b["latency_ms"] - f["latency_ms"], 98.8 - 51.2)
            self.assertLess(b["latency_ms"] - f["latency_ms"], 98.8 + 51.2)


class StartupCreditBoundaryTests(unittest.TestCase):
    def test_credited_session_survives_any_run_of_chunks_no_slower_than_the_first(self):
        for first in (0.33, 0.40, 0.43, 0.46, 0.4607):
            for later in ([first] * 9, [first, 0.0, first, first * 0.5, first, first, 0.1, first, first]):
                for r in sweep([first] + later, EXPECTED_S):
                    with self.subTest(first=first, later=later[:2]):
                        self.assertGreater(r["credit_ms"], 0.0)
                        self.assertEqual((r["steady_underflows"], r["intact"]), (0, True))

    def test_credit_boundary_is_the_earliest_time_the_next_chunk_can_be_ready(self):
        next_ready = (CHUNK // BLOCK) * BLOCK / SR  # 9 blocks = 460.8 ms
        self.assertGreater(simulate([next_ready - 1e-4, 0.27], EXPECTED_S)["credit_ms"], 0.0)
        self.assertEqual(simulate([next_ready, 0.27], EXPECTED_S)["credit_ms"], 0.0)
        self.assertEqual(simulate([next_ready + 1e-4, 0.27], EXPECTED_S)["credit_ms"], 0.0)

    def test_reserve_of_one_block_is_needed_and_fifty_ms_is_not_enough(self):
        # Equal 450 ms chunks: the credit cap (one 51.2 ms block of reserve) never underflows; a
        # 50 ms or smaller reserve does at some phases (for 50 ms in a ~4 ms window: fine grid). At exactly
        # one block the margin is zero, so a write at the same instant as a callback is excluded (half-step
        # offset); real callback jitter is not covered by this reserve.
        fine_phases = [(i + 0.5) * BLOCK_S / 128 for i in range(128)]
        for reserve_s, expect_underflow in ((0.0, True), (0.03, True), (0.05, True), (BLOCK_S, False)):
            runs = [simulate([0.45] * 8, EXPECTED_S, phase_s=ph, force_credit_s=PREBUFFER_S - reserve_s)
                    for ph in fine_phases]
            self.assertEqual(any(r["steady_underflows"] for r in runs), expect_underflow, reserve_s)
            self.assertTrue(all(r["intact"] for r in runs))

    def test_cap_and_zero_credit(self):
        stream = AudioOutputStream(sample_rate=SR, channels=1, block_size=BLOCK, prebuffer_seconds=PREBUFFER_S)
        self.assertAlmostEqual(stream.apply_prebuffer_credit(5.0), PREBUFFER_S - BLOCK_S)
        for prebuffer in (0.0, BLOCK_S, 0.04):  # no room for a reserve: never credited
            s = AudioOutputStream(sample_rate=SR, channels=1, block_size=BLOCK, prebuffer_seconds=prebuffer)
            self.assertEqual(s.apply_prebuffer_credit(1.0), 0.0)
        # Lateness inside the 50 ms tolerance: nothing credited.
        self.assertEqual(simulate([EXPECTED_S + 0.049, 0.27], EXPECTED_S)["credit_ms"], 0.0)

    def test_unknown_input_block_size_keeps_the_full_prebuffer(self):
        out = AudioOutputStream(sample_rate=SR, channels=1, block_size=BLOCK, prebuffer_seconds=PREBUFFER_S)
        p = RealTimeVoiceChangerPipeline(IdentityConverter(), input_stream=object(), output_stream=out,
                                         expected_inference_s=EXPECTED_S)
        p._apply_prebuffer_credit(0.45)
        self.assertEqual((out.prebuffer_credit_s, p.metrics.prebuffer_credit_ms), (0.0, 0.0))


class StartupCreditSessionTests(unittest.TestCase):
    def test_short_session_stopped_during_the_prebuffer_plays_everything(self):
        r = simulate([0.45], EXPECTED_S, stop_after=1)
        self.assertTrue(r["intact"])
        self.assertEqual((r["stream"].steady_underflow_count, r["silence_ms"]), (0, 0.0))

    def test_shutdown_drain_is_not_a_steady_underflow(self):
        r = simulate([0.45, 0.27], EXPECTED_S, stop_after=2)
        self.assertTrue(r["intact"])
        self.assertEqual(r["stream"].steady_underflow_count, 0)
        self.assertGreater(r["stream"].shutdown_underflow_count, 0)

    def test_repeated_sessions_are_independent(self):
        stream = AudioOutputStream(sample_rate=SR, channels=1, block_size=BLOCK, prebuffer_seconds=PREBUFFER_S)
        first = simulate([0.45] * 3, EXPECTED_S, stream=stream)
        second = simulate([0.53] * 3, EXPECTED_S, stream=stream)
        third = simulate([0.45] * 3, EXPECTED_S, stream=stream)
        self.assertAlmostEqual(first["credit_ms"], 98.8, places=6)
        self.assertEqual(second["credit_ms"], 0.0)  # not carried over from the first session
        self.assertEqual(third["latency_ms"], first["latency_ms"])
        for r in (first, second, third):
            self.assertEqual((r["steady_underflows"], r["intact"]), (0, True))


if __name__ == "__main__":
    unittest.main()
