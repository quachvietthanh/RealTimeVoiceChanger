"""Simulation telemetry: absolute progress schedule and per-write buffer records (offline, deterministic)."""

import sys
import unittest
from pathlib import Path
from unittest import mock

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np

from voice_changer.audio.output_stream import AudioOutputStream
from voice_changer.audio.simulated import SimulatedOutputStream, WriteTelemetry, next_tick

SR = 40000
BLOCK = 2048


class NextTickTests(unittest.TestCase):
    def test_returns_the_first_tick_strictly_after_now(self):
        self.assertEqual(next_tick(10.0, 1.0, 10.0), (1, 11.0))
        self.assertEqual(next_tick(10.0, 1.0, 10.4), (1, 11.0))
        self.assertEqual(next_tick(10.0, 1.0, 11.0), (2, 12.0))  # exactly on a tick: the next one
        self.assertEqual(next_tick(10.0, 1.0, 9.0), (1, 11.0))  # before the start
        self.assertEqual(next_tick(0.0, 0.5, 3.74), (8, 4.0))

    def test_skips_missed_ticks(self):
        self.assertEqual(next_tick(0.0, 1.0, 5.2), (6, 6.0))

    def test_rejects_non_positive_period(self):
        for period in (0.0, -1.0):
            with self.assertRaises(ValueError):
                next_tick(0.0, period, 1.0)

    def test_per_iteration_overhead_does_not_accumulate(self):
        # Virtual clock: each sample wakes 0-15 ms late and then works 2.5 ms (the measured kind of
        # overhead). Sampling relative to the previous iteration drifts by all of it; the absolute
        # schedule keeps every sample within one iteration's lateness of its tick.
        lateness = [0.001 * (i % 16) for i in range(300)]
        work = 0.0025
        now, absolute = 0.0, []
        _, deadline = next_tick(0.0, 1.0, now)
        for late in lateness:
            now = max(now, deadline) + late
            absolute.append((now, deadline))
            now += work
            _, deadline = next_tick(0.0, 1.0, now)
        relative, now = [], 0.0
        for late in lateness:
            now += 1.0 + late
            relative.append(now)
            now += work
        for k, (t, tick) in enumerate(absolute, start=1):
            self.assertAlmostEqual(tick, float(k))
            self.assertLess(t - tick, 0.016)
        self.assertGreater(relative[-1] - 300.0, 2.0)  # ~2.3 s of drift after 300 samples


class WriteTelemetryTests(unittest.TestCase):
    @staticmethod
    def row(before, after, callbacks=0, **extra):
        return {"t_s": 0.0, "buffer_before_ms": before, "buffer_after_ms": after,
                "callbacks_during_write": callbacks, "steady_underflow_count": 0, **extra}

    def test_is_bounded_and_counts_evictions(self):
        tel = WriteTelemetry(max_rows=3)
        for i in range(5):
            tel.record(self.row(i, i + 500))
        self.assertEqual([r["index"] for r in tel.rows], [3, 4, 5])
        self.assertEqual((tel.evicted, tel.summary()["rows"]), (2, 3))
        tel.reset()
        self.assertEqual((len(tel.rows), tel.evicted), (0, 0))
        tel.record(self.row(0, 500))
        self.assertEqual(tel.rows[0]["index"], 1)

    def test_rejects_empty_capacity(self):
        with self.assertRaises(ValueError):
            WriteTelemetry(max_rows=0)

    def test_empty_summary(self):
        self.assertEqual(WriteTelemetry().summary(), {"rows": 0, "evicted": 0, "inexact_rows": 0})

    def test_summary_statistics_and_trend(self):
        tel = WriteTelemetry()
        befores = [200.0, 180.0, 160.0, 140.0, 120.0, 100.0, 80.0, 60.0]  # drifting down 20 ms per write
        for i, b in enumerate(befores):
            tel.record(self.row(b, b + 500.0, callbacks=1 if i == 3 else 0, queue_depth=i % 3,
                                chunks_dropped=0, overflow_count=0))
        s = tel.summary()
        self.assertEqual(s["buffer_before_ms"], {"min": 60.0, "mean": 130.0, "max": 200.0})
        self.assertEqual(s["buffer_after_ms"], {"min": 560.0, "mean": 630.0, "max": 700.0})
        self.assertEqual(s["before_trend_ms"], (70.0 - 190.0))  # last 2 vs first 2
        self.assertEqual((s["inexact_rows"], s["max_queue_depth"]), (1, 2))
        self.assertEqual(s["last"], {"steady_underflow_count": 0, "chunks_dropped": 0, "overflow_count": 0})

    def test_oscillation_without_drift_has_zero_trend(self):
        tel = WriteTelemetry()
        for i in range(48):  # each quarter (12 rows) holds whole 4-write cycles
            b = 120.0 + 90.0 * (i % 4)  # varies within each cycle, same distribution throughout
            tel.record(self.row(b, b + 500.0))
        self.assertEqual(tel.summary()["before_trend_ms"], 0.0)


class SimulatedOutputTelemetryTests(unittest.TestCase):
    def make(self, telemetry):
        out = SimulatedOutputStream(sample_rate=SR, channels=1, block_size=BLOCK, prebuffer_seconds=0.0,
                                    telemetry=telemetry)
        out._reset_state()  # what start() does, without the clock thread
        out._is_active = True
        return out

    def pull(self, out, n):
        buf = np.zeros((BLOCK, 1), dtype=np.float32)
        for _ in range(n):
            out._audio_callback(buf, BLOCK, {}, None)
            out._callbacks += 1  # as the clock thread does

    def test_records_levels_counters_and_probe_fields(self):
        tel = WriteTelemetry(probe=lambda: {"queue_depth": 1, "chunks_dropped": 0})
        out = self.make(tel)
        out.write_chunk(np.ones(20000, dtype=np.float32))
        self.pull(out, 3)
        out.write_chunk(np.ones(20000, dtype=np.float32))
        first, second = tel.rows
        self.assertEqual((first["index"], first["buffer_before_ms"], first["buffer_after_ms"]), (1, 0.0, 500.0))
        self.assertEqual((second["buffer_before_ms"], second["buffer_after_ms"]), (346.4, 846.4))  # 500 - 3 x 51.2
        for r in (first, second):
            self.assertEqual(r["callbacks_during_write"], 0)
            self.assertEqual((r["queue_depth"], r["chunks_dropped"]), (1, 0))
            self.assertEqual((r["steady_underflow_count"], r["overflow_count"]), (0, 0))
        self.assertEqual(out.buffered_samples, 40000 - 3 * BLOCK)  # telemetry does not alter the buffer

    def test_marks_rows_where_a_callback_ran_between_the_reads(self):
        tel = WriteTelemetry()
        out = self.make(tel)
        original = AudioOutputStream.write_chunk

        def write_then_callback(self_, data):
            original(self_, data)
            self.pull(self_, 1)

        with mock.patch.object(AudioOutputStream, "write_chunk", write_then_callback):
            out.write_chunk(np.ones(20000, dtype=np.float32))
        self.assertEqual(tel.rows[0]["callbacks_during_write"], 1)
        self.assertEqual(tel.summary()["inexact_rows"], 1)

    def test_without_telemetry_or_when_inactive_nothing_is_recorded(self):
        out = self.make(None)
        out.write_chunk(np.ones(100, dtype=np.float32))
        self.assertEqual(out.buffered_samples, 100)
        tel = WriteTelemetry()
        idle = SimulatedOutputStream(sample_rate=SR, channels=1, block_size=BLOCK, telemetry=tel)
        idle.write_chunk(np.ones(100, dtype=np.float32))  # not started: dropped, as before
        self.assertEqual((len(tel.rows), idle.buffered_samples), (0, 0))

    def test_start_resets_the_session_telemetry(self):
        tel = WriteTelemetry()
        tel.record({"buffer_before_ms": 0.0, "buffer_after_ms": 1.0, "callbacks_during_write": 0})
        out = SimulatedOutputStream(sample_rate=SR, channels=1, block_size=BLOCK, telemetry=tel)
        out.start()
        try:
            self.assertEqual(len(tel.rows), 0)
        finally:
            out.stop()


if __name__ == "__main__":
    unittest.main()
