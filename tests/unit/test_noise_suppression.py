"""Noise suppression: suppressor state/latency/accuracy and the pipeline stage (toggle, ramp, failure fallback)."""

import sys
import unittest
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np

from voice_changer.audio.noise_suppression import (
    MAX_CONSECUTIVE_FAILURES,
    NOISE_SUPPRESSION_MODES,
    NoiseSuppressionSettings,
    NoiseSuppressionStage,
    SpectralNoiseSuppressor,
    validate_noise_suppression,
)

SR = 40000
CHUNK = 20000


def noisy_bursts(seconds=8.0, noise_std=0.03, seed=0):
    """200 Hz + 600 Hz 'voiced' bursts (1 s on / 1 s off) in white noise. Returns (clean, noise)."""
    t = np.arange(int(SR * seconds)) / SR
    gate = (np.sin(2 * np.pi * 0.5 * t) > 0).astype(np.float32)
    clean = ((0.2 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.sin(2 * np.pi * 600 * t)) * gate).astype(np.float32)
    noise = np.random.default_rng(seed).normal(0, noise_std, len(t)).astype(np.float32)
    return clean, noise


def run_chunks(proc, x, size=CHUNK):
    return np.concatenate([proc.process(x[i:i + size]) for i in range(0, len(x), size)])


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


class SuppressorTests(unittest.TestCase):
    def test_unity_gain_is_exact_delayed_copy_for_any_call_sizes(self):
        x = np.random.default_rng(1).normal(0, 0.1, SR * 2).astype(np.float32)
        s = SpectralNoiseSuppressor(SR)
        s._gain_floor = 1.0  # gain 1 everywhere: tests framing, windowing and overlap-add only
        out, i = [], 0
        for n in (20000, 777, 1, 5000, 34222, 20000):  # partial frames and odd sizes
            out.append(s.process(x[i:i + n]))
            self.assertEqual(len(out[-1]), n)
            i += n
        y = np.concatenate(out)
        lat = s.latency_samples
        self.assertEqual(lat, 800)  # 20 ms at 40 kHz
        np.testing.assert_allclose(y[lat:], x[: i - lat], atol=1e-6)
        self.assertLess(np.abs(y[:lat]).max(), 1e-9)

    def test_each_mode_attenuates_noise_by_its_floor_and_keeps_speech_level(self):
        clean, noise = noisy_bursts()
        for mode, floor_db in NOISE_SUPPRESSION_MODES.items():
            with self.subTest(mode=mode):
                s = SpectralNoiseSuppressor(SR, mode)
                y = run_chunks(s, clean + noise)[s.latency_samples:]
                pause = slice(int(SR * 5.2), int(SR * 5.9))  # noise only, after the estimate converged
                speech = slice(int(SR * 4.2), int(SR * 4.9))
                self.assertAlmostEqual(rms_db(y[pause]) - rms_db(noise[pause]), floor_db, delta=1.0)
                self.assertLess(abs(rms_db(y[speech]) - rms_db(clean[speech])), 0.5)
                self.assertTrue(np.all(np.isfinite(y)))
                self.assertLess(np.abs(y).max(), 1.0)

    def test_state_carries_across_chunks(self):
        # Chunked processing must equal one call over the whole signal (no per-chunk reset or seam).
        clean, noise = noisy_bursts(seconds=3.0)
        x = clean + noise
        whole = SpectralNoiseSuppressor(SR).process(x)
        chunked = run_chunks(SpectralNoiseSuppressor(SR), x, size=7919)
        np.testing.assert_allclose(chunked, whole, atol=1e-6)

    def test_reset_keeps_or_clears_noise_estimate(self):
        clean, noise = noisy_bursts(seconds=3.0)
        s = SpectralNoiseSuppressor(SR)
        s.process(clean + noise)
        learned = s._noise.copy()
        s.reset(keep_noise_estimate=True)
        np.testing.assert_array_equal(s._noise, learned)
        self.assertEqual(len(s._pending), 0)
        self.assertFalse(np.any(s._ola))
        s.reset()
        self.assertIsNone(s._noise)

    def test_rejects_non_finite_input_and_unknown_mode(self):
        s = SpectralNoiseSuppressor(SR)
        with self.assertRaises(ValueError):
            s.process(np.array([0.0, np.nan], dtype=np.float32))
        with self.assertRaises(ValueError):
            SpectralNoiseSuppressor(SR, "max")
        self.assertTrue(validate_noise_suppression(NoiseSuppressionSettings(mode="extreme")))
        self.assertEqual(validate_noise_suppression(NoiseSuppressionSettings(True, "light")), [])


class StageTests(unittest.TestCase):
    def test_disabled_is_a_pure_delay_and_does_not_run_the_suppressor(self):
        x = np.random.default_rng(2).normal(0, 0.1, CHUNK * 3).astype(np.float32)
        stage = NoiseSuppressionStage(SR)
        stage._suppressor.process = lambda a: (_ for _ in ()).throw(AssertionError("must not run"))
        y = run_chunks(stage, x)
        lat = stage.latency_samples
        np.testing.assert_array_equal(y[lat:], x[:-lat])
        self.assertFalse(stage.active)

    def test_enable_primes_one_chunk_then_ramps_in_without_timeline_shift(self):
        clean, noise = noisy_bursts(seconds=4.0)
        x = clean + noise
        stage = NoiseSuppressionStage(SR, NoiseSuppressionSettings(enabled=True, mode="strong"))
        lat = stage.latency_samples
        first = stage.process(x[:CHUNK])
        np.testing.assert_array_equal(first[lat:], x[: CHUNK - lat])  # priming chunk: dry
        self.assertFalse(stage.active)
        second = stage.process(x[CHUNK:2 * CHUNK])
        self.assertTrue(stage.active)
        # Inside the 20 ms ramp the output moves smoothly from dry to wet: no jump bigger than the signal's own.
        self.assertLess(np.abs(np.diff(np.concatenate([first[-50:], second[:900]]))).max(),
                        np.abs(np.diff(x)).max() * 1.5)

    def test_live_disable_ramps_out_and_returns_to_exact_dry(self):
        x = np.random.default_rng(3).normal(0, 0.05, CHUNK * 4).astype(np.float32)
        stage = NoiseSuppressionStage(SR, NoiseSuppressionSettings(enabled=True))
        stage.process(x[:CHUNK])
        stage.process(x[CHUNK:2 * CHUNK])
        stage.update(NoiseSuppressionSettings(enabled=False))
        stage.process(x[2 * CHUNK:3 * CHUNK])  # ramp out
        self.assertFalse(stage.active)
        last = stage.process(x[3 * CHUNK:])
        lat = stage.latency_samples
        np.testing.assert_array_equal(last, x[3 * CHUNK - lat: 4 * CHUNK - lat])

    def test_mode_change_applies_live(self):
        stage = NoiseSuppressionStage(SR, NoiseSuppressionSettings(enabled=True, mode="light"))
        x = np.zeros(CHUNK, dtype=np.float32)
        stage.process(x)
        stage.update(NoiseSuppressionSettings(enabled=True, mode="strong"))
        stage.process(x)
        self.assertEqual(stage._suppressor.mode, "strong")
        with self.assertRaises(ValueError):
            stage.update(NoiseSuppressionSettings(enabled=True, mode="bogus"))

    def test_failure_falls_back_to_dry_and_disables_after_repeats(self):
        x = np.random.default_rng(4).normal(0, 0.05, CHUNK * 6).astype(np.float32)
        stage = NoiseSuppressionStage(SR, NoiseSuppressionSettings(enabled=True))
        calls = {"n": 0}

        def broken(a):
            calls["n"] += 1
            raise RuntimeError("denoiser exploded")

        stage._suppressor.process = broken
        y = run_chunks(stage, x)
        lat = stage.latency_samples
        np.testing.assert_array_equal(y[lat:], x[:-lat])  # always the intact delayed input
        self.assertTrue(stage.failed)
        self.assertEqual(stage.errors, MAX_CONSECUTIVE_FAILURES)
        self.assertEqual(calls["n"], MAX_CONSECUTIVE_FAILURES)  # stops trying once failed
        self.assertIn("denoiser exploded", stage.last_error)
        stage.update(NoiseSuppressionSettings(enabled=True))  # user re-arms it
        self.assertFalse(stage.failed)

    def test_session_reset_clears_audio_and_failure(self):
        stage = NoiseSuppressionStage(SR, NoiseSuppressionSettings(enabled=True))
        stage.process(np.ones(CHUNK, dtype=np.float32))
        stage.failed, stage.errors = True, 3
        stage.reset()
        self.assertFalse(stage.failed)
        self.assertEqual(stage.errors, 0)
        self.assertFalse(np.any(stage._delay))
        out = stage.process(np.zeros(CHUNK, dtype=np.float32))
        self.assertFalse(np.any(out))  # nothing from before the reset leaks out


if __name__ == "__main__":
    unittest.main()
