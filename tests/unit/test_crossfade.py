"""ChunkStitcher: model input with left context, SOLA alignment, crossfade and exact chunk lengths."""

import sys
import unittest
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np

from voice_changer.audio.crossfade import ChunkStitcher

N = 2000  # chunk
CTX, XF, SEARCH = 800, 200, 40


def tone(n: int) -> np.ndarray:
    t = np.arange(n) / 40000.0
    return (0.3 * np.sin(2 * np.pi * 173 * t) + 0.2 * np.sin(2 * np.pi * 411 * t + 1.0)
            + 0.1 * np.sin(2 * np.pi * 887 * t + 2.0)).astype(np.float32)


def run(stitcher: ChunkStitcher, x: np.ndarray, convert=lambda k, m: m):
    outs, offsets = [], []
    for k in range(len(x) // stitcher.chunk_samples):
        m = stitcher.model_input(x[k * stitcher.chunk_samples:(k + 1) * stitcher.chunk_samples])
        outs.append(stitcher.join(convert(k, m)))
        offsets.append(stitcher.last_offset)
    return np.concatenate(outs), offsets


class ChunkStitcherTests(unittest.TestCase):
    def test_passthrough_when_disabled(self):
        x = tone(5 * N)
        s = ChunkStitcher(N)
        self.assertEqual(s.model_input_samples, N)
        self.assertEqual(s.nominal_delay_samples, 0)
        out, _ = run(s, x)
        np.testing.assert_array_equal(out, x)

    def test_model_input_carries_left_context(self):
        x = tone(3 * N)
        s = ChunkStitcher(N, CTX, XF, SEARCH)
        first = s.model_input(x[:N])
        np.testing.assert_array_equal(first[:CTX], np.zeros(CTX, dtype=np.float32))
        second = s.model_input(x[N:2 * N])
        self.assertEqual(len(second), CTX + N)
        np.testing.assert_array_equal(second, x[N - CTX:2 * N])

    def test_every_join_returns_exactly_one_chunk(self):
        s = ChunkStitcher(N, CTX, XF, SEARCH)
        for length in (CTX + N, CTX + N - 7, CTX + N + 9):  # exact, short and long converter output
            self.assertEqual(len(s.join(np.ones(length, dtype=np.float32))), N)

    def test_identity_without_search_is_exact_delayed_copy(self):
        x = tone(8 * N)
        out, _ = run(ChunkStitcher(N, CTX, XF, 0), x)
        np.testing.assert_allclose(out[XF:], x[: len(x) - XF], atol=1e-6)

    def test_sola_compensates_per_call_lag(self):
        # Each call's output is late by a different lag (as independent inferences can be); SOLA must pick
        # offsets that cancel it, so both crossfaded estimates line up and the output stays smooth.
        lags = [0, 15, 5, 20, 12, 0, 18, 8]  # offsets stay inside the 0..SEARCH window
        x = tone((len(lags)) * N)

        def lagged(k, m):
            return np.concatenate([np.zeros(lags[k], dtype=np.float32), m[: len(m) - lags[k]]])

        s = ChunkStitcher(N, CTX, XF, SEARCH)
        out, offsets = run(s, x, lagged)
        for k in range(1, len(lags)):
            self.assertEqual(offsets[k] - offsets[k - 1], lags[k] - lags[k - 1], f"seam {k}")
        max_step = np.abs(np.diff(x)).max()
        self.assertLessEqual(np.abs(np.diff(out[N:])).max(), max_step * 1.05)

    def test_without_sola_misaligned_estimates_leave_a_seam(self):
        # Control for the test above: plain end-to-end joining of lagged outputs jumps at the seams.
        lags = [0, 25, 5, 30]
        x = tone(len(lags) * N)

        def lagged(k, m):
            return np.concatenate([np.zeros(lags[k], dtype=np.float32), m[: len(m) - lags[k]]])

        out, _ = run(ChunkStitcher(N), x, lagged)
        self.assertGreater(np.abs(np.diff(out)).max(), np.abs(np.diff(x)).max() * 3)

    def test_reset_forgets_history_and_tail(self):
        x = tone(3 * N)
        s = ChunkStitcher(N, CTX, XF, SEARCH)
        run(s, x)
        s.reset()
        np.testing.assert_array_equal(s.model_input(x[:N])[:CTX], np.zeros(CTX, dtype=np.float32))
        out = s.join(np.ones(CTX + N, dtype=np.float32))
        np.testing.assert_array_equal(out, np.ones(N, dtype=np.float32))  # no fade against a stale tail

    def test_invalid_lengths_rejected(self):
        with self.assertRaises(ValueError):
            ChunkStitcher(0)
        with self.assertRaises(ValueError):
            ChunkStitcher(N, CTX, -1, 0)
        with self.assertRaises(ValueError):
            ChunkStitcher(N, 100, XF, SEARCH)  # context shorter than crossfade + search
        with self.assertRaises(ValueError):
            ChunkStitcher(N, CTX, 0, SEARCH)  # search without crossfade
        with self.assertRaises(ValueError):
            ChunkStitcher(N, 3 * N, 2 * N, 0)  # crossfade longer than a chunk


if __name__ == "__main__":
    unittest.main()
