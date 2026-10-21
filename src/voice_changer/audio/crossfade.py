"""Joining independently converted chunks into a continuous output stream.

Each streaming chunk is converted by a separate model call, so consecutive outputs do not line up at
the chunk boundary: the waveform jumps and the level steps (docs/overlap-crossfade-report.md measured
boundary jumps ~28x the within-chunk sample-to-sample change). ChunkStitcher removes that seam:

1. Every call also sees `context_samples` of the previous input (left context).
2. The last `crossfade + search` samples of each output are held back instead of played.
3. The next call's estimate of that span is aligned to the held tail by a small lag search (SOLA:
   synchronous overlap-add, offset 0..search samples maximising normalised cross-correlation),
   then the two are crossfaded.

Every call to join() returns exactly `chunk_samples` samples, so buffer accounting is unchanged. The
added algorithmic delay is `crossfade + search - offset` samples (between crossfade and
crossfade + search). With all three lengths 0 the stitcher is a pass-through (the original behaviour).
Pure NumPy; called from the inference worker, never from an audio callback.
"""

from __future__ import annotations

import numpy as np


class ChunkStitcher:
    """Builds model input with left context and joins model outputs with SOLA crossfading."""

    def __init__(self, chunk_samples: int, context_samples: int = 0, crossfade_samples: int = 0,
                 search_samples: int = 0) -> None:
        """
        Raises:
            ValueError: if a length is negative, the crossfade does not fit in a chunk, or the context is
                shorter than crossfade + search (the span the next call must re-estimate).
        """
        if chunk_samples <= 0:
            raise ValueError(f"chunk_samples must be positive, got {chunk_samples}")
        if min(context_samples, crossfade_samples, search_samples) < 0:
            raise ValueError("context, crossfade and search lengths must be >= 0")
        if crossfade_samples == 0 and search_samples > 0:
            raise ValueError("search_samples needs crossfade_samples > 0")
        if crossfade_samples + search_samples > chunk_samples:
            raise ValueError("crossfade + search must not exceed the chunk length")
        if context_samples < crossfade_samples + search_samples:
            raise ValueError(
                f"context_samples ({context_samples}) must be >= crossfade + search "
                f"({crossfade_samples + search_samples})"
            )
        self.chunk_samples = chunk_samples
        self.context_samples = context_samples
        self.crossfade_samples = crossfade_samples
        self.search_samples = search_samples
        t = (np.arange(crossfade_samples, dtype=np.float64) + 0.5) / max(crossfade_samples, 1)
        # Equal-gain sin^2 fades: after SOLA alignment the two estimates are largely coherent.
        self._fade_in = (np.sin(0.5 * np.pi * t) ** 2).astype(np.float32)
        self._fade_out = (1.0 - self._fade_in).astype(np.float32)
        self.last_offset = 0
        self.reset()

    @property
    def model_input_samples(self) -> int:
        return self.context_samples + self.chunk_samples

    @property
    def nominal_delay_samples(self) -> int:
        """Algorithmic delay at the centre of the search range."""
        return self.crossfade_samples + self.search_samples // 2 if self.crossfade_samples else 0

    def reset(self) -> None:
        """Forget history (start of a session, or after a gap such as a dropped or failed chunk)."""
        self._history = np.zeros(self.context_samples, dtype=np.float32)
        self._tail: np.ndarray | None = None

    def model_input(self, chunk: np.ndarray) -> np.ndarray:
        """Return [left context | chunk] for the converter and remember the input for the next call."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if self.context_samples == 0:
            return chunk
        x = np.concatenate([self._history, chunk])
        self._history = x[-self.context_samples:].copy()
        return x

    def join(self, converted: np.ndarray) -> np.ndarray:
        """Turn the converter output for model_input() into exactly `chunk_samples` new output samples.

        `converted` should have model_input_samples samples; it is trimmed or zero-padded at the end
        otherwise (the caller counts such mismatches).
        """
        y = np.asarray(converted, dtype=np.float32).reshape(-1)
        n_in = self.model_input_samples
        if len(y) != n_in:
            y = y[:n_in] if len(y) > n_in else np.pad(y, (0, n_in - len(y)))
        if self.crossfade_samples == 0:
            return y[self.context_samples:].copy()

        xf, search, n = self.crossfade_samples, self.search_samples, self.chunk_samples
        # Region covering [held span of the previous call | this chunk], in this call's timeline.
        r = y[self.context_samples - xf - search:]
        if self._tail is None:
            offset = search // 2
            out = r[offset: offset + n].copy()
        else:
            offset = self._best_offset(r[: xf + search], self._tail)
            out = r[offset: offset + n].copy()
            out[:xf] = out[:xf] * self._fade_in + self._tail * self._fade_out
        self._tail = r[offset + n: offset + n + xf].copy()
        self.last_offset = offset
        return out

    @staticmethod
    def _best_offset(window: np.ndarray, tail: np.ndarray) -> int:
        """Offset in window where `tail` fits best (max normalised cross-correlation)."""
        xf = len(tail)
        if len(window) == xf:
            return 0
        num = np.correlate(window.astype(np.float64), tail.astype(np.float64), mode="valid")
        energy = np.convolve(window.astype(np.float64) ** 2, np.ones(xf), mode="valid")
        score = num / np.sqrt(energy + 1e-8)
        return int(np.argmax(score))
