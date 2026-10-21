"""Optional microphone noise suppression before RVC inference (NumPy only, no extra dependency).

SpectralNoiseSuppressor is a classic streaming speech enhancer:
- STFT with 20 ms sqrt-Hann frames and 10 ms hop (50 % overlap, perfect reconstruction at unity gain);
- noise power tracked continuously by MCRA (minima-controlled recursive averaging, Cohen & Berdugo 2002),
  so it adapts to stationary and slowly changing noise (fans, hum, hiss, room tone) without a "learn
  noise" step;
- decision-directed a-priori SNR and a Wiener gain (Ephraim & Malah 1984), floored at the mode's maximum
  attenuation. The floor is the only user-facing parameter; it limits how far noise (and weak speech
  components) can be pulled down, which bounds musical-noise and breath/consonant damage.

It is a speech *enhancer*, not a gate: it does not remove non-stationary noise (keyboard clicks, voices,
music) and it attenuates noise during speech only as far as the floor allows.

All state (analysis frame, overlap-add tail, noise estimate, previous gain) persists across calls, so
500 ms streaming chunks are processed as one continuous signal. Output has exactly as many samples as
input, delayed by `latency_samples` (one frame, 20 ms at 40 kHz) for any call sizes.

NoiseSuppressionStage wraps it for the pipeline worker: an always-present delay line (so enabling and
disabling never shifts the timeline), a short wet/dry ramp on changes, per-chunk timing, and failure
fallback to the delayed dry signal. Runs on the inference worker thread, never in an audio callback.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Mode -> maximum attenuation (gain floor) in dB.
NOISE_SUPPRESSION_MODES: dict[str, float] = {"light": -6.0, "medium": -12.0, "strong": -18.0}
DEFAULT_NOISE_SUPPRESSION_MODE = "medium"
# Consecutive failures after which the stage stops trying and passes audio through.
MAX_CONSECUTIVE_FAILURES = 3


@dataclass(frozen=True)
class NoiseSuppressionSettings:
    """Microphone noise suppression. Off by default."""

    enabled: bool = False
    mode: str = DEFAULT_NOISE_SUPPRESSION_MODE


def validate_noise_suppression(settings: NoiseSuppressionSettings) -> list[str]:
    problems = []
    if not isinstance(settings.enabled, bool):
        problems.append(f"enabled must be a bool, got {settings.enabled!r}")
    if settings.mode not in NOISE_SUPPRESSION_MODES:
        problems.append(f"mode must be one of {list(NOISE_SUPPRESSION_MODES)}, got {settings.mode!r}")
    return problems


class SpectralNoiseSuppressor:
    """Stateful STFT/MCRA/Wiener noise suppressor for mono float32 audio."""

    # MCRA and decision-directed constants (per 10 ms hop).
    ALPHA_S = 0.8  # time smoothing of the periodogram for minimum tracking
    ALPHA_P = 0.2  # smoothing of the speech-presence probability
    ALPHA_D = 0.95  # noise update rate when speech is absent
    DELTA = 5.0  # S / S_min above this = speech present
    MIN_WINDOW_S = 0.8  # minimum-tracking sub-window (the minimum follows rising noise within ~2x this)
    ALPHA_DD = 0.98  # decision-directed weight
    XI_MIN_DB = -25.0

    def __init__(self, sample_rate: int = 40000, mode: str = DEFAULT_NOISE_SUPPRESSION_MODE) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        self.sample_rate = sample_rate
        self.hop = int(round(sample_rate * 0.01))
        self.frame = 2 * self.hop
        n = np.arange(self.frame)
        self._window = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * n / self.frame))  # sqrt periodic Hann
        self._min_window_frames = max(int(round(self.MIN_WINDOW_S * sample_rate / self.hop)), 1)
        self._xi_min = 10 ** (self.XI_MIN_DB / 10)
        self.set_mode(mode)
        self.reset()

    @property
    def latency_samples(self) -> int:
        return self.frame

    def set_mode(self, mode: str) -> None:
        if mode not in NOISE_SUPPRESSION_MODES:
            raise ValueError(f"unknown noise suppression mode {mode!r}")
        self.mode = mode
        self._gain_floor = 10 ** (NOISE_SUPPRESSION_MODES[mode] / 20)

    def reset(self, keep_noise_estimate: bool = False) -> None:
        """Clear audio state. keep_noise_estimate keeps the learned noise (after a gap in the same session)."""
        self._frame_buf = np.zeros(self.frame, dtype=np.float64)
        self._pending = np.zeros(0, dtype=np.float64)
        self._ola = np.zeros(self.frame, dtype=np.float64)
        # Output FIFO starts with one hop of silence: with the OLA's own (frame - hop) delay, every call
        # can return exactly as many samples as it received, for any call size.
        self._out = np.zeros(self.hop, dtype=np.float64)
        if not keep_noise_estimate or not hasattr(self, "_noise"):
            bins = self.frame // 2 + 1
            self._noise: Optional[np.ndarray] = None
            self._s = np.zeros(bins)
            self._s_min = np.zeros(bins)
            self._s_tmp = np.zeros(bins)
            self._p = np.zeros(bins)
            self._frames_in_window = 0
        self._prev_gain_sq_gamma: Optional[np.ndarray] = None

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Denoise the next samples of the stream. Returns len(audio) samples, delayed by latency_samples."""
        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(x)):
            raise ValueError("non-finite input samples")
        data = np.concatenate([self._pending, x])
        n_hops = len(data) // self.hop
        produced = [self._out]
        for k in range(n_hops):
            produced.append(self._process_hop(data[k * self.hop:(k + 1) * self.hop]))
        self._pending = data[n_hops * self.hop:]
        out_all = np.concatenate(produced)
        out, self._out = out_all[: len(x)], out_all[len(x):]
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("non-finite output samples")
        return out.astype(np.float32)

    def _process_hop(self, hop: np.ndarray) -> np.ndarray:
        self._frame_buf = np.concatenate([self._frame_buf[self.hop:], hop])
        spec = np.fft.rfft(self._frame_buf * self._window)
        power = spec.real ** 2 + spec.imag ** 2
        gain = self._gain(power)
        frame_out = np.fft.irfft(spec * gain, n=self.frame) * self._window
        self._ola += frame_out
        out = self._ola[: self.hop].copy()
        self._ola = np.concatenate([self._ola[self.hop:], np.zeros(self.hop)])
        return out

    def _gain(self, power: np.ndarray) -> np.ndarray:
        # MCRA noise tracking.
        smoothed = np.convolve(power, (0.25, 0.5, 0.25), mode="same")
        if self._noise is None:
            self._noise = np.maximum(power, 1e-12)
            self._s = smoothed.copy()
            self._s_min = smoothed.copy()
            self._s_tmp = smoothed.copy()
        else:
            self._s = self.ALPHA_S * self._s + (1 - self.ALPHA_S) * smoothed
            self._s_min = np.minimum(self._s_min, self._s)
            self._s_tmp = np.minimum(self._s_tmp, self._s)
            self._frames_in_window += 1
            if self._frames_in_window >= self._min_window_frames:
                self._s_min = np.minimum(self._s_tmp, self._s)
                self._s_tmp = self._s.copy()
                self._frames_in_window = 0
            present = (self._s > self.DELTA * np.maximum(self._s_min, 1e-12)).astype(np.float64)
            self._p = self.ALPHA_P * self._p + (1 - self.ALPHA_P) * present
            alpha = self.ALPHA_D + (1 - self.ALPHA_D) * self._p
            self._noise = np.maximum(alpha * self._noise + (1 - alpha) * power, 1e-12)
        # Decision-directed a-priori SNR, Wiener gain with a floor.
        gamma = power / self._noise
        ml = np.maximum(gamma - 1.0, 0.0)
        if self._prev_gain_sq_gamma is None:
            xi = np.maximum(ml, self._xi_min)
        else:
            xi = np.maximum(self.ALPHA_DD * self._prev_gain_sq_gamma + (1 - self.ALPHA_DD) * ml, self._xi_min)
        gain = np.maximum(xi / (1.0 + xi), self._gain_floor)
        self._prev_gain_sq_gamma = gain ** 2 * gamma
        return gain


class NoiseSuppressionStage:
    """Pipeline stage: delay line + optional suppressor, live enable/mode changes, failure fallback.

    The output is always the input delayed by `latency_samples`, whether suppression is on or off, so
    toggling never shifts or duplicates audio. Changes are applied at the start of the next chunk with a
    20 ms wet/dry ramp. Enabling runs the suppressor for one chunk with dry output first, so its overlap
    buffer and noise estimate are primed before it is heard.
    """

    def __init__(self, sample_rate: int = 40000, settings: Optional[NoiseSuppressionSettings] = None,
                 ramp_s: float = 0.02) -> None:
        settings = settings or NoiseSuppressionSettings()
        problems = validate_noise_suppression(settings)
        if problems:
            raise ValueError("; ".join(problems))
        self.sample_rate = sample_rate
        self._suppressor = SpectralNoiseSuppressor(sample_rate, settings.mode)
        self.latency_samples = self._suppressor.latency_samples
        self._ramp = max(int(round(ramp_s * sample_rate)), 1)
        self._lock = threading.Lock()
        self._settings = settings
        self.reset()

    @property
    def settings(self) -> NoiseSuppressionSettings:
        with self._lock:
            return self._settings

    @property
    def active(self) -> bool:
        """True while denoised audio is being output (fully or during a ramp)."""
        return self._mix > 0.0

    def update(self, settings: NoiseSuppressionSettings) -> None:
        """Change enable/mode; applied at the next chunk. Thread-safe. Clears a previous failure."""
        problems = validate_noise_suppression(settings)
        if problems:
            raise ValueError("; ".join(problems))
        with self._lock:
            self._settings = settings
            self.failed = False
            self._consecutive_failures = 0

    def reset(self, keep_noise_estimate: bool = False) -> None:
        """New session (full reset) or a gap in the stream (keep the learned noise only)."""
        self._suppressor.reset(keep_noise_estimate=keep_noise_estimate)
        self._delay = np.zeros(self.latency_samples, dtype=np.float32)
        self._mix = 0.0
        self._primed = False
        if not keep_noise_estimate:
            self.failed = False
            self.last_error = ""
            self.errors = 0
            self._consecutive_failures = 0
            self.process_times_ms: list[float] = []

    def process(self, chunk: np.ndarray) -> np.ndarray:
        x = np.asarray(chunk, dtype=np.float32).reshape(-1)
        buf = np.concatenate([self._delay, x])
        dry, self._delay = buf[: len(x)], buf[len(x):]

        settings = self.settings
        want = settings.enabled and not self.failed
        if not want and self._mix == 0.0:
            self._primed = False
            return dry

        t0 = time.perf_counter()
        try:
            if self._suppressor.mode != settings.mode:
                self._suppressor.set_mode(settings.mode)
            wet = self._suppressor.process(x)
            self._consecutive_failures = 0
        except Exception as err:  # fall back to the delayed dry signal; never pass corrupt audio on
            self.errors += 1
            self._consecutive_failures += 1
            self.last_error = f"{type(err).__name__}: {err}"
            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                self.failed = True
            self._suppressor.reset(keep_noise_estimate=False)
            self._mix, self._primed = 0.0, False
            return dry
        finally:
            self.process_times_ms.append((time.perf_counter() - t0) * 1000.0)

        if want and not self._primed:
            self._primed = True  # first chunk after enabling: prime state, keep output dry
            return dry
        target = 1.0 if want else 0.0
        if target == self._mix:
            return wet if target == 1.0 else dry
        n = min(self._ramp, len(x))
        mix = np.full(len(x), target, dtype=np.float32)
        mix[:n] = np.linspace(self._mix, target, n, endpoint=False, dtype=np.float32)
        self._mix = target
        if target == 0.0:
            self._primed = False
        return (dry * (1.0 - mix) + wet * mix).astype(np.float32)
