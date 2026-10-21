"""A/B test for the 16 kHz reflection padding applied before RVC feature extraction.

Hypothesis under test (thuat-toan.md): the symmetric (80, 80) padding shifts the
converted audio by about 5 ms relative to the input and creates discontinuities
at chunk boundaries. Variants compared:

    A  (80, 80)  current production padding
    B  (0, 160)  no left padding, extra right padding
    C  (0, 0)    control: no padding

For each variant the same input is converted two ways:
    whole    - one convert_chunk() call over the full signal (reference)
    chunked  - 0.5 s chunks converted independently and concatenated (as in the pipeline)

Metrics:
    lag_ms            - timing offset of the output onset envelope vs the input (1 ms frames)
    chunk_vs_whole    - best-lag normalized RMSE between chunked and whole output
    boundary_ratio    - sample-to-sample jump at each chunk boundary divided by the median
                        jump in the neighbourhood (1.0 = no click)

Usage (from the repository root, with the project venv):
    .venv\\Scripts\\python.exe scripts\\ab_padding_test.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from voice_changer.conversion.base import ConversionConfig  # noqa: E402
from voice_changer.conversion.rvc.adapter import RVCAdapter  # noqa: E402

SR = 40000
CHUNK = 20000  # 0.5 s at 40 kHz, same as the pipeline default
VARIANTS = {
    "A_80_80_current": (80, 80),
    "B_0_160": (0, 160),
    "C_0_0_control": (0, 0),
}
ENV_FRAME = 40  # 1 ms envelope frames at 40 kHz
STFT_N = 1024  # 25.6 ms analysis window
STFT_HOP = 512
SEEDS = (1234, 2024, 7)


def make_signal(seconds: float, seed: int = 1234) -> np.ndarray:
    """Deterministic harmonic signal with pitch glide and syllable-like onsets."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f0 = 120.0 * (1.0 + 0.04 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 20))
    # Syllable-like on/off gating (0.4 s period) gives clear onsets for timing measurement
    gate = (np.sin(2 * np.pi * t / 0.4) > -0.2).astype(np.float64)
    gate = np.convolve(gate, np.ones(400) / 400, mode="same")
    x = x * gate + 0.002 * rng.standard_normal(n)
    return (x / np.abs(x).max() * 0.5).astype(np.float32)


def frame_rms(x: np.ndarray, frame: int = ENV_FRAME) -> np.ndarray:
    frames = len(x) // frame
    return np.sqrt((x[: frames * frame].reshape(frames, frame) ** 2).mean(axis=1))


def onset_times_ms(x: np.ndarray, rel_threshold: float = 0.3, smooth_frames: int = 20, min_gap_ms: int = 200) -> np.ndarray:
    """Syllable onset times (ms): rising edges of a 20 ms-smoothed RMS envelope above a fraction of its peak.

    Edges closer than min_gap_ms are merged so that intra-syllable ripple is not counted as onsets.
    """
    e = frame_rms(x)
    e = np.convolve(e, np.ones(smooth_frames) / smooth_frames, mode="same")
    above = e > rel_threshold * e.max()
    rising = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    kept: list[int] = []
    for t in rising:
        if not kept or t - kept[-1] >= min_gap_ms:
            kept.append(int(t))
    return np.asarray(kept, dtype=np.float64)  # one frame = 1 ms


def envelope_lag_ms(x_in: np.ndarray, x_out: np.ndarray, max_ms: int = 30) -> dict:
    """Whole-file lag (ms) of the output RMS envelope relative to the input envelope.

    Positive = output later than input. Used for continuous speech where onset matching is unreliable.
    """
    e_in = frame_rms(x_in)
    e_out = frame_rms(x_out)
    m = min(len(e_in), len(e_out))
    e_in = e_in[:m] - e_in[:m].mean()
    e_out = e_out[:m] - e_out[:m].mean()
    best_lag, best_val = 0, -np.inf
    for lag in range(-max_ms, max_ms + 1):
        if lag >= 0:
            a, b = e_in[: m - lag], e_out[lag:m]
        else:
            a, b = e_in[-lag:m], e_out[: m + lag]
        val = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        if val > best_val:
            best_lag, best_val = lag, val
    return {"mean_ms": float(best_lag), "corr": round(best_val, 4)}


def load_speech(path: Path) -> np.ndarray:
    """Read a WAV, mix to mono, resample to 40 kHz, and cap the peak at 0.95 (no clipping)."""
    import librosa
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    x = librosa.resample(mono, orig_sr=sr, target_sr=SR).astype(np.float32)
    peak = float(np.abs(x).max())
    if peak > 0.95:
        x = x * (0.95 / peak)
    return x


def timing_offset_ms(x_in: np.ndarray, x_out: np.ndarray) -> dict:
    """Mean output-minus-input onset delay in ms, using onsets matched in order."""
    t_in = onset_times_ms(x_in)
    t_out = onset_times_ms(x_out)
    m = min(len(t_in), len(t_out))
    if m == 0:
        return {"mean_ms": None, "std_ms": None, "n_onsets": 0}
    d = t_out[:m] - t_in[:m]
    return {"mean_ms": round(float(d.mean()), 3), "std_ms": round(float(d.std()), 3), "n_onsets": int(m)}


def boundary_excess(x: np.ndarray, boundary: int, window: int = 480) -> float | None:
    """Sample jump at a chunk boundary, relative to the local median jump; None in silence."""
    seg = x[boundary - window : boundary + window]
    if np.sqrt(np.mean(seg ** 2)) < 0.02 * np.abs(x).max():
        return None
    jump = abs(float(x[boundary]) - float(x[boundary - 1]))
    diffs = np.abs(np.diff(seg))
    diffs = np.delete(diffs, window - 1)  # drop the boundary jump itself
    return jump / (float(np.median(diffs)) + 1e-9)


def log_spectrogram(x: np.ndarray) -> np.ndarray:
    window = np.hanning(STFT_N)
    frames = 1 + (len(x) - STFT_N) // STFT_HOP
    idx = np.arange(STFT_N)[None, :] + STFT_HOP * np.arange(frames)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * window, axis=1))
    return 20 * np.log10(spec + 1e-6)


def spectral_distance(a: np.ndarray, b: np.ndarray, boundaries: list[int]) -> dict:
    """Mean log-spectral distance (dB) per frame; split into frames near and away from boundaries."""
    la, lb = log_spectrogram(a), log_spectrogram(b)
    m = min(len(la), len(lb))
    dist = np.mean(np.abs(la[:m] - lb[:m]), axis=1)
    centers = STFT_HOP * np.arange(m) + STFT_N // 2
    near = np.zeros(m, dtype=bool)
    for bnd in boundaries:
        near |= np.abs(centers - bnd) < STFT_N
    return {
        "lsd_db_all": round(float(dist.mean()), 3),
        "lsd_db_near_boundary": round(float(dist[near].mean()), 3) if near.any() else None,
        "lsd_db_away_from_boundary": round(float(dist[~near].mean()), 3) if (~near).any() else None,
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def run_variant(adapter: RVCAdapter, x: np.ndarray, pad: tuple[int, int]) -> dict:
    adapter.pad_16k = pad
    cfg = ConversionConfig(pitch_shift=10.0, f0_method="rmvpe", index_rate=0.5, filter_radius=2, protect_rate=0.4)

    t0 = time.perf_counter()
    whole = np.asarray(adapter.convert_chunk(x, cfg, input_sr=SR), dtype=np.float32)
    chunks = []
    for start in range(0, len(x), CHUNK):
        piece = x[start : start + CHUNK]
        chunks.append(np.asarray(adapter.convert_chunk(piece, cfg, input_sr=SR), dtype=np.float32))
    chunked = np.concatenate(chunks)
    elapsed = time.perf_counter() - t0

    boundaries = list(range(CHUNK, min(len(whole), len(chunked)), CHUNK))
    excess = []
    for b in boundaries:
        if b + 480 >= min(len(whole), len(chunked)):
            continue
        rc = boundary_excess(chunked, b)
        rw = boundary_excess(whole, b)
        if rc is not None and rw is not None:
            excess.append(rc / max(rw, 1e-9))

    return {
        "pad_left": pad[0],
        "pad_right": pad[1],
        "len_input": len(x),
        "len_whole": len(whole),
        "len_chunked": len(chunked),
        "timing_whole_vs_input": timing_offset_ms(x, whole),
        "timing_chunked_vs_input": timing_offset_ms(x, chunked),
        "envelope_lag_whole_vs_input": envelope_lag_ms(x, whole),
        "envelope_lag_chunked_vs_input": envelope_lag_ms(x, chunked),
        "spectral_whole_vs_chunked": spectral_distance(whole, chunked, boundaries),
        "boundary_jump_ratio_chunked_over_whole": {
            "n_active_boundaries": len(excess),
            "mean": round(float(np.mean(excess)), 3) if excess else None,
            "max": round(float(np.max(excess)), 3) if excess else None,
        },
        "convert_seconds": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B test for 16 kHz padding in RVCAdapter")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--input", type=str, default=None, help="Speech WAV; replaces the synthetic signal (single run)")
    parser.add_argument("--out", type=str, default="output/ab_padding/ab_padding_results.json")
    args = parser.parse_args()

    model = REPO_ROOT / "models" / "weights" / "vctk_female_p231.pth"
    index = REPO_ROOT / "models" / "indices" / "vctk_female_p231.index"

    adapter = RVCAdapter()
    adapter.load_model(str(model), str(index))

    results: dict = {name: {} for name in VARIANTS}
    if args.input:
        x = load_speech(REPO_ROOT / args.input)
        signal_info = {"input": args.input, "seconds": round(len(x) / SR, 3), "sample_rate": SR, "chunk_samples": CHUNK}
        for name, pad in VARIANTS.items():
            print(f"[run] speech {name} pad={pad}", flush=True)
            results[name]["speech"] = run_variant(adapter, x, pad)
            print(json.dumps(results[name]["speech"], ensure_ascii=False), flush=True)
    else:
        signal_info = {"seconds": args.seconds, "sample_rate": SR, "seeds": list(SEEDS), "chunk_samples": CHUNK}
        for seed in SEEDS:
            x = make_signal(args.seconds, seed=seed)
            for name, pad in VARIANTS.items():
                print(f"[run] seed={seed} {name} pad={pad}", flush=True)
                results[name][str(seed)] = run_variant(adapter, x, pad)
                print(json.dumps(results[name][str(seed)], ensure_ascii=False), flush=True)

    report = {
        "git_head": git_head(),
        "device": "cpu" if not __import__("torch").cuda.is_available() else "cuda",
        "model_sha256": file_sha256(model),
        "index_sha256": file_sha256(index),
        "signal": signal_info,
        "results": results,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
