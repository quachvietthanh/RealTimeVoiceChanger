"""Listening comparison for microphone noise suppression (offline, real GPU inference, same code as streaming).

Inputs (never modified):
  clean  - the existing male test input (default: the +17.3 dB gain copy used in the A/B evaluations)
  noisy  - clean + a synthetic noise bed at --snr dB, written to <out>/inputs/ (seeded: pink noise,
           50 Hz mains hum with harmonics, low-frequency fan rumble). Synthetic, not a real recording.
For each input and each setting (off, light, medium, strong) the input goes through NoiseSuppressionStage
in 500 ms chunks exactly as the pipeline worker does, then through ChunkStitcher + RVC for each profile.

Writes (refuses to overwrite <out>):
  inputs/clean.wav, inputs/noise_only.wav, inputs/noisy_snr<snr>.wav
  denoised/<input>__<setting>.wav        what RVC receives (hear the denoiser alone)
  converted/<profile>__<input>__<setting>.wav
  summary.json                           level/attenuation/F0 measures and per-chunk timing

Objective measures are proxies only; they do not replace listening (tones, consonants, breath).
Example:
    .venv\\Scripts\\python scripts/noise_suppression_compare.py --out output/noise_suppression_2026-10-21
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from voice_changer.audio.crossfade import ChunkStitcher  # noqa: E402
from voice_changer.audio.noise_suppression import (  # noqa: E402
    NOISE_SUPPRESSION_MODES,
    NoiseSuppressionSettings,
    NoiseSuppressionStage,
)
from voice_changer.service import STREAM_SAMPLE_RATE, VoiceChangerService  # noqa: E402

SR = STREAM_SAMPLE_RATE
CHUNK = SR // 2
DEFAULT_INPUT = REPO / "output/spkb_preview_v1_2026-10-21/eval_input/input_male_peak0p50.wav"
SETTINGS = {"off": NoiseSuppressionSettings(enabled=False),
            **{m: NoiseSuppressionSettings(enabled=True, mode=m) for m in NOISE_SUPPRESSION_MODES}}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def noise_bed(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1, n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / SR)
    pink = np.fft.irfft(spec / np.sqrt(np.maximum(f, 20.0)), n)  # 1/f power
    t = np.arange(n) / SR
    hum = sum(a * np.sin(2 * np.pi * 50 * k * t + k) for k, a in ((1, 1.0), (2, 0.5), (3, 0.35), (5, 0.2)))
    rumble = np.fft.irfft(np.fft.rfft(rng.normal(0, 1, n)) * (f < 300), n)
    parts = [p / np.sqrt(np.mean(p ** 2)) for p in (pink, hum, rumble)]
    bed = 0.6 * parts[0] + 0.25 * parts[1] + 0.4 * parts[2]
    return (bed / np.sqrt(np.mean(bed ** 2))).astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12))


def activity_mask(clean: np.ndarray, frame: int = 400) -> np.ndarray:
    """Per-sample speech mask from the clean input (10 ms frames above -35 dB of its peak frame)."""
    n = len(clean) // frame
    e = 10 * np.log10(np.mean(clean[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    on = e > e.max() - 35
    return np.repeat(on, frame)[: len(clean)] if n else np.zeros(len(clean), bool)


def f0(y: np.ndarray):
    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
    f, voiced, _ = librosa.pyin(y16, fmin=60, fmax=400, sr=16000, frame_length=1024, hop_length=160)
    return f, voiced


def denoise(x: np.ndarray, settings: NoiseSuppressionSettings):
    stage = NoiseSuppressionStage(SR, settings)
    out, ms = [], []
    for k in range(len(x) // CHUNK):
        t0 = time.perf_counter()
        out.append(stage.process(x[k * CHUNK:(k + 1) * CHUNK]))
        ms.append((time.perf_counter() - t0) * 1000.0)
    y = np.concatenate(out)
    lat = stage.latency_samples
    return np.concatenate([y[lat:], np.zeros(lat, np.float32)]), ms  # re-aligned to the input for analysis


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--snr", type=float, default=10.0, help="speech-to-noise ratio of the noisy copy (dB, speech frames)")
    ap.add_argument("--profiles", default="voice_a_preview,voice_b_preview")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 1
    for d in ("inputs", "denoised", "converted"):
        (out / d).mkdir(parents=True)

    inp = Path(args.input)
    audio, sr = sf.read(str(inp), dtype="float32", always_2d=True)
    clean = librosa.resample(audio.mean(axis=1), orig_sr=sr, target_sr=SR).astype(np.float32)
    clean = clean[: len(clean) // CHUNK * CHUNK]
    speech = activity_mask(clean)
    noise = noise_bed(len(clean), args.seed)
    noise *= 10 ** ((rms_db(clean[speech]) - args.snr - rms_db(noise)) / 20)
    noisy = clean + noise
    if np.abs(noisy).max() >= 0.999:
        print("noisy mix would clip; lower the input or raise --snr", file=sys.stderr)
        return 1
    snr_tag = f"noisy_snr{args.snr:g}"
    for name, y in (("clean", clean), ("noise_only", noise), (snr_tag, noisy)):
        sf.write(str(out / "inputs" / f"{name}.wav"), y, SR, subtype="FLOAT")

    summary = {"input": str(inp), "input_sha256": sha256(inp), "sample_rate": SR, "chunk": CHUNK,
               "snr_db": args.snr, "noise_seed": args.seed, "speech_fraction": float(speech.mean()),
               "denoised": {}, "conversions": []}
    f0_clean, v_clean = f0(clean)
    inputs = {"clean": clean, snr_tag: noisy}
    denoised = {}
    for iname, x in inputs.items():
        for sname, st in SETTINGS.items():
            y, ms = denoise(x, st)
            denoised[(iname, sname)] = y
            sf.write(str(out / "denoised" / f"{iname}__{sname}.wav"), y, SR, subtype="FLOAT")
            fy, vy = f0(y)
            both = v_clean & vy
            cents = 1200 * np.abs(np.log2(fy[both] / f0_clean[both])) if both.any() else np.array([])
            pause = ~speech
            pause[: SR * 2] = False  # skip the noise estimator's start-up
            summary["denoised"][f"{iname}__{sname}"] = {
                "speech_level_change_db": round(rms_db(y[speech]) - rms_db(x[speech]), 2),
                "pause_level_change_db": round(rms_db(y[pause]) - rms_db(x[pause]), 2) if pause.any() else None,
                "residual_vs_clean_db": round(rms_db((y - clean)[speech]) - rms_db(clean[speech]), 2),
                "voiced_frames_vs_clean": round(float(vy.sum() / max(v_clean.sum(), 1)), 3),
                "f0_median_abs_cents_vs_clean": round(float(np.median(cents)), 1) if len(cents) else None,
                "f0_p95_abs_cents_vs_clean": round(float(np.percentile(cents, 95)), 1) if len(cents) else None,
                "ns_ms_per_chunk": {"mean": round(float(np.mean(ms)), 2), "p95": round(float(np.percentile(ms, 95)), 2),
                                    "max": round(float(np.max(ms)), 2)},
                "peak": round(float(np.abs(y).max()), 4), "nan": int(np.sum(~np.isfinite(y))),
            }
            print(json.dumps({f"{iname}__{sname}": summary["denoised"][f"{iname}__{sname}"]}), flush=True)

    svc = VoiceChangerService()
    try:
        for pid in args.profiles.split(","):
            svc.select_profile(pid)
            conv, cfg = svc._converter, replace(svc._config)  # offline harness: drive the loaded converter
            for (iname, sname), x in denoised.items():
                stitcher = ChunkStitcher(CHUNK, int(0.2 * SR), int(0.05 * SR), int(0.01 * SR))
                parts, call_ms = [], []
                for k in range(len(x) // CHUNK):
                    m = stitcher.model_input(x[k * CHUNK:(k + 1) * CHUNK])
                    torch.manual_seed(k)  # same seeds for every setting: differences are the denoiser's
                    t0 = time.perf_counter()
                    yk = conv.convert_chunk(m, config=cfg, input_sr=SR, target_sr=SR)
                    call_ms.append((time.perf_counter() - t0) * 1000.0)
                    parts.append(stitcher.join(yk))
                y = np.concatenate(parts)
                path = out / "converted" / f"{pid}__{iname}__{sname}.wav"
                sf.write(str(path), y, SR, subtype="FLOAT")
                rec = {"profile": pid, "input": iname, "setting": sname, "config": asdict(cfg), "wav": str(path),
                       "rms_dbfs": round(rms_db(y), 2), "peak": round(float(np.abs(y).max()), 4),
                       "nan": int(np.sum(~np.isfinite(y))), "call_ms_median": round(float(np.median(call_ms[1:])), 1)}
                summary["conversions"].append(rec)
                print(json.dumps(rec), flush=True)
    finally:
        svc.shutdown()
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
