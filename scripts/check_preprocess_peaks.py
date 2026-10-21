"""Measure peaks along Applio's training preprocess path, in memory (writes nothing).

Run with the training interpreter from the repository root:
    C:\\Users\\quach\\venvs\\rvc-train\\Scripts\\python scripts/check_preprocess_peaks.py <wav> [<wav> ...]

Replicates rvc/train/preprocess/preprocess.py with process_effects=False and
noise_reduction=False (the documented settings): Applio's own `load_audio`
(soundfile read -> librosa.to_mono -> librosa.resample to the target rate), its
`Slicer`, the 3.0 s / 0.3 s overlap windowing, and the per-slice librosa resample
to 16 kHz that feeds RMVPE and ContentVec.

Exit code 1 if any stage reaches --max-peak (default 0.90).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
APPLIO_ROOT = REPO_ROOT / "third_party" / "applio"

# Constants mirrored from rvc/train/preprocess/preprocess.py (v3.2.8)
OVERLAP = 0.3
SAMPLE_RATE_16K = 16000


def _slices(audio: np.ndarray, sr: int, per: float, slicer) -> list[np.ndarray]:
    out = []
    for segment in slicer.slice(audio):
        i = 0
        while True:
            start = int(sr * (per - OVERLAP) * i)
            i += 1
            if len(segment[start:]) > (per + OVERLAP) * sr:
                out.append(segment[start : start + int(per * sr)])
            else:
                out.append(segment[start:])
                break
    return out


def measure(path: Path, sr: int, per: float) -> dict[str, tuple[float, int]]:
    """Return {stage: (peak, samples >= 0.90)} for each preprocess stage."""
    os.chdir(APPLIO_ROOT)
    sys.path.insert(0, str(APPLIO_ROOT))
    from rvc.lib.utils import load_audio
    from rvc.train.preprocess.slicer import Slicer

    def stat(x: np.ndarray) -> tuple[float, int]:
        a = np.abs(x)
        return float(a.max()), int(np.count_nonzero(a >= 0.90))

    raw, _ = sf.read(str(path), dtype="float64", always_2d=True)
    audio = load_audio(str(path), sr)
    slicer = Slicer(sr=sr, threshold=-42, min_length=1500, min_interval=400, hop_size=15, max_sil_kept=500)
    slices = _slices(audio, sr, per, slicer)
    s16 = [librosa.resample(s, orig_sr=sr, target_sr=SAMPLE_RATE_16K) for s in slices]
    return {
        "input (file)": stat(raw),
        f"load_audio mono @ {sr} Hz": stat(audio),
        f"slices @ {sr} Hz (n={len(slices)})": stat(np.concatenate(slices)),
        f"slices @ {SAMPLE_RATE_16K} Hz (n={len(s16)})": stat(np.concatenate(s16)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure peaks along the Applio preprocess path (in memory)")
    parser.add_argument("wavs", nargs="+", help="Training WAV files to analyse")
    parser.add_argument("--sr", type=int, default=40000, help="Target training sample rate")
    parser.add_argument("--per", type=float, default=3.0, help="Slice length in seconds")
    parser.add_argument("--max-peak", type=float, default=0.90, help="Fail if any stage reaches this peak")
    args = parser.parse_args()

    # Resolve every path before measure() changes the working directory to Applio
    paths = [Path(wav).resolve() for wav in args.wavs]
    ok = True
    for path in paths:
        print("=" * 70)
        print(f"Preprocess peak check: {path}")
        print("=" * 70)
        for stage, (peak, n_hot) in measure(path, args.sr, args.per).items():
            passed = peak < args.max_peak
            ok &= passed
            print(f"[{'PASS' if passed else 'FAIL'}] {stage:<32} peak={peak:.6f}  samples>=0.90: {n_hot}")
    print("-" * 70)
    print(f"Overall: {'PASS' if ok else 'FAIL'} (limit {args.max_peak})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
