"""Create a gain-reduced derivative of the manifest-verified training audio.

The source split files are never modified. The derived file is written as 32-bit float
WAV with the same sample rate and channel layout, so the only change is a single
linear gain (no requantization, no clipping, no limiting):

    derived = float32(source_pcm / 32768 * gain),  gain = target_peak / source_peak

A derived manifest records the parent manifest, input/output SHA-256, gain, audio
parameters, tool versions and the user's consent/listening confirmations.
Refuses to run if the parent manifest fails verification or if any output exists.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_dataset_manifest import DEFAULT_MANIFEST, TRAINING_KEY, sha256_of, verify_manifest  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "dataset_train_peak0p88"
DEFAULT_TARGET_PEAK = 0.88
MAX_ALLOWED_PEAK = 0.90  # docs/custom-voice-training-plan.md section 2.2


def _rel(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def derive(parent_manifest: Path, output_dir: Path, target_peak: float, consent_confirmed_on: str,
           listening_confirmed_on: str, repo_root: Path = REPO_ROOT) -> dict:
    if not 0.0 < target_peak < MAX_ALLOWED_PEAK:
        raise ValueError(f"target peak {target_peak} must be in (0, {MAX_ALLOWED_PEAK})")

    failed = [r for r in verify_manifest(parent_manifest, repo_root) if not r.passed]
    if failed:
        raise RuntimeError("parent manifest verification failed: " + ", ".join(r.name for r in failed))

    parent = json.loads(parent_manifest.read_text(encoding="utf-8"))
    segment = parent["split_specifications"]["segments"][TRAINING_KEY]
    src_path = repo_root / segment["output_path"]

    out_wav = output_dir / (src_path.stem + f"_peak{target_peak:.2f}".replace(".", "p") + ".wav")
    out_manifest = output_dir / "derived_manifest.json"
    for p in (out_wav, out_manifest):
        if p.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {p}")

    src_info = sf.info(str(src_path))
    if src_info.subtype != "PCM_16":
        raise ValueError(f"expected PCM_16 source, got {src_info.subtype}")
    src, sr = sf.read(str(src_path), dtype="float64", always_2d=True)  # exact: int16 / 32768
    src_peak = float(np.max(np.abs(src)))
    if src_peak <= 0.0:
        raise ValueError("source is silent")
    full_scale = int(np.count_nonzero(np.abs(src) >= 32767 / 32768))
    if full_scale:
        raise RuntimeError(f"source has {full_scale} full-scale samples (possible clipping); stopping")

    gain = target_peak / src_peak
    if gain > 1.0:
        raise ValueError(f"source peak {src_peak} is already below target {target_peak}; refusing to amplify")
    derived = (src * gain).astype(np.float32)
    out_peak = float(np.max(np.abs(derived)))
    if out_peak > MAX_ALLOWED_PEAK:
        raise RuntimeError(f"derived peak {out_peak} exceeds {MAX_ALLOWED_PEAK}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_wav = out_wav.with_suffix(".tmp.wav")
    sf.write(str(tmp_wav), derived, sr, subtype="FLOAT")
    tmp_wav.rename(out_wav)  # rename never replaces an existing file on Windows

    out_info = sf.info(str(out_wav))
    manifest = {
        "manifest_version": "1.0",
        "description": "Gain-reduced derivative of the manifest-verified training split (peak rule < 0.90)",
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "parent_manifest": {
            "path": _rel(parent_manifest, repo_root),
            "sha256": sha256_of(parent_manifest),
            "segment": TRAINING_KEY,
        },
        "input": {
            "path": segment["output_path"],
            "sha256": segment["sha256"],
            "file_size_bytes": int(segment["file_size_bytes"]),
            "sample_rate_hz": src_info.samplerate,
            "channels": src_info.channels,
            "sample_format": src_info.subtype,
            "frames": src_info.frames,
            "peak": src_peak,
            "peak_int16": int(round(src_peak * 32768)),
        },
        "processing": {
            "operation": "linear_gain",
            "formula": "float32(int16 / 32768 * gain)",
            "target_peak": target_peak,
            "gain": gain,
            "gain_db": float(20 * np.log10(gain)),
            "clipping_or_limiting_applied": False,
            "resampling_applied": False,
            "channel_mixing_applied": False,
        },
        "output": {
            "path": _rel(out_wav, repo_root),
            "sha256": sha256_of(out_wav),
            "file_size_bytes": out_wav.stat().st_size,
            "sample_rate_hz": out_info.samplerate,
            "channels": out_info.channels,
            "sample_format": out_info.subtype,
            "frames": out_info.frames,
            "peak": out_peak,
            "max_allowed_peak": MAX_ALLOWED_PEAK,
        },
        "tools": {
            "script": _rel(Path(__file__), REPO_ROOT),
            "script_sha256": sha256_of(Path(__file__)),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "soundfile": sf.__version__,
            "libsndfile": sf.__libsndfile_version__,
        },
        "governance": {
            "speaker_consent_confirmed_by_user": True,
            "speaker_consent_confirmed_on": consent_confirmed_on,
            "listening_check_confirmed_by_user": True,
            "listening_check_confirmed_on": listening_confirmed_on,
            "note": "Confirmations reported by the project owner; no consent document is stored in the repository.",
        },
    }
    with open(out_manifest, "x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a gain-reduced training derivative (never overwrites)")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST), help="Parent split manifest")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="New directory for the derived dataset")
    parser.add_argument("--target-peak", type=float, default=DEFAULT_TARGET_PEAK)
    parser.add_argument("--consent-confirmed-on", type=str, required=True, help="Date the owner confirmed speaker consent (YYYY-MM-DD)")
    parser.add_argument("--listening-confirmed-on", type=str, required=True, help="Date the owner confirmed the listening check (YYYY-MM-DD)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    m = derive(manifest_path, output_dir, args.target_peak, args.consent_confirmed_on, args.listening_confirmed_on)
    print(json.dumps({k: m[k] for k in ("input", "processing", "output")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
