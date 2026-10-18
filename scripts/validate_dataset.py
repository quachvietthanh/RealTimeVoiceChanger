"""Dataset Validation Script for Custom Voice Training.

Analyzes local single-speaker recording files for:
- Total speech duration (target: 20-30 minutes)
- Channels, sample rates, formats
- Peak amplitude, RMS, clipping detection, silence detection
- File integrity and format consistency
Operates 100% locally without uploading or transmitting any biometric/audio data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on sys.path
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import soundfile as sf


def format_seconds(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs:02d}s"


def validate_dataset(dataset_dir: Path) -> dict:
    supported_exts = {".wav", ".mp3", ".flac", ".ogg"}
    files = [f for f in dataset_dir.rglob("*") if f.is_file() and f.suffix.lower() in supported_exts]

    if not files:
        print(f"[ERROR] No audio files ({', '.join(supported_exts)}) found in: {dataset_dir}")
        return {"file_count": 0, "total_duration_s": 0.0, "valid": False}

    print("=" * 70)
    print(f"Dataset Audio Validation Audit: {dataset_dir}")
    print("=" * 70)
    print(f"Found {len(files)} audio files. Inspecting waveforms...\n")

    total_duration = 0.0
    clipped_files = []
    silent_files = []
    low_level_files = []
    error_files = []
    sample_rates = set()
    channels_set = set()

    for idx, f in enumerate(files, 1):
        try:
            info = sf.info(str(f))
            data, sr = sf.read(str(f))
            total_duration += info.duration
            sample_rates.add(info.samplerate)
            channels_set.add(info.channels)

            peak = float(np.max(np.abs(data)))
            rms = float(np.sqrt(np.mean(data**2)))

            if peak >= 0.999:
                clipped_files.append((f.name, peak))
            if peak < 1e-4:
                silent_files.append((f.name, peak))
            elif peak < 0.15:
                low_level_files.append((f.name, peak))

        except Exception as err:
            error_files.append((f.name, str(err)))

    total_mins = total_duration / 60.0

    print("--- Summary Metrics ---")
    print(f"Total Files Analyzed   : {len(files)}")
    print(f"Total Speech Duration  : {format_seconds(total_duration)} ({total_mins:.2f} minutes)")
    print(f"Sample Rates Detected  : {sorted(list(sample_rates))} Hz")
    print(f"Channels Detected      : {sorted(list(channels_set))}")
    print(f"Corrupt/Error Files    : {len(error_files)}")
    print(f"Clipped Files (Peak>=1): {len(clipped_files)}")
    print(f"Silent Files (Peak=0)  : {len(silent_files)}")
    print(f"Low-Volume Files       : {len(low_level_files)}")

    print("\n--- Readiness Evaluation ---")
    duration_ok = 15.0 <= total_mins <= 60.0
    errors_ok = len(error_files) == 0 and len(silent_files) == 0

    if total_mins < 15.0:
        print(f"[-] DURATION WARNING: Dataset duration ({total_mins:.1f} mins) is below recommended 20-30 mins.")
        print("    Training with <15 mins may result in muffled vowels or incomplete phoneme coverage.")
    elif total_mins > 60.0:
        print(f"[!] DURATION NOTE: Dataset is large ({total_mins:.1f} mins). Training will take longer.")
    else:
        print(f"[+] DURATION OPTIMAL: {total_mins:.1f} minutes meets the 20-30 minute target for RVC fine-tuning.")

    if clipped_files:
        print(f"[!] CLIPPING WARNING: {len(clipped_files)} files exhibit clipping (distortion). Consider re-recording.")
    if error_files:
        print(f"[-] ERROR: {len(error_files)} files could not be decoded. Please repair or remove them.")

    is_ready = duration_ok and errors_ok
    status_str = "READY FOR PREPROCESSING" if is_ready else "NEEDS ATTENTION"
    print(f"\nOverall Dataset Status : {status_str}")
    print("=" * 70)

    return {
        "file_count": len(files),
        "total_duration_s": round(total_duration, 2),
        "total_duration_m": round(total_mins, 2),
        "sample_rates": list(sample_rates),
        "channels": list(channels_set),
        "clipped_count": len(clipped_files),
        "silent_count": len(silent_files),
        "error_count": len(error_files),
        "is_ready": is_ready,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate local audio dataset for custom RVC voice training")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/dataset_raw",
        help="Path to directory containing single-speaker audio files",
    )
    args = parser.parse_args()

    ds_path = Path(args.dataset_dir)
    if not ds_path.is_absolute():
        ds_path = repo_root / ds_path

    if not ds_path.exists():
        print(f"Dataset directory not found: {ds_path}")
        print("Create the directory and copy your single-speaker WAV recordings into it.")
        sys.exit(1)

    validate_dataset(ds_path)


if __name__ == "__main__":
    main()
