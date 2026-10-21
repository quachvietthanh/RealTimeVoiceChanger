"""Read-only verification of the custom-voice train/held-out split manifest.

Checks `data/source_audio/dataset_split_manifest.json` against the audio on disk:
- SHA-256 and byte size of the source and every split output file
- Sample rate, channels, sample format and frame counts
- Half-open frame intervals: ordered, non-overlapping, inside the source
- Training and held-out files live in separate directories
- Each split file is a sample-exact copy of its source interval

With `--derived <derived_manifest.json>` it re-verifies the parent manifest and then checks
that the derived file is exactly float32(parent training audio * gain), below the peak limit,
and stored apart from the held-out audio.

Never writes, moves or deletes any audio. Exit code 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "source_audio" / "dataset_split_manifest.json"

TRAINING_KEY = "training"
HELD_OUT_KEY = "held_out_test"


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _check_file(label: str, path: Path, expected_sha: str, expected_size: int) -> list[CheckResult]:
    if not path.is_file():
        return [CheckResult(f"{label}: exists", False, f"missing: {path}")]
    actual_size = path.stat().st_size
    actual_sha = sha256_of(path)
    return [
        CheckResult(f"{label}: exists", True, str(path)),
        CheckResult(f"{label}: size", actual_size == expected_size, f"{actual_size} bytes (expected {expected_size})"),
        CheckResult(f"{label}: sha256", actual_sha == expected_sha, f"{actual_sha} (expected {expected_sha})"),
    ]


def _check_format(label: str, info: Any, sample_rate: int, channels: int, subtype: str, frames: int) -> list[CheckResult]:
    return [
        CheckResult(f"{label}: sample rate", info.samplerate == sample_rate, f"{info.samplerate} Hz (expected {sample_rate})"),
        CheckResult(f"{label}: channels", info.channels == channels, f"{info.channels} (expected {channels})"),
        CheckResult(f"{label}: sample format", info.subtype == subtype, f"{info.subtype} (expected {subtype})"),
        CheckResult(f"{label}: frames", info.frames == frames, f"{info.frames} (expected {frames})"),
    ]


def verify_manifest(manifest_path: Path, repo_root: Path = REPO_ROOT, compare_samples: bool = True) -> list[CheckResult]:
    """Run every manifest check and return the individual results."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest["source_audio"]
        split = manifest["split_specifications"]
        segments: dict = split["segments"]
        source_path = repo_root / source["path"]
        sample_rate = int(source["sample_rate_hz"])
        channels = int(source["channels"])
        subtype = str(source["sample_format"])
        total_frames = int(source["total_frames"])
    except (OSError, ValueError, KeyError, TypeError) as err:
        return [CheckResult("manifest: parse", False, f"{type(err).__name__}: {err}")]

    results = [CheckResult("manifest: parse", True, str(manifest_path))]
    results.append(CheckResult(
        "manifest: partition policy",
        split.get("partition_policy") == "half_open_frame_intervals",
        str(split.get("partition_policy")),
    ))
    results.append(CheckResult(
        "manifest: split sample rate",
        int(split.get("sample_rate_hz", -1)) == sample_rate,
        f"{split.get('sample_rate_hz')} Hz (source {sample_rate})",
    ))
    for key in (TRAINING_KEY, HELD_OUT_KEY):
        results.append(CheckResult(f"manifest: '{key}' segment present", key in segments, ""))
    if not all(key in segments for key in (TRAINING_KEY, HELD_OUT_KEY)):
        return results

    # Source file
    results += _check_file("source", source_path, source["sha256"], int(source["file_size_bytes"]))
    source_ok = source_path.is_file()
    if source_ok:
        results += _check_format("source", sf.info(str(source_path)), sample_rate, channels, subtype, total_frames)

    # Interval geometry over all segments, sorted by start frame
    intervals = []
    for name, seg in segments.items():
        start, end, count = int(seg["start_frame"]), int(seg["end_frame"]), int(seg["frame_count"])
        results.append(CheckResult(
            f"{name}: interval",
            0 <= start < end <= total_frames and end - start == count,
            f"[{start}, {end}) count={count} total={total_frames}",
        ))
        expected_start = round(float(seg["start_seconds"]) * sample_rate)
        results.append(CheckResult(
            f"{name}: start_seconds matches start_frame",
            expected_start == start,
            f"{seg['start_seconds']} s -> {expected_start} (manifest {start})",
        ))
        intervals.append((start, end, name))
    intervals.sort()
    overlaps = [f"{a[2]}/{b[2]}" for a, b in zip(intervals, intervals[1:]) if b[0] < a[1]]
    results.append(CheckResult("segments: no overlap", not overlaps, ", ".join(overlaps) or "none"))
    covered = intervals[0][0] == 0 and intervals[-1][1] == total_frames and all(
        a[1] == b[0] for a, b in zip(intervals, intervals[1:])
    )
    results.append(CheckResult("segments: cover source without gaps", covered, f"{len(intervals)} segments"))

    # Training and held-out audio must never share a directory that preprocessing could scan
    train_out = (repo_root / segments[TRAINING_KEY]["output_path"]).resolve()
    held_out = (repo_root / segments[HELD_OUT_KEY]["output_path"]).resolve()
    separated = not (held_out.is_relative_to(train_out.parent) or train_out.is_relative_to(held_out.parent))
    results.append(CheckResult(
        "training/held-out: separate directories", separated, f"{train_out.parent} | {held_out.parent}"
    ))

    # Split output files
    for key in (TRAINING_KEY, HELD_OUT_KEY):
        seg = segments[key]
        out_path = repo_root / seg["output_path"]
        file_results = _check_file(key, out_path, seg["sha256"], int(seg["file_size_bytes"]))
        results += file_results
        if not out_path.is_file():
            continue
        results += _check_format(key, sf.info(str(out_path)), sample_rate, channels, subtype, int(seg["frame_count"]))
        if compare_samples and source_ok:
            data, _ = sf.read(str(out_path), dtype="int32", always_2d=True)
            ref, _ = sf.read(str(source_path), dtype="int32", always_2d=True,
                             start=int(seg["start_frame"]), stop=int(seg["end_frame"]))
            results.append(CheckResult(
                f"{key}: sample-exact copy of source interval",
                data.shape == ref.shape and np.array_equal(data, ref),
                f"shape {data.shape} vs {ref.shape}",
            ))

    return results


def verify_derived_manifest(derived_path: Path, repo_root: Path = REPO_ROOT) -> list[CheckResult]:
    """Verify a gain-derived training file against its parent split manifest.

    The parent manifest is fully re-verified first; the derived checks are added on top.
    """
    try:
        derived = json.loads(derived_path.read_text(encoding="utf-8"))
        parent_rel = derived["parent_manifest"]["path"]
        inp, proc, out = derived["input"], derived["processing"], derived["output"]
        gain = float(proc["gain"])
        max_peak = float(out["max_allowed_peak"])
    except (OSError, ValueError, KeyError, TypeError) as err:
        return [CheckResult("derived: parse", False, f"{type(err).__name__}: {err}")]

    parent_path = repo_root / parent_rel
    results = [CheckResult("derived: parse", True, str(derived_path))]
    results += [CheckResult(f"parent/{r.name}", r.passed, r.detail) for r in verify_manifest(parent_path, repo_root)]
    if not parent_path.is_file():
        return results

    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    segment = parent["split_specifications"]["segments"].get(derived["parent_manifest"].get("segment", ""), {})
    results.append(CheckResult(
        "derived: parent manifest sha256", sha256_of(parent_path) == derived["parent_manifest"]["sha256"],
        derived["parent_manifest"]["sha256"],
    ))
    results.append(CheckResult(
        "derived: input is the parent training segment",
        inp["path"] == segment.get("output_path") and inp["sha256"] == segment.get("sha256"),
        f"{inp['path']} {inp['sha256']}",
    ))
    results.append(CheckResult(
        "derived: no clipping/limiting/resampling/mixing declared",
        proc.get("operation") == "linear_gain" and not any(
            proc.get(k) for k in ("clipping_or_limiting_applied", "resampling_applied", "channel_mixing_applied")),
        str(proc.get("operation")),
    ))

    in_path, out_path = repo_root / inp["path"], repo_root / out["path"]
    results += _check_file("derived input", in_path, inp["sha256"], int(inp["file_size_bytes"]))
    results += _check_file("derived output", out_path, out["sha256"], int(out["file_size_bytes"]))
    if not (in_path.is_file() and out_path.is_file()):
        return results

    in_info = sf.info(str(in_path))
    results += _check_format("derived output", sf.info(str(out_path)), in_info.samplerate, in_info.channels,
                             str(out["sample_format"]), in_info.frames)

    held_out = parent["split_specifications"]["segments"].get(HELD_OUT_KEY, {})
    held_dir = (repo_root / held_out["output_path"]).resolve().parent if "output_path" in held_out else None
    out_dir = out_path.resolve().parent
    results.append(CheckResult(
        "derived: separate from held-out directory",
        held_dir is not None and not (out_dir.is_relative_to(held_dir) or held_dir.is_relative_to(out_dir)),
        f"{out_dir} | {held_dir}",
    ))

    src, _ = sf.read(str(in_path), dtype="float64", always_2d=True)
    dst, _ = sf.read(str(out_path), dtype="float32", always_2d=True)
    expected = (src * gain).astype(np.float32)
    results.append(CheckResult(
        "derived: sample-exact float32(input * gain)",
        dst.shape == expected.shape and np.array_equal(dst, expected),
        f"gain={gain:.9f} shape {dst.shape}",
    ))
    peak = float(np.max(np.abs(dst)))
    results.append(CheckResult("derived: peak below limit", peak < max_peak, f"{peak:.6f} (limit {max_peak})"))
    results.append(CheckResult(
        "derived: recorded peak matches", abs(peak - float(out["peak"])) < 1e-7, f"recorded {out['peak']}"
    ))
    return results


def _print_results(title: str, results: list[CheckResult]) -> bool:
    print("=" * 70)
    print(title)
    print("=" * 70)
    for r in results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name}" + (f" - {r.detail}" if r.detail else ""))
    failed = [r for r in results if not r.passed]
    print("-" * 70)
    print(f"Overall: {'PASS' if not failed else 'FAIL'} ({len(results) - len(failed)}/{len(results)} checks passed)")
    return not failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the dataset split manifest (read-only)")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST), help="Path to dataset_split_manifest.json")
    parser.add_argument("--skip-sample-compare", action="store_true", help="Skip the sample-exact comparison against the source")
    parser.add_argument("--derived", type=str, default=None,
                        help="Verify a derived_manifest.json (and its parent) instead of the split manifest")
    args = parser.parse_args()

    if args.derived:
        derived_path = Path(args.derived)
        if not derived_path.is_absolute():
            derived_path = REPO_ROOT / derived_path
        ok = _print_results(f"Derived Dataset Manifest Verification: {derived_path}", verify_derived_manifest(derived_path))
        return 0 if ok else 1

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path

    results = verify_manifest(manifest_path, compare_samples=not args.skip_sample_compare)
    ok = _print_results(f"Dataset Split Manifest Verification: {manifest_path}", results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
