"""Standalone harness: left-context overlap between 0.5 s chunks.

Each chunk is converted with `ctx` extra samples of preceding input (left context);
the output samples that correspond to that context are discarded, so every chunk still
contributes exactly CHUNK output samples. Wall-clock inference time is measured for the
baseline and for each overlap setting; no linear extrapolation is used.

Variants:
    baseline     - 0.5 s chunks, no context (current pipeline behaviour, padding as configured)
    ctx_<ms>     - 0.5 s chunks with <ms> of left context, context output dropped

Metrics (against one whole-signal conversion as reference):
    boundary_jump_mean / max  - sample jump at chunk boundaries (absolute amplitude)
    boundary_jump_ratio       - boundary jump divided by the same jump in the whole reference
    lsd_near / lsd_away       - log-spectral distance (dB) to the reference near / away from boundaries
    inference_seconds         - total wall time of all convert_chunk calls
    rtf                       - inference_seconds / signal duration

Usage (repository root, project venv):
    .venv\\Scripts\\python.exe scripts\\overlap_test.py --overlap-ms 100 --pad 80,80
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from voice_changer.conversion.base import ConversionConfig  # noqa: E402
from voice_changer.conversion.rvc.adapter import RVCAdapter  # noqa: E402
from ab_padding_test import CHUNK, SR, boundary_excess, make_signal, spectral_distance  # noqa: E402


def chunked_convert(adapter: RVCAdapter, x: np.ndarray, cfg: ConversionConfig, ctx: int) -> tuple[np.ndarray, float]:
    """Convert x in CHUNK-sized pieces, optionally with `ctx` samples of left context."""
    pieces = []
    t0 = time.perf_counter()
    for start in range(0, len(x), CHUNK):
        ctx_eff = min(ctx, start)  # no context before the first chunk
        seg = x[start - ctx_eff : start + CHUNK]
        out = np.asarray(adapter.convert_chunk(seg, cfg, input_sr=SR), dtype=np.float32)
        pieces.append(out[ctx_eff : ctx_eff + CHUNK])
    elapsed = time.perf_counter() - t0
    return np.concatenate(pieces), elapsed


def metrics(whole: np.ndarray, chunked: np.ndarray) -> dict:
    n = min(len(whole), len(chunked))
    whole, chunked = whole[:n], chunked[:n]
    boundaries = list(range(CHUNK, n, CHUNK))
    jumps = [abs(float(chunked[b]) - float(chunked[b - 1])) for b in boundaries if b + 480 < n]
    ref_jumps = [abs(float(whole[b]) - float(whole[b - 1])) for b in boundaries if b + 480 < n]
    ratios = []
    for b in boundaries:
        if b + 480 >= n:
            continue
        rc = boundary_excess(chunked, b)
        rw = boundary_excess(whole, b)
        if rc is not None and rw is not None:
            ratios.append(rc / max(rw, 1e-9))
    spec = spectral_distance(whole, chunked, boundaries)
    return {
        "boundary_jump_mean": round(float(np.mean(jumps)), 5),
        "boundary_jump_max": round(float(np.max(jumps)), 5),
        "reference_jump_mean": round(float(np.mean(ref_jumps)), 5),
        "boundary_jump_ratio_mean": round(float(np.mean(ratios)), 3) if ratios else None,
        "lsd_near_db": spec["lsd_db_near_boundary"],
        "lsd_away_db": spec["lsd_db_away_from_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlap (left-context) test for chunked conversion")
    parser.add_argument("--overlap-ms", type=int, nargs="+", default=[100])
    parser.add_argument("--pad", type=str, default="80,80")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=str, default="output/overlap/overlap_results.json")
    args = parser.parse_args()

    model = REPO_ROOT / "models" / "weights" / "vctk_female_p231.pth"
    index = REPO_ROOT / "models" / "indices" / "vctk_female_p231.index"
    adapter = RVCAdapter()
    adapter.load_model(str(model), str(index))
    left, right = (int(v) for v in args.pad.split(","))
    adapter.pad_16k = (left, right)

    cfg = ConversionConfig(pitch_shift=10.0, f0_method="rmvpe", index_rate=0.5, filter_radius=2, protect_rate=0.4)
    x = make_signal(args.seconds, seed=args.seed)

    t0 = time.perf_counter()
    whole = np.asarray(adapter.convert_chunk(x, cfg, input_sr=SR), dtype=np.float32)
    whole_seconds = time.perf_counter() - t0

    results = {}
    baseline, base_s = chunked_convert(adapter, x, cfg, ctx=0)
    results["baseline"] = {**metrics(whole, baseline), "inference_seconds": round(base_s, 2),
                           "rtf": round(base_s / args.seconds, 3)}
    print("baseline", results["baseline"], flush=True)

    for ms in args.overlap_ms:
        ctx = int(SR * ms / 1000)
        out, secs = chunked_convert(adapter, x, cfg, ctx=ctx)
        results[f"ctx_{ms}"] = {**metrics(whole, out), "inference_seconds": round(secs, 2),
                                "rtf": round(secs / args.seconds, 3),
                                "overhead_vs_baseline": round(secs / base_s - 1, 3)}
        print(f"ctx_{ms}", results[f"ctx_{ms}"], flush=True)

    report = {
        "seed": args.seed,
        "seconds": args.seconds,
        "pad_16k": [left, right],
        "whole_reference_seconds": round(whole_seconds, 2),
        "results": results,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
