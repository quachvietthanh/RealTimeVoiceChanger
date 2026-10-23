"""Separate model quality from chunked-streaming artefacts, per voice profile (offline, real GPU inference).

For each profile the same input is converted three ways with the same settings and seeds:
  whole   - one convert_chunk call over the whole file (no chunk boundaries: the model's own quality)
  legacy  - 500 ms chunks joined end to end (the streaming behaviour before ChunkStitcher)
  stitch  - 500 ms chunks through ChunkStitcher (left context + SOLA crossfade), as the pipeline now does
The chunked variants call exactly the code the pipeline worker uses (ChunkStitcher + convert_chunk),
without sound devices or real-time pacing. Each chunk call k is seeded torch.manual_seed(1000*rep + k).

Seam metrics are reference-free: at each chunk seam vs at mid-chunk positions of the same output.
  jump_norm   |y[p]-y[p-1]| / median |diff| in +-12 ms      (~1 = no discontinuity)
  resid2_norm 2nd-difference at p / local median            (~1 = smooth)
  step_db     |level 10 ms after p - level 10 ms before p|  (dB)
Only seams with speech in the input (+-20 ms RMS > -40 dBFS) are counted. These are signal metrics, not
a listening test: they show whether seams are discontinuous, not whether the voice sounds natural.

Writes <out>/<profile>__{whole,legacy,stitch}.wav (rep 0) and <out>/summary.json. Refuses to overwrite.
Example:
    .venv\\Scripts\\python scripts/stream_quality_compare.py --out output/live_audio_diag_2026-10-21/offline_compare
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
from voice_changer.service import STREAM_SAMPLE_RATE, VoiceChangerService  # noqa: E402

SR = STREAM_SAMPLE_RATE
CHUNK = SR // 2
MS = SR // 1000
WIN = 12 * MS
DEFAULT_INPUT = REPO / "output/spkb_preview_v1_2026-10-21/eval_input/input_male_peak0p50.wav"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def seam_metrics(y: np.ndarray, p: int) -> dict[str, float]:
    d = np.abs(np.diff(y[p - WIN: p + WIN]))
    d2 = np.abs(np.diff(y[p - WIN: p + WIN], 2))
    jump = abs(float(y[p]) - float(y[p - 1]))
    resid = abs(float(y[p]) - 2 * float(y[p - 1]) + float(y[p - 2]))
    lvl = lambda s: 20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-7)
    return {"jump_norm": jump / (float(np.median(d)) + 1e-9),
            "resid2_norm": resid / (float(np.median(d2)) + 1e-9),
            "step_db": abs(lvl(y[p: p + 10 * MS]) - lvl(y[p - 10 * MS: p]))}


def stats(v: list[float]) -> dict[str, float]:
    a = np.asarray(v, dtype=np.float64)
    if len(a) == 0:
        return {"n": 0}
    return {"n": int(len(a)), "median": round(float(np.median(a)), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "max": round(float(a.max()), 3)}


def run_chunked(conv, cfg, x: np.ndarray, n_chunks: int, stitcher: ChunkStitcher, seed_base: int):
    stitcher.reset()
    out, ms, offsets = [], [], []
    for k in range(n_chunks):
        model_in = stitcher.model_input(x[k * CHUNK: (k + 1) * CHUNK])
        torch.manual_seed(seed_base + k)
        t0 = time.perf_counter()
        y = conv.convert_chunk(model_in, config=cfg, input_sr=SR, target_sr=SR)
        ms.append((time.perf_counter() - t0) * 1000.0)
        out.append(stitcher.join(y))
        offsets.append(stitcher.last_offset)
    return np.concatenate(out), ms, offsets


def analyse(y: np.ndarray, x: np.ndarray, n_chunks: int, delay: int) -> dict:
    """Seams of chunked output sit at k*CHUNK in the output timeline (the stitcher keeps chunk length)."""
    active = lambda p: 20 * np.log10(np.sqrt(np.mean(x[max(p - 20 * MS, 0): p + 20 * MS] ** 2)) + 1e-9) > -40
    seams, mids = [], []
    for k in range(1, n_chunks):
        p = k * CHUNK
        if active(p - delay):
            seams.append(seam_metrics(y, p))
            mids.append(seam_metrics(y, p - CHUNK // 2))
    res = {"active_seams": len(seams)}
    for key in ("jump_norm", "resid2_norm", "step_db"):
        res[f"seam_{key}"] = stats([s[key] for s in seams])
        res[f"mid_{key}"] = stats([m[key] for m in mids])
    res["rms_dbfs"] = round(float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)), 2)
    res["peak"] = round(float(np.abs(y).max()), 4)
    res["clipped"] = int(np.sum(np.abs(y) >= 0.999))
    res["nan"] = int(np.sum(~np.isfinite(y)))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--profiles", default="vctk_p231,voice_a_preview,voice_b_preview")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--context", type=float, default=0.2)
    ap.add_argument("--crossfade", type=float, default=0.05)
    ap.add_argument("--search", type=float, default=0.01)
    ap.add_argument("--override", action="append", default=[],
                    help="extra run 'profile:pitch:index_rate:filter_radius:protect' with these settings")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists():
        print(f"refusing to overwrite {out_dir}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True)

    inp = Path(args.input)
    audio, sr = sf.read(str(inp), dtype="float32", always_2d=True)
    x = librosa.resample(audio.mean(axis=1), orig_sr=sr, target_sr=SR).astype(np.float32)
    n_chunks = len(x) // CHUNK
    x = x[: n_chunks * CHUNK]

    runs = [(p, None) for p in args.profiles.split(",")]
    for o in args.override:
        pid, pitch, ir, fr, prot = o.split(":")
        runs.append((pid, dict(pitch_shift=float(pitch), index_rate=float(ir), filter_radius=int(fr),
                               protect_rate=float(prot))))

    summary = {"input": str(inp), "input_sha256": sha256(inp), "chunks": n_chunks, "chunk_samples": CHUNK,
               "sample_rate": SR, "reps": args.reps, "torch": torch.__version__,
               "stitcher": {"context_s": args.context, "crossfade_s": args.crossfade, "search_s": args.search},
               "runs": []}
    svc = VoiceChangerService()
    try:
        for pid, over in runs:
            svc.select_profile(pid, allow_withdrawn=True)  # diagnostics may measure withdrawn profiles
            conv = svc._converter  # offline harness: drive the loaded converter directly
            cfg = replace(svc._config, **over) if over else svc._config
            tag = pid if over is None else f"{pid}__p{cfg.pitch_shift:+g}_ir{cfg.index_rate:g}_fr{cfg.filter_radius}_pr{cfg.protect_rate:g}"
            legacy = ChunkStitcher(CHUNK)
            stitch = ChunkStitcher(CHUNK, int(args.context * SR), int(args.crossfade * SR), int(args.search * SR))
            run = {"tag": tag, "profile": pid, "config": asdict(cfg), "reps": []}
            for rep in range(args.reps):
                torch.manual_seed(1000 * rep)
                t0 = time.perf_counter()
                whole = conv.convert_chunk(x, config=cfg, input_sr=SR, target_sr=SR)
                whole_ms = (time.perf_counter() - t0) * 1000.0
                y_leg, ms_leg, _ = run_chunked(conv, cfg, x, n_chunks, legacy, 1000 * rep)
                y_st, ms_st, offs = run_chunked(conv, cfg, x, n_chunks, stitch, 1000 * rep)
                r = {"rep": rep, "whole_ms": round(whole_ms, 1),
                     "whole": analyse(whole, x, n_chunks, 0),
                     "legacy": {**analyse(y_leg, x, n_chunks, 0), "call_ms": stats(ms_leg[1:])},
                     "stitch": {**analyse(y_st, x, n_chunks, stitch.nominal_delay_samples),
                                "call_ms": stats(ms_st[1:]), "sola_offsets": stats(offs[1:])}}
                run["reps"].append(r)
                print(json.dumps({"tag": tag, "rep": rep,
                                  **{k: (r[k]["seam_jump_norm"], r[k]["seam_step_db"], r[k]["rms_dbfs"])
                                     for k in ("whole", "legacy", "stitch")}}), flush=True)
                if rep == 0:
                    for name, y in (("whole", whole), ("legacy", y_leg), ("stitch", y_st)):
                        sf.write(str(out_dir / f"{tag}__{name}.wav"), np.asarray(y, dtype=np.float32), SR,
                                 subtype="FLOAT")
            summary["runs"].append(run)
    finally:
        svc.shutdown()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
