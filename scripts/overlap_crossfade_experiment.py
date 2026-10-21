"""Controlled experiment: left context vs overlap-crossfade for independent 0.5 s chunks.

Harness only. Production padding (80, 80), cropping, feature extraction and streaming
behaviour are not modified; the adapter is used exactly as the pipeline uses it.

Configurations (identical source timeline, 64 full chunks of 20000 samples at 40 kHz):
    A_baseline     chunk converted alone                                  (production)
    B_ctx100       chunk + 100 ms preceding input; context output discarded
    C_ctx200       chunk + 200 ms preceding input; context output discarded
    D_xf50         C's raw outputs; last 50 ms of each chunk held back and crossfaded
                   (equal-gain raised cosine) with the next chunk's context output
                   for the same timeline span. +50 ms algorithmic latency.
    D_xf100        as D_xf50 with a 100 ms crossfade (+100 ms latency)
    D_xf50_eqpow   as D_xf50 with an equal-power (sin/cos) fade

D variants reuse C's inference results from the same repetition, so they add no
inference calls; their inference timing is C's timing plus measured assembly time.

Every chunk contributes exactly CHUNK output samples, and output sample i maps to input
timeline sample i in every configuration (context output is never counted as new output).

Controls: warm-up calls for every segment length, randomised config order per repetition,
per-call torch seeds (1000*rep + chunk) shared by all configs within a repetition, and a
whole-file reference conversion per repetition.

Usage (repository root):
    .venv\\Scripts\\python.exe scripts\\overlap_crossfade_experiment.py --reps 6
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from ab_padding_test import CHUNK, SR, envelope_lag_ms, file_sha256, load_speech, log_spectrogram  # noqa: E402
from voice_changer.conversion.base import ConversionConfig  # noqa: E402
from voice_changer.conversion.rvc.adapter import RVCAdapter  # noqa: E402

MS = SR // 1000
INFER_CONFIGS = {"A_baseline": 0, "B_ctx100": 100 * MS, "C_ctx200": 200 * MS}
XFADE_CONFIGS = {  # name: (source config, crossfade samples, fade shape)
    "D_xf50": ("C_ctx200", 50 * MS, "equal_gain"),
    "D_xf100": ("C_ctx200", 100 * MS, "equal_gain"),
    "D_xf50_eqpow": ("C_ctx200", 50 * MS, "equal_power"),
}
ALL_CONFIGS = list(INFER_CONFIGS) + list(XFADE_CONFIGS)
REGION_BEFORE = 100 * MS  # analysis region [b - 100 ms, b + 10 ms) covers the longest crossfade
REGION_AFTER = 10 * MS
ENV_FRAME = 5 * MS
JUMP_WIN = 480  # +-12 ms window for the local median jump (same as ab_padding_test)


# ----------------------------------------------------------------------------- inference

class RawLengthProbe:
    """Records the raw Applio output length per call to detect the adapter's edge-pad branch."""

    def __init__(self, vc_pipeline_owner) -> None:
        self.owner = vc_pipeline_owner
        self.orig = vc_pipeline_owner.pipeline
        self.lengths: list[int] = []

        def wrapped(*a, **kw):
            out = self.orig(*a, **kw)
            self.lengths.append(int(len(out)))
            return out

        vc_pipeline_owner.pipeline = wrapped


def convert_timeline(adapter, x, cfg, ctx, n_chunks, seed_base, probe):
    """Run one configuration over the timeline. Returns raw outputs, per-call ms, and stats."""
    raws, ctx_effs, call_ms, errors, len_mismatch, edge_fills = [], [], [], 0, 0, 0
    for k in range(n_chunks):
        s = k * CHUNK
        ctx_eff = min(ctx, s)
        seg = x[s - ctx_eff : s + CHUNK]
        torch.manual_seed(seed_base + k)
        n_probe = len(probe.lengths)
        t0 = time.perf_counter()
        try:
            out = np.asarray(adapter.convert_chunk(seg, cfg, input_sr=SR, target_sr=SR), dtype=np.float32)
        except Exception as err:  # recorded, never hidden
            print(f"[error] chunk {k}: {err}", flush=True)
            errors += 1
            out = np.zeros(len(seg), dtype=np.float32)
        call_ms.append((time.perf_counter() - t0) * 1000.0)
        if len(out) != len(seg):
            len_mismatch += 1
        if len(probe.lengths) > n_probe and probe.lengths[-1] < len(seg):
            edge_fills += 1
        raws.append(out)
        ctx_effs.append(ctx_eff)
    return raws, ctx_effs, call_ms, {"errors": errors, "length_mismatches": len_mismatch, "edge_pad_calls": edge_fills}


def assemble_discard(raws, ctx_effs):
    return np.concatenate([r[c : c + CHUNK] for r, c in zip(raws, ctx_effs)])


def fade_curves(n, shape):
    t = (np.arange(n) + 0.5) / n
    if shape == "equal_gain":
        w_in = np.sin(0.5 * np.pi * t) ** 2
        return 1.0 - w_in, w_in
    return np.cos(0.5 * np.pi * t), np.sin(0.5 * np.pi * t)


def assemble_crossfade(raws, ctx_effs, xf, shape):
    """Streaming-faithful overlap-add: hold back xf samples of each chunk, blend with next chunk's context."""
    w_out, w_in = fade_curves(xf, shape)
    pieces, coherence = [], []
    held = None
    for r, c in zip(raws, ctx_effs):
        valid = r[c : c + CHUNK]
        if held is None:
            pieces.append(valid[: CHUNK - xf])
        else:
            assert c >= xf, "context shorter than crossfade"
            head = r[c - xf : c]  # this call's estimate of the held span
            pieces.append(held * w_out + head * w_in)
            pieces.append(valid[: CHUNK - xf])
            denom = np.linalg.norm(held) * np.linalg.norm(head)
            coherence.append(float(np.dot(held, head) / denom) if denom > 0 else float("nan"))
        held = valid[CHUNK - xf :]
    pieces.append(held)  # flush at end of stream
    return np.concatenate(pieces), coherence


# ----------------------------------------------------------------------------- metrics

def stats(v):
    v = np.asarray([e for e in v if e is not None and np.isfinite(e)], dtype=np.float64)
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def frame_db(x, a, b):
    seg = x[a:b]
    n = len(seg) // ENV_FRAME
    rms = np.sqrt((seg[: n * ENV_FRAME].reshape(n, ENV_FRAME) ** 2).mean(axis=1))
    return 20 * np.log10(rms + 1e-7), rms


def point_metrics(y, ref, pos):
    """Adjacent-sample jump, local-normalised jump and 2nd-difference residual at sample pos."""
    jump = abs(float(y[pos]) - float(y[pos - 1]))
    win = np.abs(np.diff(y[pos - JUMP_WIN : pos + JUMP_WIN]))
    win = np.delete(win, JUMP_WIN - 1)
    resid = abs(float(y[pos]) - 2 * float(y[pos - 1]) + float(y[pos - 2]))
    d2 = np.abs(np.diff(y[pos - JUMP_WIN : pos + JUMP_WIN], 2))
    return {
        "jump": jump,
        "jump_norm": jump / (float(np.median(win)) + 1e-9),
        "resid2_norm": resid / (float(np.median(d2)) + 1e-9),
        "ref_jump": abs(float(ref[pos]) - float(ref[pos - 1])),
    }


def region_metrics(y, ref, pos):
    """Region [pos-100 ms, pos+10 ms): max normalised jump, envelope deviation and dip vs reference, LSD."""
    a, b = pos - REGION_BEFORE, pos + REGION_AFTER
    d = np.abs(np.diff(y[a:b]))
    dr = np.abs(np.diff(ref[a:b]))
    db_y, _ = frame_db(y, a, b)
    db_r, rms_r = frame_db(ref, a, b)
    active = rms_r > 1e-4
    dev = db_y - db_r
    # level step across pos: 10 ms after vs 10 ms before, minus the same step in the reference
    step = lambda s: 20 * np.log10((np.sqrt(np.mean(s[pos : pos + 400] ** 2)) + 1e-7)
                                   / (np.sqrt(np.mean(s[pos - 400 : pos] ** 2)) + 1e-7))
    la = log_spectrogram(y[a - 512 : b + 512])
    lr = log_spectrogram(ref[a - 512 : b + 512])
    m = min(len(la), len(lr))
    return {
        "region_max_jump_norm": float(d.max() / (np.median(d) + 1e-9)),
        "ref_region_max_jump_norm": float(dr.max() / (np.median(dr) + 1e-9)),
        "env_dev_db": float(np.mean(np.abs(dev[active]))) if active.any() else None,
        "env_min_dip_db": float(dev[active].min()) if active.any() else None,
        "level_step_dev_db": float(abs(step(y) - step(ref))),
        "lsd_db": float(np.mean(np.abs(la[:m] - lr[:m]))),
    }


def chunk_lag_ms(y, ref, a, b, max_lag=40):
    """Per-chunk lag (ms, 0.25 ms resolution) of a 2 ms smoothed |y| envelope vs the reference."""
    kern = np.ones(80) / 80
    ey = np.convolve(np.abs(y[a:b]), kern, mode="same")[::10]
    er = np.convolve(np.abs(ref[a:b]), kern, mode="same")[::10]
    ey, er = ey - ey.mean(), er - er.mean()
    best, best_c = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        p, q = (er[: len(er) - lag], ey[lag:]) if lag >= 0 else (er[-lag:], ey[: len(ey) + lag])
        c = float(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q) + 1e-12))
        if c > best_c:
            best, best_c = lag, c
    return best * 0.25, best_c


def silence_ms(y, ref):
    """Total ms of 10 ms frames that are near-silent in y (< -80 dBFS) but active in the reference (> -50 dBFS)."""
    n = len(y) // 400
    ry = np.sqrt((y[: n * 400].reshape(n, 400) ** 2).mean(axis=1))
    rr = np.sqrt((ref[: n * 400].reshape(n, 400) ** 2).mean(axis=1))
    return float(10 * np.sum((ry < 1e-4) & (rr > 3e-3)))


def evaluate(y, ref, x_in, n_chunks, active_mask):
    boundaries = [k * CHUNK for k in range(1, n_chunks)]
    mids = [b - CHUNK // 2 for b in boundaries]
    rows = []
    for k, (b, m) in enumerate(zip(boundaries, mids), start=1):
        pb, pm = point_metrics(y, ref, b), point_metrics(y, ref, m)
        rb, rm = region_metrics(y, ref, b), region_metrics(y, ref, m)
        rows.append({"boundary": k, "sample": b, "active": bool(active_mask[k - 1]),
                     **{f"b_{key}": v for key, v in {**pb, **rb}.items()},
                     **{f"m_{key}": v for key, v in {**pm, **rm}.items()}})
    lags = []
    for k in range(n_chunks):
        lag, corr = chunk_lag_ms(y, ref, k * CHUNK, (k + 1) * CHUNK)
        lags.append({"chunk": k, "lag_ms": lag, "corr": corr})
    summary = {
        "len_output": int(len(y)),
        "len_expected": int(n_chunks * CHUNK),
        "clipped_samples": int(np.sum(np.abs(y) >= 0.999)),
        "nan_samples": int(np.sum(~np.isfinite(y))),
        "spurious_silence_ms": silence_ms(y, ref),
        "peak": float(np.abs(y).max()),
        "rms": float(np.sqrt(np.mean(y ** 2))),
        "lag_vs_input_ms": envelope_lag_ms(x_in, y),
        "lag_vs_reference_ms": envelope_lag_ms(ref, y),
    }
    return rows, lags, summary


def f0_track(y):
    import librosa

    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
    f0, _, vprob = librosa.pyin(y16, fmin=70, fmax=900, sr=16000, frame_length=1024, hop_length=160)
    return f0, vprob  # frame i centred at i * 10 ms


def f0_metrics(f0, vp, f0_ref, vp_ref, n_chunks, thr=0.5):
    cents = 1200 * np.log2(np.where(np.isfinite(f0), f0, np.nan) / 55.0)
    cents_r = 1200 * np.log2(np.where(np.isfinite(f0_ref), f0_ref, np.nan) / 55.0)
    ok = (vp > thr) & (vp_ref > thr) & np.isfinite(cents) & np.isfinite(cents_r)
    out = {}
    for tag, offset in (("boundary", 0), ("mid", -CHUNK // 2)):
        steps, errs = [], []
        for k in range(1, n_chunks):
            fb = (k * CHUNK + offset) // 400  # 400 samples per 10 ms frame at 40 kHz
            if fb + 3 >= len(cents):
                continue
            if ok[fb - 1] and ok[fb + 1]:
                steps.append(abs(cents[fb + 1] - cents[fb - 1]))
            idx = [i for i in range(fb - 3, fb + 4) if ok[i]]
            if len(idx) >= 4:
                errs.append(float(np.mean(np.abs(cents[idx] - cents_r[idx]))))
        out[f"{tag}_step_cents"] = stats(steps)
        out[f"{tag}_err_vs_ref_cents"] = stats(errs)
    out["voiced_frames_both"] = int(ok.sum())
    return out


# ----------------------------------------------------------------------------- environment

def gpu_state():
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=5).strip()
        t, clk, util = (v.strip() for v in q.split(","))
        return {"temp_c": int(t), "sm_clock_mhz": int(clk), "util_pct": int(util)}
    except Exception:
        return None


def git_rev():
    run = lambda *a: subprocess.check_output(["git", *a], cwd=REPO_ROOT, text=True).strip()
    try:
        return {"head": run("rev-parse", "HEAD"), "status_porcelain": run("status", "--porcelain")}
    except Exception:
        return {"head": "unknown"}


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description="Context / overlap-crossfade experiment for chunked RVC conversion")
    p.add_argument("--input", default="tests/fixtures/input_male.wav")
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--warmup-calls", type=int, default=6)
    p.add_argument("--max-chunks", type=int, default=None, help="Limit chunks (smoke test only)")
    p.add_argument("--no-f0", action="store_true")
    p.add_argument("--order-seed", type=int, default=20261020)
    p.add_argument("--out", default="output/overlap_crossfade_2026-10-20")
    args = p.parse_args()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    in_path = REPO_ROOT / args.input
    model = REPO_ROOT / "models" / "weights" / "vctk_female_p231.pth"
    index = REPO_ROOT / "models" / "indices" / "vctk_female_p231.index"

    x_full = load_speech(in_path)
    n_chunks = len(x_full) // CHUNK
    if args.max_chunks:
        n_chunks = min(n_chunks, args.max_chunks)
    x = x_full[: n_chunks * CHUNK]
    print(f"[input] {in_path.name}: {len(x_full)} samples @ {SR} Hz, using {n_chunks} chunks ({len(x)} samples)", flush=True)

    # Activity per boundary from the INPUT (+-20 ms RMS above 10% of the 95th pct 40 ms frame RMS)
    fr = np.sqrt((x[: len(x) // 1600 * 1600].reshape(-1, 1600) ** 2).mean(axis=1))
    act_thr = 0.1 * np.percentile(fr, 95)
    active_mask = [float(np.sqrt(np.mean(x[b - 800 : b + 800] ** 2))) > act_thr
                   for b in (k * CHUNK for k in range(1, n_chunks))]

    adapter = RVCAdapter()
    adapter.load_model(str(model), str(index))
    assert adapter.pad_16k == (80, 80), "production padding must stay (80, 80)"
    applio_vc = adapter._applio_vc
    assert applio_vc is not None
    probe = RawLengthProbe(applio_vc.vc)
    cfg = ConversionConfig(pitch_shift=10.0, f0_method="rmvpe", index_rate=0.5, filter_radius=2, protect_rate=0.4)
    device = "cuda:" + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    meta = {
        "input": args.input, "input_sha256": file_sha256(in_path), "input_info": str(sf.info(str(in_path))),
        "samples_40k_full": int(len(x_full)), "samples_used": int(len(x)), "chunks": n_chunks,
        "boundaries": n_chunks - 1, "active_boundaries": int(sum(active_mask)), "chunk_samples": CHUNK,
        "model_sha256": file_sha256(model), "index_sha256": file_sha256(index),
        "config": cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg), "pad_16k": list(adapter.pad_16k),
        "applio_is_half": bool(getattr(applio_vc.vc, "is_half", None)),
        "device": device, "torch": torch.__version__, "numpy": np.__version__, "python": sys.version,
        "platform": platform.platform(), "processor": platform.processor(), "torch_threads": torch.get_num_threads(),
        "git": git_rev(), "reps": args.reps, "order_seed": args.order_seed,
        "infer_configs_ctx_samples": INFER_CONFIGS, "xfade_configs": XFADE_CONFIGS,
    }

    # Warm-up: every segment length the experiment will use (chunk 0 length and context lengths)
    print("[warmup]", flush=True)
    for ctx in INFER_CONFIGS.values():
        for i in range(args.warmup_calls):
            s = (1 + i % max(1, n_chunks - 1)) * CHUNK
            c = min(ctx, s)
            adapter.convert_chunk(x[s - c : s + CHUNK], cfg, input_sr=SR, target_sr=SR)
    adapter.convert_chunk(x, cfg, input_sr=SR, target_sr=SR)

    rng = random.Random(args.order_seed)
    outputs_rep0: dict = {}
    rows_all, lags_all, summaries, timings, coherences, orders = [], [], [], {c: [] for c in ALL_CONFIGS}, {}, []
    for rep in range(args.reps):
        order = list(INFER_CONFIGS)
        rng.shuffle(order)
        orders.append(order)
        torch.manual_seed(1000 * rep + 999)
        t0 = time.perf_counter()
        ref = np.asarray(adapter.convert_chunk(x, cfg, input_sr=SR, target_sr=SR), dtype=np.float32)
        ref_s = time.perf_counter() - t0
        print(f"[rep {rep}] order={order} reference {ref_s:.2f}s gpu={gpu_state()}", flush=True)
        outputs = {"reference": ref}
        raws_by_cfg = {}
        for name in order:
            g0 = gpu_state()
            raws, ctx_effs, call_ms, st = convert_timeline(adapter, x, cfg, INFER_CONFIGS[name], n_chunks, 1000 * rep, probe)
            raws_by_cfg[name] = (raws, ctx_effs)
            outputs[name] = assemble_discard(raws, ctx_effs)
            timings[name].append({"rep": rep, "call_ms": call_ms, **st, "gpu_before": g0, "gpu_after": gpu_state(),
                                  "position_in_order": order.index(name)})
            print(f"  {name}: mean {np.mean(call_ms):.1f} ms, p95 {np.percentile(call_ms, 95):.1f} ms, {st}", flush=True)
        for name, (src, xf, shape) in XFADE_CONFIGS.items():
            raws, ctx_effs = raws_by_cfg[src]
            t0 = time.perf_counter()
            y, coh = assemble_crossfade(raws, ctx_effs, xf, shape)
            asm_ms = (time.perf_counter() - t0) * 1000.0
            outputs[name] = y
            coherences.setdefault(name, []).extend(coh)
            src_t = timings[src][-1]
            timings[name].append({"rep": rep, "call_ms": src_t["call_ms"], "errors": src_t["errors"],
                                  "length_mismatches": src_t["length_mismatches"], "edge_pad_calls": src_t["edge_pad_calls"],
                                  "assembly_ms_total": asm_ms, "shares_inference_with": src})
        for name, y in outputs.items():
            rows, lags, summ = evaluate(y, ref, x, n_chunks, active_mask)
            rows_all += [{"config": name, "rep": rep, **r} for r in rows]
            lags_all += [{"config": name, "rep": rep, **lg} for lg in lags]
            summaries.append({"config": name, "rep": rep, **summ})
            if rep == 0:
                sf.write(str(out_dir / f"{name}_rep0.wav"), y, SR, subtype="FLOAT")
        if rep == 0:
            sf.write(str(out_dir / "input_40k.wav"), x, SR, subtype="FLOAT")
            outputs_rep0 = outputs

    # Reference noise floor: the whole-file reference of rep 1 evaluated against rep 0
    ref_floor = None
    if args.reps > 1:
        torch.manual_seed(1999)
        ref_b = np.asarray(adapter.convert_chunk(x, cfg, input_sr=SR, target_sr=SR), dtype=np.float32)
        ref_floor = evaluate(ref_b, outputs_rep0["reference"], x, n_chunks, active_mask)
        rows_all += [{"config": "reference_vs_reference", "rep": 0, **r} for r in ref_floor[0]]
        lags_all += [{"config": "reference_vs_reference", "rep": 0, **lg} for lg in ref_floor[1]]
        summaries.append({"config": "reference_vs_reference", "rep": 0, **ref_floor[2]})

    f0_results = {}
    if not args.no_f0:
        print("[f0] pyin on rep 0 outputs", flush=True)
        f0_ref, vp_ref = f0_track(outputs_rep0["reference"])
        for name in ALL_CONFIGS:
            f0, vp = f0_track(outputs_rep0[name])
            f0_results[name] = f0_metrics(f0, vp, f0_ref, vp_ref, n_chunks)
            print(f"  {name}: {f0_results[name]['boundary_step_cents']}", flush=True)

    # Raw outputs
    with open(out_dir / "boundary_rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
        w.writeheader()
        w.writerows(rows_all)
    with open(out_dir / "chunk_lags.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lags_all[0].keys()))
        w.writeheader()
        w.writerows(lags_all)
    raw = {"meta": meta, "orders": orders, "timings": timings, "summaries": summaries,
           "crossfade_coherence": coherences, "f0": f0_results}
    (out_dir / "raw_results.json").write_text(json.dumps(raw, indent=1, default=str), encoding="utf-8")

    # Aggregated table
    agg = {}
    for name in ALL_CONFIGS + ["reference_vs_reference"]:
        rs = [r for r in rows_all if r["config"] == name]
        if not rs:
            continue
        act = [r for r in rs if r["active"]]
        entry: dict = {"n_boundary_samples": len(rs), "n_active_boundary_samples": len(act)}
        for key in ("jump", "jump_norm", "resid2_norm", "region_max_jump_norm", "env_dev_db",
                    "env_min_dip_db", "level_step_dev_db", "lsd_db"):
            entry[f"boundary_{key}"] = stats([r[f"b_{key}"] for r in act])
            entry[f"mid_{key}"] = stats([r[f"m_{key}"] for r in act])
        entry["ref_boundary_jump"] = stats([r["b_ref_jump"] for r in act])
        entry["boundary_jump_all"] = stats([r["b_jump"] for r in rs])
        lg = [l for l in lags_all if l["config"] == name]
        entry["chunk_lag_ms"] = stats([l["lag_ms"] for l in lg])
        entry["chunk_abs_lag_ms"] = stats([abs(l["lag_ms"]) for l in lg])
        entry["chunk_lag_nonzero_fraction"] = float(np.mean([l["lag_ms"] != 0 for l in lg]))
        sm = [s for s in summaries if s["config"] == name]
        entry["summaries"] = {k: [s[k] for s in sm] for k in ("len_output", "clipped_samples", "nan_samples",
                                                               "spurious_silence_ms", "rms")}
        entry["lag_vs_input_ms"] = [s["lag_vs_input_ms"] for s in sm]
        entry["lag_vs_reference_ms"] = [s["lag_vs_reference_ms"] for s in sm]
        if name in timings and timings[name]:
            calls = [c for t in timings[name] for c in t["call_ms"]]
            st = stats(calls)
            per_rep_means = [float(np.mean(t["call_ms"])) for t in timings[name]]
            entry["inference_ms"] = st
            entry["inference_per_rep_mean_ms"] = per_rep_means
            entry["rtf_new_audio"] = {k: st[k] / 500.0 for k in ("mean", "median", "p95", "max")}
            entry["errors"] = sum(t["errors"] for t in timings[name])
            entry["length_mismatches"] = sum(t["length_mismatches"] for t in timings[name])
            entry["edge_pad_calls"] = sum(t["edge_pad_calls"] for t in timings[name])
            if "assembly_ms_total" in timings[name][0]:
                entry["assembly_ms_per_chunk"] = float(np.mean([t["assembly_ms_total"] for t in timings[name]]) / n_chunks)
        if name in coherences:
            entry["crossfade_coherence"] = stats(coherences[name])
        if name in f0_results:
            entry["f0"] = f0_results[name]
        agg[name] = entry
    base = agg["A_baseline"]["inference_ms"]["mean"]
    for name in ALL_CONFIGS:
        agg[name]["inference_overhead_vs_A"] = agg[name]["inference_ms"]["mean"] / base - 1
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "orders": orders, "aggregate": agg},
                                                     indent=1, default=str), encoding="utf-8")
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
