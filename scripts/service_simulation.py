"""Headless streaming simulation through the public VoiceChangerService API (real GPU inference, no sound devices).

A WAV file is fed in real time through SimulatedInputStream (looped to fill --seconds) and the converted
audio is consumed by SimulatedOutputStream, which runs the real output ring buffer and underflow
accounting on a clock thread. Nothing is played through the speakers. Several profiles can be run in
sequence to exercise profile switching and VRAM release.

This measures the software pipeline only: inference time/RTF, queue wait, software turnaround and
output buffering. It does not measure physical microphone-to-speaker latency.

Examples (repo root, inference venv):
    .venv\\Scripts\\python scripts/service_simulation.py --profiles voice_a_preview,voice_b_preview --seconds 30
    .venv\\Scripts\\python scripts/service_simulation.py --profiles vctk_p231 --seconds 300 --pitch 10 --json-output out.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from voice_changer.audio.pipeline import RealTimeVoiceChangerPipeline  # noqa: E402
from voice_changer.audio.simulated import SimulatedInputStream, SimulatedOutputStream  # noqa: E402
from voice_changer.service import STREAM_SAMPLE_RATE, AudioSettings, VoiceChangerService  # noqa: E402


def gpu_used_mib() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def torch_mib() -> dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {"allocated_mib": torch.cuda.memory_allocated() / 2**20, "reserved_mib": torch.cuda.memory_reserved() / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", required=True, help="comma-separated profile ids, run in this order")
    ap.add_argument("--seconds", type=float, default=30.0, help="simulated input duration per profile")
    ap.add_argument("--input", default=str(REPO / "tests/fixtures/input_male.wav"))
    ap.add_argument("--pitch", type=float, default=None, help="override the profile's pitch_shift")
    ap.add_argument("--index-rate", type=float, default=None, help="override the profile's index_rate")
    ap.add_argument("--chunk", type=float, default=0.5)
    ap.add_argument("--prebuffer", type=float, default=0.15)
    ap.add_argument("--noise-suppression", default="off", help="off, or a mode: light, medium, strong")
    ap.add_argument("--legacy-seams", action="store_true",
                    help="join chunks end to end (context/crossfade/search 0), the behaviour before ChunkStitcher")
    ap.add_argument("--capture-dir", default=None, help="write each profile's streamed output WAV here (new files only)")
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()

    x, sr = sf.read(args.input, dtype="float32", always_2d=True)
    x = librosa.resample(x.mean(axis=1), orig_sr=sr, target_sr=STREAM_SAMPLE_RATE)
    n = int(args.seconds * STREAM_SAMPLE_RATE)
    feed = np.tile(x, int(np.ceil(n / len(x))))[:n]
    capture_dir = Path(args.capture_dir) if args.capture_dir else None
    if capture_dir:
        capture_dir.mkdir(parents=True, exist_ok=True)

    streams: dict[str, object] = {}

    def pipeline_factory(**kw):
        inp = SimulatedInputStream(feed, sample_rate=STREAM_SAMPLE_RATE)
        out = SimulatedOutputStream(sample_rate=STREAM_SAMPLE_RATE, channels=2, block_size=2048,
                                    prebuffer_seconds=kw["prebuffer_duration_s"], capture=capture_dir is not None)
        streams.update(inp=inp, out=out)
        return RealTimeVoiceChangerPipeline(input_stream=inp, output_stream=out, **kw)

    svc = VoiceChangerService(pipeline_factory=pipeline_factory)
    transitions: list[str] = []
    svc.add_status_listener(lambda s: transitions.append(s.state.value))
    results = {"input": args.input, "seconds_per_profile": args.seconds, "chunk_s": args.chunk,
               "prebuffer_s": args.prebuffer, "legacy_seams": args.legacy_seams,
               "noise_suppression": args.noise_suppression, "gpu_used_mib_before": gpu_used_mib(), "runs": []}

    for run_no, pid in enumerate(args.profiles.split(","), start=1):
        t_load = time.perf_counter()
        svc.select_profile(pid)
        load_s = time.perf_counter() - t_load
        overrides = {k: v for k, v in (("pitch_shift", args.pitch), ("index_rate", args.index_rate)) if v is not None}
        if overrides:
            svc.update_config(**overrides)
        vram_loaded = {"gpu_used_mib": gpu_used_mib(), **torch_mib()}
        seams = dict(context_s=0.0, crossfade_s=0.0, sola_search_s=0.0) if args.legacy_seams else {}
        if args.noise_suppression == "off":
            svc.set_noise_suppression(enabled=False)
        else:
            svc.set_noise_suppression(enabled=True, mode=args.noise_suppression)
        cpu0, wall0 = time.process_time(), time.perf_counter()
        svc.start(AudioSettings(chunk_duration_s=args.chunk, prebuffer_s=args.prebuffer, **seams))
        inp = streams["inp"]
        expected_chunks = n // int(args.chunk * STREAM_SAMPLE_RATE)
        while not inp.finished.wait(1.0):
            st = svc.get_status()
            m = svc.get_metrics() or {}
            print(f"  [{pid}] {st.state.value} converted={m.get('chunks_converted')}/{expected_chunks} "
                  f"buffer={m.get('current_buffer_ms')}ms steady_underflows={m.get('steady_underflows')}", end="\r", flush=True)
            if st.state.value == "error":
                print(f"\nsession error: {st.last_error}")
                break
        # let the last chunk finish inference and play out
        deadline = time.perf_counter() + 3.0
        while (svc.get_metrics() or {}).get("chunks_converted", 0) + (svc.get_metrics() or {}).get("chunks_dropped", 0) \
                < expected_chunks and time.perf_counter() < deadline:
            time.sleep(0.05)
        time.sleep(args.prebuffer + 0.2)
        status = svc.get_status()
        vram_running = {"gpu_used_mib": gpu_used_mib(), **torch_mib()}
        svc.stop()
        cpu_s, wall_s = time.process_time() - cpu0, time.perf_counter() - wall0
        metrics = svc.get_metrics()
        run = {"profile_id": pid, "load_s": round(load_s, 2), "config": svc.get_config().__dict__,
               "expected_chunks": expected_chunks, "final_state_before_stop": status.state.value,
               "last_error": status.last_error.__dict__ if status.last_error else None,
               "vram_after_load": vram_loaded, "vram_end_of_run": vram_running, "metrics": metrics,
               # whole process (all threads, incl. the simulated devices' clock threads), in cores
               "process_cpu_s": round(cpu_s, 1), "wall_s": round(wall_s, 1), "avg_cpu_cores": round(cpu_s / wall_s, 3)}
        if capture_dir:
            out_wav = capture_dir / f"{run_no:02d}_stream_{pid}_{int(args.seconds)}s.wav"
            if out_wav.exists():
                sys.exit(f"{out_wav} exists; refusing to overwrite")
            sf.write(str(out_wav), streams["out"].captured()[:, 0], STREAM_SAMPLE_RATE, subtype="FLOAT")
            run["captured_wav"] = str(out_wav)
        results["runs"].append(run)
        print(f"\n[{pid}] load {load_s:.1f}s | " + json.dumps({k: metrics[k] for k in (
            "chunks_captured", "chunks_converted", "chunks_dropped", "steady_underflows", "startup_underflows",
            "output_overflows", "inference_errors", "sample_count_mismatches", "avg_inference_ms", "p95_inference_ms",
            "max_inference_ms", "rtf", "avg_queue_wait_ms", "avg_software_e2e_ms", "p95_software_e2e_ms",
            "avg_buffer_occupancy_ms", "algorithmic_delay_ms", "estimated_software_latency_ms", "stitch_resets", "noise_suppression_enabled",
            "avg_noise_suppression_ms", "p95_noise_suppression_ms", "max_noise_suppression_ms",
            "noise_suppression_errors")}) + f" | cpu {run['avg_cpu_cores']} cores")
        print(f"[{pid}] VRAM after load {vram_loaded} | end of run {vram_running}")

    svc.shutdown()
    results["vram_after_shutdown"] = {"gpu_used_mib": gpu_used_mib(), **torch_mib()}
    results["transitions"] = transitions
    print(f"after shutdown: {results['vram_after_shutdown']}")
    if args.json_output:
        p = Path(args.json_output)
        if p.exists():
            sys.exit(f"{p} exists; refusing to overwrite")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    ok = all(r["final_state_before_stop"] == "running" and r["metrics"]["inference_errors"] == 0 for r in results["runs"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
