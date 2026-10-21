"""Measure what real sound devices actually deliver at the engine rate, with and without inference load.

Opens the real input and output devices at 40 kHz (the rate the pipeline requests), runs them full-duplex
for --seconds, and records per-callback timing and PortAudio status flags. The output plays SILENCE only.
Input audio is never stored: only its level (RMS/peak, for headroom) is kept.

Phase "idle": devices only. Phase "load" (unless --no-load): the same while a worker thread runs real RVC
chunk inference back to back on the test fixture, to see whether inference starves the audio callbacks
(GIL contention shows up as late callbacks, input overflows and output underflows).

Reports per phase and direction: requested vs measured sample rate (frames / wall clock), block sizes,
callback interval distribution, status flags, and the stream latency PortAudio reports. Software metrics
only; acoustic latency needs a physical loopback (docs/audio-routing.md).

Example (repo root, inference venv):
    .venv\\Scripts\\python scripts/device_probe.py --input 1 --output 4 --seconds 20 --json-output out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402

RATE = 40000


def dist(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    a = np.asarray(values, dtype=np.float64)
    return {k: round(float(v), 3) for k, v in (("mean", a.mean()), ("p50", np.percentile(a, 50)),
                                                ("p95", np.percentile(a, 95)), ("p99", np.percentile(a, 99)),
                                                ("max", a.max()), ("min", a.min()))}


class Recorder:
    def __init__(self) -> None:
        self.times: list[float] = []
        self.frames: list[int] = []
        self.flags: dict[str, int] = {}
        self.peak = 0.0
        self.sumsq = 0.0
        self.n = 0

    def note(self, frames: int, status: sd.CallbackFlags) -> None:
        self.times.append(time.perf_counter())
        self.frames.append(frames)
        if status:
            for name in ("input_overflow", "input_underflow", "output_overflow", "output_underflow", "priming_output"):
                if getattr(status, name, False):
                    self.flags[name] = self.flags.get(name, 0) + 1

    def report(self, block: int) -> dict:
        if len(self.times) < 3:
            return {"callbacks": len(self.times)}
        # Rate from callbacks 1..N (the first callback's timestamp starts the clock).
        elapsed = self.times[-1] - self.times[0]
        frames = sum(self.frames[1:])
        intervals = np.diff(self.times) * 1000.0
        out = {
            "callbacks": len(self.times),
            "measured_rate_hz": round(frames / elapsed, 1) if elapsed > 0 else None,
            "block_sizes": sorted(set(self.frames)),
            "callback_interval_ms": dist(intervals.tolist()),
            "nominal_interval_ms": round(block / RATE * 1000.0, 2),
            "late_callbacks_gt_2x": int(np.sum(intervals > 2 * block / RATE * 1000.0)),
            "status_flags": self.flags,
        }
        if self.n:
            out["input_rms_dbfs"] = round(10 * np.log10(self.sumsq / self.n + 1e-20), 1)
            out["input_peak_dbfs"] = round(20 * np.log10(self.peak + 1e-12), 1)
        return out


def run_phase(args, label: str, load_fn=None) -> dict:
    rin, rout = Recorder(), Recorder()

    def in_cb(indata, frames, t, status):
        rin.note(frames, status)
        x = indata[:, 0]
        rin.peak = max(rin.peak, float(np.abs(x).max()))
        rin.sumsq += float(np.dot(x, x))
        rin.n += len(x)

    def out_cb(outdata, frames, t, status):
        rout.note(frames, status)
        outdata.fill(0.0)

    stop = threading.Event()
    load_stats: list[float] = []
    load_thread = None
    if load_fn is not None:
        def loop():
            while not stop.is_set():
                t0 = time.perf_counter()
                load_fn()
                load_stats.append((time.perf_counter() - t0) * 1000.0)
        load_thread = threading.Thread(target=loop, daemon=True)

    istream = sd.InputStream(device=args.input, channels=1, samplerate=RATE, blocksize=args.block,
                             dtype="float32", callback=in_cb)
    ostream = sd.OutputStream(device=args.output, channels=2, samplerate=RATE, blocksize=args.block,
                              dtype="float32", callback=out_cb)
    info = {"input_latency_s": istream.latency, "output_latency_s": ostream.latency,
            "input_samplerate": istream.samplerate, "output_samplerate": ostream.samplerate}
    with istream, ostream:
        if load_thread:
            load_thread.start()
        time.sleep(args.seconds)
        stop.set()
    if load_thread:
        load_thread.join(timeout=5)
    res = {"phase": label, **info, "input": rin.report(args.block), "output": rout.report(args.block)}
    if load_fn is not None:
        res["inference_ms"] = dist(load_stats)
        res["inference_calls"] = len(load_stats)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=int, required=True, help="PortAudio input device id")
    ap.add_argument("--output", type=int, required=True, help="PortAudio output device id (plays silence)")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--block", type=int, default=2048, help="block size, as the pipeline uses")
    ap.add_argument("--profile", default="vctk_p231", help="profile for the inference-load phase")
    ap.add_argument("--no-load", action="store_true", help="skip the inference-load phase")
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()

    devs = sd.query_devices()
    result = {
        "devices": {k: {"id": i, "name": devs[i]["name"], "host_api": sd.query_hostapis(devs[i]["hostapi"])["name"],
                        "default_samplerate": devs[i]["default_samplerate"]}
                    for k, i in (("input", args.input), ("output", args.output))},
        "requested_rate_hz": RATE, "block": args.block, "seconds_per_phase": args.seconds, "phases": [],
    }
    for k in ("input", "output"):
        try:
            (sd.check_input_settings if k == "input" else sd.check_output_settings)(
                device=getattr(args, k), samplerate=RATE, channels=1 if k == "input" else 2, dtype="float32")
            result["devices"][k]["accepts_40k"] = True
        except Exception as err:  # noqa: BLE001 - report, then stop
            result["devices"][k]["accepts_40k"] = f"{type(err).__name__}: {err}"
            print(json.dumps(result, indent=2))
            return 2

    result["phases"].append(run_phase(args, "idle"))

    if not args.no_load:
        import soundfile as sf
        import librosa
        from voice_changer.service import VoiceChangerService

        svc = VoiceChangerService()
        svc.select_profile(args.profile)
        audio, sr = sf.read(str(REPO / "tests/fixtures/input_male.wav"), dtype="float32", always_2d=True)
        audio = librosa.resample(audio.mean(axis=1), orig_sr=sr, target_sr=RATE)
        chunk = audio[RATE * 2: RATE * 2 + RATE // 2]
        conv, cfg = svc._converter, svc._config  # probe only: drive the loaded converter directly

        def load():
            conv.convert_chunk(chunk, config=cfg, input_sr=RATE, target_sr=RATE)

        result["phases"].append(run_phase(args, "load", load))
        svc.shutdown()

    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.json_output:
        out = Path(args.json_output)
        if out.exists():
            print(f"refusing to overwrite {out}", file=sys.stderr)
            return 1
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
