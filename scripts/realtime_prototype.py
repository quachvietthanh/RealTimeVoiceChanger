"""Milestone 3 Real-Time Voice Conversion Prototype Script.

Tests the real-time audio pipeline: non-blocking microphone capture (or simulated fixture input),
asynchronous bounded-queue neural inference on Quadro T2000 GPU, and continuous output playback.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

# Ensure src/ is on sys.path
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import soundfile as sf
import torch

from voice_changer.audio.devices import AudioDeviceManager
from voice_changer.audio.pipeline import RealTimeVoiceChangerPipeline
from voice_changer.conversion.base import ConversionConfig
from voice_changer.tuning.language import DEFAULT_LANGUAGE, apply_language_profile, get_language_profile
from voice_changer.conversion.rvc.adapter import RVCAdapter


def get_gpu_vram_mb() -> tuple[float, float]:
    """Return (allocated_mb, reserved_mb) if CUDA is available."""
    if torch.cuda.is_available():
        alloc = float(torch.cuda.memory_allocated() / (1024 * 1024))
        res = float(torch.cuda.memory_reserved() / (1024 * 1024))
        return alloc, res
    return 0.0, 0.0


def get_gpu_memory_mb() -> float:
    """Return currently allocated GPU memory in MB if CUDA is available."""
    alloc, _ = get_gpu_vram_mb()
    return alloc


def main() -> None:
    parser = argparse.ArgumentParser(description="RealTimeVoiceChanger Milestone 3 Prototype")
    parser.add_argument("--list-devices", action="store_true", help="List audio input and output devices and exit")
    parser.add_argument("--input-device", type=int, default=None, help="Input device ID (default: system default mic)")
    parser.add_argument("--output-device", type=int, default=None, help="Output device ID (default: system default)")
    parser.add_argument("--duration", type=float, default=10.0, help="Test session duration in seconds (default: 10s)")
    parser.add_argument("--chunk-size", type=float, default=0.5, help="Chunk duration in seconds (default: 0.5s)")
    parser.add_argument("--pitch-shift", type=float, default=10.0, help="Pitch shift in semitones (default: +10)")
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE, help="Speech language profile: vi or en (default: en)")
    parser.add_argument("--index-rate", type=float, default=None, help="Feature retrieval index rate (default: from --language profile)")
    parser.add_argument("--filter-radius", type=int, default=None, help="Explicit filter radius for F0 median filter (default: from --language profile)")
    parser.add_argument("--model-path", type=str, default="models/weights/vctk_female_p231.pth", help="Path to RVC model .pth")
    parser.add_argument("--index-path", type=str, default="models/indices/vctk_female_p231.index", help="Path to Faiss .index")
    parser.add_argument("--simulated-input", type=str, default=None, help="Optional WAV file to stream in real-time as simulated mic")
    parser.add_argument("--prebuffer", type=float, default=0.15, help="Startup prebuffer duration in seconds (default: 0.15s)")
    parser.add_argument("--json-output", type=str, default=None, help="Optional path to write structured test results JSON")
    args = parser.parse_args()

    device_mgr = AudioDeviceManager()

    if args.list_devices:
        print("=" * 60)
        print("Available Audio Input Devices (Microphones):")
        print("=" * 60)
        for dev in device_mgr.list_input_devices():
            print(f"[{dev.device_id:2d}] {dev.name} ({dev.host_api}, {dev.max_input_channels} in, default {dev.default_sample_rate} Hz)")

        print("\n" + "=" * 60)
        print("Available Audio Output Devices (Speakers / Virtual Cables):")
        print("=" * 60)
        for dev in device_mgr.list_output_devices():
            print(f"[{dev.device_id:2d}] {dev.name} ({dev.host_api}, {dev.max_output_channels} out, default {dev.default_sample_rate} Hz)")
        return

    print("=" * 70)
    print("RealTimeVoiceChanger: Milestone 3 Real-Time Prototype Test")
    print("=" * 70)

    in_dev = None
    if args.input_device is not None:
        in_dev = next((d for d in device_mgr.list_input_devices() if d.device_id == args.input_device), None)
    if in_dev is None and args.simulated_input is None:
        in_dev = device_mgr.get_default_device(is_input=True)

    out_dev = None
    if args.output_device is not None:
        out_dev = next((d for d in device_mgr.list_output_devices() if d.device_id == args.output_device), None)
    if out_dev is None:
        out_dev = device_mgr.get_default_device(is_input=False)

    print(f"Input  Device: {in_dev.name if in_dev else ('[Simulated File] ' + str(args.simulated_input))} (ID: {in_dev.device_id if in_dev else 'N/A'})")
    print(f"Output Device: {out_dev.name if out_dev else 'Default'} (ID: {out_dev.device_id if out_dev else 'N/A'})")
    print(f"Target Model : {args.model_path}")
    print(f"Index File   : {args.index_path}")
    print(f"Chunk Size   : {args.chunk_size:.2f}s ({int(40000 * args.chunk_size)} samples)")
    print(f"Pitch Shift  : +{args.pitch_shift:.1f} semitones")
    print(f"Duration     : {args.duration:.1f}s")
    print("=" * 70)

    print("\n[1/3] Loading and warming up RVC voice model...")
    t_load_start = time.perf_counter()
    adapter = RVCAdapter()
    adapter.load_model(args.model_path, args.index_path)
    t_load = time.perf_counter() - t_load_start
    alloc_loaded, res_loaded = get_gpu_vram_mb()
    print(f"      Model preloaded in {t_load:.2f}s. Model info: {adapter.model_info}")
    print(f"      GPU VRAM after loading: Allocated={alloc_loaded:.1f} MB, Reserved={res_loaded:.1f} MB")

    print("\n[2/3] Initializing real-time pipeline...")
    cfg = apply_language_profile(
        ConversionConfig(pitch_shift=args.pitch_shift, f0_method="rmvpe"),
        args.language,
        index_rate=args.index_rate,
    )
    if args.filter_radius is not None:
        cfg.filter_radius = args.filter_radius
    print(f"Language     : {get_language_profile(args.language).name} (index_rate={cfg.index_rate}, filter_radius={cfg.filter_radius}, protect_rate={cfg.protect_rate})")

    pipeline = RealTimeVoiceChangerPipeline(
        converter=adapter,
        conversion_config=cfg,
        input_device_id=in_dev.device_id if in_dev else None,
        output_device_id=out_dev.device_id if out_dev else None,
        sample_rate=40000,
        chunk_duration_s=args.chunk_size,
        max_queue_size=4,
        prebuffer_duration_s=args.prebuffer,
    )

    print("      Running initial GPU kernel warmup...")
    dummy_input = np.zeros(int(40000 * args.chunk_size), dtype=np.float32)
    _ = adapter.convert_chunk(dummy_input, config=cfg, input_sr=40000, target_sr=40000)
    print("      Warmup complete.")

    print(f"\n[3/3] Running prototype session for {args.duration:.1f} seconds...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    alloc_start, res_start = get_gpu_vram_mb()

    t_wall_start = time.perf_counter()
    t_cpu_start = time.process_time()

    if args.simulated_input:
        print(f"      Streaming simulated audio from: {args.simulated_input}")
        sim_data, sim_sr = sf.read(args.simulated_input)
        if sim_data.ndim > 1:
            sim_data = sim_data[:, 0]
        if sim_sr != 40000:
            import librosa
            sim_data = librosa.resample(sim_data, orig_sr=sim_sr, target_sr=40000)

        pipeline.output_stream.start()
        pipeline._running = True
        pipeline._worker_thread = threading.Thread(target=pipeline._worker_loop, daemon=True)
        pipeline._worker_thread.start()
        pipeline.metrics.start_time = time.perf_counter()

        chunk_samples = int(40000 * args.chunk_size)
        chunks_in_file = len(sim_data) // chunk_samples
        start_t = time.perf_counter()
        c_idx = 0

        while time.perf_counter() - start_t < args.duration:
            idx_in_file = c_idx % chunks_in_file
            chunk_slice = sim_data[idx_in_file * chunk_samples : (idx_in_file + 1) * chunk_samples]

            t_first = start_t + c_idx * args.chunk_size
            target_time = start_t + (c_idx + 1) * args.chunk_size
            sleep_remaining = target_time - time.perf_counter()
            if sleep_remaining > 0:
                time.sleep(sleep_remaining)
            t_ready = time.perf_counter()

            pipeline.metrics.total_chunks_captured += 1
            pipeline._input_queue.put((chunk_slice, t_first, t_ready))
            c_idx += 1

            if c_idx > 0 and c_idx % 30 == 0:
                elapsed_now = time.perf_counter() - start_t
                buffered_ms = pipeline.output_stream.buffered_seconds * 1000.0
                conv = pipeline.metrics.total_chunks_converted
                drops = pipeline.output_stream.steady_underflow_count
                print(f"      [Progress] {elapsed_now:5.1f}s / {args.duration:.1f}s | In: {c_idx:4d} | Out: {conv:4d} | Buffer: {buffered_ms:5.1f} ms | Steady Drops: {drops}")

        pipeline.output_stream.prepare_shutdown()
        time.sleep(args.chunk_size * 1.2)
        pipeline.stop()
    else:
        pipeline.start()
        start_t = time.perf_counter()
        while time.perf_counter() - start_t < args.duration:
            time.sleep(0.5)
            buffered = pipeline.output_stream.buffered_seconds
            print(f"      [Live] Elapsed: {time.perf_counter() - start_t:.1f}s | Output Buffer: {buffered:.2f}s | Chunks converted: {pipeline.metrics.total_chunks_converted}", end="\r")
        print()
        pipeline.stop()

    t_wall_end = time.perf_counter()
    t_cpu_end = time.process_time()
    total_wall = max(t_wall_end - t_wall_start, 1e-6)
    total_cpu = max(t_cpu_end - t_cpu_start, 0.0)
    avg_cpu = (total_cpu / total_wall) * 100.0

    alloc_end, res_end = get_gpu_vram_mb()
    alloc_peak = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else alloc_end
    res_peak = float(torch.cuda.max_memory_reserved() / (1024 * 1024)) if torch.cuda.is_available() else res_end

    summary = pipeline.metrics.summary()

    chunk_dur_ms = args.chunk_size * 1000.0
    buffer_ahead_ms = max(0.0, summary["avg_buffer_occupancy_ms"] - chunk_dur_ms)
    estimated_acoustic_lat = round(summary["median_software_e2e_ms"] + buffer_ahead_ms, 1)

    print("\n" + "=" * 70)
    print("PROTOTYPE EXECUTION RESULTS & TELEMETRY")
    print("=" * 70)
    print(f"Session Duration        : {summary['session_duration_s']} s")
    print(f"Chunks Captured         : {summary['chunks_captured']}")
    print(f"Chunks Converted        : {summary['chunks_converted']}")
    print(f"Chunks Dropped (Backlog): {summary['chunks_dropped']}")
    if args.simulated_input:
        print("Input Overflows         : N/A (simulated input fixture; hardware microphone stream inactive)")
    else:
        print(f"Input Overflows         : {summary['input_overflows']}")
    print(f"Output Underflows Total : {summary['output_underflows']}")
    print(f"  - Startup Silence     : {summary['startup_underflows']} (expected delay before 1st chunk)")
    print(f"  - Steady-State Drops  : {summary['steady_underflows']} (active speech buffer glitches)")
    if pipeline.output_stream.steady_underflow_timestamps:
        print(f"  - Steady Timestamps   : {[round(t - t_wall_start, 3) for t in pipeline.output_stream.steady_underflow_timestamps]}")
    print(f"  - Shutdown Drain      : {summary['shutdown_underflows']} (trailing silence after session)")
    print(f"Inference Errors        : {summary['inference_errors']}")
    print(f"Sample Count Mismatches : {summary.get('sample_count_mismatches', 0)}")
    print(f"Chunk Sample Counts     : Expected={summary.get('expected_chunk_samples', 0)}, Last Converted={summary.get('last_converted_chunk_samples', 0)}")
    print(f"Chunk Duration          : {chunk_dur_ms:.1f} ms")
    print(f"Inference Latency (Avg) : {summary['avg_inference_ms']} ms")
    print(f"Inference Latency (Med) : {summary['median_inference_ms']} ms (Min: {summary['min_inference_ms']} ms, Max: {summary['max_inference_ms']} ms)")
    print(f"Inference Latency (P95) : {summary['p95_inference_ms']} ms")
    print(f"Real-Time Factor (RTF)  : {summary['avg_inference_ms'] / chunk_dur_ms:.3f}")
    print(f"Queue Wait Delay (Med)  : {summary['median_queue_wait_ms']} ms (P95: {summary['p95_queue_wait_ms']} ms)")
    print(f"Initial Speech Onset    : {summary['speech_onset_latency_ms']} ms (First spoken sample -> 1st chunk in buffer)")
    print(f"Measured Softw. E2E Med : {summary['median_software_e2e_ms']} ms (First sample captured -> Output written)")
    print(f"Measured Softw. E2E P95 : {summary['p95_software_e2e_ms']} ms")
    print(f"Output Buffer Post-Write: {summary['avg_buffer_occupancy_ms']} ms (Peak buffer depth right after write)")
    print(f"Output Queue Ahead      : ~{buffer_ahead_ms:.1f} ms (Audio ahead in buffer waiting for playback)")
    print(f"Estimated Acoustic Lat  : ~{estimated_acoustic_lat} ms (Software E2E + Output Queue Ahead; excl. hardware DAC/driver)")
    print(f"GPU Allocated VRAM      : Start={alloc_start:.1f} MB | Peak={alloc_peak:.1f} MB | End={alloc_end:.1f} MB")
    print(f"GPU Reserved VRAM       : Start={res_start:.1f} MB | Peak={res_peak:.1f} MB | End={res_end:.1f} MB")
    print(f"Process CPU Utilization : {avg_cpu:.1f} %")
    print("=" * 70)

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_payload = {
            "summary": {
                **summary,
                "input_overflows": "N/A" if args.simulated_input else summary["input_overflows"],
            },
            "timing": {
                "total_wall_s": round(total_wall, 3),
                "total_cpu_s": round(total_cpu, 3),
                "process_cpu_percent": round(avg_cpu, 1),
                "buffer_ahead_ms": round(buffer_ahead_ms, 1),
                "estimated_acoustic_lat_ms": estimated_acoustic_lat,
                "rtf": round(summary['avg_inference_ms'] / chunk_dur_ms, 4),
            },
            "gpu": {
                "allocated_start_mb": round(alloc_start, 1),
                "allocated_peak_mb": round(alloc_peak, 1),
                "allocated_end_mb": round(alloc_end, 1),
                "reserved_start_mb": round(res_start, 1),
                "reserved_peak_mb": round(res_peak, 1),
                "reserved_end_mb": round(res_end, 1),
                "vram_start_mb": round(alloc_start, 1),
                "vram_peak_mb": round(alloc_peak, 1),
                "vram_end_mb": round(alloc_end, 1),
            },
            "config": {
                "duration": args.duration,
                "chunk_size": args.chunk_size,
                "pitch_shift": args.pitch_shift,
                "language": args.language,
                "index_rate": cfg.index_rate,
                "filter_radius": cfg.filter_radius,
                "protect_rate": cfg.protect_rate,
                "model_path": args.model_path,
                "index_path": args.index_path,
                "simulated_input": args.simulated_input,
                "prebuffer": args.prebuffer,
            },
            "steady_underflow_timestamps": [
                round(t - t_wall_start, 3) for t in pipeline.output_stream.steady_underflow_timestamps
            ],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(telemetry_payload, f, indent=2)
        print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()
