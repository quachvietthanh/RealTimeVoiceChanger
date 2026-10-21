"""Milestone 4A.1: Vocal Separation and Audio Preview Script.

Separates vocal audio from background accompaniment using torchaudio's pretrained
Hybrid Demucs (HDEMUCS_HIGH_MUSDB) model with seamless overlap-add crossfading.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
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
import torchaudio


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def dbfs(linear_val: float) -> float:
    """Convert linear amplitude to dBFS."""
    if linear_val <= 1e-9:
        return -120.0
    return float(20.0 * np.log10(linear_val))


def separate_vocals(
    input_path: Path,
    output_path: Path,
    device_name: str = "cuda",
    chunk_len_s: float = 10.0,
    overlap_s: float = 2.0,
    force: bool = False,
) -> dict:
    """Separate vocals from music and save to output_path."""
    if not input_path.exists():
        print(f"[ERROR] Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    if input_path.resolve() == output_path.resolve():
        print(f"[ERROR] Output path cannot be the same as input path: {input_path}", file=sys.stderr)
        sys.exit(1)

    if output_path.exists() and not force:
        print(f"[WARNING] Output file already exists: {output_path}. Overwriting with newly generated preview.")

    print("=" * 70)
    print("RealTimeVoiceChanger: Milestone 4A.1 Vocal Separation")
    print("=" * 70)

    # 1. Inspect input file
    in_info = sf.info(str(input_path))
    in_sha = compute_file_sha256(input_path)
    print(f"Input File   : {input_path}")
    print(f"  Size       : {input_path.stat().st_size:,} bytes")
    print(f"  SHA-256    : {in_sha}")
    print(f"  Format     : {in_info.format} ({in_info.subtype})")
    print(f"  Channels   : {in_info.channels}")
    print(f"  Sample Rate: {in_info.samplerate} Hz")
    print(f"  Duration   : {in_info.duration:.2f}s ({in_info.frames:,} frames)")

    # 2. Select and load separation model
    print("\n[1/3] Loading torchaudio HDEMUCS_HIGH_MUSDB model...")
    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB
    model_sample_rate = bundle.sample_rate

    # Check device
    if device_name == "cuda" and not torch.cuda.is_available():
        print("  [!] CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"  Inference Device: {device}")

    t_load_start = time.perf_counter()
    try:
        model = bundle.get_model()
    except Exception as e:
        print(f"[ERROR] Failed to load model weights: {e}", file=sys.stderr)
        sys.exit(1)

    model = model.to(device).eval()
    t_load = time.perf_counter() - t_load_start

    # Record model metadata
    model_sources = getattr(model, "sources", ["drums", "bass", "other", "vocals"])
    if "vocals" not in model_sources:
        print(f"[ERROR] Model sources does not contain 'vocals': {model_sources}", file=sys.stderr)
        sys.exit(1)
    vocal_idx = model_sources.index("vocals")

    checkpoint_path = Path(torch.hub.get_dir()) / "torchaudio" / Path(bundle._model_path)
    ckpt_sha = compute_file_sha256(checkpoint_path) if checkpoint_path.exists() else "N/A"

    print(f"  Model loaded in {t_load:.2f}s.")
    print(f"  Model Sources   : {model_sources} (Vocal index: {vocal_idx})")
    print(f"  Model SampleRate: {model_sample_rate} Hz")
    print(f"  Checkpoint Path : {checkpoint_path}")
    print(f"  Checkpoint SHA  : {ckpt_sha}")

    # 3. Read and prepare input audio
    print("\n[2/3] Processing audio separation with overlap-add crossfading...")
    data, sr = sf.read(str(input_path), dtype="float32", always_2d=True)
    # data is (frames, channels)
    num_frames, num_channels = data.shape

    # Ensure stereo as required by HDemucs
    if num_channels == 1:
        data = np.repeat(data, 2, axis=1)
        num_channels = 2

    # Resample if sample rate doesn't match model
    if sr != model_sample_rate:
        print(f"  Resampling from {sr} Hz to {model_sample_rate} Hz...")
        data_t = torch.from_numpy(data.T)
        resampler = torchaudio.transforms.Resample(sr, model_sample_rate)
        data_t = resampler(data_t)
        data = data_t.numpy().T
        sr = model_sample_rate
        num_frames = data.shape[0]

    # Convert to tensor (1, channels, frames)
    chunk_samples = int(chunk_len_s * sr)
    overlap_samples = int(overlap_s * sr)
    hop_samples = chunk_samples - overlap_samples

    output_vocals = np.zeros((num_frames, 2), dtype=np.float32)
    weight_buffer = np.zeros(num_frames, dtype=np.float32)

    # Crossfade windows (cosine squared / sine squared for constant power & unit sum)
    fade_in = np.sin(np.linspace(0, np.pi / 2, overlap_samples, dtype=np.float32)) ** 2
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap_samples, dtype=np.float32)) ** 2

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_sep_start = time.perf_counter()

    num_chunks = int(np.ceil((num_frames - overlap_samples) / hop_samples))
    print(f"  Audio duration: {num_frames / sr:.2f}s | Chunk size: {chunk_len_s}s | Overlap: {overlap_s}s | Total chunks: {num_chunks}")

    chunk_idx = 0
    for start_idx in range(0, num_frames, hop_samples):
        end_idx = min(start_idx + chunk_samples, num_frames)
        cur_len = end_idx - start_idx
        chunk_idx += 1

        chunk_data = data[start_idx:end_idx] # (cur_len, 2)
        chunk_t = torch.from_numpy(chunk_data.T).unsqueeze(0).to(device) # (1, 2, cur_len)

        with torch.no_grad():
            try:
                # out shape: (1, num_sources, 2, cur_len)
                sources = model(chunk_t)
            except torch.cuda.OutOfMemoryError as oom:
                print(f"\n[ERROR] CUDA OutOfMemoryError on chunk {chunk_idx}: {oom}", file=sys.stderr)
                print("Consider reducing --chunk-size or falling back to CPU.", file=sys.stderr)
                sys.exit(2)

            vocal_stem = sources[0, vocal_idx].cpu().numpy().T # (cur_len, 2)

        # Build blend window for this chunk
        w = np.ones(cur_len, dtype=np.float32)
        if start_idx > 0:
            w[:overlap_samples] *= fade_in[:min(overlap_samples, cur_len)]
        if end_idx < num_frames:
            w[-overlap_samples:] *= fade_out[-min(overlap_samples, cur_len):]

        output_vocals[start_idx:end_idx] += vocal_stem * w[:, None]
        weight_buffer[start_idx:end_idx] += w

        if chunk_idx % 10 == 0 or chunk_idx == num_chunks:
            elapsed = time.perf_counter() - t_sep_start
            pct = (end_idx / num_frames) * 100
            print(f"    [{chunk_idx:2d}/{num_chunks:2d}] Processed {end_idx / sr:5.1f}s / {num_frames / sr:.1f}s ({pct:5.1f}%) in {elapsed:.1f}s")

    # Normalize by accumulated weights
    weight_buffer = np.maximum(weight_buffer, 1e-8)
    output_vocals /= weight_buffer[:, None]

    t_sep = time.perf_counter() - t_sep_start
    rtf = t_sep / (num_frames / sr)

    peak_gpu_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        if device.type == "cuda"
        else 0.0
    )
    peak_res_mb = (
        float(torch.cuda.max_memory_reserved(device) / (1024 * 1024))
        if device.type == "cuda"
        else 0.0
    )

    print(f"  Separation complete in {t_sep:.2f}s (RTF: {rtf:.3f}).")
    if device.type == "cuda":
        print(f"  GPU Peak VRAM: Allocated={peak_gpu_mb:.1f} MB, Reserved={peak_res_mb:.1f} MB")

    # 4. Save separated vocal preview
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write as 16-bit PCM WAV to match input specification
    sf.write(str(output_path), output_vocals, sr, subtype="PCM_16")
    out_sha = compute_file_sha256(output_path)
    out_size = output_path.stat().st_size

    # 5. Validate output preview
    print("\n[3/3] Validating separated vocal preview...")
    out_info = sf.info(str(output_path))
    vocal_data, _ = sf.read(str(output_path), dtype="float32", always_2d=True)

    peak_l = float(np.max(np.abs(vocal_data[:, 0])))
    peak_r = float(np.max(np.abs(vocal_data[:, 1])))
    peak_max = max(peak_l, peak_r)
    rms_l = float(np.sqrt(np.mean(vocal_data[:, 0] ** 2)))
    rms_r = float(np.sqrt(np.mean(vocal_data[:, 1] ** 2)))
    rms_avg = (rms_l + rms_r) / 2.0

    nan_count = int(np.sum(np.isnan(vocal_data)))
    inf_count = int(np.sum(np.isinf(vocal_data)))
    clipped_count = int(np.sum(np.abs(vocal_data) >= 0.999))

    # Silence analysis on preview
    mono_vocal = np.mean(vocal_data, axis=1)
    frame_sz = int(sr * 0.05) # 50 ms
    hop_sz = int(sr * 0.025)  # 25 ms
    num_f = (len(mono_vocal) - frame_sz) // hop_sz + 1
    f_rms = np.array([
        np.sqrt(np.mean(mono_vocal[i * hop_sz : i * hop_sz + frame_sz] ** 2))
        for i in range(num_f)
    ])
    f_db = np.array([dbfs(r) for r in f_rms])

    active_frames_40db = np.sum(f_db >= -40.0)
    active_speech_s = float(active_frames_40db * hop_sz / sr)
    silent_frames_40db = np.sum(f_db < -40.0)
    silent_s = float(silent_frames_40db * hop_sz / sr)

    # Pause floor comparison
    pause_db_p5 = float(np.percentile(f_db, 5))
    pause_db_median = float(np.median(f_db))

    timeline_match = (out_info.frames == in_info.frames)

    print(f"Output File  : {output_path}")
    print(f"  Size       : {out_size:,} bytes")
    print(f"  SHA-256    : {out_sha}")
    print(f"  Format     : {out_info.format} ({out_info.subtype})")
    print(f"  Channels   : {out_info.channels}")
    print(f"  Sample Rate: {out_info.samplerate} Hz")
    print(f"  Duration   : {out_info.duration:.2f}s ({out_info.frames:,} frames)")
    print(f"  Peak       : {dbfs(peak_max):.2f} dBFS (Linear: {peak_max:.4f})")
    print(f"  RMS        : {dbfs(rms_avg):.2f} dBFS")
    print(f"  Clipped    : {clipped_count} samples")
    print(f"  NaN / Inf  : {nan_count} / {inf_count}")
    print(f"  Timeline Match: {'EXACT MATCH' if timeline_match else 'MISMATCH'}")
    print(f"  Speech Activity (>= -40 dBFS): {active_speech_s:.2f}s ({active_speech_s / 60:.2f} min)")
    print(f"  Pause / Silence (< -40 dBFS) : {silent_s:.2f}s")
    print(f"  Noise Floor in Pauses (P5)   : {pause_db_p5:.2f} dBFS")
    print("=" * 70)

    report = {
        "status": "COMPLETED",
        "input": {
            "path": str(input_path),
            "size_bytes": input_path.stat().st_size,
            "sha256": in_sha,
            "sample_rate": in_info.samplerate,
            "channels": in_info.channels,
            "duration_s": in_info.duration,
            "frames": in_info.frames,
        },
        "model": {
            "name": "torchaudio.pipelines.HDEMUCS_HIGH_MUSDB",
            "model_type": str(type(model)),
            "sample_rate": model_sample_rate,
            "sources": model_sources,
            "vocal_index": vocal_idx,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": ckpt_sha,
            "checkpoint_url": f"https://download.pytorch.org/torchaudio/{bundle._model_path}",
        },
        "separation_runtime": {
            "device": str(device),
            "chunk_len_s": chunk_len_s,
            "overlap_s": overlap_s,
            "total_chunks": num_chunks,
            "separation_time_s": round(t_sep, 2),
            "rtf": round(rtf, 4),
            "peak_gpu_allocated_mb": round(peak_gpu_mb, 1),
            "peak_gpu_reserved_mb": round(peak_res_mb, 1),
        },
        "output": {
            "path": str(output_path),
            "size_bytes": out_size,
            "sha256": out_sha,
            "format": out_info.format,
            "subtype": out_info.subtype,
            "sample_rate": out_info.samplerate,
            "channels": out_info.channels,
            "duration_s": out_info.duration,
            "frames": out_info.frames,
            "timeline_matches_source": timeline_match,
            "peak_dbfs": round(dbfs(peak_max), 2),
            "rms_dbfs": round(dbfs(rms_avg), 2),
            "clipped_samples": clipped_count,
            "nan_samples": nan_count,
            "inf_samples": inf_count,
            "active_speech_duration_s": round(active_speech_s, 2),
            "silent_duration_s": round(silent_s, 2),
            "pause_noise_floor_dbfs": round(pause_db_p5, 2),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 4A.1: Vocal Separation Preview Generator")
    parser.add_argument(
        "--input",
        type=str,
        default="data/source_audio/female_source.wav",
        help="Path to input audio file (default: data/source_audio/female_source.wav)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/source_audio/female_source_vocal_preview.wav",
        help="Path to output vocal preview WAV (default: data/source_audio/female_source_vocal_preview.wav)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device: cuda or cpu (default: cuda if available)",
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=10.0,
        help="Chunk duration in seconds for VRAM management (default: 10.0s)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=2.0,
        help="Overlap duration in seconds for crossfading (default: 2.0s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing output file",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = repo_root / in_path

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    separate_vocals(
        input_path=in_path,
        output_path=out_path,
        device_name=args.device,
        chunk_len_s=args.chunk_size,
        overlap_s=args.overlap,
        force=args.force,
    )


if __name__ == "__main__":
    main()
