"""Detailed Stage-by-Stage Latency Profiler for Real-Time RVC Inference.

Measures individual components separately with CUDA synchronization:
1. Input audio resampling (40k -> 16k)
2. Butterworth filtering & padding
3. RMVPE pitch (F0) extraction
4. ContentVec feature extraction
5. Faiss index retrieval (Disk vs In-Memory)
6. Generator synthesis (net_g)
7. CUDA memory cache clearing overhead
8. Output audio formatting & resampling
"""

# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import faiss
import librosa
import numpy as np
from scipy import signal  # type: ignore
import soundfile as sf
import torch
import torch.nn.functional as F

from voice_changer.conversion.rvc.adapter import RVCAdapter


def format_stats(arr: list[float]) -> str:
    n = len(arr)
    label = f"N={n:2d}" if n > 0 else "N= 0"
    return (
        f"({label}) Mean={np.mean(arr):6.2f} ms | "
        f"Median={np.median(arr):6.2f} ms | "
        f"P95={np.percentile(arr, 95):6.2f} ms | "
        f"Min={np.min(arr):6.2f} ms | "
        f"Max={np.max(arr):6.2f} ms"
    )


def run_profiling(iterations: int = 15, chunk_duration_s: float = 0.5) -> None:
    print("=" * 80)
    print(f"RVC Inference Stage-by-Stage Latency Profiler ({chunk_duration_s:.2f}s chunk, {iterations} trials)")
    print("=" * 80)

    # 1. Load Audio Fixture
    fixture_path = repo_root / "tests" / "fixtures" / "input_male.wav"
    audio_full, sr_full = sf.read(str(fixture_path))
    if audio_full.ndim > 1:
        audio_full = audio_full[:, 0]

    # Target chunk at 40k
    samples_40k = int(40000 * chunk_duration_s)
    if sr_full != 40000:
        chunk_40k = librosa.resample(audio_full[: int(sr_full * chunk_duration_s)], orig_sr=sr_full, target_sr=40000)
    else:
        chunk_40k = audio_full[:samples_40k].astype(np.float32)

    # 2. Load Adapter & Engine
    print("\nLoading RVCAdapter and warming up GPU...")
    adapter = RVCAdapter()
    model_path = "models/weights/vctk_female_p231.pth"
    index_path = "models/indices/vctk_female_p231.index"
    adapter.load_model(model_path, index_path)
    vc = adapter._ensure_applio_initialized()
    if not vc.hubert_model:
        vc.load_hubert("contentvec")

    # Preload index for comparison
    t_idx_0 = time.perf_counter()
    preloaded_index = faiss.read_index(index_path)
    preloaded_big_npy = preloaded_index.reconstruct_n(0, preloaded_index.ntotal)
    t_idx_load = (time.perf_counter() - t_idx_0) * 1000

    print(f"Engine warmed up. Preloaded index in memory: {t_idx_load:.1f} ms")

    # High pass filter coefficients (Applio constants)
    filt = signal.butter(5, 48, btype="high", fs=16000)
    assert filt is not None
    bh, ah = filt[0], filt[1]

    # Metric accumulators
    t_resample_in: list[float] = []
    t_filter_pad: list[float] = []
    t_rmvpe: list[float] = []
    t_contentvec: list[float] = []
    t_faiss_disk: list[float] = []
    t_faiss_mem: list[float] = []
    t_gen_minpad: list[float] = []
    t_gen_01s_pad: list[float] = []
    t_gen_10s_pad: list[float] = []
    t_empty_cache: list[float] = []
    t_total_unoptimized: list[float] = []
    t_total_optimized: list[float] = []

    # Two thorough warm-up passes
    print("Executing 2 thorough warm-up cycles (GPU kernels + allocator)...")
    for _ in range(2):
        _ = adapter.convert_chunk(chunk_40k, input_sr=40000, target_sr=40000)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    print(f"Warmup complete. Executing {iterations} warmed-up profiling passes...\n")

    for trial in range(iterations):
        # Step 1: Input Resampling (40000 -> 16000 Hz)
        t0 = time.perf_counter()
        audio_16k = librosa.resample(chunk_40k, orig_sr=40000, target_sr=16000)
        audio_max = np.abs(audio_16k).max() / 0.95
        if audio_max > 1.0:
            audio_16k = audio_16k / audio_max
        t_resample_in.append((time.perf_counter() - t0) * 1000)

        # Step 2: High-pass Filter & Edge Padding
        t0 = time.perf_counter()
        audio_filt = signal.filtfilt(bh, ah, audio_16k)
        audio_pad = np.pad(audio_filt, (vc.vc.window // 2, vc.vc.window // 2), mode="reflect")
        p_len = audio_pad.shape[0] // vc.vc.window
        t_filter_pad.append((time.perf_counter() - t0) * 1000)

        # Step 3: RMVPE Pitch Extraction
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pitch, pitchf = vc.vc.get_f0(
            None,
            audio_pad,
            p_len,
            10,  # pitch shift
            "rmvpe",
            3,   # filter radius
            128, # hop length
            False,
            1.0,
            None,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_rmvpe.append((time.perf_counter() - t0) * 1000)

        # Step 4: ContentVec Feature Extraction
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with torch.no_grad():
            feats = (
                torch.from_numpy(audio_pad).half()
                if vc.vc.is_half
                else torch.from_numpy(audio_pad).float()
            )
            feats = feats.view(1, -1).to(vc.vc.device)
            feats = vc.hubert_model(feats)["last_hidden_state"]
            if vc.version == "v1":
                feats = vc.hubert_model.final_proj(feats[0]).unsqueeze(0)
            feats0 = feats.clone()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_contentvec.append((time.perf_counter() - t0) * 1000)

        # Step 5A: Faiss Disk Reload (unoptimized baseline behavior)
        t0 = time.perf_counter()
        disk_idx = faiss.read_index(index_path)
        disk_npy = disk_idx.reconstruct_n(0, disk_idx.ntotal)
        t_faiss_disk.append((time.perf_counter() - t0) * 1000)

        # Step 5B: Faiss In-Memory Search (optimized behavior)
        t0 = time.perf_counter()
        feats_retrieved = vc.vc._retrieve_speaker_embeddings(feats, preloaded_index, preloaded_big_npy, 0.75)
        t_faiss_mem.append((time.perf_counter() - t0) * 1000)

        # Feature upsampling and protection setup
        feats_interp = F.interpolate(feats_retrieved.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        p_len_actual = min(audio_pad.shape[0] // vc.vc.window, feats_interp.shape[1])
        feats0_interp = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        pitch_t = torch.tensor(pitch, device=vc.vc.device).unsqueeze(0).long()
        pitchf_t = torch.tensor(pitchf, device=vc.vc.device).unsqueeze(0).float()
        pitch_sub = pitch_t[:, :p_len_actual]
        pitchf_sub = pitchf_t[:, :p_len_actual]

        protect = 0.33
        if protect < 0.5:
            pitchff = pitchf_sub.clone()
            pitchff[pitchf_sub > 0] = 1
            pitchff[pitchf_sub < 1] = protect
            feats_final = feats_interp * pitchff.unsqueeze(-1) + feats0_interp * (1 - pitchff.unsqueeze(-1))
            feats_final = feats_final.to(feats0_interp.dtype)
        else:
            feats_final = feats_interp

        p_len_t = torch.tensor([p_len_actual], device=vc.vc.device).long()
        sid = torch.tensor(0, device=vc.vc.device).unsqueeze(0).long()

        # Step 6A: Generator Synthesis with Minimal Padding (window/2 = 80 samples)
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with torch.no_grad():
            _ = vc.net_g.infer(feats_final, p_len_t, pitch_sub, pitchf_sub, sid)[0][0, 0].data.cpu().float().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gen_minpad.append((time.perf_counter() - t0) * 1000)

        # Step 6B: Generator Synthesis with 0.1s Streaming Padding (1600 samples @ 16k -> 4000 samples @ 40k)
        pad_16k_01s = 1600
        pad_frames_01s = pad_16k_01s // vc.vc.window * 2  # upsample scale factor 2
        feats_01s = F.pad(feats_final, (0, 0, pad_frames_01s, pad_frames_01s), mode="replicate")
        p_len_01s = torch.tensor([feats_01s.shape[1]], device=vc.vc.device).long()
        pitch_01s = F.pad(pitch_sub, (pad_frames_01s, pad_frames_01s), mode="replicate")
        pitchf_01s = F.pad(pitchf_sub, (pad_frames_01s, pad_frames_01s), mode="replicate")

        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with torch.no_grad():
            _ = vc.net_g.infer(feats_01s, p_len_01s, pitch_01s, pitchf_01s, sid)[0][0, 0].data.cpu().float().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gen_01s_pad.append((time.perf_counter() - t0) * 1000)

        # Step 6C: Generator Synthesis with 1.0s Unoptimized Baseline Padding (16000 samples @ 16k -> 40000 @ 40k)
        pad_16k_10s = 16000
        pad_frames_10s = pad_16k_10s // vc.vc.window * 2
        feats_10s = F.pad(feats_final, (0, 0, pad_frames_10s, pad_frames_10s), mode="replicate")
        p_len_10s = torch.tensor([feats_10s.shape[1]], device=vc.vc.device).long()
        pitch_10s = F.pad(pitch_sub, (pad_frames_10s, pad_frames_10s), mode="replicate")
        pitchf_10s = F.pad(pitchf_sub, (pad_frames_10s, pad_frames_10s), mode="replicate")

        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with torch.no_grad():
            _ = vc.net_g.infer(feats_10s, p_len_10s, pitch_10s, pitchf_10s, sid)[0][0, 0].data.cpu().float().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gen_10s_pad.append((time.perf_counter() - t0) * 1000)

        # Step 7: CUDA Empty Cache overhead
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        t_empty_cache.append((time.perf_counter() - t0) * 1000)

        # Composite Totals
        unopt = t_resample_in[-1] + t_filter_pad[-1] + t_rmvpe[-1] + t_contentvec[-1] + t_faiss_disk[-1] + t_gen_10s_pad[-1]
        opt = t_resample_in[-1] + t_filter_pad[-1] + t_rmvpe[-1] + t_contentvec[-1] + t_faiss_mem[-1] + t_gen_01s_pad[-1]
        t_total_unoptimized.append(unopt)
        t_total_optimized.append(opt)

    # Print Breakdown
    print("=" * 80)
    print("STAGE-BY-STAGE LATENCY BREAKDOWN (BENCHMARK RESULTS WITH SYNCHRONIZATION)")
    print("=" * 80)
    print(f"1. Input Resampling (40k->16k CPU)     : {format_stats(t_resample_in)}")
    print(f"2. High-pass Filter & Pad (CPU)        : {format_stats(t_filter_pad)}")
    print(f"3. RMVPE F0 Extraction (GPU)           : {format_stats(t_rmvpe)}")
    print(f"4. ContentVec Extraction (GPU)         : {format_stats(t_contentvec)}")
    print(f"5A. Faiss Disk Reload (Unoptimized)    : {format_stats(t_faiss_disk)}")
    print(f"5B. Faiss In-Memory Search (Optimized) : {format_stats(t_faiss_mem)}")
    print("-" * 80)
    print(f"6A. Generator (Minimal pad: window//2) : {format_stats(t_gen_minpad)}")
    print(f"6B. Generator (Optimized 0.1s pad)     : {format_stats(t_gen_01s_pad)}")
    print(f"6C. Generator (Unoptimized 1.0s pad)   : {format_stats(t_gen_10s_pad)}")
    print("-" * 80)
    print(f"7. torch.cuda.empty_cache() Overhead   : {format_stats(t_empty_cache)}")
    print("=" * 80)
    print(f"TOTAL Inference (Unoptimized Pipeline) : {format_stats(t_total_unoptimized)}")
    print(f"TOTAL Inference (Optimized Pipeline)   : {format_stats(t_total_optimized)}")
    print("=" * 80)


if __name__ == "__main__":
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    run_profiling(iterations=trials, chunk_duration_s=dur)
