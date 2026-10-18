"""Tuning Experiments: Offline RVC Female Voice Naturalness Optimization.

Executes:
- Experiment A: Pitch comparison (+8, +9, +10, +11 semitones) vs baseline (+12 semitones)
- Experiment B: Index rate comparison (0.45, 0.55, 0.65, 0.75) at selected optimal pitch (+10 semitones)
Records:
- Duration, sample rate, peak, RMS, clipping, silence
- Inference time, RTF, peak VRAM
"""

from __future__ import annotations

import json
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

from voice_changer.conversion.base import ConversionConfig
from voice_changer.conversion.rvc.adapter import RVCAdapter


def measure_audio(audio_path: Path) -> dict:
    info = sf.info(str(audio_path))
    data, sr = sf.read(str(audio_path))
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data**2)))
    clipping = peak >= 0.999
    is_silent = peak < 1e-4
    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration": round(info.duration, 3),
        "peak": round(peak, 5),
        "rms": round(rms, 5),
        "clipping": clipping,
        "is_silent": is_silent,
    }


def run_conversion_run(
    adapter: RVCAdapter,
    input_wav: Path,
    output_wav: Path,
    pitch: float,
    index_rate: float,
    protect_rate: float = 0.33,
    filter_radius: int = 3,
) -> dict:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    if output_wav.exists():
        print(f"Skipping existing: {output_wav.name}")
        audio_metrics = measure_audio(output_wav)
        return {
            "output_path": str(output_wav),
            "pitch": pitch,
            "index_rate": index_rate,
            "skipped": True,
            **audio_metrics,
        }

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
        vram_before = torch.cuda.memory_allocated(0) / (1024 * 1024)
    else:
        vram_before = 0.0

    cfg = ConversionConfig(
        pitch_shift=pitch,
        f0_method="rmvpe",
        index_rate=index_rate,
        filter_radius=filter_radius,
        protect_rate=protect_rate,
    )

    in_info = sf.info(str(input_wav))
    input_duration = in_info.duration

    start = time.perf_counter()
    adapter.convert_file(str(input_wav), str(output_wav), cfg)
    elapsed = time.perf_counter() - start

    if torch.cuda.is_available():
        vram_peak = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
        vram_after = torch.cuda.memory_allocated(0) / (1024 * 1024)
    else:
        vram_peak = 0.0
        vram_after = 0.0

    rtf = elapsed / input_duration
    audio_metrics = measure_audio(output_wav)

    res = {
        "output_path": str(output_wav.relative_to(repo_root)),
        "pitch": pitch,
        "index_rate": index_rate,
        "inference_time_s": round(elapsed, 3),
        "input_duration_s": round(input_duration, 3),
        "rtf": round(rtf, 3),
        "vram_before_mib": round(vram_before, 2),
        "vram_peak_mib": round(vram_peak, 2),
        "vram_after_mib": round(vram_after, 2),
        **audio_metrics,
    }
    return res


def main():
    input_wav = repo_root / "tests" / "fixtures" / "input_male.wav"
    model_path = repo_root / "models" / "weights" / "vctk_female_p231.pth"
    index_path = repo_root / "models" / "indices" / "vctk_female_p231.index"

    assert input_wav.exists(), f"Input WAV not found: {input_wav}"
    assert model_path.exists(), f"Model not found: {model_path}"
    assert index_path.exists(), f"Index not found: {index_path}"

    print(f"Loading RVC model: {model_path.name}")
    print(f"Loading Index: {index_path.name}")
    adapter = RVCAdapter()
    adapter.load_model(str(model_path), str(index_path))

    results = {
        "experiment_a_pitch": [],
        "experiment_b_index": [],
    }

    print("\n" + "=" * 65)
    print("STARTING EXPERIMENT A: PITCH COMPARISON")
    print("=" * 65)
    pitch_dir = repo_root / "output" / "quality_tests" / "pitch"

    pitch_values = [8.0, 9.0, 10.0, 11.0]
    for p in pitch_values:
        p_int = int(p)
        out_file = pitch_dir / f"pitch_plus_{p_int:02d}.wav"
        print(f"\n[Exp A] Running Pitch Shift: +{p_int} semitones (Index Rate: 0.75)...")
        r = run_conversion_run(
            adapter=adapter,
            input_wav=input_wav,
            output_wav=out_file,
            pitch=p,
            index_rate=0.75,
        )
        print(f"  -> Path: {r['output_path']}")
        print(f"  -> Inference Time: {r.get('inference_time_s')}s | RTF: {r.get('rtf')}")
        print(f"  -> Peak VRAM: {r.get('vram_peak_mib')} MiB")
        print(f"  -> Peak: {r['peak']} | RMS: {r['rms']} | Clipping: {r['clipping']}")
        results["experiment_a_pitch"].append(r)

    print("\n" + "=" * 65)
    print("STARTING EXPERIMENT B: INDEX RATE COMPARISON (Selected Pitch: +10 semitones)")
    print("=" * 65)
    index_dir = repo_root / "output" / "quality_tests" / "index"

    index_values = [0.45, 0.55, 0.65, 0.75]
    for idx_rate in index_values:
        idx_str = f"{int(idx_rate*100):02d}"
        out_file = index_dir / f"index_rate_0_{idx_str}_pitch10.wav"
        print(f"\n[Exp B] Running Index Rate: {idx_rate} (Pitch Shift: +10 semitones)...")
        r = run_conversion_run(
            adapter=adapter,
            input_wav=input_wav,
            output_wav=out_file,
            pitch=10.0,
            index_rate=idx_rate,
        )
        print(f"  -> Path: {r['output_path']}")
        print(f"  -> Inference Time: {r.get('inference_time_s')}s | RTF: {r.get('rtf')}")
        print(f"  -> Peak VRAM: {r.get('vram_peak_mib')} MiB")
        print(f"  -> Peak: {r['peak']} | RMS: {r['rms']} | Clipping: {r['clipping']}")
        results["experiment_b_index"].append(r)

    summary_path = repo_root / "output" / "quality_tests" / "tuning_metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics summary to: {summary_path}")


if __name__ == "__main__":
    main()
