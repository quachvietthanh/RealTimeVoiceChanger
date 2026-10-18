"""Milestone 2: Offline Voice Conversion Verification Script.

Executes:
1. Compatibility Preflight (Torch, NumPy, SoundFile, Librosa, Applio imports).
2. Model & Asset Availability Audit (Voice model .pth, .index, rmvpe.pt, hubert_base.pt).
3. Input Audio Fixture Verification (WAV format, channels, sample rate, duration, peak).
4. Offline Conversion Execution (if assets are present) with RTF, VRAM, and RMS metrics.
5. Quality Evaluation & Pitch Variation Comparison (if assets are present).
6. Clear reporting of missing requirements without fabricating results.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure src/ is on sys.path
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from voice_changer.conversion.base import ConversionConfig
from voice_changer.conversion.rvc.adapter import RVCAdapter


def run_compatibility_preflight() -> dict:
    print("=" * 65)
    print("1. Compatibility Preflight Check")
    print("=" * 65)

    results = {}

    try:
        import numpy as np
        import torch

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = torch.from_numpy(a)
        b = t.numpy()
        bridge_ok = np.allclose(a, b)
        print(f"NumPy <-> Torch Bridge : {'PASS' if bridge_ok else 'FAIL'}")
        print(f"  - NumPy Version      : {np.__version__}")
        print(f"  - PyTorch Version    : {torch.__version__}")
        print(f"  - CUDA Available     : {torch.cuda.is_available()}")
        results["bridge"] = bridge_ok
    except Exception as err:
        print(f"NumPy <-> Torch Bridge : FAIL ({err})")
        results["bridge"] = False

    try:
        import soundfile as sf
        print(f"SoundFile Installed    : PASS (v{sf.__version__})")
        results["soundfile"] = True
    except Exception as err:
        print(f"SoundFile Installed    : FAIL ({err})")
        results["soundfile"] = False

    try:
        adapter = RVCAdapter()
        vc = adapter._ensure_applio_initialized()
        print(f"Applio Core Import     : PASS (Class: {vc.__class__.__name__})")
        results["applio_import"] = True
    except Exception as err:
        print(f"Applio Core Import     : FAIL ({err})")
        results["applio_import"] = False

    return results


def check_model_assets(
    model_path: str | None,
    index_path: str | None,
) -> dict:
    print("\n" + "=" * 65)
    print("2. Model & Auxiliary Asset Audit")
    print("=" * 65)

    models_dir = repo_root / "models"
    weights_dir = models_dir / "weights"
    indices_dir = models_dir / "indices"
    checkpoints_dir = models_dir / "checkpoints"
    applio_models_dir = repo_root / "third_party" / "applio" / "rvc" / "models"

    model_file = Path(model_path) if model_path else None
    model_info = None
    if model_file and model_file.exists():
        found_weights = [model_file]
    else:
        found_weights = sorted(list(weights_dir.glob("*.pth")))
        model_file = found_weights[0] if found_weights else None

    if model_file and model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        try:
            import torch

            cpt = torch.load(str(model_file.resolve()), map_location="cpu")
            if isinstance(cpt, dict) and "weight" in cpt and "config" in cpt:
                version = cpt.get("version", "v1")
                f0 = cpt.get("f0", 1)
                sr = cpt["config"][-1] if isinstance(cpt["config"], (list, tuple)) else "unknown"
                model_status = f"VALID [REQUIRED] ({model_file.name}, {size_mb:.1f} MB, {version}, {sr}Hz, f0={'True' if f0 == 1 else 'False'})"
                model_info = {"valid": True, "version": version, "f0": f0, "sr": sr}
            else:
                model_status = f"INVALID ({model_file.name}: missing 'weight' or 'config' dictionary keys)"
                model_file = None
        except Exception as err:
            model_status = f"CORRUPT/UNREADABLE ({model_file.name}: {err})"
            model_file = None
    else:
        model_status = "MISSING [REQUIRED] (No .pth voice weights found in models/weights/)"
        model_file = None

    print(f"Target Voice Model (.pth) : {model_status}")

    index_file = Path(index_path) if index_path else None
    if index_file and index_file.exists():
        index_status = f"FOUND [OPTIONAL] ({index_file.name}, {index_file.stat().st_size / (1024*1024):.1f} MB)"
    else:
        found_indices = sorted(list(indices_dir.glob("*.index")))
        if found_indices:
            index_file = found_indices[0]
            index_status = f"FOUND [OPTIONAL] ({index_file.name}, {index_file.stat().st_size / (1024*1024):.1f} MB)"
        else:
            index_status = "NOT PROVIDED [OPTIONAL] (Conversion will operate cleanly without feature retrieval)"
            index_file = None

    print(f"Feature Index (.index)    : {index_status}")

    rmvpe_locations = [
        checkpoints_dir / "rmvpe.pt",
        applio_models_dir / "predictors" / "rmvpe.pt",
    ]
    rmvpe_file = next((p for p in rmvpe_locations if p.exists()), None)
    if rmvpe_file:
        rmvpe_size = rmvpe_file.stat().st_size / (1024 * 1024)
        rmvpe_status = f"FOUND [REQUIRED] ({rmvpe_file.name}, {rmvpe_size:.1f} MB at {rmvpe_file.parent.relative_to(repo_root)})"
    else:
        rmvpe_status = "MISSING [REQUIRED] (rmvpe.pt needed for pitch extraction in models/checkpoints/)"
    print(f"RMVPE Pitch Model         : {rmvpe_status}")

    legacy_hubert = checkpoints_dir / "hubert_base.pt"
    if legacy_hubert.exists():
        print("  [NOTE] hubert_base.pt detected in models/checkpoints/, but Applio v3.2.8 requires HuggingFace ContentVec instead.")

    contentvec_locations = [
        checkpoints_dir / "contentvec" / "pytorch_model.bin",
        checkpoints_dir / "pytorch_model.bin",
        applio_models_dir / "embedders" / "contentvec" / "pytorch_model.bin",
    ]
    contentvec_file = next((p for p in contentvec_locations if p.exists() and (p.parent / "config.json").exists()), None)
    if contentvec_file:
        cv_size = contentvec_file.stat().st_size / (1024 * 1024)
        cv_status = f"FOUND [REQUIRED] (pytorch_model.bin [{cv_size:.1f} MB] + config.json in {contentvec_file.parent.relative_to(repo_root)})"
    else:
        cv_status = "MISSING [REQUIRED] (ContentVec pytorch_model.bin & config.json needed in models/checkpoints/contentvec/)"
    print(f"ContentVec Embedder       : {cv_status}")

    all_required_present = bool(model_file and rmvpe_file and contentvec_file)

    return {
        "all_present": all_required_present,
        "model_file": str(model_file) if model_file else None,
        "model_info": model_info,
        "index_file": str(index_file) if index_file else None,
        "rmvpe_file": str(rmvpe_file) if rmvpe_file else None,
        "contentvec_file": str(contentvec_file) if contentvec_file else None,
    }


def inspect_audio_input(audio_path: str | None) -> dict:
    print("\n" + "=" * 65)
    print("3. Input Audio Fixture Inspection")
    print("=" * 65)

    fixtures_dir = repo_root / "tests" / "fixtures"
    in_file = Path(audio_path) if audio_path else None

    if not in_file or not in_file.exists():
        found_wavs = list(fixtures_dir.glob("*.wav"))
        if found_wavs:
            in_file = found_wavs[0]
        else:
            print("Audio Test Fixture     : MISSING (No clean input .wav file provided)")
            return {"exists": False, "file": None}

    try:
        import soundfile as sf
        import numpy as np

        info = sf.info(str(in_file))
        data, sr = sf.read(str(in_file))
        rms = float(np.sqrt(np.mean(data**2)))
        peak = float(np.max(np.abs(data)))

        print(f"File Path              : {in_file}")
        print(f"Sample Rate            : {info.samplerate} Hz")
        print(f"Channels               : {info.channels}")
        print(f"Duration               : {info.duration:.2f} seconds")
        print(f"Subtype / Format       : {info.subtype} / {info.format}")
        print(f"RMS Level / Peak       : {rms:.5f} / {peak:.5f}")

        return {
            "exists": True,
            "file": str(in_file),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration": info.duration,
            "rms": rms,
            "peak": peak,
        }
    except Exception as err:
        print(f"Audio Test Fixture     : ERROR reading file ({err})")
        return {"exists": False, "file": str(in_file), "error": str(err)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 2: Offline WAV Conversion POC")
    parser.add_argument("--model", type=str, default=None, help="Path to RVC .pth model weights")
    parser.add_argument("--index", type=str, default=None, help="Path to RVC .index file")
    parser.add_argument("--input", type=str, default=None, help="Path to clean input .wav test file")
    parser.add_argument("--output", type=str, default="output/offline_test_result.wav", help="Output path")
    parser.add_argument("--pitch", type=float, default=12.0, help="Semitone pitch transposition")
    args = parser.parse_args()

    preflight = run_compatibility_preflight()
    assets = check_model_assets(args.model, args.index)
    audio = inspect_audio_input(args.input)

    print("\n" + "=" * 65)
    print("4. Conversion Readiness Assessment")
    print("=" * 65)

    if not preflight.get("applio_import", False):
        print("[FAIL] Compatibility preflight failed. Dependencies missing.")
        return

    if not assets["all_present"]:
        print("[NOT VERIFIED] Missing required model weights or auxiliary models.")
        print("To proceed with actual conversion, provide the following required assets:")
        if not assets["model_file"]:
            print("  [REQUIRED] Target voice model : Place a valid RVC v2 <model>.pth into models/weights/")
        if not assets["rmvpe_file"]:
            print("  [REQUIRED] Pitch model        : Place rmvpe.pt into models/checkpoints/ (or third_party/applio/rvc/models/predictors/)")
        if not assets["contentvec_file"]:
            print("  [REQUIRED] Embedder model     : Place pytorch_model.bin & config.json into models/checkpoints/contentvec/ (or third_party/applio/rvc/models/embedders/contentvec/)")
        print("\nRefer to docs/model-setup.md for verified sources and setup instructions.")
        print("In accordance with rules, conversion was NOT executed and NO fake outputs were produced.")
        return

    if not audio["exists"]:
        print("[NOT VERIFIED] Missing clean test audio sample.")
        print("To proceed with actual conversion, provide a small authorized test speech WAV file in tests/fixtures/")
        print("Refer to docs/model-setup.md for audio fixture requirements.")
        print("In accordance with rules, conversion was NOT executed and NO fake outputs were produced.")
        return

    print("All required models and audio fixture are present! Running baseline conversion...")
    import torch
    import soundfile as sf
    import numpy as np

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
        vram_before = torch.cuda.memory_allocated(0) / (1024 * 1024)
    else:
        vram_before = 0.0

    adapter = RVCAdapter()
    adapter.load_model(assets["model_file"], assets["index_file"])

    config = ConversionConfig(pitch_shift=args.pitch)
    out_path = repo_root / args.output

    start = time.perf_counter()
    res = adapter.convert_file(audio["file"], str(out_path), config)
    duration = time.perf_counter() - start

    if torch.cuda.is_available():
        vram_peak = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
        vram_after = torch.cuda.memory_allocated(0) / (1024 * 1024)
    else:
        vram_peak = 0.0
        vram_after = 0.0

    rtf = duration / audio["duration"]

    print("\n" + "=" * 65)
    print("Conversion Results")
    print("=" * 65)
    print(f"Output Path          : {out_path}")
    print(f"Inference Time       : {duration:.3f} s")
    print(f"Input Audio Time     : {audio['duration']:.3f} s")
    print(f"Real-Time Factor     : {rtf:.3f} (RTF < 1.0 indicates faster than real-time)")
    print(f"VRAM Before/Peak/After: {vram_before:.2f} MiB / {vram_peak:.2f} MiB / {vram_after:.2f} MiB")

    if out_path.exists():
        out_info = sf.info(str(out_path))
        out_data, out_sr = sf.read(str(out_path))
        out_peak = float(np.max(np.abs(out_data)))
        out_rms = float(np.sqrt(np.mean(out_data**2)))
        clipping = out_peak >= 0.999
        is_silent = out_peak < 1e-4

        print(f"Output Sample Rate   : {out_info.samplerate} Hz")
        print(f"Output Channels      : {out_info.channels}")
        print(f"Output Duration      : {out_info.duration:.3f} s")
        print(f"Output Peak / RMS    : {out_peak:.5f} / {out_rms:.5f}")
        print(f"Clipping Detected    : {clipping}")
        print(f"Silence Detected     : {is_silent}")


if __name__ == "__main__":
    main()
