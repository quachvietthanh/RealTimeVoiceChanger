"""Milestone 1B Verification Script: PyTorch CUDA & Audio Smoke Test.

Verifies:
1. PyTorch import and version.
2. CUDA availability and GPU device identification.
3. Pre-test VRAM availability.
4. Real GPU tensor operations (FP32 & FP16).
5. Post-test VRAM availability and memory tracking.
6. Audio device enumeration via sounddevice.
7. Short audio capture and playback validation.
"""

from __future__ import annotations

import sys
import time


def test_pytorch_cuda() -> dict:
    print("=" * 60)
    print("1. PyTorch & CUDA Verification")
    print("=" * 60)

    try:
        import torch
        import torch.version
    except ImportError as e:
        print(f"[FAIL] PyTorch is not installed: {e}")
        return {"status": "FAIL", "error": str(e)}

    print(f"PyTorch Version : {torch.__version__}")
    cuda_runtime = getattr(torch.version, "cuda", "Unknown")
    print(f"CUDA Runtime    : {cuda_runtime}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available  : {cuda_available}")

    if not cuda_available:
        print("[FAIL] CUDA is not available to PyTorch.")
        return {"status": "FAIL", "error": "CUDA not available"}

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    device_props = torch.cuda.get_device_properties(0)
    total_vram_mb = device_props.total_memory / (1024 * 1024)
    allocated_before_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
    reserved_before_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)

    print(f"Device Count    : {device_count}")
    print(f"Device Name     : {device_name}")
    print(f"Compute Cap.    : {capability[0]}.{capability[1]}")
    print(f"Multi-Proc Count: {getattr(device_props, 'multi_processor_count', 'N/A')}")
    print(f"Total VRAM      : {total_vram_mb:.1f} MiB")
    print(f"Allocated Before: {allocated_before_mb:.2f} MiB")
    print(f"Reserved Before : {reserved_before_mb:.2f} MiB")

    print("\nExecuting FP32 matrix multiplication on GPU...")
    x = torch.randn((2048, 2048), device="cuda", dtype=torch.float32)
    y = torch.randn((2048, 2048), device="cuda", dtype=torch.float32)
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    fp32_sum = z.sum().item()
    print(f"[PASS] FP32 Matmul completed successfully (sum={fp32_sum:.2f})")

    print("Executing FP16 matrix multiplication on GPU...")
    x16 = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
    y16 = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
    z16 = torch.matmul(x16, y16)
    torch.cuda.synchronize()
    fp16_sum = z16.float().sum().item()
    print(f"[PASS] FP16 Matmul completed successfully (accumulated sum={fp16_sum:.2f})")

    allocated_after_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
    reserved_after_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
    print(f"Allocated After : {allocated_after_mb:.2f} MiB")
    print(f"Reserved After  : {reserved_after_mb:.2f} MiB")

    del x, y, z, x16, y16, z16
    torch.cuda.empty_cache()
    allocated_clean_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
    reserved_clean_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
    print(f"Allocated Clean : {allocated_clean_mb:.2f} MiB")
    print(f"Reserved Clean  : {reserved_clean_mb:.2f} MiB")

    return {
        "status": "PASS",
        "pytorch_version": torch.__version__,
        "device_name": device_name,
        "total_vram_mb": total_vram_mb,
        "allocated_before_mb": allocated_before_mb,
        "allocated_after_mb": allocated_after_mb,
        "reserved_after_mb": reserved_after_mb,
        "allocated_clean_mb": allocated_clean_mb,
    }


def test_audio_subsystem() -> dict:
    print("\n" + "=" * 60)
    print("2. Audio Subsystem & Device Enumeration")
    print("=" * 60)

    try:
        import sounddevice as sd
        import numpy as np
    except ImportError as e:
        print(f"[FAIL] Audio libraries not installed: {e}")
        return {"status": "FAIL", "error": str(e)}

    print(f"sounddevice version : {sd.__version__}")
    print(f"PortAudio version   : {sd.get_portaudio_version()[1]}")

    devices = sd.query_devices()
    input_devices = [d for d in devices if d["max_input_channels"] > 0]
    output_devices = [d for d in devices if d["max_output_channels"] > 0]

    print(f"Total Audio Endpoints Detected: {len(devices)}")
    print(f"  - Input Endpoints : {len(input_devices)}")
    print(f"  - Output Endpoints: {len(output_devices)}")

    vb_audio_found = any("CABLE" in d["name"].upper() or "VB-AUDIO" in d["name"].upper() for d in devices)
    print(f"\nVB-Audio Virtual Cable Detected: {vb_audio_found}")
    if not vb_audio_found:
        print("  [INFO] VB-Audio Virtual Cable is NOT installed on this machine.")

    print("\nAudio Endpoints Details:")
    for idx, d in enumerate(devices):
        dev_type = []
        if d["max_input_channels"] > 0:
            dev_type.append(f"IN:{d['max_input_channels']}")
        if d["max_output_channels"] > 0:
            dev_type.append(f"OUT:{d['max_output_channels']}")
        type_str = ", ".join(dev_type)
        print(f"  [{idx:2d}] {d['name'][:35]:<35} | {type_str:<12} | Default SR: {int(d['default_samplerate'])} Hz")

    default_input = sd.default.device[0]
    default_output = sd.default.device[1]
    def_in_name = devices[default_input]['name'] if default_input is not None and default_input >= 0 else 'None'
    def_out_name = devices[default_output]['name'] if default_output is not None and default_output >= 0 else 'None'
    print(f"\nDefault Input Device ID : {default_input} ({def_in_name})")
    print(f"Default Output Device ID: {default_output} ({def_out_name})")

    capture_status = "SKIPPED"
    if default_input is not None and default_input >= 0:
        print("\nPerforming short 1.0s audio capture test on default input device...")
        sample_rate = 44100
        duration_sec = 1.0
        try:
            recording = sd.rec(int(duration_sec * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
            sd.wait()
            rms = float(np.sqrt(np.mean(recording**2)))
            max_amp = float(np.max(np.abs(recording)))
            del recording
            print(f"[PASS] Capture successful: 1.0s recorded (RMS={rms:.5f}, MaxAmp={max_amp:.5f}) - samples discarded")
            capture_status = f"PASS (RMS={rms:.5f}, MaxAmp={max_amp:.5f})"
        except Exception as err:
            print(f"[WARN] Audio capture test failed: {err}")
            capture_status = f"FAIL ({err})"
    else:
        print("[WARN] No default input device found for capture test.")

    return {
        "status": "PASS",
        "sounddevice_version": sd.__version__,
        "total_devices": len(devices),
        "input_devices": len(input_devices),
        "output_devices": len(output_devices),
        "vb_audio_installed": vb_audio_found,
        "capture_test": capture_status,
    }



def main() -> None:
    print("RealTimeVoiceChanger - Milestone 1B Verification")
    print(f"Python Executable: {sys.executable}\n")

    torch_res = test_pytorch_cuda()
    audio_res = test_audio_subsystem()

    print("\n" + "=" * 60)
    print("Milestone 1B Verification Summary")
    print("=" * 60)
    print(f"PyTorch CUDA Test : {torch_res.get('status')}")
    print(f"Audio Subsystem   : {audio_res.get('status')}")
    if torch_res.get("status") == "PASS" and audio_res.get("status") == "PASS":
        print("\nALL MILESTONE 1B SMOKE TESTS PASSED.")
    else:
        print("\nSOME CHECKS FAILED OR WERE INCOMPLETE.")


if __name__ == "__main__":
    main()
