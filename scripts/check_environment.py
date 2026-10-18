"""Diagnostic script to inspect local Python, OS, and GPU environment.

This script uses the standard library to inspect system capabilities
without requiring third-party machine learning dependencies.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def check_python_environment() -> None:
    print("=" * 60)
    print("Python & System Environment Report")
    print("=" * 60)
    os_name = platform.system()
    os_rel = platform.release()
    try:
        build = int(platform.version().split(".")[-1])
        if os_name == "Windows" and build >= 22000:
            os_rel = f"11 (Build {build})"
    except (ValueError, IndexError):
        pass

    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"Python Executable : {sys.executable}")
    print(f"OS Platform       : {os_name} {os_rel}")
    print(f"Platform String   : {platform.platform()}")
    print(f"Machine / Arch    : {platform.machine()}")
    print(f"Processor         : {platform.processor()}")



def check_gpu_environment() -> None:
    print("\n" + "=" * 60)
    print("GPU / Compute Acceleration Check")
    print("=" * 60)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        print(f"nvidia-smi found at: {nvidia_smi}")
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            print("Detected GPU(s):")
            for line in result.stdout.strip().splitlines():
                print(f"  - {line.strip()}")
        except Exception as err:
            print(f"Error querying nvidia-smi: {err}")
    else:
        print("nvidia-smi: Not found in system PATH (CPU mode or discrete driver unlisted)")


def check_installed_audio_and_ml_packages() -> None:
    print("\n" + "=" * 60)
    print("Audio & ML Package Status (Optional Check)")
    print("=" * 60)
    packages_to_check = ["torch", "torchaudio", "sounddevice", "pyaudio", "onnxruntime"]
    for pkg in packages_to_check:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed")
            print(f"  - {pkg:<15}: Available ({version})")
        except ImportError:
            print(f"  - {pkg:<15}: Not installed (deferred to Phase 1)")


def check_applio_patch() -> None:
    print("\n" + "=" * 60)
    print("Applio Core v3.2.8 Streaming Optimization Patch Check")
    print("=" * 60)
    try:
        from scripts.setup_applio_patch import check_patch_status
        status = check_patch_status()
        if status == 0:
            print("  - Streaming Patch : Applied and verified")
        elif status == 1:
            print("  - Streaming Patch : NOT APPLIED (Run: python scripts/setup_applio_patch.py --apply)")
        else:
            print("  - Streaming Patch : Applio directory not found")
    except Exception as err:
        print(f"  - Error checking patch status: {err}")


def main() -> None:
    check_python_environment()
    check_gpu_environment()
    check_installed_audio_and_ml_packages()
    check_applio_patch()
    print("\nEnvironment check complete.")


if __name__ == "__main__":
    main()
