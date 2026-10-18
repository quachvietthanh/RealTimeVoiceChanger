"""Diagnostic script to inspect local Python, OS, and GPU environment.

This script uses the standard library to inspect system capabilities
without requiring third-party machine learning dependencies.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def check_python_environment() -> None:
    print("=" * 60)
    print("Python & System Environment Report")
    print("=" * 60)
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"Python Executable : {sys.executable}")
    print(f"Platform          : {platform.platform()}")
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


def main() -> None:
    check_python_environment()
    check_gpu_environment()
    check_installed_audio_and_ml_packages()
    print("\nEnvironment check complete.")


if __name__ == "__main__":
    main()
