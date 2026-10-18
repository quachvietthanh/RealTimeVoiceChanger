"""Environment and hardware diagnostics.

Inspects host operating system, Python runtime, and GPU capabilities.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from typing import Dict


def get_system_info() -> Dict[str, str]:
    """Gather core operating system and Python environment details."""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def get_gpu_info() -> Dict[str, str]:
    """Check for NVIDIA GPU availability via nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"status": "nvidia-smi not found in PATH (CPU mode)"}

    try:
        res = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return {
            "status": "Available",
            "gpus": "; ".join(line.strip() for line in res.stdout.strip().splitlines()),
        }
    except Exception as err:
        return {"status": f"Query error: {err}"}


def run_environment_summary() -> None:
    """Print an environment diagnostics summary to standard output."""
    sys_info = get_system_info()
    gpu_info = get_gpu_info()

    print("=" * 60)
    print("Environment Diagnostics Summary")
    print("=" * 60)
    for key, val in sys_info.items():
        print(f"  {key:<18}: {val}")
    print("\nGPU / Compute:")
    for key, val in gpu_info.items():
        print(f"  {key:<18}: {val}")
    print("=" * 60)
