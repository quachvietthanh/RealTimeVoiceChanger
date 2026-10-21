"""Preflight checks for the dedicated RVC training environment (rvc-train).

Run with the training interpreter, from the repository root:
    C:\\Users\\quach\\venvs\\rvc-train\\Scripts\\python scripts/check_training_env.py

Checks (read-only, no training is started):
- Interpreter is Python 3.11.x and is not the repository inference `.venv`
- Pinned versions: torch/torchaudio 2.5.1+cu121, numpy 1.26.4, matplotlib 3.7.2
- CUDA is available and a small FP32/FP16 tensor workload runs on the GPU
- Every third-party module imported on the Applio training path is importable
- Applio's spectrogram logger (rvc/train/utils.py, `fig.canvas.tostring_rgb()`) runs
- Applio v2/40000 config requests fp16 training
- Pretrained G/D and RMVPE/ContentVec files match their pinned SHA-256

Exit code 0 only when every check passes.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
APPLIO_ROOT = REPO_ROOT / "third_party" / "applio"
SCRIPTS_DIR = REPO_ROOT / "scripts"

PINNED_VERSIONS = {
    "torch": "2.5.1+cu121",
    "torchaudio": "2.5.1+cu121",
    "numpy": "1.26.4",
    "matplotlib": "3.7.2",
}

# Third-party modules imported by preprocess.py, extract.py, train.py, extract_index.py
# and their rvc.* dependencies (rvc/lib/utils.py imports wget and pydub at module level).
TRAINING_IMPORTS = [
    "torch", "torchaudio", "numpy", "scipy", "scipy.io.wavfile", "scipy.signal", "soundfile",
    "librosa", "noisereduce", "torchcrepe", "tqdm", "faiss", "sklearn.cluster", "matplotlib.pyplot",
    "torch.utils.tensorboard", "transformers", "pydub", "wget", "distutils.util",
]


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _run(name: str, fn: Callable[[], tuple[bool, str]]) -> CheckResult:
    try:
        passed, detail = fn()
    except Exception as err:  # each check reports its own failure instead of aborting the run
        return CheckResult(name, False, f"{type(err).__name__}: {err}")
    return CheckResult(name, passed, detail)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_interpreter() -> tuple[bool, str]:
    prefix = Path(sys.prefix).resolve()
    is_repo_venv = prefix == (REPO_ROOT / ".venv").resolve()
    ok = sys.version_info[:2] == (3, 11) and not is_repo_venv
    note = " (this is the inference .venv)" if is_repo_venv else ""
    return ok, f"Python {sys.version.split()[0]} at {sys.executable}{note}"


def check_version(package: str, expected: str) -> tuple[bool, str]:
    actual = importlib.metadata.version(package)
    return actual == expected, f"{actual} (expected {expected})"


def check_cuda() -> tuple[bool, str]:
    import torch
    import torch.version

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    device = torch.device("cuda:0")
    a = torch.randn(256, 256, device=device)
    b = (a @ a.T).sum().item()
    with torch.autocast("cuda", dtype=torch.float16):
        h = (a @ a).dtype
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(0)
    ok = h == torch.float16 and b == b  # b == b rejects NaN
    return ok, (f"{props.name}, {props.total_memory // 2**20} MiB, cc {props.major}.{props.minor}, "
                f"CUDA {torch.version.cuda}, autocast dtype {h}")


def check_imports() -> tuple[bool, str]:
    failed = []
    for name in TRAINING_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as err:
            failed.append(f"{name} ({type(err).__name__}: {err})")
    return not failed, "; ".join(failed) or f"{len(TRAINING_IMPORTS)} modules imported"


def check_logger() -> tuple[bool, str]:
    import numpy as np

    spec = importlib.util.spec_from_file_location("applio_train_utils", APPLIO_ROOT / "rvc" / "train" / "utils.py")
    assert spec is not None and spec.loader is not None
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    image = utils.plot_spectrogram_to_numpy(np.random.default_rng(0).random((80, 120)).astype(np.float32))
    ok = image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3 and image.max() > 0
    return ok, f"plot_spectrogram_to_numpy -> {image.dtype} {image.shape}"


def check_fp16_config() -> tuple[bool, str]:
    path = APPLIO_ROOT / "rvc" / "configs" / "v2" / "40000.json"
    fp16 = json.loads(path.read_text(encoding="utf-8"))["train"]["fp16_run"]
    return fp16 is True, f"{path.relative_to(REPO_ROOT)} train.fp16_run = {fp16}"


def check_assets() -> tuple[bool, str]:
    pretraineds = _load_script("download_pretraineds")
    auxiliary = _load_script("download_auxiliary_assets")
    expected = [(pretraineds.DEST_DIR / name, meta["expected_size"], meta["expected_sha256"])
                for name, meta in pretraineds.ASSETS.items()]
    expected += [(a.applio_sync_path, a.expected_size, a.expected_sha256) for a in auxiliary.ASSETS]
    bad = []
    for path, size, sha in expected:
        if not path.is_file():
            bad.append(f"missing {path.name}")
        elif path.stat().st_size != size or _sha256(path) != sha:
            bad.append(f"mismatch {path.name}")
    return not bad, "; ".join(bad) or f"{len(expected)} files match pinned size and SHA-256"


def run_checks() -> list[CheckResult]:
    results = [_run("interpreter", check_interpreter)]
    for package, expected in PINNED_VERSIONS.items():
        results.append(_run(f"version: {package}", lambda p=package, e=expected: check_version(p, e)))
    results += [
        _run("cuda: small tensor workload", check_cuda),
        _run("imports: training path", check_imports),
        _run("logger: rvc/train/utils.py plot_spectrogram_to_numpy", check_logger),
        _run("config: fp16_run", check_fp16_config),
        _run("assets: pretrained + auxiliary checksums", check_assets),
    ]
    return results


def main() -> int:
    results = run_checks()
    print("=" * 70)
    print("Training Environment Preflight")
    print("=" * 70)
    for r in results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name}" + (f" - {r.detail}" if r.detail else ""))
    failed = [r for r in results if not r.passed]
    print("-" * 70)
    print(f"Overall: {'PASS' if not failed else 'FAIL'} ({len(results) - len(failed)}/{len(results)} checks passed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
