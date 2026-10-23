"""Release preflight: static checks of an installation, without GPU, audio devices or model loading.

Answers "is everything this copy needs present and consistent?" before anyone listens to it: Python and
dependency versions, the Applio checkout and its streaming patch, auxiliary model files, the voice profile
registry and its files, and the bundled GUI assets. Packages are located with importlib.util.find_spec (not
imported); torch is imported only to read the pinned auxiliary hashes when verify_hashes=True, and CUDA is
never initialized. Nothing is written.

    voice-changer --preflight [--verify-hashes]
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable, Optional

from voice_changer.profiles.registry import REPO_ROOT

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

# Versions of the verified inference environment (docs/rvc-engine-specification.md, README).
TESTED_PYTHON = (3, 11)
# distribution name -> import name, tested version prefix ("" = presence only)
RUNTIME_PACKAGES: dict[str, tuple[str, str]] = {
    "torch": ("torch", "2.5.1"),
    "torchaudio": ("torchaudio", "2.5.1"),
    "numpy": ("numpy", ""),
    "scipy": ("scipy", ""),
    "soundfile": ("soundfile", ""),
    "librosa": ("librosa", ""),
    "faiss-cpu": ("faiss", ""),
    "sounddevice": ("sounddevice", ""),
    "transformers": ("transformers", "4."),
    # Imported at module level by the Applio modules the adapter loads (rvc.infer.infer, rvc.infer.pipeline,
    # rvc.lib.utils, rvc.lib.predictors.FCPE), so required even though only RMVPE and none of their features is used.
    "torchcrepe": ("torchcrepe", ""),
    "noisereduce": ("noisereduce", ""),
    "pedalboard": ("pedalboard", ""),
    "pydub": ("pydub", ""),
    "wget": ("wget", ""),
    "einops": ("einops", ""),
    "local-attention": ("local_attention", ""),
}
APPLIO_PATCH_MARKER = "def set_padding("  # added by patches/applio_streaming.patch


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # PASS, WARN or FAIL
    detail: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_python(version: tuple[int, ...] = tuple(sys.version_info[:3])) -> Check:
    text = ".".join(map(str, version))
    if tuple(version[:2]) == TESTED_PYTHON:
        return Check("python", PASS, f"Python {text}")
    return Check("python", WARN, f"Python {text}; only {TESTED_PYTHON[0]}.{TESTED_PYTHON[1]}.x is tested")


def check_packages(find_spec: Callable[[str], object] = importlib.util.find_spec,
                   version_of: Callable[[str], str] = metadata.version) -> list[Check]:
    checks = []
    for dist, (module, tested) in RUNTIME_PACKAGES.items():
        if find_spec(module) is None:
            checks.append(Check(f"package:{dist}", FAIL, "not installed (see README, Installation)"))
            continue
        try:
            installed = version_of(dist)
        except metadata.PackageNotFoundError:
            installed = "unknown"
        if tested and not installed.startswith(tested):
            checks.append(Check(f"package:{dist}", WARN, f"{installed} installed; tested with {tested}*"))
        else:
            checks.append(Check(f"package:{dist}", PASS, installed))
    return checks


def check_applio(root: Path) -> Check:
    pipeline = root / "third_party" / "applio" / "rvc" / "infer" / "pipeline.py"
    if not pipeline.is_file():
        return Check("applio", FAIL, f"Applio core not found at {pipeline.parent.parent.parent} "
                                     "(docs/model-setup.md)")
    if APPLIO_PATCH_MARKER not in pipeline.read_text(encoding="utf-8", errors="replace"):
        return Check("applio", FAIL, "streaming patch not applied: run scripts/setup_applio_patch.py")
    return Check("applio", PASS, "present, streaming patch applied")


def _pinned_auxiliary_hashes() -> dict[str, str]:
    from voice_changer.conversion.rvc.contentvec import CONTENTVEC_CONFIG_SHA256, CONTENTVEC_WEIGHTS_SHA256
    from voice_changer.conversion.rvc.rmvpe import RMVPE_SHA256

    return {"rmvpe.pt": RMVPE_SHA256, "contentvec/pytorch_model.bin": CONTENTVEC_WEIGHTS_SHA256,
            "contentvec/config.json": CONTENTVEC_CONFIG_SHA256}


def check_auxiliary_assets(root: Path, verify_hashes: bool = False,
                           pinned: Optional[Callable[[], dict[str, str]]] = None) -> list[Check]:
    checkpoints = root / "models" / "checkpoints"
    names = ("rmvpe.pt", "contentvec/pytorch_model.bin", "contentvec/config.json")
    hashes: dict[str, str] = {}
    if verify_hashes:
        try:
            hashes = (pinned or _pinned_auxiliary_hashes)()
        except Exception as err:  # torch or Applio missing: the package checks name the cause
            return [Check(f"aux:{n}", FAIL, f"cannot read the pinned hash ({type(err).__name__}: {err})")
                    for n in names]
    checks = []
    for name in names:
        path = checkpoints / name
        if not path.is_file():
            checks.append(Check(f"aux:{name}", FAIL, f"missing: {path} (scripts/download_auxiliary_assets.py)"))
        elif verify_hashes and _sha256(path) != hashes[name]:
            checks.append(Check(f"aux:{name}", FAIL, f"does not match its pinned SHA-256: {path}"))
        else:
            checks.append(Check(f"aux:{name}", PASS, "SHA-256 verified" if verify_hashes else "present"))
    return checks


def check_profiles(root: Path, registry_path: Optional[Path] = None, verify_hashes: bool = False) -> list[Check]:
    from voice_changer.profiles.registration import in_training_cache
    from voice_changer.profiles.registry import ProfileRegistry, ProfileRegistryError

    path = registry_path or root / "config" / "voice_profiles.json"
    try:
        registry = ProfileRegistry.from_file(path, root)
    except ProfileRegistryError as err:
        return [Check("registry", FAIL, str(err))]
    checks = [Check("registry", PASS, f"{len(registry.list_profiles())} profiles in {path}")]
    for p in registry.list_profiles():
        name = f"profile:{p.profile_id}"
        if p.is_planned:
            checks.append(Check(name, PASS, "planned: no model, listed as not loadable"))
            continue
        notes = [f"status {p.status}"]
        if p.is_withdrawn:
            notes.append("withdrawn")
        if p.data_rights == "unverified":
            notes.append("data rights unverified: do not distribute")
        if not p.files_present():
            checks.append(Check(name, WARN, "; ".join(notes + ["model or index file not on this machine"])))
            continue
        status = WARN if p.data_rights == "unverified" else PASS
        if in_training_cache(p.model_path, root) or in_training_cache(p.index_path, root):
            status = WARN
            notes.append("reads from third_party/applio/logs/ (scripts/relocate_profile_assets.py)")
        for kind, file, pin in (("model", p.model_path, p.model_sha256), ("index", p.index_path, p.index_sha256)):
            if file is None:
                continue
            if not pin:
                status = WARN if status == PASS else status
                notes.append(f"{kind} SHA-256 not pinned")
            elif verify_hashes and _sha256(file) != pin:
                status = FAIL
                notes.append(f"{kind} does not match its pinned SHA-256")
        if verify_hashes and status != FAIL:
            notes.append("hashes verified")
        checks.append(Check(name, status, "; ".join(notes)))
    default = registry.default_profile_id
    if default is None:
        checks.append(Check("default_profile", WARN, "no default profile"))
    elif not registry.get(default).files_present():
        checks.append(Check("default_profile", FAIL, f"default '{default}' has no files on this machine"))
    else:
        checks.append(Check("default_profile", PASS, default))
    return checks


def check_gui_assets() -> Check:
    try:
        from voice_changer.gui.presentation import verify_asset_manifest
    except Exception as err:
        return Check("gui_assets", FAIL, f"GUI package cannot be imported ({type(err).__name__}: {err})")
    problems = verify_asset_manifest()
    if problems:
        return Check("gui_assets", FAIL, "; ".join(problems))
    return Check("gui_assets", PASS, "asset manifest matches the bundled files")


def run_preflight(root: Path = REPO_ROOT, registry_path: Optional[Path] = None,
                  verify_hashes: bool = False) -> list[Check]:
    """All checks, in display order. A FAIL means this copy cannot be expected to work."""
    from voice_changer import __version__

    checks = [Check("version", PASS, __version__), check_python()]
    checks += check_packages()
    checks.append(check_applio(root))
    checks += check_auxiliary_assets(root, verify_hashes)
    checks += check_profiles(root, registry_path, verify_hashes)
    checks.append(check_gui_assets())
    return checks


def format_report(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = [f"{c.status:4s}  {c.name:{width}s}  {c.detail}" for c in checks]
    counts = {s: sum(c.status == s for c in checks) for s in (PASS, WARN, FAIL)}
    lines.append(f"{counts[PASS]} passed, {counts[WARN]} warnings, {counts[FAIL]} failed")
    return "\n".join(lines)
