"""Assemble the application folder that a Windows installer would ship (everything except the Python runtime).

The folder keeps the repository layout because the code locates its files from `__file__`: the profile registry
resolves config/ and models/ from src/voice_changer/profiles/registry.py, and the RVC adapter finds
third_party/applio from its own location. Copied:

  src/voice_changer/            application code and GUI assets (no __pycache__)
  config/voice_profiles.json    the registry, unchanged (profiles whose files are not shipped show "files missing")
  third_party/applio/rvc/       Applio inference code, already carrying the streaming patch; no rvc/models/,
                                no logs/ (training cache), no Gradio UI, no .git
  requirements/, pyproject.toml, README.md
  --with-aux-models             RMVPE + ContentVec into models/checkpoints/ AND their Applio location, so the
                                adapter's first load needs no write access to the install folder
  --profile ID                  that profile's model and index (refused while they are in the training cache or
                                unless the profile's data_rights are owner_recorded, consented or licensed; a
                                missing field ('unspecified') or 'unverified' is refused like any other unknown
                                state. --allow-unverified-rights overrides this for a local test build only, and
                                BUILD_MANIFEST.json then marks the folder "redistributable": false)

Not included, by design: a Python runtime and packages (install them with requirements/inference-win-py311-cu121.txt),
voice models not named with --profile, data/, output/, tests, training checkpoints and pretrained D/G weights.
BUILD_MANIFEST.json records the source commit, a dirty-tree flag and the SHA-256 of every file.
Nothing in the repository is modified; the output folder must not exist yet (or be empty).

    .venv\\Scripts\\python scripts/build_app_folder.py --out dist\\RealTimeVoiceChanger --with-aux-models
    dist\\RealTimeVoiceChanger\\...  then: <python> dist\\RealTimeVoiceChanger\\src\\voice_changer\\main.py --preflight
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from voice_changer import __version__  # noqa: E402
from voice_changer.profiles.registration import VERIFIED_RIGHTS, in_training_cache  # noqa: E402
from voice_changer.profiles.registry import ProfileRegistry  # noqa: E402

TOP_LEVEL_FILES = ("pyproject.toml", "README.md", "config/voice_profiles.json", "config/README.md",
                   "requirements/inference-win-py311-cu121.txt", "third_party/applio/LICENSE")
AUX_FILES = {  # models/checkpoints/<name> -> Applio location the adapter reads from
    "rmvpe.pt": "third_party/applio/rvc/models/predictors/rmvpe.pt",
    "contentvec/pytorch_model.bin": "third_party/applio/rvc/models/embedders/contentvec/pytorch_model.bin",
    "contentvec/config.json": "third_party/applio/rvc/models/embedders/contentvec/config.json",
}
EXCLUDED = [
    "third_party/applio/logs/ (training cache)", "third_party/applio/rvc/models/ (model files; see --with-aux-models)",
    "third_party/applio/.git, tabs/, core.py, app.py (Gradio UI and training front end)",
    "data/, output/, tests/, docs/, scripts/, prompts/, .venv/", "__pycache__/ and *.pyc",
]


class BuildError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _code_files(root: Path, skip_dirs: Iterable[str] = ()) -> list[Path]:
    skip = set(skip_dirs)
    return sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                  and p.suffix != ".pyc" and not any(s in p.relative_to(root).parts[:1] for s in skip))


def _git(*args: str, cwd: Path = REPO) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def plan_files(with_aux_models: bool, profile_ids: list[str], allow_unverified_rights: bool,
               repo: Path = REPO) -> list[tuple[Path, str]]:
    """(source, destination relative to the output folder) for every file to ship."""
    files: list[tuple[Path, str]] = []
    for rel in TOP_LEVEL_FILES:
        if not (repo / rel).is_file():
            raise BuildError(f"required file missing: {rel}")
        files.append((repo / rel, rel))
    for src in _code_files(repo / "src" / "voice_changer"):
        files.append((src, src.relative_to(repo).as_posix()))
    rvc = repo / "third_party" / "applio" / "rvc"
    if not (rvc / "infer" / "infer.py").is_file():
        raise BuildError("third_party/applio/rvc is missing (docs/model-setup.md)")
    for src in _code_files(rvc, skip_dirs=("models",)):
        files.append((src, src.relative_to(repo).as_posix()))
    if with_aux_models:
        for name, applio_rel in AUX_FILES.items():
            src = repo / "models" / "checkpoints" / name
            if not src.is_file():
                raise BuildError(f"auxiliary model missing: {src} (scripts/download_auxiliary_assets.py)")
            files += [(src, f"models/checkpoints/{name}"), (src, applio_rel)]
    registry = ProfileRegistry.from_file(repo / "config" / "voice_profiles.json", repo)
    for pid in profile_ids:
        profile = registry.get(pid)
        if profile.model_path is None or not profile.files_present():
            raise BuildError(f"profile '{pid}' has no model files on this machine")
        if profile.data_rights not in VERIFIED_RIGHTS and not allow_unverified_rights:
            raise BuildError(f"profile '{pid}' has data rights '{profile.data_rights}', not one of {VERIFIED_RIGHTS}; "
                             "it must not be distributed (--allow-unverified-rights for a local test build only)")
        for path in (profile.model_path, profile.index_path):
            if path is None:
                continue
            if in_training_cache(path, repo):
                raise BuildError(f"profile '{pid}' reads {path} from the training cache; run "
                                 f"scripts/relocate_profile_assets.py --profile {pid} first")
            files.append((path, path.resolve().relative_to(repo.resolve()).as_posix()))
    return files


def build(out: Path, with_aux_models: bool = False, profile_ids: Optional[list[str]] = None,
          allow_unverified_rights: bool = False, repo: Path = REPO) -> dict:
    out = out.resolve()
    if out == repo.resolve() or out in repo.resolve().parents:
        raise BuildError("the output folder must not be the repository or one of its parents")
    if out.exists() and any(out.iterdir()):
        raise BuildError(f"output folder is not empty: {out}")
    files = plan_files(with_aux_models, profile_ids or [], allow_unverified_rights, repo)
    registry = ProfileRegistry.from_file(repo / "config" / "voice_profiles.json", repo)
    unverified = [pid for pid in profile_ids or [] if registry.get(pid).data_rights not in VERIFIED_RIGHTS]
    records = []
    for src, rel in files:
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        records.append({"path": rel, "size": dest.stat().st_size, "sha256": _sha256(dest)})
    status = _git("status", "--porcelain", cwd=repo)
    manifest = {
        "app": "RealTimeVoiceChanger", "version": __version__,
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commit": _git("rev-parse", "HEAD", cwd=repo),
        "source_tree_dirty": bool(status) if status is not None else None,
        "applio_commit": _git("rev-parse", "HEAD", cwd=repo / "third_party" / "applio"),
        "builder_python": sys.version.split()[0],
        "options": {"with_aux_models": with_aux_models, "profiles": profile_ids or []},
        "redistributable": not unverified,
        "profiles_without_verified_rights": unverified,
        "not_included": ["Python runtime and packages (requirements/inference-win-py311-cu121.txt)",
                         "voice models not listed under options.profiles"],
        "excluded": EXCLUDED,
        "files": sorted(records, key=lambda r: r["path"]),
    }
    (out / "BUILD_MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="output folder (must not exist or be empty)")
    ap.add_argument("--with-aux-models", action="store_true", help="include RMVPE and ContentVec (~559 MB)")
    ap.add_argument("--profile", action="append", default=[], help="ship this profile's model and index")
    ap.add_argument("--allow-unverified-rights", action="store_true", help="local test builds only")
    args = ap.parse_args()
    try:
        manifest = build(args.out, args.with_aux_models, args.profile, args.allow_unverified_rights)
    except (BuildError, KeyError) as err:
        print(f"REFUSED: {err}", file=sys.stderr)
        return 1
    total = sum(f["size"] for f in manifest["files"])
    print(f"built {args.out}: {len(manifest['files'])} files, {total / 1e6:.1f} MB, version {manifest['version']}, "
          f"commit {manifest['source_commit']}{' (dirty tree)' if manifest['source_tree_dirty'] else ''}")
    if not manifest["redistributable"]:
        print("LOCAL TEST BUILD ONLY, do not distribute: data rights not verified for "
              + ", ".join(manifest["profiles_without_verified_rights"]), file=sys.stderr)
    print("next: install the Python runtime, then run <python> src\\voice_changer\\main.py --preflight in that folder")
    return 0


if __name__ == "__main__":
    sys.exit(main())
