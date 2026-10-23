"""Register a trained model and index for a voice profile (for example a 'planned' Jett or Sage entry).

register_model() edits only the chosen entry of the registry JSON: it pins the files by SHA-256, sets the
status to 'preview' and validates the result exactly as select_profile would, before anything is written.
It refuses when:
  - the profile is not 'planned' (unless replace=True; an 'approved' or 'prototype' entry is never replaced),
  - a file is missing, not an RVC checkpoint / Faiss index, incompatible, or trained with an unsupported embedder,
  - the model or index is already registered for another profile (same path or same content),
  - a file lives under third_party/applio/logs/ (a training cache that cleanups have hit before),
  - the data rights are not owner_recorded/consented/licensed, unless allow_unverified_rights=True.
Writing keeps a timestamped backup of the previous file and replaces it atomically. Model files are never
copied, moved or modified.

relocate_profile_assets() moves an already registered profile off the training cache: it copies (never moves)
the registered files to models/, proves each copy identical to the pinned SHA-256, and only then repoints
model_path/index_path, keeping every other field of the entry.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from voice_changer.profiles.registry import (
    DATA_RIGHTS,
    DEFAULT_REGISTRY_PATH,
    REPO_ROOT,
    ProfileRegistry,
    ProfileRegistryError,
    ProfileValidation,
)

VERIFIED_RIGHTS = ("owner_recorded", "consented", "licensed")
TRAINING_CACHE = Path("third_party") / "applio" / "logs"


class RegistrationError(ValueError):
    """The model cannot be registered for the profile; nothing was written."""


@dataclass(frozen=True)
class RegistrationResult:
    profile_id: str
    entry: dict[str, Any]
    validation: ProfileValidation
    written: bool
    backup_path: Optional[Path]


def format_registry(data: dict[str, Any]) -> str:
    """Serialize registry JSON in the repository's style: one profile field per line, nested objects inline."""
    top = [f'  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}' for k, v in data.items() if k != "profiles"]
    profiles = []
    for p in data.get("profiles", []):
        fields = ",\n".join(f'      {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}' for k, v in p.items())
        profiles.append("    {\n" + fields + "\n    }")
    top.append('  "profiles": [\n' + ",\n".join(profiles) + "\n  ]")
    return "{\n" + ",\n".join(top) + "\n}\n"


def _repo_relative(path: Path, base_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def register_model(
    profile_id: str,
    model_path: Path,
    index_path: Optional[Path],
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    base_dir: Path = REPO_ROOT,
    data_rights: Optional[str] = None,
    allow_unverified_rights: bool = False,
    allow_no_index: bool = False,
    replace: bool = False,
    write: bool = False,
) -> RegistrationResult:
    """Validate and (with write=True) register `model_path`/`index_path` for `profile_id`.

    Raises:
        RegistrationError: any refusal listed in the module docstring; the registry file is unchanged.
    """
    from voice_changer.conversion.rvc.assets import sha256_file

    registry_path = Path(registry_path)
    original = registry_path.read_bytes()
    data = json.loads(original.decode("utf-8"))
    try:
        current = ProfileRegistry.from_data(data, base_dir)
        profile = current.get(profile_id)
    except (ProfileRegistryError, KeyError) as err:
        raise RegistrationError(str(err).strip("'\"")) from err

    if profile.status in ("approved", "prototype"):
        raise RegistrationError(f"profile '{profile_id}' is '{profile.status}'; register a new profile id instead")
    if profile.status != "planned" and not replace:
        raise RegistrationError(f"profile '{profile_id}' already has a model (status '{profile.status}'); "
                                "pass --replace (replace=True) to register a different one")
    if index_path is None and not allow_no_index:
        raise RegistrationError("an index is required (pass --no-index / allow_no_index=True to register a model without one)")

    files = {"model": Path(model_path), "index": Path(index_path) if index_path is not None else None}
    for kind, path in files.items():
        if path is None:
            continue
        if not path.is_file():
            raise RegistrationError(f"{kind} file not found: {path}")
        rel = _repo_relative(path, base_dir)
        if Path(rel).parts[:3] == TRAINING_CACHE.parts:
            raise RegistrationError(f"{kind} file is inside {TRAINING_CACHE.as_posix()}/ (a training cache); "
                                    "copy it to models/weights/ or models/indices/ first")

    rights = data_rights if data_rights is not None else profile.data_rights
    if rights not in DATA_RIGHTS:
        raise RegistrationError(f"data_rights must be one of {DATA_RIGHTS}")
    if rights not in VERIFIED_RIGHTS and not allow_unverified_rights:
        raise RegistrationError(f"data rights are '{rights}'; set them to one of {VERIFIED_RIGHTS} once verified, "
                                "or pass --allow-unverified-rights (allow_unverified_rights=True) for a local, never-distributed test")

    entry = next(p for p in data["profiles"] if p.get("id") == profile_id)
    entry.update({
        "status": "preview",
        "data_rights": rights,
        "model_path": _repo_relative(files["model"], base_dir),
        "model_sha256": sha256_file(files["model"]),
    })
    if files["index"] is not None:
        entry.update({"index_path": _repo_relative(files["index"], base_dir),
                      "index_sha256": sha256_file(files["index"])})
    else:
        entry.pop("index_path", None)
        entry.pop("index_sha256", None)
    entry["provenance"] = {**entry.get("provenance", {}), "registered_on": datetime.now().strftime("%Y-%m-%d")}

    try:
        candidate = ProfileRegistry.from_data(data, base_dir)
    except ProfileRegistryError as err:
        raise RegistrationError(str(err)) from err
    validation = candidate.validate(profile_id)
    if not validation.ok:
        raise RegistrationError("; ".join(validation.errors))

    backup = _write_registry(registry_path, original, data, base_dir) if write else None
    return RegistrationResult(profile_id, entry, validation, write, backup)


def _write_registry(registry_path: Path, original: bytes, data: dict[str, Any], base_dir: Path) -> Path:
    """Keep a timestamped, byte-exact backup of `original`, then atomically replace the registry with `data`.

    The new file keeps the original line endings (CRLF only if the original used them); text-mode writes on
    Windows would otherwise turn every LF into CRLF, so the diff and a restore from the backup stay exact.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = registry_path.with_name(f"{registry_path.name}.bak-{stamp}")
    try:
        with open(backup, "xb") as f:  # never overwrite an earlier backup
            f.write(original)
    except FileExistsError:
        raise RegistrationError(f"backup {backup} already exists; retry in a second") from None
    newline = "\r\n" if b"\r\n" in original else "\n"
    tmp = registry_path.with_name(registry_path.name + ".tmp")
    try:
        tmp.write_bytes(format_registry(data).replace("\n", newline).encode("utf-8"))
        ProfileRegistry.from_file(tmp, base_dir)  # the written file must load
        os.replace(tmp, registry_path)
    finally:
        tmp.unlink(missing_ok=True)
    return backup


# ------------------------------------------------------------------------------ relocation out of the cache
def in_training_cache(path: Optional[Path], base_dir: Path = REPO_ROOT) -> bool:
    """True if `path` lies under third_party/applio/logs/ of `base_dir`."""
    if path is None:
        return False
    return Path(_repo_relative(path, base_dir)).parts[:3] == TRAINING_CACHE.parts


def profiles_in_training_cache(registry: ProfileRegistry, base_dir: Path = REPO_ROOT) -> list[str]:
    """Ids of registered profiles whose model or index is read from the training cache."""
    return [p.profile_id for p in registry.list_profiles()
            if in_training_cache(p.model_path, base_dir) or in_training_cache(p.index_path, base_dir)]


@dataclass(frozen=True)
class RelocationStep:
    kind: str  # "model" or "index"
    source: Path  # currently registered file
    destination: Path
    action: str  # "copy" (needed), "copied" (done now), "present" (identical copy already there), "unchanged"


@dataclass(frozen=True)
class RelocationResult:
    profile_id: str
    steps: tuple[RelocationStep, ...]
    entry: dict[str, Any]  # the entry as it would be (dry run) or was written; unchanged while copies are missing
    validation: Optional[ProfileValidation]  # None until every destination file exists
    written: bool
    backup_path: Optional[Path]


def relocate_profile_assets(
    profile_id: str,
    model_dest: Optional[Path] = None,
    index_dest: Optional[Path] = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    base_dir: Path = REPO_ROOT,
    copy: bool = False,
    write: bool = False,
) -> RelocationResult:
    """Point a registered profile at byte-identical copies of its files outside the training cache.

    Defaults: models/weights/<file name> and models/indices/<file name>. With copy=True, a missing destination
    is filled by copying the registered file (to a temporary name, verified, then renamed); the registered
    file itself is never moved, modified or deleted, and an existing destination is never overwritten. With
    write=True, and only when every destination matches the pinned SHA-256 and the result validates, the
    entry's model_path/index_path are changed and the old paths recorded under provenance.relocated_from.
    Status, defaults, data rights, withdrawal and hashes are left exactly as they are.

    Raises:
        RegistrationError: nothing was copied over an existing file and the registry is unchanged.
    """
    import shutil

    from voice_changer.conversion.rvc.assets import sha256_file

    registry_path = Path(registry_path)
    original = registry_path.read_bytes()
    data = json.loads(original.decode("utf-8"))
    try:
        profile = ProfileRegistry.from_data(data, base_dir).get(profile_id)
    except (ProfileRegistryError, KeyError) as err:
        raise RegistrationError(str(err).strip("'\"")) from err
    if profile.model_path is None:
        raise RegistrationError(f"profile '{profile_id}' has no model to relocate (status '{profile.status}')")

    plan: list[tuple[str, Path, Path, str]] = [
        ("model", profile.model_path, Path(model_dest) if model_dest else base_dir / "models" / "weights" /
         profile.model_path.name, profile.model_sha256 or "")]
    if profile.index_path is not None:
        plan.append(("index", profile.index_path, Path(index_dest) if index_dest else base_dir / "models" /
                     "indices" / profile.index_path.name, profile.index_sha256 or ""))
    elif index_dest is not None:
        raise RegistrationError(f"profile '{profile_id}' has no index; do not pass an index destination")

    steps: list[RelocationStep] = []
    for kind, source, dest, pinned in plan:
        if not pinned:
            raise RegistrationError(f"{kind} SHA-256 is not pinned for '{profile_id}'; a copy cannot be proven "
                                    "identical (register the hash first)")
        if dest.resolve() == source.resolve():
            steps.append(RelocationStep(kind, source, dest, "unchanged"))
            continue
        if in_training_cache(dest, base_dir):
            raise RegistrationError(f"{kind} destination is inside {TRAINING_CACHE.as_posix()}/: {dest}")
        try:
            dest.resolve().relative_to(base_dir.resolve())
        except ValueError:
            raise RegistrationError(f"{kind} destination must be inside {base_dir}: {dest}") from None
        if dest.exists():
            if not dest.is_file() or sha256_file(dest) != pinned:
                raise RegistrationError(f"{kind} destination already exists and is not an identical copy; "
                                        f"it is left untouched: {dest}")
            steps.append(RelocationStep(kind, source, dest, "present"))
            continue
        if not source.is_file():
            raise RegistrationError(f"registered {kind} file not found, nothing to copy: {source}")
        if not copy:
            steps.append(RelocationStep(kind, source, dest, "copy"))
            continue
        if sha256_file(source) != pinned:
            raise RegistrationError(f"registered {kind} file no longer matches its pinned SHA-256: {source}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".partial")
        if tmp.exists():
            raise RegistrationError(f"leftover temporary file {tmp}; check and remove it first")
        try:
            shutil.copyfile(source, tmp)
            if sha256_file(tmp) != pinned:
                raise RegistrationError(f"copied {kind} does not match the pinned SHA-256; nothing was registered")
            os.rename(tmp, dest)  # fails instead of overwriting if dest appeared meanwhile
        finally:
            tmp.unlink(missing_ok=True)
        steps.append(RelocationStep(kind, source, dest, "copied"))

    entry = next(p for p in data["profiles"] if p.get("id") == profile_id)
    ready = all(s.action in ("present", "copied", "unchanged") for s in steps)
    moving = [s for s in steps if s.action != "unchanged"]
    validation = None
    if ready and moving:
        entry["provenance"] = {**entry.get("provenance", {}), "relocated_from": {
            f"{s.kind}_path": entry[f"{s.kind}_path"] for s in moving}}
        for s in moving:
            entry[f"{s.kind}_path"] = _repo_relative(s.destination, base_dir)
        try:
            candidate = ProfileRegistry.from_data(data, base_dir)
        except ProfileRegistryError as err:
            raise RegistrationError(str(err)) from err
        validation = candidate.validate(profile_id)
        if not validation.ok:
            raise RegistrationError("; ".join(validation.errors))
    elif write and moving:
        raise RegistrationError("copy the files first (copy=True / --copy); the registry was not changed")

    backup = _write_registry(registry_path, original, data, base_dir) if write and moving else None
    return RelocationResult(profile_id, tuple(steps), entry, validation, backup is not None, backup)
