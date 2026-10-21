"""Voice profile registry.

A voice profile binds one RVC checkpoint to its own retrieval index and default conversion settings.
Profiles are declared in a JSON file (default: config/voice_profiles.json). Each declared model and
index can be pinned by SHA-256, so a profile can only be loaded with the exact files it was
registered with: a speaker's index can never silently be paired with another speaker's model.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from voice_changer.conversion.base import ConversionConfig, validate_conversion_config

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "config" / "voice_profiles.json"

# Quality status. Only the project owner may set "approved", after listening.
PROFILE_STATUSES = ("prototype", "preview", "approved")
SCHEMA_VERSION = 1


class ProfileRegistryError(ValueError):
    """The registry file is missing, malformed, or declares an invalid profile."""


@dataclass(frozen=True)
class VoiceProfile:
    """One selectable voice: model checkpoint, matching index and intended settings."""

    profile_id: str
    display_name: str
    description: str
    model_path: Path
    index_path: Optional[Path]
    defaults: ConversionConfig
    status: str
    language: str = "vi"
    model_sha256: Optional[str] = None
    index_sha256: Optional[str] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class ProfileValidation:
    """Result of checking a profile's assets on disk. `ok` means it can be loaded."""

    profile_id: str
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    model_version: Optional[str] = None
    model_sample_rate: Optional[int] = None
    index_dim: Optional[int] = None
    index_vectors: Optional[int] = None


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def _parse_profile(raw: Mapping[str, Any], base: Path) -> VoiceProfile:
    pid = raw.get("id")
    if not isinstance(pid, str) or not pid or not pid.replace("_", "").replace("-", "").isalnum():
        raise ProfileRegistryError(f"profile id must be a non-empty [A-Za-z0-9_-] string, got {pid!r}")
    for key in ("display_name", "model_path", "status"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ProfileRegistryError(f"profile '{pid}': '{key}' must be a non-empty string")
    if raw["status"] not in PROFILE_STATUSES:
        raise ProfileRegistryError(f"profile '{pid}': status must be one of {PROFILE_STATUSES}")
    defaults_raw = raw.get("defaults", {})
    try:
        defaults = ConversionConfig(**defaults_raw)
    except TypeError as err:
        raise ProfileRegistryError(f"profile '{pid}': invalid defaults: {err}") from err
    problems = validate_conversion_config(defaults)
    if problems:
        raise ProfileRegistryError(f"profile '{pid}': invalid defaults: {'; '.join(problems)}")
    index = raw.get("index_path")
    return VoiceProfile(
        profile_id=pid,
        display_name=raw["display_name"],
        description=str(raw.get("description", "")),
        model_path=_resolve(base, raw["model_path"]),
        index_path=_resolve(base, index) if index else None,
        defaults=defaults,
        status=raw["status"],
        language=str(raw.get("language", "vi")),
        model_sha256=raw.get("model_sha256") or None,
        index_sha256=raw.get("index_sha256") or None,
        provenance=dict(raw.get("provenance", {})),
        notes=str(raw.get("notes", "")),
    )


class ProfileRegistry:
    """Loads, lists and validates voice profiles. Thread-safe for concurrent reads and validation."""

    def __init__(self, profiles: list[VoiceProfile], default_profile_id: Optional[str] = None) -> None:
        ids = [p.profile_id for p in profiles]
        if len(set(ids)) != len(ids):
            raise ProfileRegistryError(f"duplicate profile ids: {ids}")
        if default_profile_id is not None and default_profile_id not in ids:
            raise ProfileRegistryError(f"default_profile '{default_profile_id}' is not a registered profile")
        self._profiles = {p.profile_id: p for p in profiles}
        self._default = default_profile_id
        self._hash_cache: dict[tuple[str, int, int], str] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_file(cls, path: Path = DEFAULT_REGISTRY_PATH, base_dir: Optional[Path] = None) -> "ProfileRegistry":
        """Load a registry JSON file. Relative asset paths resolve against `base_dir` (default: repo root)."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError as err:
            raise ProfileRegistryError(f"voice profile registry not found: {path}") from err
        except json.JSONDecodeError as err:
            raise ProfileRegistryError(f"voice profile registry is not valid JSON: {path}: {err}") from err
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ProfileRegistryError(f"unsupported registry schema_version {data.get('schema_version')!r}")
        base = base_dir or REPO_ROOT
        profiles = [_parse_profile(p, base) for p in data.get("profiles", [])]
        return cls(profiles, data.get("default_profile"))

    @property
    def default_profile_id(self) -> Optional[str]:
        return self._default

    def list_profiles(self) -> list[VoiceProfile]:
        return list(self._profiles.values())

    def get(self, profile_id: str) -> VoiceProfile:
        try:
            return self._profiles[profile_id]
        except KeyError:
            raise KeyError(f"unknown voice profile '{profile_id}'. Registered: {sorted(self._profiles)}") from None

    def _sha(self, path: Path) -> str:
        from voice_changer.conversion.rvc.assets import sha256_file

        st = path.stat()
        key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
        with self._lock:
            cached = self._hash_cache.get(key)
        if cached is None:
            cached = sha256_file(path)
            with self._lock:
                self._hash_cache[key] = cached
        return cached

    def validate(self, profile_id: str) -> ProfileValidation:
        """Check that the profile's files exist, match their pinned hashes and fit together.

        Reads the checkpoint and index on the CPU only; nothing is loaded onto the GPU.
        """
        from voice_changer.conversion.rvc.assets import check_pair, inspect_checkpoint, inspect_index

        profile = self.get(profile_id)
        errors: list[str] = []
        warnings: list[str] = []
        ckpt = idx = None

        if not profile.model_path.is_file():
            errors.append(f"model file not found: {profile.model_path}")
        else:
            if profile.model_sha256 and self._sha(profile.model_path) != profile.model_sha256:
                errors.append(f"model file does not match its registered SHA-256: {profile.model_path}")
            elif not profile.model_sha256:
                warnings.append("model SHA-256 is not pinned; file identity is unverified")
            try:
                ckpt = inspect_checkpoint(profile.model_path)
            except ValueError as err:
                errors.append(f"model file {profile.model_path}: {err}")

        if profile.index_path is None:
            warnings.append("no retrieval index: index_rate has no effect for this profile")
        elif not profile.index_path.is_file():
            errors.append(f"index file not found: {profile.index_path}")
        else:
            if profile.index_sha256 and self._sha(profile.index_path) != profile.index_sha256:
                errors.append(f"index file does not match its registered SHA-256: {profile.index_path}")
            elif not profile.index_sha256:
                warnings.append("index SHA-256 is not pinned; model/index pairing is unverified")
            try:
                idx = inspect_index(profile.index_path)
            except ValueError as err:
                errors.append(f"index file {profile.index_path}: {err}")

        if ckpt is not None:
            errors.extend(check_pair(ckpt, idx))
            if ckpt.f0 != 1:
                errors.append("checkpoint has no pitch guidance (f0=0); pitch shifting is not supported")
        if profile.status != "approved":
            warnings.append(f"quality status is '{profile.status}', not approved by the project owner")

        return ProfileValidation(
            profile_id=profile_id,
            ok=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
            model_version=ckpt.version if ckpt else None,
            model_sample_rate=ckpt.sample_rate if ckpt else None,
            index_dim=idx.dim if idx else None,
            index_vectors=idx.ntotal if idx else None,
        )
