"""Presentation-only rules and metadata for the desktop GUI.

Everything here decides how backend facts are *shown*; nothing here decides what can be loaded. Loadability
comes only from ``ProfileSummary.selectable`` and the live ``EngineStatus`` (docs/gui-backend-contract.md),
and ``config/voice_profiles.json`` stays the only profile registry.

The data files under ``gui/assets/`` are read from disk only (never from the network):

* ``voice_presentation.json``: Vietnamese tagline, monogram and color per registered profile id.
* ``character_catalog.json``: the VALORANT agent roster, shown as a collapsible "planned" collection. An entry
  can point at a registered profile (``linked_profile_id``); a link to an id the registry does not have is
  dropped, so an entry can never become loadable by a typo.
* ``asset_manifest.json``: source, license and SHA-256 of every bundled file; an image is shown only when its
  entry is redistributable.

Catalog entries may also show a personal picture from the user's own avatars folder (``user_avatars_dir``),
which is outside the package and never shipped.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from voice_changer.audio.devices import AudioDevice
from voice_changer.gui.theme import Colors
from voice_changer.service import AudioSettings, EngineState, ProfileSummary

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
VOICE_PRESENTATION_FILE = ASSETS_DIR / "voice_presentation.json"
CHARACTER_CATALOG_FILE = ASSETS_DIR / "character_catalog.json"
ICONS_DIR = ASSETS_DIR / "icons"

_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
DEFAULT_AVATAR_COLOR = "#334155"


# ----------------------------------------------------------------------------------------------- avatars
@dataclass(frozen=True)
class AvatarSpec:
    """An original monogram avatar: initials on a colored disc. ``image`` is an optional local PNG."""

    glyph: str
    color: str
    image: Optional[Path] = None


def monogram(name: str) -> str:
    """Two-letter initials: "Voice B (preview)" -> "VB", "KAY/O" -> "KO", "Jett" -> "JE"."""
    words = [w for w in re.split(r"[^0-9A-Za-zÀ-ỹ]+", name) if w]
    if not words:
        return "?"
    if len(words) >= 2:
        return (words[0][0] + words[1][0]).upper()
    return words[0][:2].upper()


def _valid_color(value: Any, fallback: str) -> str:
    return value if isinstance(value, str) and _HEX_COLOR.match(value) else fallback


def _local_asset(value: Any) -> Optional[Path]:
    """A PNG inside ASSETS_DIR that the asset manifest allows to be shipped, or None.

    URLs, absolute paths, paths leaving the assets folder and images without a redistributable manifest
    entry are refused, so a dropped-in picture of unknown origin falls back to the monogram.
    """
    if not isinstance(value, str) or not value or "://" in value:
        return None
    rel = Path(value)
    if rel.is_absolute() or rel.suffix.lower() != ".png":
        return None
    path = (ASSETS_DIR / rel).resolve()
    if ASSETS_DIR not in path.parents:
        return None
    record = approved_assets().get(path.relative_to(ASSETS_DIR).as_posix())
    if record is None:
        logger.warning("Image %s has no redistributable entry in %s; using the monogram", value,
                       ASSET_MANIFEST_FILE.name)
        return None
    return path


# ---------------------------------------------------------------------------------------- asset manifest
ASSET_MANIFEST_FILE = ASSETS_DIR / "asset_manifest.json"
# License states under which a bundled file may be shown and shipped; anything else is refused at runtime.
REDISTRIBUTABLE_LICENSES = ("project_original", "licensed", "permission_granted")
ASSET_LICENSES = REDISTRIBUTABLE_LICENSES + ("unverified", "not_redistributable")
_ASSET_TEXT_FIELDS = ("id", "path", "kind", "used_for", "source", "license", "attribution", "verification",
                      "fallback")


@dataclass(frozen=True)
class AssetRecord:
    """One bundled file: where it came from, whether it may be redistributed, and what replaces it if absent."""

    asset_id: str
    path: str  # POSIX path relative to ASSETS_DIR
    kind: str
    used_for: str
    source: str
    license: str
    attribution: str
    verification: str
    fallback: str
    sha256: Optional[str]

    @property
    def redistributable(self) -> bool:
        return self.license in REDISTRIBUTABLE_LICENSES


def _parse_asset_records(data: Optional[Mapping[str, Any]]) -> tuple[list[AssetRecord], list[str]]:
    records: list[AssetRecord] = []
    problems: list[str] = []
    if data is None:
        return records, ["asset manifest is missing or unreadable"]
    if data.get("schema_version") != 1:
        problems.append(f"unsupported asset manifest schema_version {data.get('schema_version')!r}")
    raw = data.get("assets")
    if not isinstance(raw, list):
        return records, problems + ["asset manifest has no 'assets' list"]
    for entry in raw:
        if not isinstance(entry, dict) or not all(isinstance(entry.get(k), str) for k in _ASSET_TEXT_FIELDS):
            problems.append(f"malformed asset entry {entry!r}")
            continue
        sha = entry.get("sha256")
        if sha is not None and not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
            problems.append(f"{entry['id']}: sha256 must be 64 lowercase hex characters or null")
            continue
        records.append(AssetRecord(entry["id"], entry["path"], entry["kind"], entry["used_for"], entry["source"],
                                   entry["license"], entry["attribution"], entry["verification"],
                                   entry["fallback"], sha))
    return records, problems


def load_asset_manifest(path: Path = ASSET_MANIFEST_FILE) -> dict[str, AssetRecord]:
    """Well-formed manifest records keyed by their path relative to the assets folder ({} if unreadable)."""
    records, _ = _parse_asset_records(_read_json(path))
    return {r.path: r for r in records}


@functools.lru_cache(maxsize=1)
def approved_assets() -> Mapping[str, AssetRecord]:
    """Records of the bundled manifest that may be shown and shipped (read once per process)."""
    return {p: r for p, r in load_asset_manifest().items() if r.redistributable}


def verify_asset_manifest(path: Path = ASSET_MANIFEST_FILE, assets_dir: Path = ASSETS_DIR) -> list[str]:
    """Every problem that would make the manifest an untrustworthy record of ``assets_dir`` (empty when sound).

    Checks: local relative paths only, unique ids and paths, known license values, attribution recorded for
    licensed or permitted third-party files, each listed file present with its pinned SHA-256, and every file in
    the folder listed (the manifest itself and Python bytecode caches excepted).
    """
    records, problems = _parse_asset_records(_read_json(path))
    assets_dir = assets_dir.resolve()
    seen_ids: set[str] = set()
    listed: set[str] = set()
    for r in records:
        if r.asset_id in seen_ids:
            problems.append(f"duplicate asset id {r.asset_id}")
        seen_ids.add(r.asset_id)
        rel = Path(r.path)
        target = (assets_dir / rel).resolve()
        if "://" in r.path or rel.is_absolute() or assets_dir not in target.parents:
            problems.append(f"{r.asset_id}: path must be relative and inside the assets folder: {r.path!r}")
            continue
        if r.path in listed:
            problems.append(f"{r.asset_id}: path listed twice: {r.path}")
        listed.add(r.path)
        if r.license not in ASSET_LICENSES:
            problems.append(f"{r.asset_id}: license must be one of {ASSET_LICENSES}, got {r.license!r}")
        if r.license in ("licensed", "permission_granted") and not r.attribution.strip():
            problems.append(f"{r.asset_id}: a {r.license} asset must record its attribution requirement")
        if not target.is_file():
            problems.append(f"{r.asset_id}: listed file is missing: {r.path}")
        elif r.sha256 is not None and hashlib.sha256(target.read_bytes()).hexdigest() != r.sha256:
            problems.append(f"{r.asset_id}: file does not match its recorded SHA-256 (replaced without "
                            f"updating the manifest?): {r.path}")
        elif r.sha256 is None and r.kind != "data":
            problems.append(f"{r.asset_id}: only 'data' files may omit sha256")
    manifest_rel = path.resolve().relative_to(assets_dir).as_posix() if assets_dir in path.resolve().parents else None
    for f in sorted(p for p in assets_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        rel = f.relative_to(assets_dir).as_posix()
        if rel != manifest_rel and rel not in listed:
            problems.append(f"file is not recorded in the asset manifest: {rel}")
    return problems


# ------------------------------------------------------------------------------------ voice presentation
@dataclass(frozen=True)
class VoicePresentation:
    tagline: str
    avatar: AvatarSpec


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.warning("Presentation file %s not found; using fallbacks.", path)
        return None
    except (OSError, ValueError) as err:
        logger.warning("Could not read presentation file %s: %s", path, err)
        return None
    if not isinstance(data, dict):
        logger.warning("Presentation file %s is not a JSON object; ignored.", path)
        return None
    return data


def load_voice_presentation(path: Path = VOICE_PRESENTATION_FILE) -> dict[str, VoicePresentation]:
    """Per-profile presentation, keyed by profile id. Bad entries are skipped; a missing file gives {}."""
    data = _read_json(path)
    voices = data.get("voices") if data else None
    if not isinstance(voices, dict):
        return {}
    result: dict[str, VoicePresentation] = {}
    for pid, entry in voices.items():
        if not isinstance(pid, str) or not isinstance(entry, dict):
            logger.warning("Skipping malformed voice presentation entry %r", pid)
            continue
        tagline = entry.get("tagline_vi") if isinstance(entry.get("tagline_vi"), str) else ""
        glyph = entry.get("monogram") if isinstance(entry.get("monogram"), str) and entry["monogram"] else ""
        result[pid] = VoicePresentation(
            tagline=tagline,
            avatar=AvatarSpec(glyph=glyph[:3], color=_valid_color(entry.get("color"), DEFAULT_AVATAR_COLOR),
                              image=_local_asset(entry.get("avatar"))),
        )
    return result


def language_label(language: str) -> str:
    return {"vi": "Tiếng Việt", "en": "Tiếng Anh"}.get(language, language or "Không rõ ngôn ngữ")


def voice_presentation_for(profile: ProfileSummary,
                           table: Mapping[str, VoicePresentation]) -> VoicePresentation:
    """The table entry for ``profile``, filled in from the profile itself where the table has nothing."""
    entry = table.get(profile.profile_id)
    glyph = entry.avatar.glyph if entry and entry.avatar.glyph else monogram(profile.display_name)
    color = entry.avatar.color if entry else DEFAULT_AVATAR_COLOR
    tagline = entry.tagline if entry and entry.tagline else f"Giọng {language_label(profile.language).lower()}"
    return VoicePresentation(tagline=tagline, avatar=AvatarSpec(glyph, color, entry.avatar.image if entry else None))


# ------------------------------------------------------------------------------------ personal avatars
_CATALOG_ID = re.compile(r"^([a-z0-9_-]+)\.([a-z0-9_-]+)$")


def user_avatars_dir() -> Path:
    """Per-user folder of personal character pictures (%LOCALAPPDATA%\\RealTimeVoiceChanger\\avatars).

    Pictures here belong to the user: they are never bundled, shipped or recorded in the asset manifest, so
    third-party artwork the user chooses to use stays on their machine (scripts/fetch_character_avatars.py).
    """
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "RealTimeVoiceChanger" / "avatars"
    return Path.home() / ".real_time_voice_changer" / "avatars"


def character_avatar_image(catalog_id: str, avatars_dir: Optional[Path] = None) -> Optional[Path]:
    """The personal PNG for a catalog entry ("valorant.jett" -> avatars/valorant/jett.png), or None."""
    match = _CATALOG_ID.match(catalog_id)
    if match is None:
        return None
    path = (avatars_dir or user_avatars_dir()) / match.group(1) / f"{match.group(2)}.png"
    return path if path.is_file() else None


# ------------------------------------------------------------------------------------ character catalog
@dataclass(frozen=True)
class CatalogRole:
    label: str
    color: str


@dataclass(frozen=True)
class CatalogEntry:
    catalog_id: str
    name: str
    role: str
    linked_profile_id: Optional[str]


@dataclass(frozen=True)
class CharacterCatalog:
    franchise: str = ""
    source_url: str = ""
    checked_on: str = ""
    disclaimer: str = ""
    roles: Mapping[str, CatalogRole] = field(default_factory=dict)
    entries: tuple[CatalogEntry, ...] = ()

    def role(self, role_id: str) -> CatalogRole:
        return self.roles.get(role_id, CatalogRole(role_id, DEFAULT_AVATAR_COLOR))


def load_character_catalog(path: Path = CHARACTER_CATALOG_FILE) -> CharacterCatalog:
    """Read the catalog; malformed or duplicate entries are skipped with a warning. Missing file: empty catalog."""
    data = _read_json(path)
    if not data:
        return CharacterCatalog()
    roles: dict[str, CatalogRole] = {}
    raw_roles = data.get("roles") if isinstance(data.get("roles"), dict) else {}
    for rid, r in raw_roles.items():
        if isinstance(rid, str) and isinstance(r, dict) and isinstance(r.get("label_vi"), str):
            roles[rid] = CatalogRole(r["label_vi"], _valid_color(r.get("color"), DEFAULT_AVATAR_COLOR))
    entries: list[CatalogEntry] = []
    seen: set[str] = set()
    raw_entries = data.get("entries") if isinstance(data.get("entries"), list) else []
    for e in raw_entries:
        if not isinstance(e, dict):
            logger.warning("Skipping malformed catalog entry %r", e)
            continue
        cid, name, role, link = e.get("id"), e.get("name"), e.get("role"), e.get("linked_profile_id")
        if not (isinstance(cid, str) and cid and isinstance(name, str) and name and role in roles):
            logger.warning("Skipping catalog entry with missing id/name or unknown role: %r", e)
            continue
        if link is not None and not (isinstance(link, str) and link):
            logger.warning("Catalog entry %s has an invalid linked_profile_id %r; treated as unlinked", cid, link)
            link = None
        if cid in seen:
            logger.warning("Skipping duplicate catalog entry %s", cid)
            continue
        seen.add(cid)
        entries.append(CatalogEntry(cid, name, role, link))
    text = {k: data.get(k) if isinstance(data.get(k), str) else "" for k in
            ("franchise", "roster_source", "roster_checked_on", "disclaimer_vi")}
    return CharacterCatalog(text["franchise"], text["roster_source"], text["roster_checked_on"],
                            text["disclaimer_vi"], roles, tuple(entries))


@dataclass(frozen=True)
class ResolvedCatalogEntry:
    entry: CatalogEntry
    profile: Optional[ProfileSummary]  # the registered profile the entry is linked to, if that id exists


def resolve_catalog(catalog: CharacterCatalog,
                    profiles: Iterable[ProfileSummary]) -> tuple[list[ResolvedCatalogEntry], list[str]]:
    """Attach each entry to its registered profile. Returns (entries, rejected link ids).

    A ``linked_profile_id`` that is not a registered profile id is rejected (logged and returned) and the
    entry is shown as having no voice preset; it is never offered for loading.
    """
    by_id = {p.profile_id: p for p in profiles}
    resolved: list[ResolvedCatalogEntry] = []
    rejected: list[str] = []
    for entry in catalog.entries:
        profile = None
        if entry.linked_profile_id is not None:
            profile = by_id.get(entry.linked_profile_id)
            if profile is None:
                logger.warning("Catalog entry %s links to unknown profile %r; shown without a preset",
                               entry.catalog_id, entry.linked_profile_id)
                rejected.append(entry.linked_profile_id)
        resolved.append(ResolvedCatalogEntry(entry, profile))
    return resolved, rejected


# ------------------------------------------------------------------------------- availability and rights
@dataclass(frozen=True)
class Availability:
    key: str  # active, withdrawn, planned, files_missing, preview, prototype, approved, unknown, no_preset
    label: str
    bg: str
    fg: str
    loadable: bool  # the GUI may offer "use"/"load"; only ever True for ProfileSummary.selectable profiles


NO_PRESET = Availability("no_preset", "Chưa có preset giọng", Colors.BG_ELEVATED, Colors.TEXT_MUTED, False)


def _is_planned(profile: Any) -> bool:
    return getattr(profile, "status", "") == "planned" or getattr(profile, "readiness", None) == "no_model"


def availability(profile: Optional[ProfileSummary], active_id: Optional[str] = None) -> Availability:
    """How a profile's availability is shown, in the Library and in Studio alike.

    ``active_id`` must come from the live EngineStatus (loaded model), never from ProfileSummary.is_active,
    which is a snapshot. Withdrawn, planned and missing-file profiles differ in color as well as wording.
    """
    if profile is None:
        return NO_PRESET
    selectable = bool(getattr(profile, "selectable", False))
    if getattr(profile, "withdrawn_reason", ""):
        return Availability("withdrawn", "⛔ Đã thu hồi", Colors.DANGER_BG, Colors.DANGER_TEXT, False)
    if _is_planned(profile):
        return Availability("planned", "🔒 Chưa có mô hình", Colors.BG_ELEVATED, Colors.TEXT_MUTED, False)
    if not selectable:
        return Availability("files_missing", "⚠ Thiếu tệp mô hình", Colors.BG_ELEVATED, Colors.DANGER_TEXT, False)
    if active_id is not None and profile.profile_id == active_id:
        return Availability("active", "● Đang dùng", Colors.SUCCESS_BG, Colors.SUCCESS_TEXT, True)
    status = getattr(profile, "status", "")
    if status == "preview":
        return Availability("preview", "Thử nghiệm", Colors.WARNING_BG, Colors.WARNING_TEXT, True)
    if status == "prototype":
        return Availability("prototype", "Bản mẫu kỹ thuật", Colors.BG_HOVER, Colors.TEXT_SECONDARY, True)
    if status == "approved":
        return Availability("approved", "✓ Đã duyệt", Colors.SUCCESS_BG, Colors.SUCCESS_TEXT, True)
    return Availability("unknown", "Chưa rõ trạng thái", Colors.BG_ELEVATED, Colors.TEXT_MUTED, True)


RIGHTS_LABELS: dict[str, str] = {
    "unverified": "Chỉ dùng cá nhân · bản quyền chưa xác minh",
    "owner_recorded": "Giọng do chủ dự án ghi âm",
    "consented": "Có sự đồng ý của người nói",
    "licensed": "Có giấy phép",
}


def rights_label(profile: Any) -> str:
    """User-facing data-rights note; empty for "unspecified" (the registry makes no claim either way)."""
    return RIGHTS_LABELS.get(getattr(profile, "data_rights", ""), "")


def readiness_text(profile: Optional[ProfileSummary], active_id: Optional[str] = None) -> str:
    """One truthful sentence on whether the profile can be used.

    File presence is not validation: select_profile() checks SHA-256 pins and compatibility when it loads.
    Only a profile the backend reports as loaded is described as checked.
    """
    if profile is None:
        return "Chưa chọn giọng."
    av = availability(profile, active_id)
    if av.key == "withdrawn":
        return "Giọng này đã bị thu hồi và không thể nạp."
    if av.key == "planned":
        return "Chưa thể chuyển giọng: giọng này chưa có mô hình."
    if av.key == "files_missing":
        return "Chưa thể chuyển giọng: thiếu tệp mô hình hoặc chỉ mục trên máy này."
    if av.key == "active":
        return "Đã nạp và kiểm tra tệp thành công."
    return "Đã có tệp trên máy · sẽ được kiểm tra đầy đủ khi nạp."


# ------------------------------------------------------------------------------------- primary action
@dataclass(frozen=True)
class PrimaryAction:
    kind: str  # load, start, stop, retry, busy, none
    text: str
    enabled: bool
    style: str


_BUSY_TEXT = {
    EngineState.LOADING: "Đang nạp giọng…",
    EngineState.STARTING: "Đang mở thiết bị âm thanh…",
    EngineState.STOPPING: "Đang dừng…",
}


def primary_action(state: Optional[EngineState], selected: Optional[ProfileSummary],
                   loaded_id: Optional[str], loading: bool = False) -> PrimaryAction:
    """The single Studio action for the real engine state and the selected voice.

    ``selected`` None means "nothing chosen beyond what is loaded": READY then still offers Start.
    A voice that is selected but not loaded offers "Nạp giọng"; choosing a voice never loads it by itself.
    """
    if loading or state in _BUSY_TEXT:
        return PrimaryAction("busy", _BUSY_TEXT.get(state, _BUSY_TEXT[EngineState.LOADING]), False,
                             "Primary")
    if state == EngineState.RUNNING:
        return PrimaryAction("stop", "■  Dừng", True, "Danger")
    wants_other = (selected is not None and selected.profile_id != loaded_id)
    if wants_other and not availability(selected).loadable:
        return PrimaryAction("none", "Không thể nạp giọng này", False, "Primary")
    if wants_other:
        return PrimaryAction("load", "Nạp giọng", True, "Primary")
    if state == EngineState.ERROR:
        return PrimaryAction("retry", "↻  Thử lại", True, "Success")
    if state == EngineState.READY and loaded_id is not None:
        return PrimaryAction("start", "▶  Bắt đầu chuyển giọng", True, "Success")
    if state == EngineState.READY and selected is None:
        return PrimaryAction("start", "▶  Bắt đầu chuyển giọng", True, "Success")
    return PrimaryAction("none", "Chọn một giọng để bắt đầu", False, "Primary")


# ------------------------------------------------------------------------------------------- routing
@dataclass(frozen=True)
class RouteSummary:
    text: str  # "<mic> → <output>"
    hint: str  # what the selection means for other applications
    to_virtual_cable: bool


def short_device_name(device: Optional[AudioDevice], limit: int = 34) -> str:
    if device is None:
        return "Chưa chọn"
    name = device.name
    return name if len(name) <= limit else name[: limit - 1] + "…"


def route_summary(inp: Optional[AudioDevice], out: Optional[AudioDevice]) -> RouteSummary:
    """Describe the route from the devices actually selected; never assume a virtual cable."""
    text = f"🎙  {short_device_name(inp)}   →   🔊  {short_device_name(out)}"
    if out is None:
        return RouteSummary(text, "Chưa chọn ngõ ra. Chọn thiết bị trong “Âm thanh & Kết nối”.", False)
    if out.is_virtual_cable:
        return RouteSummary(
            text,
            "Ngõ ra là cáp âm thanh ảo. Trong ứng dụng khác (Discord, OBS, game có voice chat), hãy chọn micro "
            "là đầu thu của cáp này (với VB-Cable: “CABLE Output”).",
            True,
        )
    return RouteSummary(
        text,
        "Đang phát ra thiết bị này để bạn nghe thử. Muốn dùng giọng trong ứng dụng khác, hãy chọn ngõ ra là "
        "một cáp âm thanh ảo.",
        False,
    )


# ------------------------------------------------------------------------------------------ telemetry
def telemetry_label(metrics: Optional[Mapping[str, Any]]) -> str:
    """Whether metrics describe the session running now or the last one that ended."""
    if not metrics:
        return "Chưa có dữ liệu: chưa có phiên chuyển giọng nào."
    sid = metrics.get("session_id")
    if metrics.get("live") is True:
        return f"Trực tiếp · phiên #{sid}" if sid is not None else "Trực tiếp"
    return f"Phiên trước (#{sid}) · đã kết thúc, không còn cập nhật" if sid is not None else "Phiên trước · đã kết thúc"


def performance_summary(metrics: Optional[Mapping[str, Any]]) -> Optional[tuple[str, str]]:
    """(text, color) for Studio while measurements are live; None when there is nothing live to show."""
    if not metrics or metrics.get("live") is not True or not metrics.get("chunks_converted"):
        return None
    rtf = metrics.get("rtf")
    underflows = metrics.get("steady_underflows", 0) or 0
    dropped = metrics.get("chunks_dropped", 0) or 0
    if (isinstance(rtf, (int, float)) and rtf > 1.0) or underflows > 0 or dropped > 0:
        return "Hiệu năng: có gián đoạn — xem “Nâng cao”", Colors.DANGER_TEXT
    if isinstance(rtf, (int, float)) and rtf >= 0.85:
        return "Hiệu năng: sát giới hạn", Colors.WARNING_TEXT
    return "Hiệu năng: ổn định", Colors.SUCCESS_TEXT


def stream_parameters(audio: Optional[AudioSettings] = None) -> list[tuple[str, str]]:
    """The buffering a session uses, read from AudioSettings (the session's own values, or the defaults)."""
    a = audio if audio is not None else AudioSettings()
    return [
        ("Độ dài chunk", f"{a.chunk_duration_s * 1000:.0f} ms"),
        ("Bộ đệm khởi động", f"{a.prebuffer_s * 1000:.0f} ms"),
        ("Hàng đợi tối đa", f"{a.max_queue_size} chunk"),
        ("Ngữ cảnh", f"{a.context_s * 1000:.0f} ms"),
        ("Crossfade", f"{a.crossfade_s * 1000:.0f} ms"),
        ("Tìm SOLA", f"{a.sola_search_s * 1000:.0f} ms"),
    ]


# ------------------------------------------------------------------------------------------ app info
APP_NAME = "RealTimeVoiceChanger"


def app_version() -> str:
    """The running package's version: ``voice_changer.__version__``, which pyproject.toml reads at build time.

    It is the same in a source run, an installed wheel and a frozen build, where installed metadata or the
    repository's pyproject.toml may be absent or describe a different copy.
    """
    from voice_changer import __version__

    return __version__


def icon_files(sizes: Sequence[int] = (16, 32, 48, 256)) -> list[Path]:
    """Existing app-icon PNGs for ``sizes`` (missing files are left out)."""
    return [p for p in (ICONS_DIR / f"app_icon_{s}.png" for s in sizes) if p.is_file()]


def windows_icon_file() -> Optional[Path]:
    path = ICONS_DIR / "app_icon.ico"
    return path if sys.platform == "win32" and path.is_file() else None
